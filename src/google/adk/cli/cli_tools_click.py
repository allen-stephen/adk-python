# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from typing import TYPE_CHECKING

import click
from click.core import ParameterSource
from fastapi import FastAPI
import uvicorn

from .. import version
from ..agents.run_config import StreamingMode
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..features import FeatureName
from ..features import override_feature_enabled
from .cli import run_cli
from .utils import envs
from .utils import logs

if TYPE_CHECKING:
  from ..evaluation.eval_metrics import EvalMetric
  from ..evaluation.eval_result import EvalCaseResult
  from ..evaluation.eval_set_results_manager import EvalSetResultsManager

LOG_LEVELS = click.Choice(
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    case_sensitive=False,
)

# Metrics that compare the agent's output against a golden/expected response.
# A persona (live audio-to-audio) run synthesizes the conversation fresh and has
# no golden reference, so these metrics cannot be scored and would hard-fail at
# scoring time. They are filtered out (with a warning) for live runs.
_REFERENCE_BASED_METRIC_NAMES = frozenset({
    "response_match_score",
    "response_evaluation_score",
    "tool_trajectory_avg_score",
    "final_response_match_v2",
})

# The managed (Gen AI Eval Service) multi-turn metrics offered for audio live
# runs via --managed_metrics. These are reference-free and valid for live.
_MANAGED_LIVE_METRIC_DEFAULTS = (
    ("multi_turn_task_success_v1", 0.7),
    ("multi_turn_trajectory_quality_v1", 0.7),
    ("safety_v1", 0.8),
)

# Native-audio Live models typically respond in 2-5s, so use a realistic default
# latency threshold. Kept in sync with the web UI's LOCAL_LIVE_METRICS.
_LIVE_LATENCY_THRESHOLD_SECONDS = 5.0

# A comfortable conversational speaking rate is ~2-3 words/second; pass at or
# below this upper bound.
_LIVE_SPEAKING_RATE_THRESHOLD_WPS = 3.5

_EVAL_RESULT_FILE_EXTENSION = ".evalset_result.json"


def _print_eval_criteria(eval_metrics: list[EvalMetric]) -> None:
  """Prints a concise list of the metrics and thresholds being used."""
  click.echo("Evaluation criteria:")
  if not eval_metrics:
    click.echo("  (none)")
    return
  for metric in eval_metrics:
    click.echo(f"  {metric.metric_name} (threshold: {metric.threshold})")


def _print_eval_run_summary(eval_results: list[EvalCaseResult]) -> None:
  """Prints the per-eval-set and overall pass/fail/not-evaluated summary.

  PASSED, FAILED and NOT_EVALUATED are reported as distinct buckets (rather than
  collapsing the latter two into a single "failed" count) so a coding agent or
  developer can distinguish a real regression from a metric that could not be
  scored.
  """
  from ..evaluation.evaluator import EvalStatus

  # eval_set_id -> [passed, failed, not_evaluated]
  eval_run_summary: dict[str, list[int]] = {}
  for eval_result in eval_results:
    counts = eval_run_summary.setdefault(eval_result.eval_set_id, [0, 0, 0])
    if eval_result.final_eval_status == EvalStatus.PASSED:
      counts[0] += 1
    elif eval_result.final_eval_status == EvalStatus.FAILED:
      counts[1] += 1
    else:
      counts[2] += 1

  click.echo(
      "*********************************************************************"
  )
  click.echo("Eval Run Summary")

  total_passed = total_failed = total_not_evaluated = 0
  for eval_set_id, counts in eval_run_summary.items():
    passed, failed, not_evaluated = counts
    total_passed += passed
    total_failed += failed
    total_not_evaluated += not_evaluated
    click.echo(f"{eval_set_id}:")
    click.secho(f"  Tests passed: {passed}", fg="green")
    click.secho(f"  Tests failed: {failed}", fg="red" if failed else None)
    if not_evaluated:
      click.secho(f"  Tests not evaluated: {not_evaluated}", fg="yellow")

  click.echo(
      "Total: "
      f"{total_passed} passed, {total_failed} failed, "
      f"{total_not_evaluated} not evaluated"
  )


def _report_written_result_files(
    *,
    eval_set_results_manager: EvalSetResultsManager,
    app_name: str,
    pre_run_result_ids: set[str],
) -> None:
  """Echoes the result files this run wrote, for easy discovery."""
  try:
    post_run_result_ids = set(
        eval_set_results_manager.list_eval_set_results(app_name)
    )
  except Exception:  # pylint: disable=broad-except
    return
  new_result_ids = sorted(post_run_result_ids - pre_run_result_ids)
  if not new_result_ids:
    return
  from ..evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
  from ..evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager

  click.echo("Result files:")
  for result_id in new_result_ids:
    location = _eval_result_location(
        eval_set_results_manager=eval_set_results_manager,
        app_name=app_name,
        result_id=result_id,
        local_cls=LocalEvalSetResultsManager,
        gcs_cls=GcsEvalSetResultsManager,
    )
    click.echo(f"  {location}")


def _eval_result_location(
    *,
    eval_set_results_manager: EvalSetResultsManager,
    app_name: str,
    result_id: str,
    local_cls: type,
    gcs_cls: type,
) -> str:
  """Returns a human-readable location for a persisted eval set result."""
  filename = result_id + _EVAL_RESULT_FILE_EXTENSION
  if isinstance(eval_set_results_manager, local_cls):
    return os.path.join(
        eval_set_results_manager._get_eval_history_dir(app_name), filename
    )
  if isinstance(eval_set_results_manager, gcs_cls):
    bucket = eval_set_results_manager.bucket_name
    return f"gs://{bucket}/{app_name}/evals/eval_history/{filename}"
  return result_id


def _write_eval_output_file(
    *, output_file: str, app_name: str, eval_results: list[EvalCaseResult]
) -> None:
  """Writes an aggregated JSON envelope of all eval results to output_file.

  The envelope has a top-level ``summary`` (overall and per-eval-set pass/fail/
  not-evaluated counts) plus the full ``eval_set_results`` list, so programmatic
  consumers can branch on the summary without re-aggregating per-case results.
  """
  from ..evaluation._eval_set_results_manager_utils import create_eval_set_result
  from ..evaluation.evaluator import EvalStatus

  # Group cases back into EvalSetResult objects, preserving the on-disk schema.
  cases_by_set: dict[str, list[EvalCaseResult]] = {}
  for eval_result in eval_results:
    cases_by_set.setdefault(eval_result.eval_set_id, []).append(eval_result)

  eval_set_results = []
  per_set_summary = {}
  total_passed = total_failed = total_not_evaluated = 0
  for eval_set_id, cases in cases_by_set.items():
    eval_set_results.append(
        create_eval_set_result(
            app_name=app_name,
            eval_set_id=eval_set_id,
            eval_case_results=cases,
        )
    )
    passed = sum(1 for c in cases if c.final_eval_status == EvalStatus.PASSED)
    failed = sum(1 for c in cases if c.final_eval_status == EvalStatus.FAILED)
    not_evaluated = len(cases) - passed - failed
    per_set_summary[eval_set_id] = {
        "passed": passed,
        "failed": failed,
        "not_evaluated": not_evaluated,
    }
    total_passed += passed
    total_failed += failed
    total_not_evaluated += not_evaluated

  envelope = {
      "summary": {
          "passed": total_passed,
          "failed": total_failed,
          "not_evaluated": total_not_evaluated,
          "per_eval_set": per_set_summary,
      },
      "eval_set_results": [
          result.model_dump(mode="json", by_alias=True)
          for result in eval_set_results
      ],
  }
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(envelope, f, indent=2)


def _make_watch_progress_callback():
  """Builds an async progress callback that streams a live run to the console.

  Audio transports emit a small set of progress events as a conversation
  unfolds; this prints them as an alternating, color-coded transcript so the
  user can inspect both the simulated-user input and the agent-under-test output
  in real time. Returns a coroutine callback suitable for
  `LocalEvalService(audio_progress_callback=...)`.
  """

  async def _on_progress(event: dict) -> None:
    event_type = event.get("type")
    if event_type == "turn_started":
      click.secho(f"\n── turn {event.get('turn_index')} ──", fg="white")
    elif event_type == "transcript_update":
      speaker = event.get("speaker")
      text = event.get("text", "")
      if speaker == "persona":
        click.secho(f"  user  ▶ {text}", fg="green")
      else:
        click.secho(f"  agent ◀ {text}", fg="blue")
    elif event_type == "barge_in":
      listen = event.get("listen_seconds")
      detail = f" after {listen}s" if listen is not None else ""
      click.secho(f"  ⚡ user barged in{detail} (agent cut off)", fg="magenta")
    elif event_type == "conversation_complete":
      click.secho(
          f"\n✓ conversation complete: {event.get('turns')} turn(s),"
          f" reason={event.get('termination_reason')}\n",
          fg="cyan",
      )

  return _on_progress


def _resolve_live_eval_metrics(
    eval_metrics: list[EvalMetric],
    *,
    config_file_path: str | None,
    managed_metrics: bool,
) -> list[EvalMetric]:
  """Returns the metrics to use for an audio (live) eval run.

  An audio run scores a freshly generated conversation and has no golden
  references, so reference-based metrics do not apply:

  - When no config file was supplied, we default to the local acoustic metrics
    and add the managed multi-turn metrics only when `managed_metrics` is set.
  - When a config file was supplied, we honor it but drop any reference-based
    metrics (with a warning) so the run does not hard-fail at scoring time.

  Args:
    eval_metrics: Metrics resolved from the config (or default config).
    config_file_path: The explicit eval config path, if any.
    managed_metrics: Whether to add the managed multi-turn metrics (only used
      when no config file is supplied).

  Returns:
    The metrics to run for the live audio eval.
  """
  from ..evaluation.eval_metrics import EvalMetric

  if not config_file_path:
    resolved = [
        EvalMetric(
            metric_name="response_latency_v1",
            threshold=_LIVE_LATENCY_THRESHOLD_SECONDS,
        ),
        EvalMetric(
            metric_name="speaking_rate_v1",
            threshold=_LIVE_SPEAKING_RATE_THRESHOLD_WPS,
        ),
    ]
    if managed_metrics:
      resolved.extend(
          EvalMetric(metric_name=name, threshold=threshold)
          for name, threshold in _MANAGED_LIVE_METRIC_DEFAULTS
      )
    metric_names = ", ".join(m.metric_name for m in resolved)
    print(f"Audio live run: using reference-free metrics [{metric_names}].")
    return resolved

  # A config file was supplied: honor it, but drop reference-based metrics that
  # cannot be scored without a golden reference.
  resolved = []
  for metric in eval_metrics:
    if metric.metric_name in _REFERENCE_BASED_METRIC_NAMES:
      print(
          f"Skipping metric '{metric.metric_name}' for the audio live run: it"
          " requires a golden/expected response, which a live run does not"
          " have."
      )
      continue
    resolved.append(metric)
  return resolved


def _logging_options():
  """Decorator to add logging options to click commands."""

  def decorator(func):
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        show_default=True,
        default=False,
        help="Enable verbose (DEBUG) logging. Shortcut for --log_level DEBUG.",
    )
    @click.option(
        "--log_level",
        type=LOG_LEVELS,
        default="INFO",
        help="Optional. Set the logging level",
    )
    @functools.wraps(func)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
      # If verbose flag is set and log level is not set, set log level to DEBUG.
      log_level_source = ctx.get_parameter_source("log_level")
      if (
          kwargs.pop("verbose", False)
          and log_level_source == ParameterSource.DEFAULT
      ):
        kwargs["log_level"] = "DEBUG"
      return func(*args, **kwargs)

    return wrapper

  return decorator


def _apply_feature_overrides(
    *,
    enable_features: tuple[str, ...] = (),
    disable_features: tuple[str, ...] = (),
) -> None:
  """Apply feature overrides from CLI flags.

  Args:
    enable_features: Tuple of feature names to enable.
    disable_features: Tuple of feature names to disable.
  """
  feature_overrides: dict[str, bool] = {}

  for features_str in enable_features:
    for feature_name_str in features_str.split(","):
      feature_name_str = feature_name_str.strip()
      if feature_name_str:
        feature_overrides[feature_name_str] = True

  for features_str in disable_features:
    for feature_name_str in features_str.split(","):
      feature_name_str = feature_name_str.strip()
      if feature_name_str:
        feature_overrides[feature_name_str] = False

  # Apply all overrides
  for feature_name_str, enabled in feature_overrides.items():
    try:
      feature_name = FeatureName(feature_name_str)
      override_feature_enabled(feature_name, enabled)
    except ValueError:
      valid_names = ", ".join(f.value for f in FeatureName)
      click.secho(
          f"WARNING: Unknown feature name '{feature_name_str}'. "
          f"Valid names are: {valid_names}",
          fg="yellow",
          err=True,
      )


def feature_options():
  """Decorator to add feature override options to click commands."""

  def decorator(func):
    @click.option(
        "--enable_features",
        help=(
            "Optional. Comma-separated list of feature names to enable. "
            "This provides an alternative to environment variables for "
            "enabling experimental features. Example: "
            "--enable_features=JSON_SCHEMA_FOR_FUNC_DECL,PROGRESSIVE_SSE_STREAMING"
        ),
        multiple=True,
    )
    @click.option(
        "--disable_features",
        help=(
            "Optional. Comma-separated list of feature names to disable. "
            "This provides an alternative to environment variables for "
            "disabling features. Example: "
            "--disable_features=JSON_SCHEMA_FOR_FUNC_DECL,PROGRESSIVE_SSE_STREAMING"
        ),
        multiple=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      enable_features = kwargs.pop("enable_features", ())
      disable_features = kwargs.pop("disable_features", ())
      if enable_features or disable_features:
        _apply_feature_overrides(
            enable_features=enable_features,
            disable_features=disable_features,
        )
      return func(*args, **kwargs)

    return wrapper

  return decorator


class HelpfulCommand(click.Command):
  """Command that shows full help on error instead of just the error message.

  A custom Click Command class that overrides the default error handling
  behavior to display the full help text when a required argument is missing,
  followed by the error message. This provides users with better context
  about command usage without needing to run a separate --help command.

  Args:
    *args: Variable length argument list to pass to the parent class.
    **kwargs: Arbitrary keyword arguments to pass to the parent class.

  Returns:
    None. Inherits behavior from the parent Click Command class.

  Returns:
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @staticmethod
  def _format_missing_arg_error(click_exception):
    """Format the missing argument error with uppercase parameter name.

    Args:
      click_exception: The MissingParameter exception from Click.

    Returns:
      str: Formatted error message with uppercase parameter name.
    """
    name = click_exception.param.name
    return f"Missing required argument: {name.upper()}"

  def parse_args(self, ctx, args):
    """Override the parse_args method to show help text on error.

    Args:
      ctx: Click context object for the current command.
      args: List of command-line arguments to parse.

    Returns:
      The parsed arguments as returned by the parent class's parse_args method.

    Raises:
      click.MissingParameter: When a required parameter is missing, but this
        is caught and handled by displaying the help text before exiting.
    """
    try:
      return super().parse_args(ctx, args)
    except click.MissingParameter as exc:
      error_message = self._format_missing_arg_error(exc)

      click.echo(ctx.get_help())
      click.secho(f"\nError: {error_message}", fg="red", err=True)
      ctx.exit(2)


logger = logging.getLogger("google_adk." + __name__)


_ADK_WEB_WARNING = (
    "ADK Web is for development purposes. It has access to all data and"
    " should not be used in production."
)


def _warn_if_with_ui(with_ui: bool) -> None:
  """Warn when deploying with the developer UI enabled."""
  if with_ui:
    click.secho(f"WARNING: {_ADK_WEB_WARNING}", fg="yellow", err=True)


@click.group(context_settings={"max_content_width": 240})
@click.version_option(version.__version__)
def main():
  """Agent Development Kit CLI tools."""
  pass


@main.group()
def deploy():
  """Deploys agent to hosted environments."""
  pass


@main.group()
def conformance():
  """Conformance testing tools for ADK."""
  pass


@conformance.command("record", cls=HelpfulCommand)
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument(
    "streaming-mode",
    type=click.Choice(
        [str(m.value) for m in StreamingMode], case_sensitive=False
    ),
    callback=lambda ctx, param, value: next(
        (m for m in StreamingMode if str(m.value).lower() == value.lower()),
        value,
    ),
)
@click.pass_context
def cli_conformance_record(
    ctx,
    paths: tuple[str, ...],
    streaming_mode: StreamingMode,
):
  """Generate ADK conformance test YAML files from TestCaseInput specifications.

  NOTE: this is work in progress.

  This command reads TestCaseInput specifications from input.yaml files,
  executes the specified test cases against agents, and generates conformance
  test files with recorded agent interactions as test.yaml files.

  Expected directory structure:
  category/name/input.yaml (TestCaseInput) -> category/name/test.yaml (TestCase)

  PATHS: One or more directories containing test case specifications.
  If no paths are provided, defaults to 'tests/' directory.

  Examples:

  Use default directory: adk conformance record

  Custom directories: adk conformance record tests/core tests/tools
  """

  try:
    from .conformance.cli_record import run_conformance_record
  except ImportError as e:
    click.secho(
        f"Error: Missing conformance testing dependencies: {e}",
        fg="red",
        err=True,
    )
    click.secho(
        "Please install the required conformance testing package dependencies.",
        fg="yellow",
        err=True,
    )
    ctx.exit(1)

  # Default to tests/ directory if no paths provided
  test_paths = [Path(p) for p in paths] if paths else [Path("tests").resolve()]
  asyncio.run(run_conformance_record(test_paths, streaming_mode))


@conformance.command("test", cls=HelpfulCommand)
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
)
@click.option(
    "--mode",
    type=click.Choice(["replay", "live"], case_sensitive=False),
    default="replay",
    show_default=True,
    help=(
        "Test mode: 'replay' verifies against recorded interactions, 'live'"
        " runs evaluation-based verification."
    ),
)
@click.option(
    "--generate_report",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to generate a Markdown report of the test results.",
)
@click.option(
    "--report_dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    help=(
        "Optional. Directory to store the generated report. Defaults to current"
        " directory."
    ),
)
@click.option(
    "--streaming-mode",
    type=click.Choice(
        [str(m.value) for m in StreamingMode], case_sensitive=False
    ),
    callback=lambda ctx, param, value: next(
        (m for m in StreamingMode if str(m.value).lower() == value.lower()),
        value,
    )
    if value is not None
    else None,
    required=False,
    default=None,
)
@click.pass_context
def cli_conformance_test(
    ctx,
    paths: tuple[str, ...],
    mode: str,
    generate_report: bool,
    report_dir: str | None = None,
    streaming_mode: StreamingMode | None = None,
):
  """Run conformance tests to verify agent behavior consistency.

  Validates that agents produce consistent outputs by comparing against recorded
  interactions or evaluating live execution results.

  PATHS can be any number of folder paths. Each folder can either:
  - Contain a spec.yaml file directly (single test case)
  - Contain subdirectories with spec.yaml files (multiple test cases)

  If no paths are provided, defaults to searching for the 'tests' folder.

  TEST MODES:

  \b
  replay  : Verifies agent interactions match previously recorded behaviors
            exactly. Compares LLM requests/responses and tool calls/results.
  live    : Runs evaluation-based verification (not yet implemented)

  DIRECTORY STRUCTURE:

  Test cases must follow this structure:

  \b
  category/
    test_name/
      spec.yaml                     # Test specification
      generated-recordings.yaml     # Recorded interactions (replay mode)
      generated-session.yaml        # Session data (replay mode)
      generated-recordings-sse.yaml # Recorded SSE interactions (replay mode)
      generated-session-sse.yaml    # SSE Session data (replay mode)

  REPORT GENERATION:

  Use --generate_report to create a Markdown report of test results.
  Use --report_dir to specify where the report should be saved.

  EXAMPLES:

  \b
  # Run all tests in current directory's 'tests' folder
  adk conformance test

  \b
  # Run tests from specific folders
  adk conformance test tests/core tests/tools

  \b
  # Run a single test case
  adk conformance test tests/core/description_001

  \b
  # Run in live mode (when available)
  adk conformance test --mode=live tests/core

  \b
  # Generate a test report
  adk conformance test --generate_report

  \b
  # Generate a test report in a specific directory
  adk conformance test --generate_report --report_dir=reports
  """
  try:
    from .conformance.cli_test import run_conformance_test
  except ImportError as e:
    click.secho(
        f"Error: Missing conformance testing dependencies: {e}",
        fg="red",
        err=True,
    )
    click.secho(
        "Please install the required conformance testing package dependencies.",
        fg="yellow",
        err=True,
    )
    ctx.exit(1)

  # Convert to Path objects, use default if empty (paths are already resolved
  # by Click)
  test_paths = [Path(p) for p in paths] if paths else [Path("tests").resolve()]

  asyncio.run(
      run_conformance_test(
          test_paths=test_paths,
          mode=mode.lower(),
          generate_report=generate_report,
          report_dir=report_dir,
          streaming_mode=streaming_mode,
      )
  )


@main.command("create", cls=HelpfulCommand)
@click.option(
    "--model",
    type=str,
    help="Optional. The model used for the root agent.",
)
@click.option(
    "--api_key",
    type=str,
    help=(
        "Optional. The API Key needed to access the model, e.g. Google AI API"
        " Key."
    ),
)
@click.option(
    "--project",
    type=str,
    help="Optional. The Google Cloud Project for using VertexAI as backend.",
)
@click.option(
    "--region",
    type=str,
    help="Optional. The Google Cloud Region for using VertexAI as backend.",
)
@click.option(
    "--type",
    type=click.Choice(["CODE", "CONFIG"], case_sensitive=False),
    help=(
        "EXPERIMENTAL Optional. Type of agent to create: 'config' or 'code'."
        " 'config' is not ready for use so it defaults to 'code'. It may change"
        " later once 'config' is ready for use."
    ),
    default="CODE",
    show_default=True,
    hidden=True,  # Won't show in --help output. Not ready for use.
)
@click.argument("app_name", type=str, required=True)
def cli_create_cmd(
    app_name: str,
    model: str | None,
    api_key: str | None,
    project: str | None,
    region: str | None,
    type: str | None,
):
  """Creates a new app in the current folder with prepopulated agent template.

  APP_NAME: required, the folder of the agent source code.

  Example:

    adk create path/to/my_app
  """
  from . import cli_create

  cli_create.run_cmd(
      app_name,
      model=model,
      google_api_key=api_key,
      google_cloud_project=project,
      google_cloud_region=region,
      type=type,
  )


def validate_exclusive(ctx, param, value):
  # Store the validated parameters in the context
  if not hasattr(ctx, "exclusive_opts"):
    ctx.exclusive_opts = {}

  # If this option has a value and we've already seen another exclusive option
  if value is not None and any(ctx.exclusive_opts.values()):
    exclusive_opt = next(key for key, val in ctx.exclusive_opts.items() if val)
    raise click.UsageError(
        f"Options '{param.name}' and '{exclusive_opt}' cannot be set together."
    )

  # Record this option's value
  ctx.exclusive_opts[param.name] = value is not None
  return value


def adk_services_options(*, default_use_local_storage: bool = True):
  """Decorator to add ADK services options to click commands."""

  def decorator(func):
    @click.option(
        "--session_service_uri",
        help=textwrap.dedent("""\
            Optional. The URI of the session service.
            If set, ADK uses this service.

            \b
            If unset, ADK chooses a default session service (see
            --use_local_storage).
            - Use 'agentengine://<agent_engine>' to connect to Agent Engine
              sessions. <agent_engine> can either be the full qualified resource
              name 'projects/abc/locations/us-central1/reasoningEngines/123' or
              the resource id '123'.
            - Use 'memory://' to run with the in-memory session service.
            - Use 'sqlite://<path_to_sqlite_file>' to connect to a SQLite DB.
            - See https://docs.sqlalchemy.org/en/20/core/engines.html#backend-specific-urls
              for supported database URIs."""),
    )
    @click.option(
        "--artifact_service_uri",
        type=str,
        help=textwrap.dedent(
            """\
            Optional. The URI of the artifact service.
            If set, ADK uses this service.

            \b
            If unset, ADK chooses a default artifact service (see
            --use_local_storage).
            - Use 'gs://<bucket_name>' to connect to the GCS artifact service.
            - Use 'memory://' to force the in-memory artifact service.
            - Use 'file://<path>' to store artifacts in a custom local directory."""
        ),
        default=None,
    )
    @click.option(
        "--use_local_storage/--no_use_local_storage",
        default=default_use_local_storage,
        show_default=True,
        help=(
            "Optional. Whether to use local .adk storage when "
            "--session_service_uri and --artifact_service_uri are unset. "
            "Cannot be combined with explicit service URIs. When the agents "
            "directory isn't writable (common in Cloud Run/Kubernetes), ADK "
            "falls back to in-memory unless overridden by "
            "ADK_FORCE_LOCAL_STORAGE=1 or ADK_DISABLE_LOCAL_STORAGE=1."
        ),
    )
    @click.option(
        "--memory_service_uri",
        type=str,
        help=textwrap.dedent("""\
            Optional. The URI of the memory service.
            If set, ADK uses this service.

            \b
            If unset, ADK chooses a default memory service.
            - Use 'rag://<rag_corpus_id>' to connect to Vertex AI Rag Memory Service.
            - Use 'agentengine://<agent_engine>' to connect to Agent Engine
              sessions. <agent_engine> can either be the full qualified resource
              name 'projects/abc/locations/us-central1/reasoningEngines/123' or
              the resource id '123'.
            - Use 'memory://' to force the in-memory memory service."""),
        default=None,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      ctx = click.get_current_context(silent=True)
      if ctx is not None:
        use_local_storage_source = ctx.get_parameter_source("use_local_storage")
        if use_local_storage_source != ParameterSource.DEFAULT and (
            kwargs.get("session_service_uri") is not None
            or kwargs.get("artifact_service_uri") is not None
        ):
          raise click.UsageError(
              "--use_local_storage/--no_use_local_storage cannot be used with "
              "--session_service_uri or --artifact_service_uri."
          )
      return func(*args, **kwargs)

    return wrapper

  return decorator


@main.command("run", cls=HelpfulCommand)
@feature_options()
@adk_services_options(default_use_local_storage=True)
@_logging_options()
@click.option(
    "--save_session",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to save the session to a json file on exit.",
)
@click.option(
    "--session_id",
    type=str,
    help=(
        "Optional. The session ID to save the session to on exit when"
        " --save_session is set to true. User will be prompted to enter a"
        " session ID if not set."
    ),
)
@click.option(
    "--replay",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "The json file that contains the initial state of the session and user"
        " queries. A new session will be created using this state. And user"
        " queries are run against the newly created session. Users cannot"
        " continue to interact with the agent."
    ),
    callback=validate_exclusive,
)
@click.option(
    "--resume",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "The json file that contains a previously saved session (by"
        " --save_session option). The previous session will be re-displayed."
        " And user can continue to interact with the agent."
    ),
    callback=validate_exclusive,
)
@click.option(
    "--state",
    type=str,
    help="Optional. Initial state for the run as a JSON string.",
)
@click.option(
    "--timeout",
    type=str,
    help="Optional. Timeout for a single turn or query (e.g., 30s, 5m).",
)
@click.option(
    "--in_memory",
    is_flag=True,
    help="Optional. Do not persist session data (use in-memory storage).",
)
@click.option(
    "--jsonl",
    is_flag=True,
    help="Optional. Output structured JSONL instead of human-readable text.",
)
@click.option(
    "--default_llm_model",
    type=str,
    help=(
        "Optional. Sets the default LLM model used when the agent does not set"
        " a model explicitly."
    ),
    default=None,
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("query", type=str, required=False)
def cli_run(
    agent: str,
    query: Optional[str],
    save_session: bool,
    session_id: Optional[str],
    replay: Optional[str],
    resume: Optional[str],
    state: Optional[str] = None,
    timeout: Optional[str] = None,
    in_memory: bool = False,
    jsonl: bool = False,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = True,
    default_llm_model: Optional[str] = None,
    log_level: str = "INFO",
):
  """Runs an agent. If no query is provided, enters interactive mode.

  AGENT: The path to the agent source code folder.
  QUERY: Optional. The user message to send to the agent for a single-step run.

  Example:

    adk run path/to/my_agent
    adk run path/to/my_agent "hello"
  """
  logs.log_to_tmp_folder(level=getattr(logging, log_level.upper()))

  agent_parent_folder = os.path.dirname(agent)
  agent_folder_name = os.path.basename(agent)

  # If query is provided, we run in single-step mode (JSONL output)
  if query is not None:
    from .cli import run_once_cli

    exit_code = asyncio.run(
        run_once_cli(
            agent_parent_dir=agent_parent_folder,
            agent_folder_name=agent_folder_name,
            query=query,
            state_str=state,
            session_id=session_id,
            replay=replay,
            timeout=timeout,
            in_memory=in_memory,
            jsonl=jsonl,
            session_service_uri=session_service_uri,
            artifact_service_uri=artifact_service_uri,
            memory_service_uri=memory_service_uri,
            use_local_storage=use_local_storage,
            default_llm_model=default_llm_model,
        )
    )
    sys.exit(exit_code)
  else:
    # Legacy interactive mode
    asyncio.run(
        run_cli(
            agent_parent_dir=agent_parent_folder,
            agent_folder_name=agent_folder_name,
            input_file=replay,
            saved_session_file=resume,
            save_session=save_session,
            session_id=session_id,
            state_str=state,
            timeout=timeout,
            in_memory=in_memory,
            jsonl=jsonl,
            session_service_uri=session_service_uri,
            artifact_service_uri=artifact_service_uri,
            memory_service_uri=memory_service_uri,
            use_local_storage=use_local_storage,
            default_llm_model=default_llm_model,
        )
    )


@main.command(
    "test",
    cls=HelpfulCommand,
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": True,
        "ignore_unknown_options": True,
    },
)
@click.argument(
    "folder",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=".",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Rebuild test files by running the real agent with user messages.",
)
@click.pass_context
def cli_test(ctx, folder: str, rebuild: bool):
  """Runs pytest on agent test JSON files under the specified folder.

  FOLDER: The path to the folder containing agents and tests.
  Defaults to the current directory if not specified.

  Example:
      adk test path/to/agents
  """
  import sys

  if rebuild:
    from .agent_test_runner import rebuild_tests

    click.echo(f"Rebuilding tests in {folder}...")
    rebuild_tests(folder)
    sys.exit(0)

  # Parse arguments to separate pytest args (after --) from regular args
  pytest_args = []
  if "--" in ctx.args:
    separator_index = ctx.args.index("--")
    pytest_args = ctx.args[separator_index + 1 :]
    regular_args = ctx.args[:separator_index]

    if regular_args:
      click.secho(
          "Error: Unexpected arguments after folder and before '--':"
          f" {' '.join(regular_args)}. \nOnly arguments after '--' are passed"
          " to pytest.",
          fg="red",
          err=True,
      )
      ctx.exit(2)
  else:
    # If no '--', all remaining arguments are passed to pytest
    pytest_args = ctx.args

  import subprocess

  os.environ["ADK_TEST_FOLDER"] = folder

  current_dir = Path(__file__).parent
  test_runner_path = current_dir / "agent_test_runner.py"

  if not test_runner_path.exists():
    click.secho(
        f"Error: Test runner not found at {test_runner_path}",
        fg="red",
        err=True,
    )
    sys.exit(1)

  click.echo(f"Running tests in {folder} using runner {test_runner_path}...")

  result = subprocess.run([
      sys.executable,
      "-m",
      "pytest",
      str(test_runner_path),
      "-v",
      "-s",
      *pytest_args,
  ])
  sys.exit(result.returncode)


def eval_options():
  """Decorator to add common eval options to click commands."""

  def decorator(func):
    @click.option(
        "--eval_storage_uri",
        type=str,
        help=(
            "Optional. The evals storage URI to store agent evals,"
            " supported URIs: gs://<bucket name>."
        ),
        default=None,
    )
    @click.option(
        "--log_level",
        type=LOG_LEVELS,
        default="INFO",
        help="Optional. Set the logging level",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)

    return wrapper

  return decorator


@main.command("eval", cls=HelpfulCommand)
@feature_options()
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("eval_set_file_path_or_id", nargs=-1)
@click.option("--config_file_path", help="Optional. The path to config file.")
@click.option(
    "--print_detailed_results",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to print detailed results on console or not.",
)
@click.option(
    "--use_live",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Run inference using the Live API (bidirectional streaming)."
        " Required for Live API models (e.g. gemini-*-live-*)."
    ),
)
@click.option(
    "--live_transport",
    type=click.Choice(["text", "tts", "native_audio"]),
    default="text",
    show_default=True,
    help=(
        "Optional. How user turns are carried to the agent under test. 'text'"
        " runs the standard path; 'tts' synthesizes each user turn to audio"
        " against a live agent (works with fixed scripts and simulated users);"
        " 'native_audio' drives a native-audio persona that hears and speaks"
        " (supports reactive barge-in)."
    ),
)
@click.option(
    "--live_run_config_file",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    default=None,
    help=(
        "Optional. Path to a JSON file with run-level voice settings"
        " (VoiceProfile: voice_name, language_code, transport, audio_realism,"
        " barge_in) applied uniformly to the run's audio cases. Voice/realism/"
        "barge-in are run configuration, not eval-case data. When the file sets"
        " 'transport' it takes precedence over --live_transport."
    ),
)
@click.option(
    "--watch",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. With an audio --live_transport, stream the live conversation"
        " to the console as it unfolds."
    ),
)
@click.option(
    "--managed_metrics",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. With an audio --live_transport, also score with the managed"
        " Gen AI Eval Service multi-turn metrics (requires a GCP project;"
        " slower). By default only the local latency metric is used."
    ),
)
@click.option(
    "--live_timeout_seconds",
    type=int,
    default=None,
    help="Optional. Per-turn timeout in seconds when running with --use_live.",
)
@click.option(
    "--output_file",
    type=str,
    default=None,
    help=(
        "Optional. Write the aggregated eval results to this path as JSON. The"
        " file contains a top-level summary plus the full EvalSetResult for"
        " each eval set, intended for CI and programmatic (e.g. coding agent)"
        " consumption."
    ),
)
@eval_options()
@click.pass_context
def cli_eval(
    ctx: click.Context,
    agent_module_file_path: str,
    eval_set_file_path_or_id: list[str],
    config_file_path: str,
    print_detailed_results: bool,
    use_live: bool = False,
    live_transport: str = "text",
    live_run_config_file: str | None = None,
    watch: bool = False,
    managed_metrics: bool = False,
    live_timeout_seconds: int | None = None,
    output_file: str | None = None,
    eval_storage_uri: str | None = None,
    log_level: str = "INFO",
):
  """Evaluates an agent given the eval sets.

  AGENT_MODULE_FILE_PATH: The path to the __init__.py file that contains a
  module by the name "agent". "agent" module contains a root_agent.

  EVAL_SET_FILE_PATH_OR_ID: You can specify one or more eval set file paths or
  eval set id.

  Mixing of eval set file paths with eval set ids is not allowed.

  *Eval Set File Path*
  For each file, all evals will be run by default.

  If you want to run only specific evals from an eval set, first create a comma
  separated list of eval names and then add that as a suffix to the eval set
  file name, demarcated by a `:`.

  For example, we have `sample_eval_set_file.json` file that has following the
  eval cases:
  sample_eval_set_file.json:
    |....... eval_1
    |....... eval_2
    |....... eval_3
    |....... eval_4
    |....... eval_5

  sample_eval_set_file.json:eval_1,eval_2,eval_3

  This will only run eval_1, eval_2 and eval_3 from sample_eval_set_file.json.

  *Eval Set ID*
  For each eval set, all evals will be run by default.

  If you want to run only specific evals from an eval set, first create a comma
  separated list of eval names and then add that as a suffix to the eval set
  file name, demarcated by a `:`.

  For example, we have `sample_eval_set_id` that has following the eval cases:
  sample_eval_set_id:
    |....... eval_1
    |....... eval_2
    |....... eval_3
    |....... eval_4
    |....... eval_5

  If we did:
      sample_eval_set_id:eval_1,eval_2,eval_3

  This will only run eval_1, eval_2 and eval_3 from sample_eval_set_id.

  CONFIG_FILE_PATH: The path to config file.

  PRINT_DETAILED_RESULTS: Prints detailed results on the console.
  """
  envs.load_dotenv_for_agent(agent_module_file_path, ".")
  logs.setup_adk_logger(getattr(logging, log_level.upper()))

  try:
    import importlib  # noqa: F401

    from ..evaluation.base_eval_service import InferenceConfig
    from ..evaluation.base_eval_service import InferenceRequest
    from ..evaluation.custom_metric_evaluator import _CustomMetricEvaluator
    from ..evaluation.eval_config import get_eval_metrics_from_config
    from ..evaluation.eval_config import get_evaluation_criteria_or_default
    from ..evaluation.eval_result import EvalCaseResult
    from ..evaluation.evaluator import EvalStatus
    from ..evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from ..evaluation.local_eval_service import LocalEvalService
    from ..evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
    from ..evaluation.local_eval_sets_manager import load_eval_set_from_file
    from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
    from ..evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
    from ..evaluation.simulation.user_simulator_provider import UserSimulatorProvider
    from .cli_eval import _collect_eval_results
    from .cli_eval import _collect_inferences
    from .cli_eval import get_default_metric_info
    from .cli_eval import get_root_agent
    from .cli_eval import parse_and_get_evals_to_run
    from .cli_eval import pretty_print_eval_result
  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  from ..evaluation.simulation.voice_profile import LiveTransport
  from ..evaluation.simulation.voice_profile import VoiceProfile

  eval_config = get_evaluation_criteria_or_default(config_file_path)
  eval_metrics = get_eval_metrics_from_config(eval_config)
  _print_eval_criteria(eval_metrics)

  # Run-level voice settings: a --live_run_config_file takes precedence over a
  # voice_profile embedded in the eval config file.
  voice_profile = eval_config.voice_profile
  if live_run_config_file:
    with open(live_run_config_file, "r") as f:
      voice_profile = VoiceProfile.model_validate_json(f.read())

  # Transport is run configuration: the voice profile may pin it, otherwise the
  # --live_transport flag is used.
  transport = LiveTransport(live_transport)
  if voice_profile is not None and voice_profile.transport is not None:
    transport = voice_profile.transport
  is_audio_transport = transport in (
      LiveTransport.TTS,
      LiveTransport.NATIVE_AUDIO,
  )

  if is_audio_transport:
    # An audio run scores the freshly generated conversation, so reference-based
    # metrics do not apply. Resolve the valid live metric set (local latency by
    # default, managed multi-turn metrics opt-in) and filter reference-based
    # metrics out of any supplied config.
    eval_metrics = _resolve_live_eval_metrics(
        eval_metrics,
        config_file_path=config_file_path,
        managed_metrics=managed_metrics,
    )

  inference_config_kwargs = {
      "use_live": use_live,
      "live_transport": transport,
      "voice_profile": voice_profile,
  }
  if live_timeout_seconds is not None:
    inference_config_kwargs["live_timeout_seconds"] = live_timeout_seconds
  inference_config = InferenceConfig(**inference_config_kwargs)

  audio_progress_callback = None
  if watch and is_audio_transport:
    click.secho(
        f"\n🎙  Live eval over '{transport.value}' transport: a simulated user"
        " speaks with the agent under test.\n",
        fg="cyan",
    )
    audio_progress_callback = _make_watch_progress_callback()

  root_agent = get_root_agent(agent_module_file_path)
  app_name = os.path.basename(agent_module_file_path)
  agents_dir = os.path.dirname(agent_module_file_path)
  eval_sets_manager = None
  eval_set_results_manager = None

  if eval_storage_uri:
    from .utils import evals

    gcs_eval_managers = evals.create_gcs_eval_managers_from_uri(
        eval_storage_uri
    )
    eval_sets_manager = gcs_eval_managers.eval_sets_manager
    eval_set_results_manager = gcs_eval_managers.eval_set_results_manager
  else:
    eval_set_results_manager = LocalEvalSetResultsManager(agents_dir=agents_dir)

  # Snapshot existing result ids so we can report the files written by this run.
  try:
    pre_run_result_ids = set(
        eval_set_results_manager.list_eval_set_results(app_name)
    )
  except Exception:  # pylint: disable=broad-except
    pre_run_result_ids = set()

  inference_requests = []
  eval_set_file_or_id_to_evals = parse_and_get_evals_to_run(
      eval_set_file_path_or_id
  )

  # Check if the first entry is a file that exists, if it does then we assume
  # rest of the entries are also files. We enforce this assumption in the if
  # block.
  if eval_set_file_or_id_to_evals and os.path.exists(
      list(eval_set_file_or_id_to_evals.keys())[0]
  ):
    eval_sets_manager = InMemoryEvalSetsManager()

    # Read the eval_set files and get the cases.
    for (
        eval_set_file_path,
        eval_case_ids,
    ) in eval_set_file_or_id_to_evals.items():
      try:
        eval_set = load_eval_set_from_file(
            eval_set_file_path, eval_set_file_path
        )
      except FileNotFoundError as fne:
        raise click.ClickException(
            f"`{eval_set_file_path}` should be a valid eval set file."
        ) from fne

      eval_sets_manager.create_eval_set(
          app_name=app_name, eval_set_id=eval_set.eval_set_id
      )
      for eval_case in eval_set.eval_cases:
        eval_sets_manager.add_eval_case(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            eval_case=eval_case,
        )
      inference_requests.append(
          InferenceRequest(
              app_name=app_name,
              eval_set_id=eval_set.eval_set_id,
              eval_case_ids=eval_case_ids,
              inference_config=inference_config,
          )
      )
  else:
    # We assume that what we have are eval set ids instead.
    eval_sets_manager = (
        eval_sets_manager
        if eval_storage_uri
        else LocalEvalSetsManager(agents_dir=agents_dir)
    )

    for eval_set_id_key, eval_case_ids in eval_set_file_or_id_to_evals.items():
      inference_requests.append(
          InferenceRequest(
              app_name=app_name,
              eval_set_id=eval_set_id_key,
              eval_case_ids=eval_case_ids,
              inference_config=inference_config,
          )
      )

  # Transport is run configuration, so whether this is an audio run is decided
  # once by the resolved transport (above) rather than per case.
  run_has_audio = is_audio_transport

  if watch and not run_has_audio:
    click.secho(
        "--watch has no effect without an audio --live_transport; ignoring.",
        fg="yellow",
    )

  # Audio runs persist per-turn user/agent audio to the artifact service, and the
  # eval result stores references (app/user/session/filename) to that audio. Use
  # a durable, per-agent artifact service rooted at the same local agents dir the
  # results are written to, so the audio remains fetchable after this process
  # exits (e.g. for playback in `adk web`). Without it the default in-memory
  # artifact service drops the audio on exit, leaving the persisted result's
  # references dangling (404 on load). Only built for audio runs; text runs do
  # not produce artifacts. GCS-backed eval storage manages its own artifacts.
  artifact_service = None
  if run_has_audio and not eval_storage_uri:
    from .utils.service_factory import create_artifact_service_from_options

    artifact_service = create_artifact_service_from_options(
        base_dir=agents_dir,
    )

  user_simulator_provider = UserSimulatorProvider(
      user_simulator_config=eval_config.user_simulator_config
  )

  try:
    metric_evaluator_registry = DEFAULT_METRIC_EVALUATOR_REGISTRY
    if eval_config.custom_metrics:
      for (
          metric_name,
          config,
      ) in eval_config.custom_metrics.items():
        if config.metric_info:
          metric_info = config.metric_info.model_copy()
          metric_info.metric_name = metric_name
        else:
          metric_info = get_default_metric_info(
              metric_name=metric_name, description=config.description
          )

        metric_evaluator_registry.register_evaluator(
            metric_info, _CustomMetricEvaluator
        )

    eval_service = LocalEvalService(
        root_agent=root_agent,
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
        artifact_service=artifact_service,
        user_simulator_provider=user_simulator_provider,
        metric_evaluator_registry=metric_evaluator_registry,
        audio_progress_callback=audio_progress_callback,
    )

    inference_results = asyncio.run(
        _collect_inferences(
            inference_requests=inference_requests, eval_service=eval_service
        )
    )
    eval_results = asyncio.run(
        _collect_eval_results(
            inference_results=inference_results,
            eval_service=eval_service,
            eval_metrics=eval_metrics,
        )
    )
  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  if print_detailed_results:
    for eval_result in eval_results:
      eval_result: EvalCaseResult
      click.echo(
          "********************************************************************"
      )
      pretty_print_eval_result(eval_result)

  _print_eval_run_summary(eval_results)

  # Report the result files this run wrote, so devs and coding agents can find
  # the persisted structured results without guessing timestamped filenames.
  _report_written_result_files(
      eval_set_results_manager=eval_set_results_manager,
      app_name=app_name,
      pre_run_result_ids=pre_run_result_ids,
  )

  if output_file:
    _write_eval_output_file(
        output_file=output_file,
        app_name=app_name,
        eval_results=eval_results,
    )
    click.secho(f"Wrote eval results JSON to: {output_file}", fg="cyan")

  # Exit non-zero when any case did not pass, so CI and coding-agent loops can
  # branch on success. NOT_EVALUATED is treated as non-passing for this purpose.
  any_not_passed = any(
      eval_result.final_eval_status != EvalStatus.PASSED
      for eval_result in eval_results
  )
  if any_not_passed:
    ctx.exit(1)


@main.command("optimize", cls=HelpfulCommand)
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.option(
    "--sampler_config_file_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="The path to the local eval sampler config file.",
)
@click.option(
    "--optimizer_config_file_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=(
        "Optional. The path to the GEPA optimizer config file. If not provided,"
        " the default config will be used."
    ),
)
@click.option(
    "--print_detailed_results",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Set to enable detailed printing of GEPA optimization"
        " results to the console."
    ),
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    show_default=True,
    default="INFO",
    help="Optional. Set the logging level",
)
def cli_optimize(
    agent_module_file_path: str,
    sampler_config_file_path: str,
    optimizer_config_file_path: str,
    print_detailed_results: bool,
    log_level: str = "INFO",
):
  """Optimizes the root agent instructions using the GEPA optimizer.

  AGENT_MODULE_FILE_PATH: The path to the __init__.py file that contains a
  module by the name "agent". "agent" module contains a root_agent.

  SAMPLER_CONFIG_FILE_PATH: The path to the config for the LocalEvalSampler,
  which contains the eval config and the eval sets to use for training and
  validation during optimization.

  OPTIMIZER_CONFIG_FILE_PATH: Optional. The path to the config for the
  GEPARootAgentPromptOptimizer. If not provided, the default config will be
  used.

  PRINT_DETAILED_RESULTS: Optional. Enables printing detailed results exposed by
  the GEPA optimizer to the console.

  LOG_LEVEL: Optional. Set the logging level.
  """
  envs.load_dotenv_for_agent(agent_module_file_path, ".")
  logs.setup_adk_logger(getattr(logging, log_level.upper()))

  try:
    from ..evaluation.custom_metric_evaluator import _CustomMetricEvaluator  # noqa: F401
    from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
    from ..optimization.gepa_root_agent_prompt_optimizer import GEPARootAgentPromptOptimizer
    from ..optimization.gepa_root_agent_prompt_optimizer import GEPARootAgentPromptOptimizerConfig
    from ..optimization.local_eval_sampler import LocalEvalSampler
    from ..optimization.local_eval_sampler import LocalEvalSamplerConfig
    from .cli_eval import _collect_eval_results  # noqa: F401
    from .cli_eval import _collect_inferences  # noqa: F401
    from .cli_eval import get_root_agent

  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  with open(sampler_config_file_path, "r", encoding="utf-8") as f:
    content = f.read()
    sampler_config = LocalEvalSamplerConfig.model_validate_json(content)

  if optimizer_config_file_path:
    with open(optimizer_config_file_path, "r", encoding="utf-8") as f:
      content = f.read()
      optimizer_config = GEPARootAgentPromptOptimizerConfig.model_validate_json(
          content
      )
  else:
    optimizer_config = GEPARootAgentPromptOptimizerConfig()

  root_agent = get_root_agent(agent_module_file_path)
  app_name = os.path.basename(agent_module_file_path)
  agents_dir = os.path.dirname(agent_module_file_path)
  if app_name != sampler_config.app_name:
    raise click.ClickException(
        f"App name in the agent module file path ({app_name}) does not match"
        f" the app name in the sampler config file ({sampler_config.app_name})."
    )
  eval_sets_manager = LocalEvalSetsManager(agents_dir=agents_dir)

  sampler = LocalEvalSampler(sampler_config, eval_sets_manager)
  optimizer = GEPARootAgentPromptOptimizer(optimizer_config)

  optimization_result = asyncio.run(optimizer.optimize(root_agent, sampler))
  best_idx = optimization_result.gepa_result["best_idx"]

  click.echo("=" * 80)
  click.echo("Optimized root agent instructions:")
  click.echo("-" * 80)
  click.echo(
      optimization_result.optimized_agents[best_idx].optimized_agent.instruction
  )

  if print_detailed_results:
    click.echo("=" * 80)
    if optimization_result.gepa_result:
      click.echo("Detailed GEPA optimization metrics:")
      click.echo("-" * 80)
      click.echo(json.dumps(optimization_result.gepa_result, indent=2))
    else:
      click.echo("Detailed GEPA optimization metrics are not available.")

  click.echo("=" * 80)


@main.group("eval_set")
def eval_set():
  """Manage Eval Sets."""
  pass


@eval_set.command("create", cls=HelpfulCommand)
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("eval_set_id", type=str, required=True)
@eval_options()
def cli_create_eval_set(
    agent_module_file_path: str,
    eval_set_id: str,
    eval_storage_uri: str | None = None,
    log_level: str = "INFO",
):
  """Creates an empty EvalSet given the agent_module_file_path and eval_set_id."""
  from .cli_eval import get_eval_sets_manager

  logs.setup_adk_logger(getattr(logging, log_level.upper()))
  app_name = os.path.basename(agent_module_file_path)
  agents_dir = os.path.dirname(agent_module_file_path)
  eval_sets_manager = get_eval_sets_manager(eval_storage_uri, agents_dir)

  try:
    eval_sets_manager.create_eval_set(
        app_name=app_name, eval_set_id=eval_set_id
    )
    click.echo(f"Eval set '{eval_set_id}' created for app '{app_name}'.")
  except ValueError as e:
    raise click.ClickException(str(e))


@eval_set.command("add_eval_case", cls=HelpfulCommand)
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("eval_set_id", type=str, required=True)
@click.option(
    "--scenarios_file",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help="A path to file containing JSON serialized ConversationScenarios.",
    required=True,
)
@click.option(
    "--session_input_file",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help="Path to session file containing SessionInput in JSON format.",
    required=True,
)
@eval_options()
def cli_add_eval_case(
    agent_module_file_path: str,
    eval_set_id: str,
    scenarios_file: str,
    eval_storage_uri: str | None = None,
    session_input_file: str | None = None,
    log_level: str = "INFO",
):
  """Adds eval cases to the given eval set.

  There are several ways that an eval case can be created, for now this method
  only supports adding one using a conversation scenarios file.

  If an eval case for the generated id already exists, then we skip adding it.
  """
  logs.setup_adk_logger(getattr(logging, log_level.upper()))
  try:
    from ..evaluation.conversation_scenarios import ConversationScenarios
    from ..evaluation.eval_case import EvalCase
    from ..evaluation.eval_case import SessionInput
    from .cli_eval import get_eval_sets_manager

  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  app_name = os.path.basename(agent_module_file_path)
  agents_dir = os.path.dirname(agent_module_file_path)
  eval_sets_manager = get_eval_sets_manager(eval_storage_uri, agents_dir)

  try:
    with open(session_input_file, "r") as f:
      session_input = SessionInput.model_validate_json(f.read())

    with open(scenarios_file, "r") as f:
      conversation_scenarios = ConversationScenarios.model_validate_json(
          f.read()
      )

    for scenario in conversation_scenarios.scenarios:
      scenario_str = json.dumps(
          scenario.model_dump(exclude_none=True), sort_keys=True
      )
      eval_id = hashlib.sha256(scenario_str.encode("utf-8")).hexdigest()[:8]
      eval_case = EvalCase(
          eval_id=eval_id,
          conversation_scenario=scenario,
          session_input=session_input,
          creation_timestamp=datetime.now().timestamp(),
      )

      if (
          eval_sets_manager.get_eval_case(
              app_name=app_name, eval_set_id=eval_set_id, eval_case_id=eval_id
          )
          is None
      ):
        eval_sets_manager.add_eval_case(
            app_name=app_name, eval_set_id=eval_set_id, eval_case=eval_case
        )
        click.echo(
            f"Eval case '{eval_case.eval_id}' added to eval set"
            f" '{eval_set_id}'."
        )
      else:
        click.echo(
            f"Eval case '{eval_case.eval_id}' already exists in eval set"
            f" '{eval_set_id}', skipped adding."
        )
  except Exception as e:
    raise click.ClickException(f"Failed to add eval case(s): {e}") from e


@eval_set.command("generate_eval_cases", cls=HelpfulCommand)
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("eval_set_id", type=str, required=True)
@click.option(
    "--user_simulation_config_file",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "A path to file containing JSON serialized "
        "UserScenarioGenerationConfig dict."
    ),
    required=True,
)
@eval_options()
def cli_generate_eval_cases(
    agent_module_file_path: str,
    eval_set_id: str,
    user_simulation_config_file: str,
    eval_storage_uri: str | None = None,
    log_level: str = "INFO",
):
  """Generates eval cases dynamically and adds them to the given eval set.

  Uses Vertex AI Eval SDK to generate conversation scenarios based on an
  Agent's info and definitions. It will automatically create the empty eval_set
  if it has not been created in advance.

  Args:
    agent_module_file_path: The path to the agent module file.
    eval_set_id: The id of the eval set to generate cases for.
    user_simulation_config_file: The path to the user simulation config file.
    eval_storage_uri: The eval storage uri.
    log_level: The log level.
  """
  logs.setup_adk_logger(getattr(logging, log_level.upper()))
  try:
    from ..evaluation._scenario_generation_helper import generate_and_add_eval_cases
    from ..evaluation.conversation_scenarios import ConversationGenerationConfig
    from .cli_eval import get_eval_sets_manager
    from .cli_eval import get_root_agent
    from .utils.state import create_empty_state

  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  app_name = os.path.basename(agent_module_file_path)
  agents_dir = os.path.dirname(agent_module_file_path)

  try:
    eval_sets_manager = get_eval_sets_manager(eval_storage_uri, agents_dir)
    root_agent = get_root_agent(agent_module_file_path)

    with open(user_simulation_config_file, "r") as f:
      config = ConversationGenerationConfig.model_validate_json(f.read())

    click.echo("Generating scenarios utilizing Vertex AI Eval SDK...")
    added_eval_ids = generate_and_add_eval_cases(
        root_agent=root_agent,
        config=config,
        eval_sets_manager=eval_sets_manager,
        app_name=app_name,
        eval_set_id=eval_set_id,
        initial_session_state=create_empty_state(root_agent),
    )
    for eval_id in added_eval_ids:
      click.echo(f"Eval case '{eval_id}' added to eval set '{eval_set_id}'.")
  except Exception as e:
    raise click.ClickException(f"Failed to generate eval case(s): {e}") from e


def web_options():
  """Decorator to add web UI options to click commands."""

  def decorator(func):
    @click.option(
        "--logo-text",
        type=str,
        help="Optional. The text to display in the logo of the web UI.",
        default=None,
    )
    @click.option(
        "--logo-image-url",
        type=str,
        help=(
            "Optional. The URL of the image to display in the logo of the"
            " web UI."
        ),
        default=None,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)

    return wrapper

  return decorator


def _deprecate_parameter(ctx, param, value):
  if value:
    click.echo(
        click.style(
            f"WARNING: --{param} is deprecated and will be removed. Please"
            " leave it unspecified.",
            fg="yellow",
        ),
        err=True,
    )
  return value


def _deprecate_trace_to_cloud(ctx, param, value):
  if value:
    click.echo(
        click.style(
            f"WARNING: --{param} is deprecated and will be removed. Please"
            " use --otel_to_cloud instead.",
            fg="yellow",
        ),
        err=True,
    )
  return value


def fast_api_common_options():
  """Decorator to add common fast api options to click commands."""

  def decorator(func):
    func = _logging_options()(func)

    @click.option(
        "--host",
        type=str,
        help="Optional. The binding host of the server",
        default="127.0.0.1",
        show_default=True,
    )
    @click.option(
        "--port",
        type=int,
        help="Optional. The port of the server",
        default=8000,
    )
    @click.option(
        "--allow_origins",
        help=(
            "Optional. Origins to allow for CORS. Can be literal origins"
            " (e.g., 'https://example.com') or regex patterns prefixed with"
            " 'regex:' (e.g., 'regex:https://.*\\.example\\.com')."
        ),
        multiple=True,
    )
    @click.option(
        "--trace_to_cloud",
        is_flag=True,
        show_default=True,
        default=False,
        help="Optional. Whether to enable cloud trace for telemetry.",
    )
    @click.option(
        "--otel_to_cloud",
        is_flag=True,
        show_default=True,
        default=False,
        help=(
            "Optional. Whether to write OTel data to Google Cloud"
            " Observability services - Cloud Trace and Cloud Logging."
        ),
    )
    @click.option(
        "--reload/--no-reload",
        default=True,
        help=(
            "Optional. Whether to enable auto reload for server. Not supported"
            " for Cloud Run."
        ),
    )
    @click.option(
        "--a2a",
        is_flag=True,
        show_default=True,
        default=False,
        help="Optional. Whether to enable A2A endpoint.",
    )
    @click.option(
        "--reload_agents",
        is_flag=True,
        default=False,
        show_default=True,
        help="Optional. Whether to enable live reload for agents changes.",
    )
    @click.option(
        "--eval_storage_uri",
        type=str,
        help=(
            "Optional. The evals storage URI to store agent evals,"
            " supported URIs: gs://<bucket name>."
        ),
        default=None,
    )
    @click.option(
        "--extra_plugins",
        help=(
            "Optional. Comma-separated list of extra plugin classes or"
            " instances to enable (e.g., my.module.MyPluginClass or"
            " my.module.my_plugin_instance)."
        ),
        multiple=True,
    )
    @click.option(
        "--url_prefix",
        type=str,
        help=(
            "Optional. URL path prefix when the application is mounted behind a"
            " reverse proxy or API gateway (e.g., '/api/v1', '/adk'). This"
            " ensures generated URLs and redirects work correctly when the app"
            " is not served at the root path. Must start with '/' if provided."
        ),
        default=None,
    )
    # Parsed into list[str] by the wrapper below (server commands need a list).
    @click.option(
        "--trigger_sources",
        type=str,
        help=(
            "Optional. Comma-separated list of trigger sources to enable"
            " (e.g., 'pubsub,eventarc'). Registers /apps/{app_name}/trigger/*"
            " endpoints for batch and event-driven agent invocations."
        ),
        default=None,
    )
    @functools.wraps(func)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
      # Parse comma-separated trigger_sources into a list.
      trigger_sources = kwargs.get("trigger_sources")
      if trigger_sources is not None:
        kwargs["trigger_sources"] = [
            s.strip() for s in trigger_sources.split(",") if s.strip()
        ]

      return func(*args, **kwargs)

    return wrapper

  return decorator


def _check_windows_reload(reload: bool) -> bool:
  """Checks if reload is enabled on Windows and forces it to False if so."""
  if sys.platform == "win32" and reload:
    click.secho(
        "WARNING: The --reload flag is not supported on Windows because it"
        " forces Uvicorn to use SelectorEventLoop, which does not support"
        " subprocesses (needed for executing tools). Forcing --no-reload.",
        fg="yellow",
        err=True,
    )
    return False
  return reload


@main.command("web")
@feature_options()
@fast_api_common_options()
@web_options()
@adk_services_options(default_use_local_storage=True)
@click.option(
    "--default_llm_model",
    type=str,
    help=(
        "Optional. Sets the default LLM model used when the agent does not set"
        " a model explicitly."
    ),
    default=None,
)
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd,
)
def cli_web(
    agents_dir: str,
    default_llm_model: Optional[str] = None,
    eval_storage_uri: Optional[str] = None,
    log_level: str = "INFO",
    allow_origins: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    url_prefix: str | None = None,
    trace_to_cloud: bool = False,
    otel_to_cloud: bool = False,
    reload: bool = True,
    session_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    use_local_storage: bool = True,
    a2a: bool = False,
    reload_agents: bool = False,
    extra_plugins: list[str] | None = None,
    logo_text: str | None = None,
    logo_image_url: str | None = None,
    trigger_sources: list[str] | None = None,
):
  """Starts a FastAPI server with Web UI for agents.

  AGENTS_DIR: The directory of agents (where each subdirectory is a single
  agent containing `agent.py` or `root_agent.yaml` files) or a path pointing
  directly to a single agent folder.

  Example:

    adk web --session_service_uri=[uri] --port=[port] path/to/agents_dir
  """
  reload = _check_windows_reload(reload)
  logs.setup_adk_logger(getattr(logging, log_level.upper()))

  @asynccontextmanager
  async def _lifespan(app: FastAPI):
    click.secho(
        f"""
+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
|                                                                             |
| For local testing, access at http://{host}:{port}.{" "*(29 - len(str(port)))}|
+-----------------------------------------------------------------------------+
""",
        fg="green",
    )
    yield  # Startup is done, now app is running
    click.secho(
        """
+-----------------------------------------------------------------------------+
| ADK Web Server shutting down...                                             |
+-----------------------------------------------------------------------------+
""",
        fg="green",
    )

  from .fast_api import get_fast_api_app

  app = get_fast_api_app(
      agents_dir=agents_dir,
      session_service_uri=session_service_uri,
      artifact_service_uri=artifact_service_uri,
      memory_service_uri=memory_service_uri,
      use_local_storage=use_local_storage,
      eval_storage_uri=eval_storage_uri,
      allow_origins=allow_origins,
      web=True,
      trace_to_cloud=trace_to_cloud,
      otel_to_cloud=otel_to_cloud,
      lifespan=_lifespan,
      a2a=a2a,
      host=host,
      port=port,
      url_prefix=url_prefix,
      reload_agents=reload_agents,
      extra_plugins=extra_plugins,
      logo_text=logo_text,
      logo_image_url=logo_image_url,
      trigger_sources=trigger_sources,
      default_llm_model=default_llm_model,
  )
  config = uvicorn.Config(
      app,
      host=host,
      port=port,
      reload=reload,
  )

  server = uvicorn.Server(config)
  server.run()


@main.command("api_server")
@feature_options()
# The directory of agents, where each subdirectory is a single agent.
# By default, it is the current working directory
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd(),
)
@fast_api_common_options()
@adk_services_options(default_use_local_storage=True)
@click.option(
    "--auto_create_session",
    is_flag=True,
    default=False,
    help=(
        "Automatically create a session if it doesn't exist when calling /run."
    ),
)
@click.option(
    "--with_ui",
    is_flag=True,
    default=False,
    help="Serve ADK Web UI if set.",
)
@click.option(
    "--gemini_enterprise_app_name",
    type=str,
    default=None,
    help=(
        "The app_name to register with Gemini Enterprise via"
        " https://docs.cloud.google.com/gemini/enterprise/docs/register-and-manage-an-adk-agent"
    ),
)
@click.option(
    "--express_mode",
    is_flag=True,
    default=False,
    help=(
        "Whether or not to initialize the server in express mode. This is only"
        " supported when gemini_enterprise_app_name is set. Defaults to"
        " False."
    ),
)
def cli_api_server(
    agents_dir: str,
    eval_storage_uri: str | None = None,
    log_level: str = "INFO",
    allow_origins: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    url_prefix: str | None = None,
    trace_to_cloud: bool = False,
    otel_to_cloud: bool = False,
    reload: bool = True,
    session_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    use_local_storage: bool = True,
    a2a: bool = False,
    reload_agents: bool = False,
    extra_plugins: list[str] | None = None,
    auto_create_session: bool = False,
    trigger_sources: list[str] | None = None,
    with_ui: bool = False,
    gemini_enterprise_app_name: str | None = None,
    express_mode: bool = False,
):
  """Starts a FastAPI server for agents.

  AGENTS_DIR: The directory of agents (where each subdirectory is a single
  agent containing `agent.py` or `root_agent.yaml` files) or a path pointing
  directly to a single agent folder.

  Example:

    adk api_server --session_service_uri=[uri] --port=[port] path/to/agents_dir
  """
  reload = _check_windows_reload(reload)
  if express_mode and not gemini_enterprise_app_name:
    raise click.UsageError(
        "--express_mode is only supported when --gemini_enterprise_app_name is"
        " set."
    )

  logs.setup_adk_logger(getattr(logging, log_level.upper()))

  from .fast_api import get_fast_api_app

  config = uvicorn.Config(
      get_fast_api_app(
          agents_dir=agents_dir,
          session_service_uri=session_service_uri,
          artifact_service_uri=artifact_service_uri,
          memory_service_uri=memory_service_uri,
          use_local_storage=use_local_storage,
          eval_storage_uri=eval_storage_uri,
          allow_origins=allow_origins,
          web=with_ui,
          trace_to_cloud=trace_to_cloud,
          otel_to_cloud=otel_to_cloud,
          a2a=a2a,
          host=host,
          port=port,
          url_prefix=url_prefix,
          reload_agents=reload_agents,
          extra_plugins=extra_plugins,
          auto_create_session=auto_create_session,
          trigger_sources=trigger_sources,
          gemini_enterprise_app_name=gemini_enterprise_app_name,
          express_mode=express_mode,
      ),
      host=host,
      port=port,
      reload=reload,
  )
  server = uvicorn.Server(config)
  server.run()


@deploy.command(
    "cloud_run",
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
    },
)
@click.option(
    "--project",
    type=str,
    help=(
        "Required. Google Cloud project to deploy the agent. When absent,"
        " default project from gcloud config is used."
    ),
)
@click.option(
    "--region",
    type=str,
    help=(
        "Required. Google Cloud region to deploy the agent. When absent,"
        " gcloud run deploy will prompt later."
    ),
)
@click.option(
    "--service_name",
    type=str,
    default="adk-default-service-name",
    help=(
        "Optional. The service name to use in Cloud Run (default:"
        " 'adk-default-service-name')."
    ),
)
@click.option(
    "--app_name",
    type=str,
    default="",
    help=(
        "Optional. App name of the ADK API server (default: the folder name"
        " of the AGENT source code)."
    ),
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Optional. The port of the ADK API server (default: 8000).",
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Whether to enable Cloud Trace export for Cloud Run"
        " deployments."
    ),
)
@click.option(
    "--otel_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Whether to enable OpenTelemetry export to GCP for Cloud Run"
        " deployments."
    ),
)
@click.option(
    "--with_ui",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Deploy ADK Web UI if set. (default: deploy ADK API server"
        " only). WARNING: The web UI is for development and testing only — do"
        " not use in production."
    ),
)
@click.option(
    "--temp_folder",
    type=str,
    default=os.path.join(
        tempfile.gettempdir(),
        "cloud_run_deploy_src",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    help=(
        "Optional. Temp folder for the generated Cloud Run source files"
        " (default: a timestamped folder in the system temp directory)."
    ),
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    default="INFO",
    help="Optional. Set the logging level",
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.option(
    "--adk_version",
    type=str,
    default=version.__version__,
    show_default=True,
    help=(
        "Optional. The ADK version used in Cloud Run deployment. (default: the"
        " version in the dev environment)"
    ),
)
@click.option(
    "--a2a",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable A2A endpoint.",
)
# Kept as raw str (not parsed to list) — interpolated directly into Dockerfile CMD.
@click.option(
    "--trigger_sources",
    type=str,
    help=(
        "Optional. Comma-separated list of trigger sources to enable"
        " (e.g., 'pubsub,eventarc'). Registers /trigger/* endpoints"
        " for batch and event-driven agent invocations."
    ),
    default=None,
)
@click.option(
    "--allow_origins",
    help=(
        "Optional. Origins to allow for CORS. Can be literal origins"
        " (e.g., 'https://example.com') or regex patterns prefixed with"
        " 'regex:' (e.g., 'regex:https://.*\\.example\\.com')."
    ),
    multiple=True,
)
# TODO: Add eval_storage_uri option back when evals are supported in Cloud Run.
@adk_services_options(default_use_local_storage=False)
@click.pass_context
def cli_deploy_cloud_run(
    ctx,
    agent: str,
    project: str | None,
    region: str | None,
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    otel_to_cloud: bool,
    with_ui: bool,
    adk_version: str,
    log_level: str,
    allow_origins: Optional[list[str]] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = False,
    a2a: bool = False,
    trigger_sources: str | None = None,
):
  """Deploys an agent to Cloud Run.

  AGENT: The path to the agent source code folder.

  Use '--' to separate gcloud arguments from adk arguments.

  Examples:

    adk deploy cloud_run --project=[project] --region=[region] path/to/my_agent

    adk deploy cloud_run --project=[project] --region=[region] path/to/my_agent
      -- --no-allow-unauthenticated --min-instances=2
  """

  _warn_if_with_ui(with_ui)

  # Parse arguments to separate gcloud args (after --) from regular args
  gcloud_args = []
  if "--" in ctx.args:
    separator_index = ctx.args.index("--")
    gcloud_args = ctx.args[separator_index + 1 :]
    regular_args = ctx.args[:separator_index]

    # If there are regular args before --, that's an error
    if regular_args:
      click.secho(
          "Error: Unexpected arguments after agent path and before '--':"
          f" {' '.join(regular_args)}. \nOnly arguments after '--' are passed"
          " to gcloud.",
          fg="red",
          err=True,
      )
      ctx.exit(2)
  else:
    # No -- separator, treat all args as an error to enforce the new behavior
    if ctx.args:
      click.secho(
          f"Error: Unexpected arguments: {' '.join(ctx.args)}. \nUse '--' to"
          " separate gcloud arguments, e.g.: adk deploy cloud_run [options]"
          " agent_path -- --min-instances=2",
          fg="red",
          err=True,
      )
      ctx.exit(2)

  try:
    from . import cli_deploy

    cli_deploy.to_cloud_run(
        agent_folder=agent,
        project=project,
        region=region,
        service_name=service_name,
        app_name=app_name,
        temp_folder=temp_folder,
        port=port,
        trace_to_cloud=trace_to_cloud,
        otel_to_cloud=otel_to_cloud,
        allow_origins=allow_origins,
        with_ui=with_ui,
        log_level=log_level,
        verbosity=log_level,
        adk_version=adk_version,
        session_service_uri=session_service_uri,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=memory_service_uri,
        use_local_storage=use_local_storage,
        a2a=a2a,
        trigger_sources=trigger_sources,
        extra_gcloud_args=tuple(gcloud_args),
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)


@main.group()
def migrate():
  """ADK migration commands."""
  pass


@migrate.command("session", cls=HelpfulCommand)
@click.option(
    "--source_db_url",
    required=True,
    help=(
        "SQLAlchemy URL of source database in database session service, e.g."
        " sqlite:///source.db."
    ),
)
@click.option(
    "--dest_db_url",
    required=True,
    help=(
        "SQLAlchemy URL of destination database in database session service,"
        " e.g. sqlite:///dest.db."
    ),
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(  # type: ignore[untyped-decorator]
    "--allow-unsafe-unpickling",
    "--allow_unsafe_unpickling",
    is_flag=True,
    default=False,
    help=(
        "Optional. Allow unsafe pickle loading for trusted legacy session"
        " databases."
    ),
)
def cli_migrate_session(
    *,
    source_db_url: str,
    dest_db_url: str,
    log_level: str,
    allow_unsafe_unpickling: bool,
):
  """Migrates a session database to the latest schema version."""
  logs.setup_adk_logger(getattr(logging, log_level.upper()))
  try:
    from ..sessions.migration import migration_runner

    migration_runner.upgrade(
        source_db_url,
        dest_db_url,
        allow_unsafe_unpickling=allow_unsafe_unpickling,
    )
    click.secho("Migration check and upgrade process finished.", fg="green")
  except Exception as e:
    click.secho(f"Migration failed: {e}", fg="red", err=True)


@deploy.command("agent_engine")
@click.option(
    "--api_key",
    type=str,
    default=None,
    help=(
        "Optional. The API key to use for Express Mode. If not"
        " provided, the API key from the GOOGLE_API_KEY environment variable"
        " will be used. It will only be used if GOOGLE_GENAI_USE_ENTERPRISE is"
        " true. (It will override GOOGLE_API_KEY in the .env file if it"
        " exists.)"
    ),
)
@click.option(
    "--project",
    type=str,
    default=None,
    help=(
        "Optional. Google Cloud project to deploy the agent. It will override"
        " GOOGLE_CLOUD_PROJECT in the .env file (if it exists). It will be"
        " ignored if api_key is set."
    ),
)
@click.option(
    "--region",
    type=str,
    default=None,
    help=(
        "Optional. Google Cloud region to deploy the agent. It will override"
        " GOOGLE_CLOUD_LOCATION in the .env file (if it exists). It will be"
        " ignored if api_key is set."
    ),
)
@click.option(
    "--staging_bucket",
    type=str,
    default=None,
    help="Deprecated. This argument is no longer required or used.",
    callback=_deprecate_parameter,
)
@click.option(
    "--agent_engine_id",
    type=str,
    default=None,
    help=(
        "Optional. ID of the Agent Engine instance to update if it exists"
        " (default: None, which means a new instance will be created). If"
        " project and region are set, this should be the resource ID, and the"
        " corresponding resource name in Agent Engine will be:"
        " `projects/{project}/locations/{region}/reasoningEngines/{agent_engine_id}`."
        " If api_key is set, then agent_engine_id is required to be the full"
        " resource name (i.e. `projects/*/locations/*/reasoningEngines/*`)."
    ),
)
@click.option(
    "--trace_to_cloud/--no-trace_to_cloud",
    type=bool,
    is_flag=True,
    show_default=True,
    default=None,
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_trace_to_cloud,
)
@click.option(
    "--otel_to_cloud",
    type=bool,
    is_flag=True,
    show_default=True,
    default=None,
    help="Optional. Whether to enable OpenTelemetry for Agent Engine.",
)
@click.option(
    "--display_name",
    type=str,
    show_default=True,
    default="",
    help="Optional. Display name of the agent in Agent Engine.",
)
@click.option(
    "--description",
    type=str,
    show_default=True,
    default="",
    help="Optional. Description of the agent in Agent Engine.",
)
@click.option(
    "--adk_app",
    type=str,
    default=None,
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
@click.option(
    "--temp_folder",
    type=str,
    default=None,
    help=(
        "Optional. Temp folder for the generated Agent Engine source files."
        " If the folder already exists, its contents will be removed."
        " (default: a timestamped folder in the current working directory)."
    ),
)
@click.option(
    "--adk_app_object",
    type=str,
    default=None,
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
@click.option(
    "--env_file",
    type=str,
    default="",
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
@click.option(
    "--requirements_file",
    type=str,
    default="",
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
@click.option(
    "--absolutize_imports",
    type=bool,
    default=False,
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
@click.option(
    "--agent_engine_config_file",
    type=str,
    default="",
    help=(
        "Optional. The filepath to the `.agent_engine_config.json` file to use."
        " The values in this file will be overridden by the values set by other"
        " flags. (default: the `.agent_engine_config.json` file in the `agent`"
        " directory, if any.)"
    ),
)
@click.option(
    "--validate-agent-import/--no-validate-agent-import",
    default=False,
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
@click.option(
    "--skip-agent-import-validation",
    "skip_agent_import_validation_alias",
    is_flag=True,
    default=False,
    help=" NOTE: This flag is deprecated and will be removed in the future.",
    callback=_deprecate_parameter,
)
# Kept as raw str (not parsed to list) — interpolated directly into Dockerfile CMD.
@click.option(
    "--trigger_sources",
    type=str,
    help=(
        "Optional. Comma-separated list of trigger sources to enable"
        " (e.g., 'pubsub,eventarc'). Registers /trigger/* endpoints"
        " for batch and event-driven agent invocations."
    ),
    default=None,
)
@click.option(
    "--adk_version",
    type=str,
    default=version.__version__,
    show_default=True,
    help=(
        "Optional. The ADK version used in Agent Engine deployment. (default: "
        " the version in the dev environment)"
    ),
)
@adk_services_options(default_use_local_storage=False)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_deploy_agent_engine(
    agent: str,
    project: str | None,
    region: str | None,
    staging_bucket: str | None,
    agent_engine_id: str | None,
    trace_to_cloud: bool | None,
    otel_to_cloud: bool | None,
    api_key: str | None,
    display_name: str,
    description: str,
    adk_app: str | None,
    adk_app_object: str | None,
    temp_folder: str | None,
    env_file: str,
    requirements_file: str,
    absolutize_imports: bool,
    agent_engine_config_file: str,
    validate_agent_import: bool = False,
    skip_agent_import_validation_alias: bool = False,
    adk_version: str | None = None,
    trigger_sources: str | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    session_service_uri: str | None = None,
    use_local_storage: bool = False,
):
  """Deploys an agent to Agent Engine.

  Example:

    \b
    # With Express Mode API Key
    adk deploy agent_engine --api_key=[api_key] my_agent

    \b
    # With Google Cloud Project and Region
    adk deploy agent_engine --project=[project] --region=[region]
      --display_name=[app_name] my_agent
  """
  logging.getLogger("vertexai_genai.agentengines").setLevel(logging.INFO)
  try:
    if validate_agent_import and skip_agent_import_validation_alias:
      raise click.UsageError(
          "Do not pass both --validate-agent-import and"
          " --skip-agent-import-validation."
      )
    from . import cli_deploy

    cli_deploy.to_agent_engine(
        agent_folder=agent,
        project=project,
        region=region,
        agent_engine_id=agent_engine_id,
        trace_to_cloud=trace_to_cloud,
        otel_to_cloud=otel_to_cloud,
        api_key=api_key,
        adk_app_object=adk_app_object,
        display_name=display_name,
        description=description,
        adk_app=adk_app,
        temp_folder=temp_folder,
        env_file=env_file,
        requirements_file=requirements_file,
        absolutize_imports=absolutize_imports,
        agent_engine_config_file=agent_engine_config_file,
        skip_agent_import_validation=not validate_agent_import,
        trigger_sources=trigger_sources,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=memory_service_uri,
        session_service_uri=session_service_uri,
        adk_version=adk_version,
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)


@deploy.command("gke")
@click.option(
    "--project",
    type=str,
    help=(
        "Required. Google Cloud project to deploy the agent. When absent,"
        " default project from gcloud config is used."
    ),
)
@click.option(
    "--region",
    type=str,
    help=(
        "Required. Google Cloud region to deploy the agent. When absent,"
        " gcloud run deploy will prompt later."
    ),
)
@click.option(
    "--cluster_name",
    type=str,
    help="Required. The name of the GKE cluster.",
)
@click.option(
    "--service_name",
    type=str,
    default="adk-default-service-name",
    help=(
        "Optional. The service name to use in GKE (default:"
        " 'adk-default-service-name')."
    ),
)
@click.option(
    "--app_name",
    type=str,
    default="",
    help=(
        "Optional. App name of the ADK API server (default: the folder name"
        " of the AGENT source code)."
    ),
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Optional. The port of the ADK API server (default: 8000).",
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable Cloud Trace for GKE.",
)
@click.option(
    "--otel_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable OpenTelemetry for GKE.",
)
@click.option(
    "--with_ui",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Deploy ADK Web UI if set. (default: deploy ADK API server"
        " only). WARNING: The web UI is for development and testing only — do"
        " not use in production."
    ),
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(
    "--service_type",
    type=click.Choice(["ClusterIP", "LoadBalancer"], case_sensitive=True),
    default="ClusterIP",
    show_default=True,
    help=(
        "Optional. The Kubernetes Service type for the deployed agent."
        " ClusterIP (default) keeps the service cluster-internal;"
        " use LoadBalancer to expose a public IP."
    ),
)
@click.option(
    "--temp_folder",
    type=str,
    default=os.path.join(
        tempfile.gettempdir(),
        "gke_deploy_src",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    help=(
        "Optional. Temp folder for the generated GKE source files"
        " (default: a timestamped folder in the system temp directory)."
    ),
)
@click.option(
    "--adk_version",
    type=str,
    default=version.__version__,
    show_default=True,
    help=(
        "Optional. The ADK version used in GKE deployment. (default: the"
        " version in the dev environment)"
    ),
)
# Kept as raw str (not parsed to list) — interpolated directly into Dockerfile CMD.
@click.option(
    "--trigger_sources",
    type=str,
    help=(
        "Optional. Comma-separated list of trigger sources to enable"
        " (e.g., 'pubsub,eventarc'). Registers /trigger/* endpoints"
        " for batch and event-driven agent invocations."
    ),
    default=None,
)
@adk_services_options(default_use_local_storage=False)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_deploy_gke(
    agent: str,
    project: str | None,
    region: str | None,
    cluster_name: str,
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    otel_to_cloud: bool,
    with_ui: bool,
    adk_version: str,
    service_type: str,
    log_level: str | None = None,
    session_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    use_local_storage: bool = False,
    trigger_sources: str | None = None,
):
  """Deploys an agent to GKE.

  AGENT: The path to the agent source code folder.

  Example:

    adk deploy gke --project=[project] --region=[region]
      --cluster_name=[cluster_name] path/to/my_agent
  """
  try:
    _warn_if_with_ui(with_ui)
    from . import cli_deploy

    cli_deploy.to_gke(
        agent_folder=agent,
        project=project,
        region=region,
        cluster_name=cluster_name,
        service_name=service_name,
        app_name=app_name,
        temp_folder=temp_folder,
        port=port,
        trace_to_cloud=trace_to_cloud,
        otel_to_cloud=otel_to_cloud,
        with_ui=with_ui,
        log_level=log_level,
        adk_version=adk_version,
        service_type=service_type,
        session_service_uri=session_service_uri,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=memory_service_uri,
        use_local_storage=use_local_storage,
        trigger_sources=trigger_sources,
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)
