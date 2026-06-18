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

import abc
from dataclasses import dataclass
import logging
import math
import os
from typing import Optional
from typing import Union

from google.genai import types as genai_types
import pandas as pd
from typing_extensions import override

from ..dependencies.vertexai import vertexai
from .app_details import AgentDetails
from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .eval_rubrics import RubricScore
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

logger = logging.getLogger("google_adk." + __name__)


@dataclass
class _BatchMetricSpec:
  """Describes a single Vertex metric to compute within a batched eval call.

  Attributes:
    adk_metric_name: The ADK-facing metric name (used as the result key, e.g.
      `multi_turn_task_success_v1`). Distinct from the Vertex SDK metric handle
      so callers can map results back to their own metric identifiers.
    metric: The Vertex SDK metric handle (a `PrebuiltMetric`/`RubricMetric` enum
      value or a metric name string) passed to `evals.evaluate()`.
    threshold: The pass/fail threshold for this metric.
  """

  adk_metric_name: str
  metric: Union[
      "vertexai.types.PrebuiltMetric", "vertexai.types.RubricMetric", str
  ]
  threshold: float


def _resolve_metric_spec_name(metric) -> str:
  """Resolves a Vertex metric handle to its result-key spec name.

  Batched results key metrics by their resolved spec name (e.g.
  `multi_turn_task_success_v1`), so we resolve handles the same way the SDK
  does to match them back up. Prebuilt/Rubric metric handles expose the spec
  name via `_get_api_metric_spec_name()`; plain strings and `Metric` objects
  fall back to their string/`name` form.
  """
  if isinstance(metric, str):
    return metric
  # LazyLoadedPrebuiltMetric (PrebuiltMetric / RubricMetric): resolve to the
  # versioned API spec name that keys the eval results.
  get_spec_name = getattr(metric, "_get_api_metric_spec_name", None)
  if callable(get_spec_name):
    spec_name = get_spec_name()
    if spec_name:
      return spec_name
  # Metric objects and other handles expose a `.name`.
  name = getattr(metric, "name", None)
  if name is not None:
    return str(name)
  return str(metric)


_ERROR_MESSAGE_SUFFIX = """
You should specify both project id and location. This metric uses Vertex Gen AI
Eval SDK, and it requires google cloud credentials.

If using an .env file add the values there, or explicitly set in the code using
the template below:

os.environ['GOOGLE_CLOUD_LOCATION'] = <LOCATION>
os.environ['GOOGLE_CLOUD_PROJECT'] = <PROJECT ID>
"""


class _VertexAiEvalFacade(Evaluator):
  """Simple facade for Vertex Gen AI Eval SDK.

  Vertex Gen AI Eval SDK exposes quite a few metrics that are valuable for
  agentic evals. This class helps us to access those metrics.

  Using this class requires a GCP project. Please set GOOGLE_CLOUD_PROJECT and
  GOOGLE_CLOUD_LOCATION in your .env file.
  """

  def __init__(
      self,
      threshold: float,
      metric_name: Union[
          vertexai.types.PrebuiltMetric, vertexai.types.RubricMetric
      ],
      expected_invocations_required=False,
  ):
    self._threshold = threshold
    self._metric_name = metric_name
    self._expected_invocations_required = expected_invocations_required

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", None)
    api_key = os.environ.get("GOOGLE_API_KEY", None)

    if api_key:
      self._client = vertexai.Client(api_key=api_key)
    elif project_id or location:
      if not project_id:
        raise ValueError("Missing project id." + _ERROR_MESSAGE_SUFFIX)
      if not location:
        raise ValueError("Missing location." + _ERROR_MESSAGE_SUFFIX)
      self._client = vertexai.Client(project=project_id, location=location)
    else:
      raise ValueError(
          "Either API Key or Google cloud Project id and location should be"
          " specified."
      )

  @abc.abstractmethod
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    """Returns EvaluationResult after performing evaluations using actual and expected invocations.

    Args:
      actual_invocations: These are the invocations that are obtained from the
        agent under test.
      expected_invocations: An optional list of invocations, if specified,
        usually act as a benchmark/golden response. If these are specified
        usually the expectation is that the length of this list and actual
        invocation is the same.
      conversation_scenario: An optional conversation scenario for multi-turn
        conversations.
    """

  def _get_text(self, content: Optional[genai_types.Content]) -> str:
    if content and content.parts:
      return "\n".join([p.text for p in content.parts if p.text])

    return ""

  def _get_score(self, eval_result) -> Optional[float]:
    if (
        eval_result
        and eval_result.summary_metrics
        and isinstance(eval_result.summary_metrics[0].mean_score, float)
        and not math.isnan(eval_result.summary_metrics[0].mean_score)
    ):
      return eval_result.summary_metrics[0].mean_score

    return None

  @staticmethod
  def _get_score_for_metric(
      eval_result, metric_spec_name: str
  ) -> Optional[float]:
    """Returns the mean score for a specific metric in a batched eval result.

    When multiple metrics are computed in a single `evals.evaluate()` call, the
    result carries one `summary_metrics` entry per metric, keyed by the metric
    spec name. This selects the matching entry rather than assuming a single
    metric was requested.
    """
    if not eval_result or not eval_result.summary_metrics:
      return None
    for summary_metric in eval_result.summary_metrics:
      if summary_metric.metric_name != metric_spec_name:
        continue
      mean_score = summary_metric.mean_score
      if isinstance(mean_score, float) and not math.isnan(mean_score):
        return mean_score
      return None
    return None

  @staticmethod
  def _get_rubric_reasoning_for_metric(
      eval_result, metric_spec_name: str
  ) -> tuple[Optional[list[RubricScore]], Optional[str]]:
    """Extracts the rubric verdicts and explanation for a metric, if any.

    The managed eval service returns per-case reasoning on
    `eval_case_results[i].response_candidate_results[j].metric_results[name]` as
    `rubric_verdicts` (each with a verdict + reasoning) and a free-text
    `explanation`. The aggregate `summary_metrics` path carries only the score,
    so this reads the detailed per-case results instead.

    Returns a tuple of (rubric_scores, explanation). Either may be None when the
    metric does not provide that detail.
    """
    if not eval_result or not getattr(eval_result, "eval_case_results", None):
      return None, None

    rubric_scores: list[RubricScore] = []
    explanation: Optional[str] = None

    for case_result in eval_result.eval_case_results:
      for candidate_result in case_result.response_candidate_results or []:
        metric_result = (candidate_result.metric_results or {}).get(
            metric_spec_name
        )
        if metric_result is None:
          continue
        if metric_result.explanation and explanation is None:
          explanation = metric_result.explanation
        for verdict in metric_result.rubric_verdicts or []:
          rubric = verdict.evaluated_rubric
          rubric_scores.append(
              RubricScore(
                  rubric_id=(
                      rubric.rubric_id if rubric and rubric.rubric_id else ""
                  ),
                  rationale=verdict.reasoning,
                  verdict=verdict.verdict,
                  score=(
                      None
                      if verdict.verdict is None
                      else (1.0 if verdict.verdict else 0.0)
                  ),
              )
          )

    return (rubric_scores or None), explanation

  def _get_eval_status(self, score: Optional[float]):
    if score is not None:
      return (
          EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED
      )

    return EvalStatus.NOT_EVALUATED

  @staticmethod
  def _get_eval_status_for_threshold(
      score: Optional[float], threshold: float
  ) -> EvalStatus:
    if score is not None:
      return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED
    return EvalStatus.NOT_EVALUATED

  def _perform_eval(self, dataset, metrics):
    """This method hides away the call to external service.

    Primarily helps with unit testing.
    """
    return self._client.evals.evaluate(
        dataset=dataset,
        metrics=metrics,
    )


class _SingleTurnVertexAiEvalFacade(_VertexAiEvalFacade):
  """A facade for single turn metrics exposed in Vertex Gen AI Eval SDK."""

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if self._expected_invocations_required and expected_invocations is None:
      raise ValueError("expected_invocations is needed by this metric.")
    del conversation_scenario  # not supported for per-invocation evaluation.

    # If expected_invocation are not required by the metric and if they are not
    # supplied, we provide a list of None.
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      prompt = self._get_text(actual.user_content)
      reference = self._get_text(expected.final_response) if expected else None
      response = self._get_text(actual.final_response)
      eval_case = {
          "prompt": prompt,
          "reference": reference,
          "response": response,
      }

      dataset = vertexai.types.EvaluationDataset(
          eval_dataset_df=pd.DataFrame([eval_case])
      )
      eval_case_result = self._perform_eval(
          dataset=dataset, metrics=[self._metric_name]
      )
      score = self._get_score(eval_case_result)
      rubric_scores, explanation = self._get_rubric_reasoning_for_metric(
          eval_case_result, _resolve_metric_spec_name(self._metric_name)
      )
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=self._get_eval_status(score),
              rubric_scores=rubric_scores,
              explanation=explanation,
          )
      )

      if score is not None:
        total_score += score
        num_invocations += 1

    if per_invocation_results:
      overall_score = (
          total_score / num_invocations if num_invocations > 0 else None
      )
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=self._get_eval_status(overall_score),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()

  def evaluate_invocations_for_metrics(
      self,
      metric_specs: list[_BatchMetricSpec],
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
  ) -> dict[str, EvaluationResult]:
    """Computes several single-turn metrics in one eval-service call per turn.

    The per-invocation dataset is identical across single-turn metrics, so we
    build it once per invocation and request all metrics together. Results are
    returned keyed by `_BatchMetricSpec.adk_metric_name`.
    """
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    spec_names = [
        _resolve_metric_spec_name(spec.metric) for spec in metric_specs
    ]
    per_metric_invocation_results: dict[str, list[PerInvocationResult]] = {
        spec.adk_metric_name: [] for spec in metric_specs
    }
    per_metric_total: dict[str, float] = {
        spec.adk_metric_name: 0.0 for spec in metric_specs
    }
    per_metric_count: dict[str, int] = {
        spec.adk_metric_name: 0 for spec in metric_specs
    }

    for actual, expected in zip(actual_invocations, expected_invocations):
      eval_case = {
          "prompt": self._get_text(actual.user_content),
          "reference": (
              self._get_text(expected.final_response) if expected else None
          ),
          "response": self._get_text(actual.final_response),
      }
      dataset = vertexai.types.EvaluationDataset(
          eval_dataset_df=pd.DataFrame([eval_case])
      )
      eval_result = self._perform_eval(
          dataset=dataset, metrics=[spec.metric for spec in metric_specs]
      )

      for spec, spec_name in zip(metric_specs, spec_names):
        score = self._get_score_for_metric(eval_result, spec_name)
        rubric_scores, explanation = self._get_rubric_reasoning_for_metric(
            eval_result, spec_name
        )
        per_metric_invocation_results[spec.adk_metric_name].append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=self._get_eval_status_for_threshold(
                    score, spec.threshold
                ),
                rubric_scores=rubric_scores,
                explanation=explanation,
            )
        )
        if score is not None:
          per_metric_total[spec.adk_metric_name] += score
          per_metric_count[spec.adk_metric_name] += 1

    results: dict[str, EvaluationResult] = {}
    for spec in metric_specs:
      invocation_results = per_metric_invocation_results[spec.adk_metric_name]
      if not invocation_results:
        results[spec.adk_metric_name] = EvaluationResult()
        continue
      count = per_metric_count[spec.adk_metric_name]
      overall_score = (
          per_metric_total[spec.adk_metric_name] / count if count > 0 else None
      )
      results[spec.adk_metric_name] = EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=self._get_eval_status_for_threshold(
              overall_score, spec.threshold
          ),
          per_invocation_results=invocation_results,
      )
    return results


class _MultiTurnVertexiAiEvalFacade(_VertexAiEvalFacade):
  """A facade for multi turn metrics exposed in Vertex Gen AI Eval SDK."""

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    del conversation_scenario

    per_invocation_results = []
    # If expected_invocation are not required by the metric and if they are not
    # supplied, we provide a list of None.
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    # We mark all the n-1 turns as NOT-EVALUATED for these metrics.
    for actual, expected in zip(
        actual_invocations[:-1], expected_invocations[:-1]
    ):
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=None,
              eval_status=self._get_eval_status(None),
          )
      )

    # Only evaluate the last turn and take into account all the previous turns.
    eval_case = vertexai.types.EvalCase(
        agent_data=_MultiTurnVertexiAiEvalFacade._get_agent_data(
            actual_invocations
        )
    )
    dataset = vertexai.types.EvaluationDataset(eval_cases=[eval_case])

    eval_case_result = self._perform_eval(
        dataset=dataset, metrics=[self._metric_name]
    )

    score = self._get_score(eval_case_result)
    rubric_scores, explanation = self._get_rubric_reasoning_for_metric(
        eval_case_result, _resolve_metric_spec_name(self._metric_name)
    )
    per_invocation_results.append(
        PerInvocationResult(
            actual_invocation=actual_invocations[-1],
            expected_invocation=expected_invocations[-1],
            score=score,
            eval_status=self._get_eval_status(score),
            rubric_scores=rubric_scores,
            explanation=explanation,
        )
    )

    if score is not None:
      return EvaluationResult(
          overall_score=score,
          overall_eval_status=self._get_eval_status(score),
          per_invocation_results=per_invocation_results,
          overall_rubric_scores=rubric_scores,
          overall_explanation=explanation,
      )

    return EvaluationResult()

  def evaluate_invocations_for_metrics(
      self,
      metric_specs: list[_BatchMetricSpec],
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
  ) -> dict[str, EvaluationResult]:
    """Computes several multi-turn metrics in a single eval-service call.

    The multi-turn dataset (the conversation `EvalCase`) is identical across
    these metrics, so we build it once and request all metrics together. Only
    the last turn is scored; earlier turns are marked NOT_EVALUATED. Results are
    returned keyed by `_BatchMetricSpec.adk_metric_name`.
    """
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    # The multi-turn (agent) metrics read `agent_data` (the full conversation
    # trace). Pointwise metrics such as safety_v1 instead read `prompt` +
    # `response`, so we also populate the final turn's prompt/response on the
    # same EvalCase. This lets reference-free single-turn metrics be computed in
    # the SAME service call as the multi-turn metrics. Multi-turn metrics ignore
    # the extra fields, so this is additive and safe.
    last_invocation = actual_invocations[-1] if actual_invocations else None
    eval_case = vertexai.types.EvalCase(
        agent_data=_MultiTurnVertexiAiEvalFacade._get_agent_data(
            actual_invocations
        ),
        prompt=(
            last_invocation.user_content
            if last_invocation is not None
            else None
        ),
        responses=(
            [
                vertexai.types.ResponseCandidate(
                    response=last_invocation.final_response
                )
            ]
            if last_invocation is not None
            and last_invocation.final_response is not None
            else None
        ),
    )
    dataset = vertexai.types.EvaluationDataset(eval_cases=[eval_case])
    eval_result = self._perform_eval(
        dataset=dataset, metrics=[spec.metric for spec in metric_specs]
    )

    results: dict[str, EvaluationResult] = {}
    for spec in metric_specs:
      spec_name = _resolve_metric_spec_name(spec.metric)
      score = self._get_score_for_metric(eval_result, spec_name)
      rubric_scores, explanation = self._get_rubric_reasoning_for_metric(
          eval_result, spec_name
      )

      per_invocation_results = []
      for actual, expected in zip(
          actual_invocations[:-1], expected_invocations[:-1]
      ):
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=None,
                eval_status=EvalStatus.NOT_EVALUATED,
            )
        )
      # The reasoning applies to the scored (last) turn.
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual_invocations[-1],
              expected_invocation=expected_invocations[-1],
              score=score,
              eval_status=self._get_eval_status_for_threshold(
                  score, spec.threshold
              ),
              rubric_scores=rubric_scores,
              explanation=explanation,
          )
      )

      if score is not None:
        results[spec.adk_metric_name] = EvaluationResult(
            overall_score=score,
            overall_eval_status=self._get_eval_status_for_threshold(
                score, spec.threshold
            ),
            per_invocation_results=per_invocation_results,
            overall_rubric_scores=rubric_scores,
            overall_explanation=explanation,
        )
      else:
        results[spec.adk_metric_name] = EvaluationResult()
    return results

  @staticmethod
  def _get_agent_data(
      actual_invocations: list[Invocation],
  ) -> vertexai.types.evals.AgentData:
    return vertexai.types.evals.AgentData(
        agents=_MultiTurnVertexiAiEvalFacade._get_agent_details(
            actual_invocations
        ),
        turns=_MultiTurnVertexiAiEvalFacade._get_turns(actual_invocations),
    )

  @staticmethod
  def _get_turns(
      actual_invocations: list[Invocation],
  ) -> list[vertexai.types.evals.ConversationTurn]:
    return [
        _MultiTurnVertexiAiEvalFacade._map_invocation_turn(index, invocation)
        for index, invocation in enumerate(actual_invocations)
    ]

  @staticmethod
  def _map_invocation_turn(
      turn_index: int,
      invocation: Invocation,
  ) -> vertexai.types.evals.ConversationTurn:
    agent_events = []
    if invocation.user_content is not None:
      agent_events.append(
          vertexai.types.evals.AgentEvent(
              author="user", content=invocation.user_content
          )
      )

    for invocation_event in invocation.intermediate_data.invocation_events:
      if invocation_event.content is None:
        # The managed eval service rejects events with empty content.
        continue
      agent_events.append(
          _MultiTurnVertexiAiEvalFacade._map_inovcation_event_to_agent_event(
              invocation_event
          )
      )

    if invocation.final_response is not None:
      agent_events.append(
          vertexai.types.evals.AgentEvent(
              author="agent", content=invocation.final_response
          )
      )

    return vertexai.types.evals.ConversationTurn(
        turn_index=turn_index,
        events=agent_events,
        turn_id=invocation.invocation_id,
    )

  @staticmethod
  def _map_inovcation_event_to_agent_event(
      invocation_event: InvocationEvent,
  ) -> vertexai.types.evals.AgentEvent:
    return vertexai.types.evals.AgentEvent(
        author=invocation_event.author, content=invocation_event.content
    )

  @staticmethod
  def _get_agent_details(
      actual_invocations: list[Invocation],
  ) -> dict[str, vertexai.types.evals.AgentConfig]:
    agent_configs = {}
    for invocation in actual_invocations:
      if invocation.app_details and invocation.app_details.agent_details:
        for (
            agent_name,
            agent_details,
        ) in invocation.app_details.agent_details.items():
          if agent_name not in agent_configs:
            agent_configs[agent_name] = (
                _MultiTurnVertexiAiEvalFacade._map_agent_details_to_agent_config(
                    agent_details
                )
            )

    return agent_configs

  @staticmethod
  def _map_agent_details_to_agent_config(
      agent_details: AgentDetails,
  ) -> vertexai.types.evals.AgentConfig:
    return vertexai.types.evals.AgentConfig(
        agent_id=agent_details.name,
        instruction=agent_details.instructions,
        tools=agent_details.tool_declarations,
    )
