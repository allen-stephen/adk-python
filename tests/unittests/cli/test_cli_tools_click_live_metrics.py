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

"""Tests for resolving the valid metric set for persona (live) eval runs."""

from google.adk.cli.cli_tools_click import _resolve_live_eval_metrics
from google.adk.evaluation.eval_metrics import EvalMetric


def _names(metrics):
  return [m.metric_name for m in metrics]


def test_default_no_config_uses_local_latency_only():
  """With no config and no managed flag, only local latency is used."""
  resolved = _resolve_live_eval_metrics(
      [], config_file_path=None, managed_metrics=False
  )

  assert _names(resolved) == ["response_latency_v1"]
  # The default latency threshold matches the realistic native-audio value.
  assert resolved[0].threshold == 5.0


def test_default_no_config_with_managed_adds_managed_set():
  """--managed_metrics adds the reference-free managed multi-turn metrics."""
  resolved = _resolve_live_eval_metrics(
      [], config_file_path=None, managed_metrics=True
  )

  assert _names(resolved) == [
      "response_latency_v1",
      "multi_turn_task_success_v1",
      "multi_turn_trajectory_quality_v1",
      "safety_v1",
  ]
  thresholds = {m.metric_name: m.threshold for m in resolved}
  assert thresholds["multi_turn_task_success_v1"] == 0.7
  assert thresholds["multi_turn_trajectory_quality_v1"] == 0.7
  assert thresholds["safety_v1"] == 0.8


def test_default_block_ignores_supplied_metrics_when_no_config():
  """When no config file is given, prior metrics are replaced by the defaults."""
  prior = [EvalMetric(metric_name="response_match_score", threshold=0.7)]

  resolved = _resolve_live_eval_metrics(
      prior, config_file_path=None, managed_metrics=False
  )

  assert _names(resolved) == ["response_latency_v1"]


def test_config_path_filters_reference_based_metrics():
  """A supplied config keeps valid metrics but drops reference-based ones."""
  configured = [
      EvalMetric(metric_name="response_latency_v1", threshold=3.0),
      EvalMetric(metric_name="multi_turn_task_success_v1", threshold=0.7),
      EvalMetric(metric_name="response_match_score", threshold=0.7),
      EvalMetric(metric_name="tool_trajectory_avg_score", threshold=1.0),
      EvalMetric(metric_name="response_evaluation_score", threshold=0.7),
      EvalMetric(metric_name="final_response_match_v2", threshold=0.7),
      EvalMetric(metric_name="safety_v1", threshold=0.8),
  ]

  resolved = _resolve_live_eval_metrics(
      configured, config_file_path="criteria.json", managed_metrics=False
  )

  # Reference-based metrics are dropped; valid (reference-free) ones are kept,
  # preserving their configured thresholds.
  assert _names(resolved) == [
      "response_latency_v1",
      "multi_turn_task_success_v1",
      "safety_v1",
  ]
  assert resolved[0].threshold == 3.0


def test_config_path_ignores_managed_flag():
  """With a config file, the managed flag does not inject extra metrics."""
  configured = [EvalMetric(metric_name="response_latency_v1", threshold=2.0)]

  resolved = _resolve_live_eval_metrics(
      configured, config_file_path="criteria.json", managed_metrics=True
  )

  assert _names(resolved) == ["response_latency_v1"]
  assert resolved[0].threshold == 2.0


def test_config_path_keeps_reference_free_tool_use_metric():
  """Reference-free metrics not in the defaults still pass through from config."""
  configured = [
      EvalMetric(metric_name="multi_turn_tool_use_quality_v1", threshold=0.7),
  ]

  resolved = _resolve_live_eval_metrics(
      configured, config_file_path="criteria.json", managed_metrics=False
  )

  assert _names(resolved) == ["multi_turn_tool_use_quality_v1"]
