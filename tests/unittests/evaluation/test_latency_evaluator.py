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

"""Tests for LatencyV1Evaluator.

Verifies that the latency metric measures the gap between the user's turn and
the agent's first response and passes/fails against a threshold.
"""

from __future__ import annotations

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.latency_evaluator import LatencyV1Evaluator
from google.genai import types


def _make_invocation(
    *, user_ts: float, first_agent_ts: float | None
) -> Invocation:
  events = []
  if first_agent_ts is not None:
    events.append(
        InvocationEvent(
            author="agent",
            content=types.Content(parts=[types.Part(text="hi")]),
            timestamp=first_agent_ts,
        )
    )
  return Invocation(
      invocation_id="inv1",
      user_content=types.Content(parts=[types.Part(text="hello")]),
      creation_timestamp=user_ts,
      intermediate_data=InvocationEvents(invocation_events=events),
  )


def test_latency_below_threshold_passes():
  """An agent that responds within the threshold passes."""
  evaluator = LatencyV1Evaluator(threshold=2.0)
  invocation = _make_invocation(user_ts=100.0, first_agent_ts=101.0)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED


def test_latency_above_threshold_fails():
  """An agent that responds slower than the threshold fails."""
  evaluator = LatencyV1Evaluator(threshold=2.0)
  invocation = _make_invocation(user_ts=100.0, first_agent_ts=105.0)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_score == 5.0
  assert result.overall_eval_status == EvalStatus.FAILED


def test_missing_agent_timestamp_is_not_evaluated():
  """An invocation with no agent response timestamp is not evaluated."""
  evaluator = LatencyV1Evaluator(threshold=2.0)
  invocation = _make_invocation(user_ts=100.0, first_agent_ts=None)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert result.per_invocation_results[0].eval_status == (
      EvalStatus.NOT_EVALUATED
  )


def test_overall_score_is_average_latency():
  """The overall score is the mean per-invocation latency."""
  evaluator = LatencyV1Evaluator(threshold=10.0)
  invocations = [
      _make_invocation(user_ts=0.0, first_agent_ts=1.0),
      _make_invocation(user_ts=0.0, first_agent_ts=3.0),
  ]

  result = evaluator.evaluate_invocations(invocations)

  assert result.overall_score == 2.0
  assert result.overall_eval_status == EvalStatus.PASSED


def test_legacy_threshold_field_is_honored():
  """A threshold set via the legacy `EvalMetric.threshold` field is used.

  The CLI and web UI populate `EvalMetric.threshold` (not `criterion`); the
  evaluator must honor it rather than silently falling back to the default.
  """
  eval_metric = EvalMetric(metric_name="response_latency_v1", threshold=5.0)
  evaluator = LatencyV1Evaluator(eval_metric=eval_metric)
  # 4s latency is under the configured 5.0 threshold -> PASS (would FAIL against
  # the 2.0 default).
  invocation = _make_invocation(user_ts=100.0, first_agent_ts=104.0)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_score == 4.0
  assert result.overall_eval_status == EvalStatus.PASSED


def test_criterion_threshold_takes_precedence():
  """A structured `criterion.threshold` is preferred over the legacy field."""
  eval_metric = EvalMetric(
      metric_name="response_latency_v1",
      threshold=10.0,
      criterion=BaseCriterion(threshold=3.0),
  )
  evaluator = LatencyV1Evaluator(eval_metric=eval_metric)
  # 4s latency exceeds the criterion's 3.0 threshold -> FAIL (even though the
  # legacy field says 10.0).
  invocation = _make_invocation(user_ts=100.0, first_agent_ts=104.0)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.FAILED


def test_defaults_to_two_seconds_when_unset():
  """With neither criterion nor threshold set, the default (2.0s) applies."""
  eval_metric = EvalMetric(metric_name="response_latency_v1")
  evaluator = LatencyV1Evaluator(eval_metric=eval_metric)
  invocation = _make_invocation(user_ts=100.0, first_agent_ts=103.0)

  result = evaluator.evaluate_invocations([invocation])

  # 3s latency exceeds the 2.0 default -> FAIL.
  assert result.overall_eval_status == EvalStatus.FAILED
