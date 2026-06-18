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

"""A local acoustic metric: response latency for live (voice) eval.

Latency is the single most-felt quality of a voice agent and it can be measured
locally, with no cloud project, from event timestamps captured during a live
run. This evaluator measures, per invocation, the time between the user's turn
and the agent's first response, and passes when it is at or below a threshold.
"""

from __future__ import annotations

import logging
from typing import ClassVar
from typing import Optional

from typing_extensions import override

from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_case import InvocationEvents
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

logger = logging.getLogger("google_adk." + __name__)

_DEFAULT_LATENCY_THRESHOLD_SECONDS = 2.0


class LatencyV1Evaluator(Evaluator):
  """Measures the agent's response latency per invocation.

  For each invocation, latency is the elapsed time (in seconds) between the
  user's turn (`Invocation.creation_timestamp`) and the agent's first response
  event. A lower latency is better, so an invocation passes when its latency is
  at or below the configured threshold (in seconds).

  This metric runs entirely locally and requires no cloud project. If no agent
  response timestamp is available for an invocation, it is reported as
  NOT_EVALUATED rather than failed.
  """

  criterion_type: ClassVar[type[BaseCriterion]] = BaseCriterion

  def __init__(
      self,
      threshold: Optional[float] = None,
      eval_metric: Optional[EvalMetric] = None,
  ):
    if threshold is not None and eval_metric:
      raise ValueError(
          "Either eval_metric should be specified or threshold should be"
          " specified."
      )

    # Resolve the threshold preferring the structured `criterion` (the modern
    # field), then the legacy `eval_metric.threshold` (which the CLI and web UI
    # actually populate), then the explicit `threshold` arg, then the default.
    # Other evaluators read `eval_metric.threshold`, so honoring it here keeps a
    # consistent, configured threshold instead of silently falling back to the
    # default.
    if eval_metric and eval_metric.criterion:
      self._threshold = eval_metric.criterion.threshold
    elif eval_metric and eval_metric.threshold is not None:
      self._threshold = eval_metric.threshold
    elif threshold is not None:
      self._threshold = threshold
    else:
      self._threshold = _DEFAULT_LATENCY_THRESHOLD_SECONDS

  @staticmethod
  def _first_agent_event_timestamp(
      invocation: Invocation,
  ) -> Optional[float]:
    intermediate_data = invocation.intermediate_data
    if not isinstance(intermediate_data, InvocationEvents):
      return None
    for event in intermediate_data.invocation_events:
      if event.timestamp is not None:
        return event.timestamp
    return None

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    per_invocation_results = []
    latencies = []

    for actual in actual_invocations:
      first_response_ts = self._first_agent_event_timestamp(actual)
      if first_response_ts is None or actual.creation_timestamp is None:
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=None,
                eval_status=EvalStatus.NOT_EVALUATED,
            )
        )
        continue

      latency = first_response_ts - actual.creation_timestamp
      latencies.append(latency)
      eval_status = (
          EvalStatus.PASSED if latency <= self._threshold else EvalStatus.FAILED
      )
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              score=latency,
              eval_status=eval_status,
          )
      )

    if not latencies:
      return EvaluationResult(
          overall_score=None,
          overall_eval_status=EvalStatus.NOT_EVALUATED,
          per_invocation_results=per_invocation_results,
      )

    overall_score = sum(latencies) / len(latencies)
    overall_eval_status = (
        EvalStatus.PASSED
        if overall_score <= self._threshold
        else EvalStatus.FAILED
    )
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=overall_eval_status,
        per_invocation_results=per_invocation_results,
    )
