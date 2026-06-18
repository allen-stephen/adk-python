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

from abc import ABC
from typing import ClassVar
from typing import Optional

from pydantic import BaseModel
from typing_extensions import TypeAlias

from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalStatus
from .eval_rubrics import RubricScore

# Redefining the type here for backward compatibility.
EvalStatus: TypeAlias = EvalStatus


class PerInvocationResult(BaseModel):
  """Metric evaluation score per invocation."""

  actual_invocation: Invocation
  expected_invocation: Optional[Invocation] = None
  score: Optional[float] = None
  eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  rubric_scores: Optional[list[RubricScore]] = None
  explanation: Optional[str] = None
  """Free-text explanation of the result, when the metric provides one."""


class EvaluationResult(BaseModel):
  overall_score: Optional[float] = None
  """Overall score, based on each invocation."""

  overall_eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  """Overall status, based on each invocation."""

  per_invocation_results: list[PerInvocationResult] = []
  """Detailed results per invocation."""

  overall_rubric_scores: Optional[list[RubricScore]] = None
  """Overall rubric, based on each invocation."""

  overall_explanation: Optional[str] = None
  """Free-text explanation of the overall result, when one is provided."""


class Evaluator(ABC):
  """A metrics evaluator interface."""

  criterion_type: ClassVar[type[BaseCriterion]] = BaseCriterion

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
    raise NotImplementedError()


class BatchableEvaluator(Evaluator):
  """An evaluator whose metric can be computed alongside others in one call.

  Several ADK metrics are backed by the same external eval service and share an
  identical request payload (the conversation/turn data), differing only by the
  metric they request. Such metrics can be computed together in a single
  service call instead of one call each, which is a large latency win when many
  metrics are run (the common case for live/voice eval).

  Evaluators that opt in expose a `batch_group_key` (metrics sharing a key can
  be sent together) and a `batch_spec` (the per-metric handle + threshold). The
  actual batched call is delegated to `evaluate_batch`, which a single member of
  the group performs on behalf of the whole group.
  """

  def get_batch_group_key(self) -> Optional[str]:
    """Returns a key identifying metrics that can be batched together.

    Metrics that return the same non-None key can be computed in one call.
    Returning None opts this metric out of batching (it will run on its own).
    """
    return None

  def get_batch_spec(self) -> object:
    """Returns the per-metric spec (handle + threshold) for batched execution."""
    raise NotImplementedError()

  def evaluate_batch(
      self,
      batch_specs: list[object],
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
  ) -> dict[str, EvaluationResult]:
    """Computes all metrics in `batch_specs` in a single service call.

    Args:
      batch_specs: The specs (from `get_batch_spec`) of every metric in the
        group, including this evaluator's own.
      actual_invocations: Invocations from the agent under test.
      expected_invocations: Optional golden invocations.

    Returns:
      A mapping from ADK metric name to its EvaluationResult.
    """
    raise NotImplementedError()
