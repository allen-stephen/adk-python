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

from typing import Optional

from typing_extensions import override

from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import BatchableEvaluator
from .evaluator import EvaluationResult
from .vertex_ai_eval_facade import _BatchMetricSpec
from .vertex_ai_eval_facade import _MultiTurnVertexiAiEvalFacade
from .vertex_ai_eval_facade import _SingleTurnVertexAiEvalFacade


class SafetyEvaluatorV1(BatchableEvaluator):
  """Evaluates safety (harmlessness) of an Agent's Response.

  The class delegates the responsibility to Vertex Gen AI Eval SDK. The V1
  suffix in the class name is added to convey that there could be other versions
  of the safety metric as well, and those metrics could use a different strategy
  to evaluate safety.

  Using this class requires a GCP project. Please set GOOGLE_CLOUD_PROJECT and
  GOOGLE_CLOUD_LOCATION in your .env file.

  Value range of the metric is [0, 1], with values closer to 1 to be more
  desirable (safe).
  """

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    from ..dependencies.vertexai import vertexai

    return _SingleTurnVertexAiEvalFacade(
        threshold=self._eval_metric.threshold,
        metric_name=vertexai.types.PrebuiltMetric.SAFETY,
    ).evaluate_invocations(
        actual_invocations, expected_invocations, conversation_scenario
    )

  @override
  def get_batch_group_key(self) -> str:
    # Safety is a pointwise metric (reads the final turn's prompt/response), but
    # the multi-turn facade also populates those fields on its EvalCase. Joining
    # the `vertex_multi_turn` group lets safety be computed in the SAME single
    # eval-service call as the multi-turn metrics, instead of a separate per-turn
    # call. (When run alone, the standalone path below still uses the single-turn
    # facade.)
    return "vertex_multi_turn"

  @override
  def get_batch_spec(self) -> _BatchMetricSpec:
    from ..dependencies.vertexai import vertexai

    return _BatchMetricSpec(
        adk_metric_name=self._eval_metric.metric_name,
        metric=vertexai.types.PrebuiltMetric.SAFETY,
        threshold=self._eval_metric.threshold,
    )

  @override
  def evaluate_batch(
      self,
      batch_specs: list[_BatchMetricSpec],
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
  ) -> dict[str, EvaluationResult]:
    from ..dependencies.vertexai import vertexai

    # Route through the multi-turn facade so safety shares the single multi-turn
    # eval-service call. The multi-turn EvalCase carries the final turn's
    # prompt/response, which is what the safety metric scores.
    return _MultiTurnVertexiAiEvalFacade(
        threshold=self._eval_metric.threshold,
        metric_name=vertexai.types.PrebuiltMetric.SAFETY,
    ).evaluate_invocations_for_metrics(
        batch_specs, actual_invocations, expected_invocations
    )
