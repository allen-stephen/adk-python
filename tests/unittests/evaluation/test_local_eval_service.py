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
from typing import Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.eval_set import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.evaluation.evaluator import BatchableEvaluator
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import Evaluator
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.local_eval_service import _add_rubrics_to_invocation
from google.adk.evaluation.local_eval_service import _copy_eval_case_rubrics_to_actual_invocations
from google.adk.evaluation.local_eval_service import _copy_invocation_rubrics_to_actual_invocations
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from google.adk.models.registry import LLMRegistry
from google.genai import types as genai_types
import pytest
from typing_extensions import override


@pytest.fixture
def mock_eval_sets_manager(mocker):
  return mocker.create_autospec(EvalSetsManager)


@pytest.fixture
def dummy_agent():
  llm = LLMRegistry.new_llm("gemini-pro")
  return LlmAgent(name="test_agent", model=llm)


@pytest.fixture
def mock_eval_set_results_manager(mocker):
  return mocker.create_autospec(EvalSetResultsManager)


@pytest.fixture
def eval_service(
    dummy_agent, mock_eval_sets_manager, mock_eval_set_results_manager
):
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeEvaluator.get_metric_info(), evaluator=FakeEvaluator
  )
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeSingleSidedEvaluator.get_metric_info(),
      evaluator=FakeSingleSidedEvaluator,
  )
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeBatchableEvaluator.get_metric_info("fake_batch_metric_a"),
      evaluator=FakeBatchableEvaluatorA,
  )
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeBatchableEvaluator.get_metric_info("fake_batch_metric_b"),
      evaluator=FakeBatchableEvaluatorB,
  )
  FakeBatchableEvaluator.reset()
  return LocalEvalService(
      root_agent=dummy_agent,
      eval_sets_manager=mock_eval_sets_manager,
      eval_set_results_manager=mock_eval_set_results_manager,
  )


class FakeEvaluator(Evaluator):

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name="fake_metric",
        description="Fake metric description",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if expected_invocations is None:
      raise ValueError("expected_invocations is required for this metric.")
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=0.9,
              eval_status=EvalStatus.PASSED,
          )
      )
    return EvaluationResult(
        overall_score=0.9,
        overall_eval_status=EvalStatus.PASSED,
        per_invocation_results=per_invocation_results,
    )


class FakeSingleSidedEvaluator(Evaluator):

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name="fake_single_sided_metric",
        description="Fake single sided metric description",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    per_invocation_results = []
    for actual in actual_invocations:
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              score=0.995,
              eval_status=EvalStatus.PASSED,
          )
      )
    return EvaluationResult(
        overall_score=0.95,
        overall_eval_status=EvalStatus.PASSED,
        per_invocation_results=per_invocation_results,
    )


class FakeBatchableEvaluator(BatchableEvaluator):
  """A batchable evaluator that records how its batched call was invoked.

  Two concrete subclasses (`FakeBatchableEvaluatorA`/`B`) share the same batch
  group key, so the service should compute both in one `evaluate_batch` call.
  """

  metric_name: str = "fake_batch_metric"

  # Class-level bookkeeping shared across instances/subclasses so tests can
  # assert how many batched calls were made and with which specs.
  batch_call_count: int = 0
  last_batch_spec_names: list[str] = []

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @classmethod
  def reset(cls) -> None:
    FakeBatchableEvaluator.batch_call_count = 0
    FakeBatchableEvaluator.last_batch_spec_names = []

  @staticmethod
  def get_metric_info(metric_name: str) -> MetricInfo:
    return MetricInfo(
        metric_name=metric_name,
        description="Fake batchable metric description",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    return EvaluationResult(
        overall_score=0.9,
        overall_eval_status=EvalStatus.PASSED,
        per_invocation_results=[
            PerInvocationResult(
                actual_invocation=actual,
                score=0.9,
                eval_status=EvalStatus.PASSED,
            )
            for actual in actual_invocations
        ],
    )

  @override
  def get_batch_group_key(self) -> str:
    return "fake_batch_group"

  @override
  def get_batch_spec(self) -> str:
    return self._eval_metric.metric_name

  @override
  def evaluate_batch(
      self,
      batch_specs: list[str],
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
  ) -> dict[str, EvaluationResult]:
    FakeBatchableEvaluator.batch_call_count += 1
    FakeBatchableEvaluator.last_batch_spec_names = list(batch_specs)
    results: dict[str, EvaluationResult] = {}
    for spec in batch_specs:
      results[spec] = EvaluationResult(
          overall_score=0.9,
          overall_eval_status=EvalStatus.PASSED,
          per_invocation_results=[
              PerInvocationResult(
                  actual_invocation=actual,
                  score=0.9,
                  eval_status=EvalStatus.PASSED,
              )
              for actual in actual_invocations
          ],
      )
    return results


class FakeBatchableEvaluatorA(FakeBatchableEvaluator):
  metric_name = "fake_batch_metric_a"


class FakeBatchableEvaluatorB(FakeBatchableEvaluator):
  metric_name = "fake_batch_metric_b"


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_batches_metrics(
    eval_service, mock_eval_sets_manager, mocker
):
  """Metrics sharing a batch group are computed in a single batched call."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=[invocation.model_copy(deep=True)],
      session_id="session1",
  )
  evaluate_config = EvaluateConfig(
      eval_metrics=[
          EvalMetric(metric_name="fake_batch_metric_a", threshold=0.5),
          EvalMetric(metric_name="fake_batch_metric_b", threshold=0.5),
      ],
      parallelism=1,
  )

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [invocation.model_copy(deep=True)]
  mock_eval_case.conversation_scenario = None
  mock_eval_case.live_persona_scenario = None
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )

  # Both metrics computed via a single batched call.
  assert FakeBatchableEvaluator.batch_call_count == 1
  assert set(FakeBatchableEvaluator.last_batch_spec_names) == {
      "fake_batch_metric_a",
      "fake_batch_metric_b",
  }
  # Both metrics still produce overall results.
  metric_names = {r.metric_name for r in result.overall_eval_metric_results}
  assert metric_names == {"fake_batch_metric_a", "fake_batch_metric_b"}
  for overall in result.overall_eval_metric_results:
    assert overall.score == 0.9
    assert overall.eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_single_batchable_metric_not_batched(
    eval_service, mock_eval_sets_manager, mocker
):
  """A lone batchable metric runs via the standard path (no batched call)."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=[invocation.model_copy(deep=True)],
      session_id="session1",
  )
  evaluate_config = EvaluateConfig(
      eval_metrics=[
          EvalMetric(metric_name="fake_batch_metric_a", threshold=0.5),
      ],
      parallelism=1,
  )

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [invocation.model_copy(deep=True)]
  mock_eval_case.conversation_scenario = None
  mock_eval_case.live_persona_scenario = None
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )

  # Single metric: no batched call; falls back to per-metric evaluation.
  assert FakeBatchableEvaluator.batch_call_count == 0
  assert len(result.overall_eval_metric_results) == 1
  assert (
      result.overall_eval_metric_results[0].metric_name == "fake_batch_metric_a"
  )


def test_safety_and_multi_turn_metrics_share_one_batch_group(eval_service):
  """Safety batches with the multi-turn metrics (single managed eval call)."""
  metrics = [
      EvalMetric(metric_name="multi_turn_task_success_v1", threshold=0.7),
      EvalMetric(metric_name="multi_turn_trajectory_quality_v1", threshold=0.7),
      EvalMetric(metric_name="safety_v1", threshold=0.8),
  ]

  batched_groups, standalone = eval_service._partition_metrics_for_batching(
      metrics
  )

  # All three managed metrics fall into a single batch group; none standalone.
  assert standalone == []
  assert len(batched_groups) == 1
  group = next(iter(batched_groups.values()))
  group_names = {m.metric_name for m in group}
  assert group_names == {
      "multi_turn_task_success_v1",
      "multi_turn_trajectory_quality_v1",
      "safety_v1",
  }


@pytest.mark.asyncio
async def test_perform_inference_success(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
    mocker,
):
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", conversation=[], session_input=None),
          EvalCase(eval_id="case2", conversation=[], session_input=None),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  mock_inference_result = mocker.MagicMock()
  eval_service._perform_inference_single_eval_item = mocker.AsyncMock(
      return_value=mock_inference_result
  )

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="test_eval_set",
      inference_config=InferenceConfig(parallelism=2),
  )

  results = []
  async for result in eval_service.perform_inference(inference_request):
    results.append(result)

  assert len(results) == 2
  assert results[0] == mock_inference_result
  assert results[1] == mock_inference_result
  mock_eval_sets_manager.get_eval_set.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set"
  )
  assert eval_service._perform_inference_single_eval_item.call_count == 2


@pytest.mark.asyncio
async def test_perform_inference_with_case_ids(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
    mocker,
):
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", conversation=[], session_input=None),
          EvalCase(eval_id="case2", conversation=[], session_input=None),
          EvalCase(eval_id="case3", conversation=[], session_input=None),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  mock_inference_result = mocker.MagicMock()
  eval_service._perform_inference_single_eval_item = mocker.AsyncMock(
      return_value=mock_inference_result
  )

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_ids=["case1", "case3"],
      inference_config=InferenceConfig(parallelism=1),
  )

  results = []
  async for result in eval_service.perform_inference(inference_request):
    results.append(result)

  assert len(results) == 2
  eval_service._perform_inference_single_eval_item.assert_any_call(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[0],
      root_agent=dummy_agent,
      use_live=False,
      live_persona_scenario=None,
      live_timeout_seconds=300,
  )
  eval_service._perform_inference_single_eval_item.assert_any_call(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[2],
      root_agent=dummy_agent,
      use_live=False,
      live_persona_scenario=None,
      live_timeout_seconds=300,
  )


@pytest.mark.asyncio
async def test_perform_inference_with_use_live(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
    mocker,
):
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", conversation=[], session_input=None),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  mock_inference_result = mocker.MagicMock()
  eval_service._perform_inference_single_eval_item = mocker.AsyncMock(
      return_value=mock_inference_result
  )

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="test_eval_set",
      inference_config=InferenceConfig(
          parallelism=1, use_live=True, live_timeout_seconds=600
      ),
  )

  results = []
  async for result in eval_service.perform_inference(inference_request):
    results.append(result)

  assert len(results) == 1
  eval_service._perform_inference_single_eval_item.assert_called_once_with(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[0],
      root_agent=dummy_agent,
      use_live=True,
      live_persona_scenario=None,
      live_timeout_seconds=600,
  )


@pytest.mark.asyncio
async def test_perform_inference_uses_persona_scenario_from_eval_case(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
    mocker,
):
  """A persona saved on the eval case drives inference without a request scenario."""
  from google.adk.evaluation.simulation.live_conversation_scenario import LiveConversationScenario
  from google.adk.evaluation.simulation.persona import Persona

  scenario = LiveConversationScenario(
      persona=Persona(id="saved", character_prompt="c", goal="g"), max_turns=2
  )
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", live_persona_scenario=scenario),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  eval_service._perform_inference_single_eval_item = mocker.AsyncMock(
      return_value=mocker.MagicMock()
  )

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="test_eval_set",
      inference_config=InferenceConfig(parallelism=1),
  )

  async for _ in eval_service.perform_inference(inference_request):
    pass

  eval_service._perform_inference_single_eval_item.assert_called_once_with(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[0],
      root_agent=dummy_agent,
      use_live=False,
      live_persona_scenario=scenario,
      live_timeout_seconds=300,
  )


@pytest.mark.asyncio
async def test_perform_inference_eval_set_not_found(
    eval_service,
    mock_eval_sets_manager,
):
  mock_eval_sets_manager.get_eval_set.return_value = None

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="not_found_set",
      inference_config=InferenceConfig(parallelism=1),
  )

  with pytest.raises(NotFoundError):
    async for _ in eval_service.perform_inference(inference_request):
      pass


@pytest.mark.asyncio
async def test_evaluate_success(
    eval_service, mock_eval_sets_manager, mock_eval_set_results_manager, mocker
):
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_results = [
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case1",
          inferences=[invocation.model_copy(deep=True)],
          session_id="session1",
      ),
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case2",
          inferences=[invocation.model_copy(deep=True)],
          session_id="session2",
      ),
  ]
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=[eval_metric], parallelism=2),
  )

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [invocation.model_copy(deep=True)]
  mock_eval_case.conversation_scenario = None
  mock_eval_case.live_persona_scenario = None
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  results = []
  async for result in eval_service.evaluate(evaluate_request):
    results.append(result)

  assert len(results) == 2
  assert isinstance(results[0], EvalCaseResult)
  assert isinstance(results[1], EvalCaseResult)
  assert mock_eval_sets_manager.get_eval_case.call_count == 2
  assert mock_eval_set_results_manager.save_eval_set_result.call_count == 1


@pytest.mark.asyncio
async def test_evaluate_eval_case_not_found(
    eval_service,
    mock_eval_sets_manager,
):
  inference_results = [
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case1",
          inferences=[],
          session_id="session1",
      ),
  ]
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=[eval_metric], parallelism=1),
  )

  mock_eval_sets_manager.get_eval_case.return_value = None

  with pytest.raises(NotFoundError):
    async for _ in eval_service.evaluate(evaluate_request):
      pass

  mock_eval_sets_manager.get_eval_case.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_single_inference_result(
    eval_service, mock_eval_sets_manager, mock_eval_set_results_manager, mocker
):
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=[
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
      ],
      session_id="session1",
  )
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_config = EvaluateConfig(eval_metrics=[eval_metric], parallelism=1)

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [
      invocation.model_copy(deep=True),
      invocation.model_copy(deep=True),
      invocation.model_copy(deep=True),
  ]
  mock_eval_case.conversation_scenario = None
  mock_eval_case.live_persona_scenario = None
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )

  assert isinstance(result, EvalCaseResult)
  assert result.eval_id == "case1"
  assert result.session_id == "session1"
  assert len(result.overall_eval_metric_results) == 1
  assert result.overall_eval_metric_results[0].metric_name == "fake_metric"
  assert result.overall_eval_metric_results[0].score == 0.9
  mock_eval_sets_manager.get_eval_case.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set", eval_case_id="case1"
  )

  assert len(result.eval_metric_result_per_invocation) == 3
  for i in range(3):
    invocation_result = result.eval_metric_result_per_invocation[i]
    assert invocation_result.actual_invocation == inference_result.inferences[i]
    assert (
        invocation_result.expected_invocation == mock_eval_case.conversation[i]
    )
    assert len(invocation_result.eval_metric_results) == 1
    metric_result = invocation_result.eval_metric_results[0]
    assert metric_result.metric_name == "fake_metric"
    assert metric_result.score == 0.9
    assert metric_result.eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_failed_without_inferences(
    eval_service, mock_eval_sets_manager, mocker
):
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=None,
      session_id="session1",
      status=InferenceStatus.FAILURE,
      error_message="auth failed",
  )
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_config = EvaluateConfig(eval_metrics=[eval_metric], parallelism=1)

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = []
  mock_eval_case.conversation_scenario = None
  mock_eval_case.live_persona_scenario = None
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )

  assert result.eval_id == "case1"
  assert result.session_id == "session1"
  assert result.final_eval_status == EvalStatus.FAILED
  assert result.overall_eval_metric_results == []
  assert result.eval_metric_result_per_invocation == []


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_for_conversation_scenario(
    eval_service, mock_eval_sets_manager, mocker
):
  """To be removed once evaluation is implemented for conversation scenarios."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=[
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
      ],
      session_id="session1",
  )
  eval_metric = EvalMetric(
      metric_name="fake_single_sided_metric", threshold=0.5
  )
  evaluate_config = EvaluateConfig(eval_metrics=[eval_metric], parallelism=1)

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = None
  mock_eval_case.conversation_scenario = mocker.MagicMock()
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )
  assert isinstance(result, EvalCaseResult)
  assert result.eval_id == "case1"
  assert result.final_eval_status == EvalStatus.PASSED
  assert len(result.overall_eval_metric_results) == 1
  assert (
      result.overall_eval_metric_results[0].metric_name
      == "fake_single_sided_metric"
  )
  assert result.overall_eval_metric_results[0].score == 0.95
  mock_eval_sets_manager.get_eval_case.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set", eval_case_id="case1"
  )

  assert len(result.eval_metric_result_per_invocation) == 3
  for i in range(3):
    invocation_result = result.eval_metric_result_per_invocation[i]
    assert invocation_result.actual_invocation == inference_result.inferences[i]
    assert invocation_result.expected_invocation is None
    assert len(invocation_result.eval_metric_results) == 1
    metric_result = invocation_result.eval_metric_results[0]
    assert metric_result.metric_name == "fake_single_sided_metric"
    assert metric_result.score == 0.995
    assert metric_result.eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_for_conversation_scenario_with_unsupported_metric(
    eval_service, mock_eval_sets_manager, mocker
):
  """To be removed once evaluation is implemented for conversation scenarios."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=[
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
      ],
      session_id="session1",
  )
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_config = EvaluateConfig(eval_metrics=[eval_metric], parallelism=1)

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.eval_id = "case1"
  mock_eval_case.conversation = None
  mock_eval_case.conversation_scenario = mocker.MagicMock()
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )
  assert isinstance(result, EvalCaseResult)
  assert result.eval_id == "case1"
  assert result.final_eval_status == EvalStatus.NOT_EVALUATED
  assert len(result.overall_eval_metric_results) == 1
  assert result.overall_eval_metric_results[0].metric_name == "fake_metric"
  assert result.overall_eval_metric_results[0].score is None
  mock_eval_sets_manager.get_eval_case.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set", eval_case_id="case1"
  )

  assert len(result.eval_metric_result_per_invocation) == 3


def test_generate_final_eval_status_doesn_t_throw_on(eval_service):
  # How to fix if this test case fails?
  # This test case has failed mainly because a new EvalStatus got added. You
  # mostly need to update _generate_final_eval_status method to handle the new
  # eval case.

  # We go over all the possible values of EvalStatus one by one and expect
  # the _generate_final_eval_status to handle it without throwing an exception.
  for status in EvalStatus:
    eval_metric_result = EvalMetricResult(
        metric_name="metric1", threshold=0.5, eval_status=status
    )
    eval_service._generate_final_eval_status([eval_metric_result])


@pytest.mark.asyncio
async def test_mcp_stdio_agent_no_runtime_error(mocker):
  """Test that LocalEvalService can handle MCP stdio agents without RuntimeError.

  This is a regression test for GitHub issue #2196:
  "RuntimeError: Attempted to exit cancel scope in a different task than it was
  entered in"

  The fix ensures that Runner.close() is called to properly cleanup MCP
  connections.
  """
  import tempfile

  from google.adk.evaluation.local_eval_service import LocalEvalService
  from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
  from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
  from mcp import StdioServerParameters

  # Mock LLM responses to avoid real API calls
  from tests.unittests.testing_utils import MockModel

  mock_responses = [
      genai_types.Content(
          parts=[genai_types.Part(text="Mocked response from test agent")]
      )
  ]
  mock_model = MockModel.create(responses=mock_responses)

  # Create a test agent with MCP stdio toolset and mocked model
  test_dir = tempfile.mkdtemp()
  try:
    agent = LlmAgent(
        model=mock_model,
        name="test_mcp_agent",
        instruction="Test agent for MCP stdio regression test.",
        tools=[
            MCPToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="npx",
                        args=[
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            test_dir,
                        ],
                    ),
                    timeout=5,
                ),
                tool_filter=["read_file", "list_directory"],
            )
        ],
    )

    # Create a mock eval sets manager that returns an eval case
    mock_eval_sets_manager = mocker.create_autospec(EvalSetsManager)
    test_eval_case = EvalCase(
        eval_id="test_mcp_case",
        conversation=[
            Invocation(
                user_content=genai_types.Content(
                    parts=[genai_types.Part(text="List directory contents")]
                ),
            )
        ],
    )
    mock_eval_sets_manager.get_eval_case.return_value = test_eval_case
    eval_set = EvalSet(
        eval_set_id="test_set",
        eval_cases=[test_eval_case],
    )
    mock_eval_sets_manager.get_eval_set.return_value = eval_set

    # Create LocalEvalService with MCP agent
    eval_service = LocalEvalService(
        root_agent=agent,
        eval_sets_manager=mock_eval_sets_manager,
    )

    # Create inference request to actually trigger the code path with the fix
    inference_request = InferenceRequest(
        app_name="test_app",
        eval_set_id="test_set",
        inference_config=InferenceConfig(parallelism=1),
    )

    # The main test: actually call perform_inference which will trigger
    # _generate_inferences_from_root_agent where the fix is located

    # Note: In Python 3.10 and 3.11, there may be asyncio.CancelledError during cleanup
    # due to anyio cancel scope context violations when MCP toolsets are cleaned up
    # via asyncio.wait_for() in different task contexts. Python 3.12+ enhanced task
    # context management (Task.get_context(), improved context propagation) resolves this.

    try:
      results = []
      async for result in eval_service.perform_inference(inference_request):
        results.append(result)
        # We should get at least one result since we mocked the LLM
        break

      # Test passes if we get here without the cancel scope RuntimeError
      # With mocked model, we should get successful inference results
      assert len(results) >= 1

    except RuntimeError as e:
      # If we get a RuntimeError about cancel scope, the fix isn't working
      if "cancel scope" in str(e) and "different task" in str(e):
        pytest.fail(f"MCP stdio RuntimeError regression detected: {e}")
      else:
        # Other RuntimeErrors might be acceptable
        pass
    except asyncio.CancelledError as e:
      # In Python 3.10 and 3.11, anyio cancel scope context violations may manifest as CancelledError
      # when MCP RequestResponder.__exit__() is called in a different task than __enter__()
      if (
          hasattr(e, "args")
          and len(e.args) > 0
          and "cancel scope" in str(e.args[0])
      ):
        pytest.fail(f"MCP stdio cancel scope error regression detected: {e}")
      else:
        # Re-raise other CancelledErrors
        raise
    except Exception as e:
      # Check if this is the specific cancel scope error we're testing for
      if "cancel scope" in str(e) and "different task" in str(e):
        pytest.fail(f"MCP stdio RuntimeError regression detected: {e}")
      # Other exceptions are acceptable for this test

    # The main goal is to ensure the test completes without the specific
    # RuntimeError about cancel scopes. If we reach here, the fix is working.

  finally:
    # Cleanup
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)


def test_add_rubrics_to_invocation_initializes_rubrics_list():
  invocation = Invocation(user_content=genai_types.Content())
  rubric = Rubric(
      rubric_id="r1", rubric_content=RubricContent(text_property="p1")
  )
  _add_rubrics_to_invocation(invocation, [rubric])
  assert invocation.rubrics == [rubric]


def test_add_rubrics_to_invocation_adds_to_existing_list():
  rubric1 = Rubric(
      rubric_id="r1", rubric_content=RubricContent(text_property="p1")
  )
  rubric2 = Rubric(
      rubric_id="r2", rubric_content=RubricContent(text_property="p2")
  )
  invocation = Invocation(user_content=genai_types.Content(), rubrics=[rubric1])
  _add_rubrics_to_invocation(invocation, [rubric2])
  assert invocation.rubrics == [rubric1, rubric2]


def test_add_rubrics_to_invocation_errors_on_duplicate_id():
  rubric1 = Rubric(
      rubric_id="r1", rubric_content=RubricContent(text_property="p1")
  )
  rubric2 = Rubric(
      rubric_id="r1", rubric_content=RubricContent(text_property="p2")
  )
  invocation = Invocation(user_content=genai_types.Content(), rubrics=[rubric1])
  with pytest.raises(ValueError):
    _add_rubrics_to_invocation(invocation, [rubric2])


def test_copy_eval_case_rubrics_to_actual_invocations():
  rubric1 = Rubric(
      rubric_id="r1", rubric_content=RubricContent(text_property="p1")
  )
  eval_case = EvalCase(
      eval_id="case1",
      conversation=[
          Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="expected invocation 1.")]
              )
          ),
          Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="expected invocation 2.")]
              )
          ),
      ],
      rubrics=[rubric1],
  )
  invocations = [
      Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="actual invocation 1.")]
          )
      ),
      Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="actual invocation 2.")]
          )
      ),
  ]
  _copy_eval_case_rubrics_to_actual_invocations(eval_case, invocations)
  assert invocations[0].rubrics == [rubric1]
  assert invocations[1].rubrics == [rubric1]


def test_copy_invocation_rubrics_to_actual_invocations():
  rubric1 = Rubric(
      rubric_id="r1", rubric_content=RubricContent(text_property="p1")
  )
  rubric2 = Rubric(
      rubric_id="r2", rubric_content=RubricContent(text_property="p2")
  )
  expected = [
      Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="expected invocation 1.")]
          ),
          rubrics=[rubric1],
      ),
      Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="expected invocation 2.")]
          ),
          rubrics=[rubric2],
      ),
  ]
  actual = [
      Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="actual invocation 1.")]
          )
      ),
      Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="actual invocation 2.")]
          )
      ),
  ]
  _copy_invocation_rubrics_to_actual_invocations(expected, actual)
  assert actual[0].rubrics == [rubric1]
  assert actual[1].rubrics == [rubric2]


@pytest.mark.asyncio
async def test_perform_inference_single_eval_item_live(
    eval_service, dummy_agent, mocker
):
  eval_case = EvalCase(eval_id="case1", conversation=[], session_input=None)
  mock_generate_live = mocker.patch(
      "google.adk.evaluation.evaluation_generator.EvaluationGenerator._generate_inferences_from_root_agent_live"
  )
  mock_generate_live.return_value = []

  eval_service._session_id_supplier = mocker.MagicMock(
      return_value="test_session_id"
  )
  mock_user_sim = mocker.MagicMock()
  eval_service._user_simulator_provider.provide = mocker.MagicMock(
      return_value=mock_user_sim
  )

  await eval_service._perform_inference_single_eval_item(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_case,
      root_agent=dummy_agent,
      use_live=True,
      live_persona_scenario=None,
      live_timeout_seconds=600,
  )

  mock_generate_live.assert_called_once_with(
      root_agent=dummy_agent,
      user_simulator=mock_user_sim,
      initial_session=None,
      session_id="test_session_id",
      session_service=eval_service._session_service,
      artifact_service=eval_service._artifact_service,
      memory_service=eval_service._memory_service,
      live_timeout_seconds=600,
  )


@pytest.mark.asyncio
async def test_perform_inference_single_eval_item_non_live(
    eval_service, dummy_agent, mocker
):
  eval_case = EvalCase(eval_id="case1", conversation=[], session_input=None)
  mock_generate = mocker.patch(
      "google.adk.evaluation.evaluation_generator.EvaluationGenerator._generate_inferences_from_root_agent"
  )
  mock_generate.return_value = []

  eval_service._session_id_supplier = mocker.MagicMock(
      return_value="test_session_id"
  )
  mock_user_sim = mocker.MagicMock()
  eval_service._user_simulator_provider.provide = mocker.MagicMock(
      return_value=mock_user_sim
  )

  await eval_service._perform_inference_single_eval_item(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_case,
      root_agent=dummy_agent,
      use_live=False,
      live_persona_scenario=None,
      live_timeout_seconds=300,
  )

  mock_generate.assert_called_once_with(
      root_agent=dummy_agent,
      user_simulator=mock_user_sim,
      initial_session=None,
      session_id="test_session_id",
      session_service=eval_service._session_service,
      artifact_service=eval_service._artifact_service,
      memory_service=eval_service._memory_service,
  )
