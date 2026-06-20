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

"""Tests for the Vertex AI Scenario Generation Facade."""

from __future__ import annotations

import os

from google.adk.agents.base_agent import BaseAgent
from google.adk.dependencies.vertexai import vertexai
from google.adk.evaluation._vertex_ai_scenario_generation_facade import ScenarioGenerator
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig
from google.adk.tools.tool_context import ToolContext
import pytest

vertexai_types = vertexai.types


def _roll_die_with_tool_context(sides: int, tool_context: ToolContext) -> int:
  """Rolls a die.

  A tool with an ADK-injected `tool_context` parameter, defined at module scope
  so its type hints resolve. Used to verify the SDK does not choke on the
  injected param after tool canonicalization.

  Args:
    sides: The number of sides.
  """
  return sides


class TestScenarioGenerator:
  """Unit tests for ScenarioGenerator."""

  def test_constructor_with_api_key(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    ScenarioGenerator()

    mock_client_cls.assert_called_once_with(api_key="test_api_key")

  def test_constructor_with_project_and_location(self, mocker):
    """Test constructor with project and location in env."""
    mocker.patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test_project",
            "GOOGLE_CLOUD_LOCATION": "test_location",
        },
        clear=True,
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    ScenarioGenerator()

    mock_client_cls.assert_called_once_with(
        project="test_project", location="test_location"
    )

  def test_constructor_with_project_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_PROJECT": "test_project"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(ValueError, match="Missing location."):
      ScenarioGenerator()

  def test_constructor_with_location_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_LOCATION": "test_location"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(ValueError, match="Missing project id."):
      ScenarioGenerator()

  def test_constructor_with_no_env_vars_raises_error(self, mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")

    with pytest.raises(
        ValueError,
        match=(
            "Either API Key or Google cloud Project id and location should be"
            " specified."
        ),
    ):
      ScenarioGenerator()

  def test_generate_scenarios(self, mocker):
    """Test scenario generation with mocked components."""
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    mock_client = mock_client_cls.return_value

    # I need to mock AgentInfo.load_from_agent(agent=agent)
    mock_agent_info_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.types.evals.AgentInfo"
    )
    mock_agent_info_cls.load_from_agent.return_value = "mock_agent_info"

    mock_generate = mocker.patch.object(
        mock_client.evals, "generate_conversation_scenarios"
    )

    mock_eval_cases = [
        mocker.Mock(
            user_scenario=mocker.Mock(
                starting_prompt="Hello", conversation_plan="Say hello"
            )
        ),
        mocker.Mock(user_scenario=None),  # testing handling of None
        mocker.Mock(
            user_scenario=mocker.Mock(
                starting_prompt="Bye", conversation_plan="Say bye"
            )
        ),
    ]
    mock_generate.return_value = mocker.Mock(eval_cases=mock_eval_cases)

    generator = ScenarioGenerator()

    # An agent whose tools are canonicalized before SDK introspection (the SDK
    # needs `_get_declaration()`-capable tools). The canonical agent copy is what
    # must be passed to `load_from_agent`.
    canonical_agent = mocker.Mock(spec=BaseAgent)
    mock_agent = mocker.Mock(spec=BaseAgent)
    mock_agent.canonical_tools = mocker.AsyncMock(return_value=["canon_tool"])
    mock_agent.model_copy.return_value = canonical_agent
    config = ConversationGenerationConfig(
        count=2,
        generation_instruction="Test generation",
        model_name="gemini-2.5-flash",
    )

    scenarios = generator.generate_scenarios(mock_agent, config)

    assert len(scenarios) == 2
    assert scenarios[0].starting_prompt == "Hello"
    assert scenarios[0].conversation_plan == "Say hello"
    assert scenarios[1].starting_prompt == "Bye"
    assert scenarios[1].conversation_plan == "Say bye"

    # The agent is copied with its canonical tools, and that copy (not the raw
    # agent) is handed to the SDK.
    mock_agent.model_copy.assert_called_once_with(
        update={"tools": ["canon_tool"]}
    )
    mock_agent_info_cls.load_from_agent.assert_called_once_with(
        agent=canonical_agent
    )

    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert kwargs["agent_info"] == "mock_agent_info"
    passed_config = kwargs["config"]
    assert passed_config.count == 2
    assert passed_config.generation_instruction == "Test generation"

  def test_generate_scenarios_passes_through_agent_without_canonical_tools(
      self, mocker
  ):
    """A non-LLM agent (no canonical_tools) is introspected as-is."""
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    mock_client = mock_client_cls.return_value
    mock_agent_info_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.types.evals.AgentInfo"
    )
    mock_agent_info_cls.load_from_agent.return_value = "mock_agent_info"
    mocker.patch.object(
        mock_client.evals,
        "generate_conversation_scenarios",
        return_value=mocker.Mock(eval_cases=[]),
    )

    generator = ScenarioGenerator()

    # A bare BaseAgent has no `canonical_tools`; it must be passed through
    # unchanged (no model_copy).
    agent = mocker.Mock(spec=BaseAgent)
    del agent.canonical_tools

    generator.generate_scenarios(agent, ConversationGenerationConfig(count=1))

    mock_agent_info_cls.load_from_agent.assert_called_once_with(agent=agent)

  def test_generate_scenarios_canonicalizes_tools_with_injected_params(
      self, mocker
  ):
    """Regression: an agent whose tool has an ADK-injected param (e.g.
    `tool_context`) must not break SDK agent introspection.

    Raw callables with injected params cannot be parsed by the SDK's naive
    function-declaration fallback. Canonicalizing the tools first lets the SDK
    use `_get_declaration()`, which strips injected params. `load_from_agent`
    runs for real here (it is local introspection) to prove no error is raised.
    """
    from google.adk.agents.llm_agent import Agent

    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    mock_client = mock_client_cls.return_value
    mocker.patch.object(
        mock_client.evals,
        "generate_conversation_scenarios",
        return_value=mocker.Mock(eval_cases=[]),
    )

    agent = Agent(
        model="gemini-2.5-flash",
        name="roller",
        tools=[_roll_die_with_tool_context],
    )

    generator = ScenarioGenerator()

    # Must not raise the "Failed to parse the parameter tool_context" ValueError.
    scenarios = generator.generate_scenarios(
        agent, ConversationGenerationConfig(count=1)
    )

    assert scenarios == []
