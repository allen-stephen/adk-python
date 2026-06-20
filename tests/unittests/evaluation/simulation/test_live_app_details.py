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

"""Tests for building live-run AppDetails from the agent under test."""

from __future__ import annotations

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import Agent
from google.adk.evaluation.simulation.live_app_details import build_app_details
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
import pytest


def _roll_die(sides: int, tool_context: ToolContext) -> int:
  """Rolls a die.

  A tool with an ADK-injected `tool_context` parameter (defined at module scope
  so its type hints resolve), used to verify canonicalization strips it.

  Args:
    sides: The number of sides.
  """
  return sides


@pytest.mark.asyncio
async def test_build_app_details_captures_instruction_and_tools():
  agent = Agent(
      model="gemini-2.5-flash",
      name="roller",
      instruction="Always call the tools; never compute yourself.",
      tools=[_roll_die],
  )

  app_details = await build_app_details(agent)

  assert list(app_details.agent_details.keys()) == ["roller"]
  details = app_details.agent_details["roller"]
  assert details.name == "roller"
  assert details.instructions == (
      "Always call the tools; never compute yourself."
  )
  # Tools are recorded as genai Tool objects (matching the managed eval API and
  # the non-live path's `config.tools`), each wrapping function declarations with
  # the injected `tool_context` parameter stripped (canonicalization).
  assert len(details.tool_declarations) == 1
  tool = details.tool_declarations[0]
  assert isinstance(tool, genai_types.Tool)
  assert [fd.name for fd in tool.function_declarations] == ["_roll_die"]


@pytest.mark.asyncio
async def test_build_app_details_without_tools():
  agent = Agent(
      model="gemini-2.5-flash",
      name="chatter",
      instruction="Just chat.",
  )

  app_details = await build_app_details(agent)

  details = app_details.agent_details["chatter"]
  assert details.instructions == "Just chat."
  assert details.tool_declarations == []


@pytest.mark.asyncio
async def test_build_app_details_non_string_instruction_is_empty():
  """A callable/provider instruction resolves only at runtime, so it is empty."""
  agent = Agent(
      model="gemini-2.5-flash",
      name="dynamic",
      instruction=lambda ctx: "resolved at runtime",
  )

  app_details = await build_app_details(agent)

  details = app_details.agent_details["dynamic"]
  assert details.instructions == ""


@pytest.mark.asyncio
async def test_build_app_details_agent_without_canonical_tools():
  """A non-LLM agent (no canonical_tools) yields an entry with empty tools."""

  class _BareAgent(BaseAgent):
    pass

  agent = _BareAgent(name="bare")

  app_details = await build_app_details(agent)

  details = app_details.agent_details["bare"]
  assert details.name == "bare"
  assert details.tool_declarations == []
