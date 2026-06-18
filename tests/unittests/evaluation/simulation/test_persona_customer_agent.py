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

"""Tests for PersonaCustomerAgentFactory."""

from __future__ import annotations

from google.adk.evaluation.simulation.persona import Persona
from google.adk.evaluation.simulation.persona_customer_agent import PersonaCustomerAgentFactory
from google.adk.models.google_llm import Gemini


def _make_persona(**overrides) -> Persona:
  defaults = dict(
      id="hungry_customer",
      character_prompt="You are a hungry customer ordering lunch.",
      goal="Order a burger and fries.",
      voice_name="Kore",
  )
  defaults.update(overrides)
  return Persona(**defaults)


def test_build_uses_persona_voice():
  """The built agent speaks with the persona's prebuilt voice."""
  agent = PersonaCustomerAgentFactory().build(_make_persona())

  assert isinstance(agent.model, Gemini)
  voice = agent.model.speech_config.voice_config.prebuilt_voice_config
  assert voice.voice_name == "Kore"


def test_system_instruction_embeds_character_and_goal():
  """The persona's character prompt and goal appear in the instruction."""
  agent = PersonaCustomerAgentFactory().build(_make_persona())

  assert "hungry customer ordering lunch" in agent.instruction
  assert "Order a burger and fries." in agent.instruction


def test_model_resolution_order():
  """Model resolves from arg, then persona.model, then factory default."""
  factory = PersonaCustomerAgentFactory(default_model="default-live")

  from_default = factory.build(_make_persona())
  from_persona = factory.build(_make_persona(model="persona-live"))
  from_arg = factory.build(
      _make_persona(model="persona-live"), model="arg-live"
  )

  assert from_default.model.model == "default-live"
  assert from_persona.model.model == "persona-live"
  assert from_arg.model.model == "arg-live"


def test_agent_name_is_namespaced_by_persona_id():
  """The agent name is derived from the persona id."""
  agent = PersonaCustomerAgentFactory().build(_make_persona())

  assert agent.name == "persona_hungry_customer"
