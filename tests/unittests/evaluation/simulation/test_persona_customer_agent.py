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

from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.simulation.persona_customer_agent import PersonaCustomerAgentFactory
from google.adk.evaluation.simulation.user_simulator_personas import UserBehavior
from google.adk.evaluation.simulation.user_simulator_personas import UserPersona
from google.adk.evaluation.simulation.voice_profile import VoiceProfile
from google.adk.models.google_llm import Gemini


def _make_scenario(**overrides) -> ConversationScenario:
  defaults = dict(
      starting_prompt="I'd like to order lunch.",
      conversation_plan="Order a burger and fries.",
  )
  defaults.update(overrides)
  return ConversationScenario(**defaults)


def test_build_uses_run_level_voice_profile_voice():
  """The built agent speaks with the run-level voice profile's prebuilt voice."""
  agent = PersonaCustomerAgentFactory().build(
      _make_scenario(), voice_profile=VoiceProfile(voice_name="Kore")
  )

  assert isinstance(agent.model, Gemini)
  voice = agent.model.speech_config.voice_config.prebuilt_voice_config
  assert voice.voice_name == "Kore"


def test_build_defaults_voice_when_no_profile():
  """A build without a voice profile still uses a default voice."""
  agent = PersonaCustomerAgentFactory().build(_make_scenario())

  voice = agent.model.speech_config.voice_config.prebuilt_voice_config
  assert voice.voice_name == "Aoede"


def test_system_instruction_embeds_plan_and_starting_prompt():
  """The scenario's plan and starting prompt appear in the instruction."""
  agent = PersonaCustomerAgentFactory().build(_make_scenario())

  assert "Order a burger and fries." in agent.instruction
  assert "I'd like to order lunch." in agent.instruction


def test_system_instruction_embeds_persona_description():
  """A user persona's description is woven into the instruction."""
  persona = UserPersona(
      id="impatient",
      description="A user who is in a rush and easily frustrated.",
      behaviors=[
          UserBehavior(
              name="Terse",
              description="Short replies.",
              behavior_instructions=["Keep it under 10 words."],
              violation_rubrics=["Response over 10 words."],
          )
      ],
  )
  agent = PersonaCustomerAgentFactory().build(
      _make_scenario(user_persona=persona)
  )

  assert "in a rush" in agent.instruction


def test_model_resolution_prefers_arg_then_default():
  """Model resolves from the explicit arg, else the factory default."""
  factory = PersonaCustomerAgentFactory(default_model="default-live")

  from_default = factory.build(_make_scenario())
  from_arg = factory.build(_make_scenario(), model="arg-live")

  assert from_default.model.model == "default-live"
  assert from_arg.model.model == "arg-live"
