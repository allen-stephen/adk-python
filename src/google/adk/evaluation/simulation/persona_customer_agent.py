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

"""Builds a native-audio Live agent that role-plays the simulated user.

For the native-audio transport the simulated user is itself a Live agent: it
hears the agent-under-test's audio and replies in kind. Its behavior is composed
from the same `ConversationScenario` that drives the text user simulator (the
starting prompt, the conversation plan, and the optional `UserPersona`), so the
conversation source is shared with non-live eval; only the voice comes from the
`VoiceProfile`.
"""

from __future__ import annotations

from typing import Optional

from google.genai import types

from ...agents.llm_agent import Agent
from ...models.google_llm import Gemini
from ...utils.feature_decorator import experimental
from ..conversation_scenarios import ConversationScenario
from .voice_profile import VoiceProfile

_DEFAULT_LIVE_MODEL = "gemini-live-2.5-flash-native-audio"

_PERSONA_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are a real person having a spoken conversation. This is NOT acting — you ARE \
this person. Speak naturally, the way a real person talks out loud.

## WHO YOU ARE
{persona}

## WHAT YOU WANT
{conversation_plan}

## HOW TO BEHAVE
- Stay in character at all times and speak conversationally.
- Open the conversation by saying, in your own words: "{starting_prompt}"
- Pursue your goal across the conversation; you do not need to say everything at \
once.
- React naturally to what the other speaker says.
- When you have fully accomplished your goal and there is nothing left to do, \
wrap up politely and stop.
- Do not narrate your actions or break character.
"""

_DEFAULT_PERSONA_DESCRIPTION = "an ordinary person with a goal to accomplish"


@experimental
class PersonaCustomerAgentFactory:
  """Creates a native-audio Live `Agent` that role-plays a conversation scenario."""

  def __init__(self, *, default_model: str = _DEFAULT_LIVE_MODEL):
    self._default_model = default_model

  def build(
      self,
      scenario: ConversationScenario,
      *,
      voice_profile: Optional[VoiceProfile] = None,
      model: Optional[str] = None,
  ) -> Agent:
    """Builds a native-audio persona agent for a scenario.

    Args:
      scenario: The conversation scenario whose plan, starting prompt, and
        persona drive the agent.
      voice_profile: The run-level voice/realism settings supplying the voice.
        When unset, a default `VoiceProfile` is used.
      model: Optional Live model override. When unset, the factory default is
        used.

    Returns:
      An `Agent` configured with the run's voice and system instruction.
    """
    voice_profile = voice_profile or VoiceProfile()
    resolved_model = model or self._default_model

    return Agent(
        name="simulated_user",
        model=Gemini(
            model=resolved_model,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_profile.voice_name,
                    )
                )
            ),
        ),
        description="Synthetic user simulator for live evaluation.",
        instruction=self.build_system_instruction(scenario),
    )

  @staticmethod
  def build_system_instruction(scenario: ConversationScenario) -> str:
    """Composes the persona agent's system instruction from the scenario."""
    persona = scenario.user_persona
    persona_text = (
        persona.description if persona else _DEFAULT_PERSONA_DESCRIPTION
    )
    return _PERSONA_SYSTEM_INSTRUCTION_TEMPLATE.format(
        persona=persona_text,
        conversation_plan=scenario.conversation_plan,
        starting_prompt=scenario.starting_prompt,
    )
