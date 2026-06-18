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

"""Builds a Live agent that role-plays a persona in audio-to-audio eval.

The persona agent is an ordinary ADK `LlmAgent` configured with a Live
(native-audio) model and the persona's prebuilt voice. All of its behavior —
how it speaks, what it wants, when it is done — lives in its system instruction,
composed from the `Persona` (LLM-first design).
"""

from __future__ import annotations

from typing import Optional

from google.genai import types

from ...agents.llm_agent import Agent
from ...models.google_llm import Gemini
from ...utils.feature_decorator import experimental
from .persona import Persona

_DEFAULT_LIVE_MODEL = "gemini-live-2.5-flash-native-audio"

_PERSONA_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are a real person having a spoken conversation. This is NOT acting — you ARE \
this person. Speak naturally, the way a real person talks out loud.

## WHO YOU ARE
{character_prompt}

## WHAT YOU WANT
{goal}

## HOW TO BEHAVE
- Stay in character at all times and speak conversationally.
- Pursue your goal across the conversation; you do not need to say everything at \
once.
- React naturally to what the other speaker says.
- When you have fully accomplished your goal and there is nothing left to do, \
wrap up politely and stop.
- Do not narrate your actions or break character.
"""


@experimental
class PersonaCustomerAgentFactory:
  """Creates a Live `Agent` that role-plays a given persona."""

  def __init__(self, *, default_model: str = _DEFAULT_LIVE_MODEL):
    self._default_model = default_model

  def build(self, persona: Persona, *, model: Optional[str] = None) -> Agent:
    """Builds a Live persona agent.

    Args:
      persona: The persona to role-play.
      model: Optional Live model override. Resolution order is this argument,
        then `persona.model`, then the factory default.

    Returns:
      An `Agent` configured with the persona's voice and system instruction.
    """
    resolved_model = model or persona.model or self._default_model

    return Agent(
        name=f"persona_{persona.id}",
        model=Gemini(
            model=resolved_model,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=persona.voice_name,
                    )
                )
            ),
        ),
        description=(
            persona.description or f"Synthetic persona '{persona.id}'."
        ),
        instruction=self.build_system_instruction(persona),
    )

  @staticmethod
  def build_system_instruction(persona: Persona) -> str:
    """Composes the persona agent's system instruction from the persona."""
    return _PERSONA_SYSTEM_INSTRUCTION_TEMPLATE.format(
        character_prompt=persona.character_prompt,
        goal=persona.goal or "Have a natural conversation.",
    )
