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

"""Typed models for persona-driven audio-to-audio live evaluation.

A live (voice) eval is fundamentally different from a text eval: instead of a
fixed list of user turns, a synthetic *persona* agent holds a real spoken
conversation with the agent under test. These models describe that persona, the
conversation it should pursue, and the optional realism knobs (barge-in,
audio degradation) that make the simulation more lifelike.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...utils.feature_decorator import experimental
from ..common import EvalBaseModel


@experimental
class BargeInConfig(EvalBaseModel):
  """Configuration for persona interruptions (barge-in).

  When enabled, the persona may start speaking before the agent under test has
  finished its turn, simulating a real user talking over the agent.
  """

  enabled: bool = Field(
      default=False,
      description="Whether the persona may interrupt the agent under test.",
  )

  probability: float = Field(
      default=0.0,
      ge=0.0,
      le=1.0,
      description="Per-turn probability that the persona interrupts the agent.",
  )

  min_listen_ms: int = Field(
      default=500,
      ge=0,
      description=(
          "Minimum milliseconds of the agent's turn the persona listens to"
          " before it may interrupt."
      ),
  )

  max_listen_ms: int = Field(
      default=2000,
      ge=0,
      description=(
          "Maximum milliseconds of the agent's turn the persona listens to"
          " before it may interrupt."
      ),
  )


@experimental
class AudioRealismConfig(EvalBaseModel):
  """Configuration for layering realism effects onto persona audio.

  Realism transforms degrade the persona's audio (e.g. noise, channel effects)
  before it reaches the agent under test, stress-testing the agent's robustness
  to imperfect real-world audio.
  """

  enabled: bool = Field(
      default=False,
      description="Whether to apply an audio-realism transform.",
  )

  intensity: float = Field(
      default=0.0,
      ge=0.0,
      le=1.0,
      description=(
          "Strength of the realism effect, from 0 (none) to 1 (maximum)."
      ),
  )

  background_noise: bool = Field(
      default=False,
      description="Whether to mix in background noise.",
  )


@experimental
class Persona(EvalBaseModel):
  """A synthetic customer/user persona that speaks with the agent under test.

  The persona's behavior is driven entirely by its `character_prompt`, which
  becomes the system instruction of a dedicated Live agent (LLM-first design).
  """

  id: str = Field(description="Stable identifier for the persona.")

  description: str = Field(
      default="",
      description="Short human-readable summary of who this persona is.",
  )

  character_prompt: str = Field(
      description=(
          "The persona's character and behavior, used verbatim as the core of"
          " the persona agent's system instruction."
      ),
  )

  goal: str = Field(
      default="",
      description=(
          "What the persona is trying to accomplish in the conversation (its"
          " intent)."
      ),
  )

  voice_name: str = Field(
      default="Aoede",
      description=(
          "The prebuilt voice the persona speaks with (e.g. 'Aoede', 'Kore')."
      ),
  )

  language_code: str = Field(
      default="en-US",
      description="BCP-47 language code the persona speaks in.",
  )

  model: Optional[str] = Field(
      default=None,
      description=(
          "Optional Live model override for the persona agent. Defaults to the"
          " same Live model as the agent under test when unset."
      ),
  )
