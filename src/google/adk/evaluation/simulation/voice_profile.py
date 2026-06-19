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

"""The voice/realism layer that lets an ordinary eval case run over audio.

Generating a conversation (a fixed script or a simulated user) is shared between
text and live eval. The *voice* a turn is spoken in, and the realism effects
applied to it, are the only live-specific data — so they live here as a small
profile that attaches to an existing `ConversationScenario` (or static
`EvalCase`) rather than as a parallel persona model.
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import Field

from ...utils.feature_decorator import experimental
from ..common import EvalBaseModel


class LiveTransport(enum.Enum):
  """How a user turn is carried to the agent under test.

  This is the seam between *what* the user says (shared user simulation) and
  *how* it reaches the agent (text vs. audio). It is selected per run on the
  eval config and may be overridden per scenario via `VoiceProfile.transport`.
  """

  TEXT = "text"
  """Text over the standard (HTTP) inference path. The default, non-live case."""

  TTS = "tts"
  """Synthesize each user turn to audio with a TTS voice, then stream it to the
  agent. Works with any conversation source (fixed script or simulated user) and
  supports timed barge-in."""

  NATIVE_AUDIO = "native_audio"
  """Drive a native-audio persona agent that hears the agent's audio and replies
  in kind. Supports true, reactive barge-in. Requires a simulated user (a
  `ConversationScenario`), not a fixed script."""


@experimental
class BargeInConfig(EvalBaseModel):
  """Configuration for the user interrupting (barging in on) the agent.

  Over the TTS transport this drives *timed* barge-in (the user starts speaking
  a configured amount into the agent's turn). Over the native-audio transport it
  enables *reactive* barge-in, where the persona interrupts in response to what
  it hears.
  """

  enabled: bool = Field(
      default=False,
      description="Whether the user may interrupt the agent under test.",
  )

  probability: float = Field(
      default=0.0,
      ge=0.0,
      le=1.0,
      description="Per-turn probability that the user interrupts the agent.",
  )

  min_listen_ms: int = Field(
      default=500,
      ge=0,
      description=(
          "Minimum milliseconds of the agent's turn the user listens to before"
          " it may interrupt."
      ),
  )

  max_listen_ms: int = Field(
      default=2000,
      ge=0,
      description=(
          "Maximum milliseconds of the agent's turn the user listens to before"
          " it may interrupt."
      ),
  )

  max_barge_ins: Optional[int] = Field(
      default=None,
      ge=0,
      description=(
          "Maximum number of barge-ins allowed across the whole conversation."
          " Once reached, no further interruptions occur even if the per-turn"
          " probability fires. When unset, there is no cap."
      ),
  )


@experimental
class AudioRealismConfig(EvalBaseModel):
  """Configuration for layering realism effects onto the user's audio.

  Realism transforms degrade the user's audio (e.g. noise, speaking-rate
  changes) before it reaches the agent under test, stress-testing the agent's
  robustness to imperfect real-world audio.
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

  speaking_rate: float = Field(
      default=1.0,
      gt=0.0,
      description=(
          "Time-stretch factor applied to the user's audio. 1.0 is unchanged,"
          " >1.0 speaks faster, <1.0 speaks slower."
      ),
  )


@experimental
class VoiceProfile(EvalBaseModel):
  """The voice and realism settings used to run a conversation over audio.

  Attaches to a `ConversationScenario` (or a static `EvalCase`) to make it
  runnable over an audio transport. It carries no conversation content of its
  own; the turns still come from user simulation or the static conversation.
  """

  voice_name: str = Field(
      default="Aoede",
      description=(
          "The prebuilt voice the user's turns are spoken with (e.g. 'Aoede',"
          " 'Kore')."
      ),
  )

  language_code: str = Field(
      default="en-US",
      description="BCP-47 language code the user speaks in.",
  )

  transport: Optional[LiveTransport] = Field(
      default=None,
      description=(
          "Optional per-scenario override of the run-level live transport. When"
          " unset, the transport chosen in the eval config is used."
      ),
  )

  audio_realism: Optional[AudioRealismConfig] = Field(
      default=None,
      description=(
          "Optional audio-realism configuration. When unset, the user's audio"
          " is passed through unmodified."
      ),
  )

  barge_in: Optional[BargeInConfig] = Field(
      default=None,
      description=(
          "Optional barge-in configuration. When unset, the conversation uses"
          " strict alternating turns."
      ),
  )
