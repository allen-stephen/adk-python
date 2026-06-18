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

"""Captured-turn types produced by a persona-driven live conversation run.

These are intermediate, in-memory results (not the eval-case contract). A
materializer converts a `LiveConversation` into `Invocation`s for scoring and
storage.
"""

from __future__ import annotations

from typing import Literal
from typing import Optional

from google.genai import types as genai_types
from pydantic import Field

from ...utils.feature_decorator import experimental
from ..app_details import AppDetails
from ..common import EvalBaseModel
from ..eval_case import AudioReference

Speaker = Literal["persona", "sut"]


@experimental
class CapturedUtterance(EvalBaseModel):
  """One speaker's spoken contribution within a turn."""

  speaker: Speaker = Field(description="Who spoke: the persona or the SUT.")

  transcript: str = Field(
      default="",
      description="The transcribed text of the spoken audio.",
  )

  audio_pcm: Optional[bytes] = Field(
      default=None,
      description="Raw PCM bytes of the spoken audio, if captured.",
  )

  audio_reference: Optional[AudioReference] = Field(
      default=None,
      description=(
          "Reference to the persisted audio artifact, set once the audio has"
          " been saved to the artifact service."
      ),
  )

  mime_type: str = Field(
      default="audio/pcm;rate=24000",
      description="MIME type of the captured audio.",
  )

  start_time: float = Field(
      default=0.0,
      description=(
          "Epoch seconds when collection of this utterance started (i.e. when"
          " the runner began waiting for this speaker). Note: this is before"
          " the speaker generated/spoke anything, so it is NOT a good latency"
          " anchor; use first_audio_time / input_sent_time instead."
      ),
  )

  end_time: float = Field(
      default=0.0,
      description="Epoch seconds when this utterance completed.",
  )

  first_audio_time: Optional[float] = Field(
      default=None,
      description=(
          "Epoch seconds when this speaker's first audio chunk arrived, i.e."
          " the moment they started responding. Used by the local latency"
          " metric as the response timestamp. None if no audio was captured."
      ),
  )

  input_sent_time: Optional[float] = Field(
      default=None,
      description=(
          "Epoch seconds when the input audio that prompted this utterance"
          " finished being sent to this participant. For the SUT this is the"
          " moment the persona's audio was delivered; the latency metric uses"
          " it as the baseline (latency = first_audio_time - input_sent_time)."
      ),
  )

  tool_calls: list[genai_types.FunctionCall] = Field(
      default_factory=list,
      description="Tool calls the SUT made during this utterance.",
  )

  tool_responses: list[genai_types.FunctionResponse] = Field(
      default_factory=list,
      description="Tool responses produced during this utterance.",
  )

  was_interrupted: bool = Field(
      default=False,
      description="Whether this utterance was cut short by a barge-in.",
  )


@experimental
class ConversationTurn(EvalBaseModel):
  """A persona utterance paired with the SUT's response to it."""

  turn_index: int = Field(description="Zero-based index of the turn.")

  persona_utterance: Optional[CapturedUtterance] = Field(
      default=None,
      description="What the persona said in this turn.",
  )

  sut_utterance: Optional[CapturedUtterance] = Field(
      default=None,
      description="How the SUT responded in this turn.",
  )


@experimental
class LiveConversation(EvalBaseModel):
  """The full captured result of a persona-driven live conversation."""

  turns: list[ConversationTurn] = Field(
      default_factory=list,
      description="The conversation turns in chronological order.",
  )

  session_id: str = Field(
      default="",
      description=(
          "The SUT session id under which audio artifacts were persisted."
      ),
  )

  termination_reason: str = Field(
      default="completed",
      description=(
          "Why the conversation ended: 'completed' (SUT signalled done),"
          " 'max_turns', or 'error'."
      ),
  )

  app_details: Optional[AppDetails] = Field(
      default=None,
      description=(
          "Details of the agent under test (root agent and any sub-agents),"
          " including instructions and tool declarations. Carried onto each"
          " materialized invocation so managed metrics (e.g."
          " trajectory/tool-use quality) can use the agent configuration during"
          " rubric generation."
      ),
  )
