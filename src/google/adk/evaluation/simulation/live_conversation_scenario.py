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

"""The scenario describing a persona-driven audio-to-audio live conversation."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from ...utils.feature_decorator import experimental
from ..common import EvalBaseModel
from .persona import AudioRealismConfig
from .persona import BargeInConfig
from .persona import Persona


@experimental
class LiveConversationScenario(EvalBaseModel):
  """Describes a single persona-driven live (voice) conversation to run.

  Unlike a text eval case (a fixed list of user turns), a live scenario only
  specifies *who* the persona is and *what* it wants; the actual turns are
  produced fresh each run by the persona agent speaking with the agent under
  test.
  """

  persona: Persona = Field(
      description="The synthetic persona that drives the conversation.",
  )

  max_turns: int = Field(
      default=10,
      ge=1,
      description=(
          "Safety cap on the number of conversation turns before the run is"
          " forcibly ended."
      ),
  )

  barge_in: Optional[BargeInConfig] = Field(
      default=None,
      description=(
          "Optional barge-in configuration. When unset, the conversation uses"
          " strict alternating turns."
      ),
  )

  audio_realism: Optional[AudioRealismConfig] = Field(
      default=None,
      description=(
          "Optional audio-realism configuration. When unset, persona audio is"
          " passed through unmodified."
      ),
  )
