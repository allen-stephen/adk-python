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

"""Converts a captured live conversation into eval `Invocation`s.

This is the bridge that lets the live-native audio-to-audio result flow into
the rest of the eval engine unchanged: once a `LiveConversation` is materialized
into `Invocation`s, both the managed multi-turn metrics (GenAI Eval Service) and
the local acoustic metrics score it, and ADK Web renders it with per-turn audio
playback.
"""

from __future__ import annotations

from google.genai import types as genai_types

from ...utils.feature_decorator import experimental
from ..app_details import AppDetails
from ..eval_case import Invocation
from ..eval_case import InvocationEvent
from ..eval_case import InvocationEvents
from .live_conversation_types import CapturedUtterance
from .live_conversation_types import ConversationTurn
from .live_conversation_types import LiveConversation

_USER_ROLE = "user"
_MODEL_ROLE = "model"


@experimental
def materialize_conversation(
    conversation: LiveConversation,
    *,
    sut_author: str = "sut",
) -> list[Invocation]:
  """Converts a `LiveConversation` into a list of `Invocation`s.

  Each turn becomes one invocation:
    - persona transcript -> `user_content` (+ `user_audio` reference)
    - SUT transcript -> `final_response` (+ `agent_audio` reference)
    - SUT tool calls/responses -> `intermediate_data` with event timestamps
      (used by the latency acoustic metric and the managed trajectory metrics).

  Args:
    conversation: The captured live conversation.
    sut_author: Author name to attribute SUT events to.

  Returns:
    The materialized invocations, in conversation order.
  """
  invocations: list[Invocation] = []
  for turn in conversation.turns:
    invocations.append(
        _materialize_turn(
            turn,
            sut_author=sut_author,
            app_details=conversation.app_details,
        )
    )
  return invocations


def _materialize_turn(
    turn: ConversationTurn,
    *,
    sut_author: str,
    app_details: AppDetails | None = None,
) -> Invocation:
  persona = turn.persona_utterance
  sut = turn.sut_utterance

  user_content = _text_content(
      persona.transcript if persona else "", role=_USER_ROLE
  )
  final_response = (
      _text_content(sut.transcript, role=_MODEL_ROLE)
      if sut and sut.transcript
      else None
  )

  return Invocation(
      invocation_id=f"turn_{turn.turn_index}",
      user_content=user_content,
      user_audio=persona.audio_reference if persona else None,
      final_response=final_response,
      agent_audio=sut.audio_reference if sut else None,
      intermediate_data=_build_intermediate_data(sut, sut_author=sut_author),
      creation_timestamp=_latency_baseline(persona, sut),
      app_details=app_details,
  )


def _latency_baseline(
    persona: CapturedUtterance | None,
    sut: CapturedUtterance | None,
) -> float:
  """Returns the latency baseline timestamp for the turn.

  The local latency metric measures `first_agent_event_timestamp -
  creation_timestamp`. The accurate baseline is when the persona's audio
  finished being delivered to the SUT (`sut.input_sent_time`); pairing it with
  the SUT's first-audio time yields the true response delay. We fall back to the
  persona's start time only when the precise send time is unavailable.
  """
  if sut is not None and sut.input_sent_time is not None:
    return sut.input_sent_time
  return persona.start_time if persona else 0.0


def _text_content(text: str, *, role: str) -> genai_types.Content:
  return genai_types.Content(role=role, parts=[genai_types.Part(text=text)])


def _build_intermediate_data(
    sut: CapturedUtterance | None,
    *,
    sut_author: str,
) -> InvocationEvents:
  """Builds invocation events from the SUT utterance.

  The first event carries the SUT's response timestamp so the latency metric can
  measure the gap from the persona's turn. Tool calls/responses are included so
  managed trajectory/tool-use metrics can evaluate them.
  """
  events: list[InvocationEvent] = []
  if sut is None:
    return InvocationEvents(invocation_events=events)

  # A leading, content-free marker whose timestamp records the SUT's first
  # response time (the moment its first audio chunk arrived), used solely by the
  # local latency metric. It is intentionally content-free so it does NOT
  # duplicate the SUT transcript (which is already carried in `final_response`);
  # the managed facade skips content-free events. Fall back to start_time only
  # when no audio was captured.
  response_timestamp = (
      sut.first_audio_time
      if sut.first_audio_time is not None
      else sut.start_time
  )
  events.append(
      InvocationEvent(
          author=sut_author,
          content=None,
          timestamp=response_timestamp,
      )
  )

  for tool_call in sut.tool_calls:
    events.append(
        InvocationEvent(
            author=sut_author,
            content=genai_types.Content(
                role=_MODEL_ROLE,
                parts=[genai_types.Part(function_call=tool_call)],
            ),
            timestamp=sut.start_time,
        )
    )
  for tool_response in sut.tool_responses:
    events.append(
        InvocationEvent(
            author=sut_author,
            content=genai_types.Content(
                role=_MODEL_ROLE,
                parts=[genai_types.Part(function_response=tool_response)],
            ),
            timestamp=sut.end_time,
        )
    )

  return InvocationEvents(invocation_events=events)
