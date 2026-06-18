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

"""Tests for materializing a LiveConversation into Invocations."""

from __future__ import annotations

from google.adk.evaluation.eval_case import AudioReference
from google.adk.evaluation.simulation.live_conversation_materializer import materialize_conversation
from google.adk.evaluation.simulation.live_conversation_types import CapturedUtterance
from google.adk.evaluation.simulation.live_conversation_types import ConversationTurn
from google.adk.evaluation.simulation.live_conversation_types import LiveConversation
from google.genai import types


def _turn(index, persona_text, sut_text, *, tool_call=None):
  persona = CapturedUtterance(
      speaker="persona",
      transcript=persona_text,
      audio_pcm=b"p",
      audio_reference=AudioReference(artifact_filename=f"t{index}_persona.pcm"),
      start_time=100.0 + index,
  )
  sut = CapturedUtterance(
      speaker="sut",
      transcript=sut_text,
      audio_pcm=b"s",
      audio_reference=AudioReference(artifact_filename=f"t{index}_sut.pcm"),
      start_time=100.5 + index,
      end_time=101.0 + index,
      # input_sent_time: persona audio delivered to the SUT at 100.4; the SUT's
      # first audio (response) arrives at 100.7. So accurate latency = 0.3.
      input_sent_time=100.4 + index,
      first_audio_time=100.7 + index,
      tool_calls=[tool_call] if tool_call else [],
  )
  return ConversationTurn(
      turn_index=index, persona_utterance=persona, sut_utterance=sut
  )


def test_each_turn_becomes_an_invocation():
  """Every captured turn maps to one invocation with paired text."""
  conversation = LiveConversation(
      turns=[
          _turn(0, "Hi, a burger please", "Sure, anything else?"),
          _turn(1, "No thanks", "Pull forward."),
      ]
  )

  invocations = materialize_conversation(conversation)

  assert len(invocations) == 2
  assert invocations[0].user_content.parts[0].text == "Hi, a burger please"
  assert invocations[0].final_response.parts[0].text == "Sure, anything else?"


def test_app_details_are_carried_onto_invocations():
  """The conversation's app_details are attached to every invocation."""
  from google.adk.evaluation.app_details import AgentDetails
  from google.adk.evaluation.app_details import AppDetails

  app_details = AppDetails(
      agent_details={
          "root_agent": AgentDetails(
              name="root_agent", instructions="be helpful"
          )
      }
  )
  conversation = LiveConversation(
      app_details=app_details,
      turns=[_turn(0, "Hi", "Hello"), _turn(1, "Bye", "Goodbye")],
  )

  invocations = materialize_conversation(conversation)

  assert len(invocations) == 2
  for invocation in invocations:
    assert invocation.app_details is app_details
    assert "root_agent" in invocation.app_details.agent_details


def test_app_details_absent_leaves_invocation_app_details_none():
  """Without app_details on the conversation, invocations have none."""
  conversation = LiveConversation(turns=[_turn(0, "Hi", "Hello")])

  invocation = materialize_conversation(conversation)[0]

  assert invocation.app_details is None


def test_audio_references_are_carried():
  """Persona/SUT audio references are attached to the invocation."""
  conversation = LiveConversation(turns=[_turn(0, "Hello", "Hi there")])

  invocation = materialize_conversation(conversation)[0]

  assert invocation.user_audio.artifact_filename == "t0_persona.pcm"
  assert invocation.agent_audio.artifact_filename == "t0_sut.pcm"


def test_sut_timestamp_enables_latency():
  """Latency uses send->first-audio, not the persona's speaking time.

  The marker event timestamp is the SUT's first-audio time and the invocation's
  creation_timestamp is when the persona audio was delivered to the SUT, so
  `first_event.timestamp - creation_timestamp` is the true response delay.
  """
  conversation = LiveConversation(turns=[_turn(0, "Hello", "Hi there")])

  invocation = materialize_conversation(conversation)[0]

  first_event = invocation.intermediate_data.invocation_events[0]
  assert first_event.timestamp == 100.7  # sut.first_audio_time
  assert invocation.creation_timestamp == 100.4  # sut.input_sent_time
  # The measured latency is the response delay (0.3s), NOT the gap from the
  # persona's wait-start (which would be 100.7 - 100.0 = 0.7s, over-counted).
  assert round(first_event.timestamp - invocation.creation_timestamp, 1) == 0.3


def test_latency_falls_back_when_precise_timestamps_missing():
  """Without first_audio_time / input_sent_time, fall back to start times."""
  persona = CapturedUtterance(
      speaker="persona", transcript="Hi", audio_pcm=b"p", start_time=50.0
  )
  sut = CapturedUtterance(
      speaker="sut",
      transcript="Hello",
      audio_pcm=b"s",
      start_time=50.5,
      end_time=51.0,
      # No input_sent_time / first_audio_time (e.g. older data).
  )
  conversation = LiveConversation(
      turns=[
          ConversationTurn(
              turn_index=0, persona_utterance=persona, sut_utterance=sut
          )
      ]
  )

  invocation = materialize_conversation(conversation)[0]

  first_event = invocation.intermediate_data.invocation_events[0]
  assert first_event.timestamp == 50.5  # falls back to sut.start_time
  assert (
      invocation.creation_timestamp == 50.0
  )  # falls back to persona.start_time


def test_tool_calls_are_materialized():
  """SUT tool calls appear as invocation events for trajectory metrics."""
  tool_call = types.FunctionCall(name="roll_die", args={"sides": 6})
  conversation = LiveConversation(
      turns=[_turn(0, "Roll a die", "You rolled a 5", tool_call=tool_call)]
  )

  invocation = materialize_conversation(conversation)[0]

  function_calls = [
      p.function_call
      for event in invocation.intermediate_data.invocation_events
      if event.content
      for p in event.content.parts
      if p.function_call
  ]
  assert len(function_calls) == 1
  assert function_calls[0].name == "roll_die"


def test_managed_facade_receives_no_null_content_or_duplicate_text():
  """The managed payload has no null-content events and no duplicated SUT text.

  The latency marker is content-free (timestamp only); it must be filtered out
  of the managed payload, and the SUT transcript must appear exactly once (as
  the agent's final response), not duplicated as an intermediate event.
  """
  from google.adk.evaluation.vertex_ai_eval_facade import _MultiTurnVertexiAiEvalFacade

  conversation = LiveConversation(turns=[_turn(0, "Hello", "Hi there friend")])
  invocations = materialize_conversation(conversation)

  # Local latency still has a timestamp to read.
  first_event = invocations[0].intermediate_data.invocation_events[0]
  assert first_event.timestamp is not None

  # Managed payload: no null content, SUT text appears exactly once.
  turn = _MultiTurnVertexiAiEvalFacade._get_turns(invocations)[0]
  for event in turn.events:
    assert event.content is not None
  agent_texts = [
      part.text
      for event in turn.events
      for part in event.content.parts or []
      if part.text == "Hi there friend"
  ]
  assert len(agent_texts) == 1


def test_materialized_conversation_maps_to_managed_facade():
  """Materialized invocations map cleanly into the managed multi-turn facade.

  This is the seam that lets the GenAI Evaluation Service score audio-to-audio
  conversations without any new managed code.
  """
  from google.adk.evaluation.vertex_ai_eval_facade import _MultiTurnVertexiAiEvalFacade

  tool_call = types.FunctionCall(name="add_item", args={"item": "burger"})
  conversation = LiveConversation(
      turns=[
          _turn(0, "I want a burger", "One burger", tool_call=tool_call),
          _turn(1, "That's all", "Pull forward"),
      ]
  )

  invocations = materialize_conversation(conversation)
  turns = _MultiTurnVertexiAiEvalFacade._get_turns(invocations)

  assert len(turns) == 2
  authors = [event.author for event in turns[0].events]
  assert authors[0] == "user"
  assert authors[-1] == "agent"
