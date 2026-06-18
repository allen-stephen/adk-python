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

"""Tests for PersonaLiveConversationRunner.

The Live API is the boundary; we mock `Runner.run_live` so each participant
emits a scripted sequence of audio + transcription + turn_complete events. We
then verify the runner captures turns, persists audio, and pairs persona/SUT
utterances correctly.
"""

from __future__ import annotations

import asyncio

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.evaluation.simulation import persona_live_conversation
from google.adk.evaluation.simulation.live_conversation_scenario import LiveConversationScenario
from google.adk.evaluation.simulation.persona import Persona
from google.adk.evaluation.simulation.persona_live_conversation import _LiveAgentSession
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


def _audio_event(speaker: str, text: str, pcm: bytes) -> Event:
  """An event carrying output audio + a final transcription + turn_complete."""
  return Event(
      author=speaker,
      content=types.Content(
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      data=pcm, mime_type="audio/pcm;rate=24000"
                  )
              )
          ]
      ),
      output_transcription=types.Transcription(text=text),
      partial=True,
      turn_complete=True,
  )


class _ScriptedLiveSession:
  """Replaces _LiveAgentSession with a scripted, no-network version."""

  def __init__(self, *, runner, session, speaker):
    self._speaker = speaker
    # Each speaker yields a fixed list of utterances, one per collect_turn call.
    self._script = _ScriptedLiveSession.scripts[speaker]
    self._index = 0
    self._finished = False

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    return None

  def send_audio(self, pcm, *, mime_type="audio/pcm;rate=16000"):
    pass

  def send_text(self, text):
    pass

  async def collect_turn(self, *, timeout_seconds=60):
    from google.adk.evaluation.simulation.live_conversation_types import CapturedUtterance

    if self._index >= len(self._script):
      self._finished = True
      return CapturedUtterance(speaker=self._speaker)
    text, pcm = self._script[self._index]
    self._index += 1
    if self._index >= len(self._script):
      self._finished = True
    return CapturedUtterance(
        speaker=self._speaker,
        transcript=text,
        audio_pcm=pcm,
        mime_type="audio/pcm;rate=24000",
    )

  @property
  def is_finished(self):
    return self._finished

  # Class-level scripts set by each test.
  scripts: dict = {}


def _make_runner(monkeypatch):
  from google.adk.agents.llm_agent import Agent

  monkeypatch.setattr(
      persona_live_conversation, "_LiveAgentSession", _ScriptedLiveSession
  )
  sut_agent = Agent(
      name="sut_agent", model="gemini-live-2.5-flash-native-audio"
  )
  return persona_live_conversation.PersonaLiveConversationRunner(
      sut_agent=sut_agent,
      app_name="test_app",
      artifact_service=InMemoryArtifactService(),
  )


@pytest.mark.asyncio
async def test_runs_alternating_turns_and_pairs_utterances(monkeypatch):
  """The runner pairs persona and SUT utterances into turns."""
  _ScriptedLiveSession.scripts = {
      "persona": [
          ("Hi, I'd like a burger", b"p0"),
          ("Yes please", b"p1"),
      ],
      "sut": [
          ("Sure, one burger. Anything else?", b"s0"),
          ("Great, pull forward.", b"s1"),
      ],
  }
  runner = _make_runner(monkeypatch)

  conversation = await runner.run(
      LiveConversationScenario(
          persona=Persona(
              id="p", character_prompt="hungry", goal="order a burger"
          ),
          max_turns=5,
      )
  )

  assert len(conversation.turns) == 2
  assert conversation.turns[0].persona_utterance.transcript == (
      "Hi, I'd like a burger"
  )
  assert conversation.turns[0].sut_utterance.transcript == (
      "Sure, one burger. Anything else?"
  )


def test_collect_agent_tree_includes_sub_agents_and_agent_tools():
  """The agent tree includes the root, sub-agents, and AgentTool-wrapped agents."""
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.tools.agent_tool import AgentTool

  wrapped = LlmAgent(name="wrapped_agent", model="gemini-2.0-flash")
  sub = LlmAgent(name="sub_agent", model="gemini-2.0-flash")
  root = LlmAgent(
      name="root_agent",
      model="gemini-2.0-flash",
      tools=[AgentTool(agent=wrapped)],
      sub_agents=[sub],
  )

  names = {a.name for a in persona_live_conversation._collect_agent_tree(root)}

  assert names == {"root_agent", "sub_agent", "wrapped_agent"}


@pytest.mark.asyncio
async def test_build_app_details_resolves_instructions_and_tools():
  """AppDetails captures instructions + tool declarations across the tree."""
  from google.adk.agents.llm_agent import LlmAgent

  def my_tool(city: str) -> str:
    """Returns the weather for a city."""
    return "sunny"

  sub = LlmAgent(
      name="sub_agent",
      model="gemini-2.0-flash",
      instruction="I am a sub agent",
  )
  root = LlmAgent(
      name="root_agent",
      model="gemini-2.0-flash",
      instruction="I am root",
      tools=[my_tool],
      sub_agents=[sub],
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name="a", user_id="u")

  app_details = await persona_live_conversation._build_app_details(
      root, session_service=session_service, session=session
  )

  assert set(app_details.agent_details.keys()) == {"root_agent", "sub_agent"}
  assert app_details.agent_details["root_agent"].instructions == "I am root"
  assert app_details.agent_details["sub_agent"].instructions == (
      "I am a sub agent"
  )
  # The root agent's function tool declaration is captured.
  root_tools = app_details.agent_details["root_agent"].tool_declarations
  declared = [
      fd.name
      for tool in root_tools
      for fd in getattr(tool, "function_declarations", None) or []
  ]
  assert "my_tool" in declared


@pytest.mark.asyncio
async def test_run_populates_conversation_app_details(monkeypatch):
  """run() captures the SUT agent's app_details on the conversation."""
  _ScriptedLiveSession.scripts = {
      "persona": [("Hi", b"p0")],
      "sut": [("Hello", b"s0")],
  }
  runner = _make_runner(monkeypatch)

  conversation = await runner.run(
      LiveConversationScenario(
          persona=Persona(id="p", character_prompt="x", goal="y"),
          max_turns=1,
      )
  )

  assert conversation.app_details is not None
  assert "sut_agent" in conversation.app_details.agent_details


@pytest.mark.asyncio
async def test_driver_stamps_input_sent_time_on_sut_turns(monkeypatch):
  """The driver records when persona audio was delivered, for latency scoring."""
  _ScriptedLiveSession.scripts = {
      "persona": [("I'd like a burger", b"p0")],
      "sut": [("Sure, one burger.", b"s0")],
  }
  runner = _make_runner(monkeypatch)

  conversation = await runner.run(
      LiveConversationScenario(
          persona=Persona(
              id="p", character_prompt="hungry", goal="order a burger"
          ),
          max_turns=1,
      )
  )

  sut_utterance = conversation.turns[0].sut_utterance
  # input_sent_time is set by the driver right after the persona audio is sent;
  # this is the latency baseline used by the materializer.
  assert sut_utterance.input_sent_time is not None


@pytest.mark.asyncio
async def test_persists_audio_with_references(monkeypatch):
  """Captured audio is saved to the artifact service and referenced."""
  _ScriptedLiveSession.scripts = {
      "persona": [("Hello", b"persona-audio")],
      "sut": [("Hi there", b"sut-audio")],
  }
  artifact_service = InMemoryArtifactService()
  runner = _make_runner(monkeypatch)
  runner._artifact_service = artifact_service

  conversation = await runner.run(
      LiveConversationScenario(
          persona=Persona(id="p", character_prompt="x", goal="y"),
          max_turns=3,
      ),
      session_id="sess1",
  )

  persona_ref = conversation.turns[0].persona_utterance.audio_reference
  assert persona_ref is not None
  # The artifact filename must be flat (no directory prefix) so the web UI,
  # which references artifacts by their bare name, can fetch it.
  assert persona_ref.artifact_filename == "turn_0_persona.pcm"
  stored = await artifact_service.load_artifact(
      app_name="test_app",
      user_id="live_eval_user",
      session_id="sess1",
      filename=persona_ref.artifact_filename,
  )
  assert stored.inline_data.data == b"persona-audio"


@pytest.mark.asyncio
async def test_stops_at_max_turns(monkeypatch):
  """The conversation is capped at max_turns."""
  _ScriptedLiveSession.scripts = {
      "persona": [("a", b"a"), ("b", b"b"), ("c", b"c"), ("d", b"d")],
      "sut": [("1", b"1"), ("2", b"2"), ("3", b"3"), ("4", b"4")],
  }
  runner = _make_runner(monkeypatch)

  conversation = await runner.run(
      LiveConversationScenario(
          persona=Persona(id="p", character_prompt="x", goal="y"),
          max_turns=2,
      )
  )

  assert len(conversation.turns) == 2
  assert conversation.termination_reason == "max_turns"


@pytest.mark.asyncio
async def test_progress_callback_receives_events(monkeypatch):
  """Progress events are streamed for observation (Watch mode)."""
  _ScriptedLiveSession.scripts = {
      "persona": [("Hello", b"p")],
      "sut": [("Hi", b"s")],
  }
  runner = _make_runner(monkeypatch)
  events = []

  async def _cb(event):
    events.append(event["type"])

  await runner.run(
      LiveConversationScenario(
          persona=Persona(id="p", character_prompt="x", goal="y"), max_turns=2
      ),
      progress_callback=_cb,
  )

  assert "turn_started" in events
  assert "transcript_update" in events
  assert "conversation_complete" in events


# --- _LiveAgentSession send/collect mechanics (real, no network) ---


def _make_live_session(speaker: str = "sut") -> _LiveAgentSession:
  """Builds a _LiveAgentSession without starting its network consumer task."""
  return _LiveAgentSession(runner=None, session=None, speaker=speaker)


def _drain_queue(queue: asyncio.Queue) -> list:
  items = []
  while not queue.empty():
    items.append(queue.get_nowait())
  return items


def test_run_config_disables_automatic_vad():
  """The relay RunConfig disables automatic voice-activity detection."""
  config = _LiveAgentSession._build_run_config()

  detection = config.realtime_input_config.automatic_activity_detection
  assert detection.disabled is True


def test_send_audio_brackets_chunks_with_activity_markers():
  """send_audio emits activity_start, audio blobs, then activity_end."""
  session = _make_live_session()
  # 1.5 chunks worth of audio so we get multiple blobs.
  pcm = b"\x01\x02" * (persona_live_conversation._AUDIO_CHUNK_BYTES // 2 + 100)

  session.send_audio(pcm, mime_type="audio/pcm;rate=16000")

  requests = _drain_queue(session._live_request_queue._queue)
  assert requests[0].activity_start is not None
  assert requests[-1].activity_end is not None
  blobs = [r for r in requests if r.blob is not None]
  assert len(blobs) >= 2
  assert b"".join(b.blob.data for b in blobs) == pcm


def test_send_audio_sends_no_trailing_silence():
  """send_audio streams only the supplied audio, with no padding."""
  session = _make_live_session()
  pcm = b"\x09\x09" * 50

  session.send_audio(pcm)

  requests = _drain_queue(session._live_request_queue._queue)
  blobs = [r for r in requests if r.blob is not None]
  assert b"".join(b.blob.data for b in blobs) == pcm


@pytest.mark.asyncio
async def test_collect_turn_records_first_audio_time():
  """first_audio_time is stamped on the first audio chunk (the response start)."""
  session = _make_live_session()
  # A text-only event precedes audio; first_audio_time must track the audio, not
  # the earlier text event.
  session._event_queue.put_nowait(
      Event(
          author="sut",
          output_transcription=types.Transcription(text="Hi"),
          partial=True,
      )
  )
  session._event_queue.put_nowait(
      Event(
          author="sut",
          content=types.Content(
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          data=b"audio", mime_type="audio/pcm;rate=24000"
                      )
                  )
              ]
          ),
          output_transcription=types.Transcription(
              text="Hi there", finished=True
          ),
          partial=False,
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=5)

  assert utterance.first_audio_time is not None
  # The response start must be at/after collection start, and the utterance
  # carries audio.
  assert utterance.first_audio_time >= utterance.start_time
  assert utterance.audio_pcm == b"audio"


@pytest.mark.asyncio
async def test_collect_turn_no_audio_leaves_first_audio_time_none():
  """A text-only (no audio) turn leaves first_audio_time unset."""
  session = _make_live_session()
  session._event_queue.put_nowait(
      Event(
          author="sut",
          output_transcription=types.Transcription(
              text="text only", finished=True
          ),
          partial=False,
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=5)

  assert utterance.first_audio_time is None


@pytest.mark.asyncio
async def test_collect_turn_uses_final_flushed_transcription():
  """The final (non-partial) transcription is the source of truth."""
  session = _make_live_session()
  session._event_queue.put_nowait(
      Event(
          author="sut",
          output_transcription=types.Transcription(text="Hel"),
          partial=True,
      )
  )
  session._event_queue.put_nowait(
      Event(
          author="sut",
          output_transcription=types.Transcription(
              text="Hello there", finished=True
          ),
          partial=False,
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=5)

  assert utterance.transcript == "Hello there"


@pytest.mark.asyncio
async def test_collect_turn_falls_back_to_partials_without_final():
  """If no final transcription arrives, partial fragments are concatenated."""
  session = _make_live_session()
  session._event_queue.put_nowait(
      Event(
          author="sut",
          output_transcription=types.Transcription(text="Good "),
          partial=True,
      )
  )
  session._event_queue.put_nowait(
      Event(
          author="sut",
          output_transcription=types.Transcription(text="bye"),
          partial=True,
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=5)

  assert utterance.transcript == "Good bye"


@pytest.mark.asyncio
async def test_collect_turn_stops_at_turn_complete():
  """collect_turn returns once a non-user turn_complete event arrives."""
  session = _make_live_session()
  session._event_queue.put_nowait(
      Event(
          author="sut",
          content=types.Content(
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          data=b"audio", mime_type="audio/pcm;rate=24000"
                      )
                  )
              ]
          ),
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=5)

  assert utterance.audio_pcm == b"audio"
  assert utterance.mime_type == "audio/pcm;rate=24000"


@pytest.mark.asyncio
async def test_collect_turn_continues_past_tool_only_completion():
  """A tool-only turn_complete is followed to capture the spoken result.

  Setup: the SUT calls a tool (turn_complete with only a function call/response,
    no audio/text), then speaks the result in a following turn.
  Act: collect a single turn.
  Assert: the returned utterance carries both the tool activity and the spoken
    audio/transcript from the follow-up turn.
  """
  session = _make_live_session()
  # Turn A: tool call + response, completes with no speech.
  session._event_queue.put_nowait(
      Event(
          author="sut",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="roll_dice", args={"sides": 20}
                      )
                  )
              ]
          ),
      )
  )
  session._event_queue.put_nowait(
      Event(
          author="sut",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name="roll_dice", response={"result": 10}
                      )
                  )
              ]
          ),
          turn_complete=True,
      )
  )
  # Turn B: the spoken result.
  session._event_queue.put_nowait(
      Event(
          author="sut",
          content=types.Content(
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          data=b"spoken", mime_type="audio/pcm;rate=24000"
                      )
                  )
              ]
          ),
          output_transcription=types.Transcription(
              text="You rolled a 10", finished=True
          ),
          partial=False,
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=5)

  assert utterance.transcript == "You rolled a 10"
  assert utterance.audio_pcm == b"spoken"
  assert len(utterance.tool_calls) == 1
  assert utterance.tool_calls[0].name == "roll_dice"


@pytest.mark.asyncio
async def test_collect_turn_returns_tool_only_when_no_speech_follows():
  """A tool-only turn with no spoken follow-up still returns the tool activity.

  The continuation waits for speech but must not hang forever; once the absolute
  timeout elapses it returns what it has (the tool call).
  """
  session = _make_live_session()
  session._event_queue.put_nowait(
      Event(
          author="sut",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="roll_dice", args={"sides": 20}
                      )
                  )
              ]
          ),
          turn_complete=True,
      )
  )

  utterance = await session.collect_turn(timeout_seconds=1)

  assert utterance.transcript == ""
  assert utterance.audio_pcm is None
  assert len(utterance.tool_calls) == 1
