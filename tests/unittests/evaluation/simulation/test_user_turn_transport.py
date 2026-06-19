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

"""Tests for the audio transports.

These tests cover the transport seam without touching the Live API: the SUT
`_LiveAgentSession` and the TTS call are replaced with scripted fakes, so the
turn loop, materialization-ready capture, and the fixed-script-over-audio path
are all exercised deterministically.
"""

from __future__ import annotations

import array

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.simulation import user_turn_transport
from google.adk.evaluation.simulation.live_conversation_types import \
    CapturedUtterance
from google.adk.evaluation.simulation.static_user_simulator import \
    StaticUserSimulator
from google.adk.evaluation.simulation.voice_profile import LiveTransport
from google.adk.evaluation.simulation.voice_profile import VoiceProfile
from google.genai import types as genai_types
import pytest


def _pcm(num_samples: int = 100) -> bytes:
  return array.array("h", [1000] * num_samples).tobytes()


def _user_invocation(text: str) -> Invocation:
  return Invocation(
      invocation_id="x",
      user_content=genai_types.Content(
          role="user", parts=[genai_types.Part(text=text)]
      ),
  )


class _FakeSutSession:
  """A scripted stand-in for the SUT `_LiveAgentSession`."""

  def __init__(self, replies):
    self._replies = list(replies)
    self._index = 0
    self.sent_audio = []
    self.is_finished = False
    self.drain_calls = 0
    self.last_input_sent_time = None

  async def __aenter__(self):
    return self

  async def __aexit__(self, *exc):
    return None

  def send_audio(self, pcm, *, mime_type):
    self.sent_audio.append(pcm)

  def send_text(self, text):
    pass

  def drain_pending_events(self):
    self.drain_calls += 1

  async def collect_turn(
      self,
      *,
      timeout_seconds=30,
      listen_window_seconds=None,
      input_sent_time=None,
  ):
    self.last_listen_window = listen_window_seconds
    self.last_input_sent_time = input_sent_time
    if self._index < len(self._replies):
      reply = self._replies[self._index]
      self._index += 1
      return reply
    return CapturedUtterance(speaker="sut")


@pytest.mark.asyncio
async def test_tts_transport_runs_fixed_script_over_audio(monkeypatch):
  """A StaticUserSimulator (fixed script) is performed over the TTS transport."""
  # Two scripted SUT replies, one per user turn.
  fake_sut = _FakeSutSession([
      CapturedUtterance(
          speaker="sut", transcript="Rolled a 7.", audio_pcm=_pcm()
      ),
      CapturedUtterance(
          speaker="sut", transcript="7 is prime.", audio_pcm=_pcm()
      ),
  ])

  def _fake_session(*args, **kwargs):
    return fake_sut

  monkeypatch.setattr(user_turn_transport, "_LiveAgentSession", _fake_session)

  transport = user_turn_transport.TtsUserTurnTransport(
      sut_agent=object(),
      app_name="app",
      tts_client=object(),
  )

  async def _fake_synthesize(text, voice_profile):
    return _pcm()

  monkeypatch.setattr(transport, "_synthesize", _fake_synthesize)

  static_conversation = [
      _user_invocation("Roll a 20-sided die."),
      _user_invocation("Is it prime?"),
  ]
  simulator = StaticUserSimulator(static_conversation=static_conversation)

  conversation = await transport.run(
      user_simulator=simulator,
      scenario=None,
      voice_profile=VoiceProfile(transport=LiveTransport.TTS),
      max_turns=5,
  )

  # One turn per scripted user message, each with persona + SUT utterances.
  assert len(conversation.turns) == 2
  assert conversation.turns[0].persona_utterance.transcript == (
      "Roll a 20-sided die."
  )
  assert conversation.turns[0].sut_utterance.transcript == "Rolled a 7."
  # The synthesized user audio was delivered to the SUT for each turn.
  assert len(fake_sut.sent_audio) == 2


@pytest.mark.asyncio
async def test_tts_transport_barges_in_when_probability_fires(monkeypatch):
  """When barge-in fires, the SUT turn is collected with a listen window."""
  fake_sut = _FakeSutSession([
      CapturedUtterance(
          speaker="sut",
          transcript="Let me roll that for you...",
          audio_pcm=_pcm(),
          was_interrupted=True,
      ),
  ])

  monkeypatch.setattr(
      user_turn_transport, "_LiveAgentSession", lambda *a, **k: fake_sut
  )
  # Force barge-in to always fire and pin the window for determinism.
  monkeypatch.setattr(
      user_turn_transport, "_maybe_barge_in", lambda *a, **k: True
  )
  monkeypatch.setattr(
      user_turn_transport, "_barge_in_listen_seconds", lambda _: 0.75
  )

  transport = user_turn_transport.TtsUserTurnTransport(
      sut_agent=object(), app_name="app", tts_client=object()
  )

  async def _fake_synthesize(text, voice_profile):
    return _pcm()

  monkeypatch.setattr(transport, "_synthesize", _fake_synthesize)

  events = []

  async def _progress(event):
    events.append(event)

  simulator = StaticUserSimulator(
      static_conversation=[_user_invocation("Roll a 20-sided die.")]
  )

  conversation = await transport.run(
      user_simulator=simulator,
      scenario=None,
      voice_profile=VoiceProfile(transport=LiveTransport.TTS),
      max_turns=5,
      progress_callback=_progress,
  )

  # The listen window was threaded into the SUT collect, the captured SUT
  # utterance is flagged interrupted, and a barge_in event was emitted.
  assert fake_sut.last_listen_window == 0.75
  assert conversation.turns[0].sut_utterance.was_interrupted is True
  barge_in_events = [e for e in events if e.get("type") == "barge_in"]
  assert len(barge_in_events) == 1
  assert barge_in_events[0]["listen_seconds"] == 0.75


def test_barge_in_listen_seconds_within_configured_range():
  from google.adk.evaluation.simulation.voice_profile import BargeInConfig

  config = BargeInConfig(
      enabled=True, probability=1.0, min_listen_ms=400, max_listen_ms=900
  )
  for _ in range(50):
    seconds = user_turn_transport._barge_in_listen_seconds(config)
    assert 0.4 <= seconds <= 0.9


def test_maybe_barge_in_respects_max_barge_ins_cap():
  from google.adk.evaluation.simulation.voice_profile import BargeInConfig

  # probability=1.0 means it would always fire, but max_barge_ins caps it.
  config = BargeInConfig(enabled=True, probability=1.0, max_barge_ins=2)
  assert user_turn_transport._maybe_barge_in(config, 0) is True
  assert user_turn_transport._maybe_barge_in(config, 1) is True
  assert user_turn_transport._maybe_barge_in(config, 2) is False
  assert user_turn_transport._maybe_barge_in(config, 5) is False


def test_maybe_barge_in_no_cap_when_unset():
  from google.adk.evaluation.simulation.voice_profile import BargeInConfig

  config = BargeInConfig(enabled=True, probability=1.0)
  assert user_turn_transport._maybe_barge_in(config, 100) is True


@pytest.mark.asyncio
async def test_tts_transport_caps_barge_ins(monkeypatch):
  """At most `max_barge_ins` interruptions occur across the conversation."""
  from google.adk.evaluation.simulation.voice_profile import BargeInConfig

  # Five turns, every SUT reply reports it was interrupted if a window was set.
  replies = [
      CapturedUtterance(
          speaker="sut", transcript=f"reply {i}", audio_pcm=_pcm()
      )
      for i in range(5)
  ]

  class _CapFakeSut(_FakeSutSession):

    async def collect_turn(
        self,
        *,
        timeout_seconds=30,
        listen_window_seconds=None,
        input_sent_time=None,
    ):
      reply = (
          replies[self._index]
          if self._index < len(replies)
          else (CapturedUtterance(speaker="sut"))
      )
      self._index += 1
      # Mirror real behavior: interrupted only when a window was provided.
      reply.was_interrupted = listen_window_seconds is not None
      return reply

  fake_sut = _CapFakeSut([])
  monkeypatch.setattr(
      user_turn_transport, "_LiveAgentSession", lambda *a, **k: fake_sut
  )
  monkeypatch.setattr(
      user_turn_transport, "_barge_in_listen_seconds", lambda _: 0.5
  )

  transport = user_turn_transport.TtsUserTurnTransport(
      sut_agent=object(), app_name="app", tts_client=object()
  )

  async def _fake_synthesize(text, voice_profile):
    return _pcm()

  monkeypatch.setattr(transport, "_synthesize", _fake_synthesize)

  events = []

  async def _progress(event):
    events.append(event)

  simulator = StaticUserSimulator(
      static_conversation=[_user_invocation(f"turn {i}") for i in range(5)]
  )

  await transport.run(
      user_simulator=simulator,
      scenario=None,
      voice_profile=VoiceProfile(
          transport=LiveTransport.TTS,
          barge_in=BargeInConfig(
              enabled=True, probability=1.0, max_barge_ins=2
          ),
      ),
      max_turns=5,
      progress_callback=_progress,
  )

  barge_in_events = [e for e in events if e.get("type") == "barge_in"]
  assert len(barge_in_events) == 2


@pytest.mark.asyncio
async def test_tts_transport_drains_and_threads_input_sent_time(monkeypatch):
  """Each turn drains stale events before sending and threads input_sent_time."""
  fake_sut = _FakeSutSession([
      CapturedUtterance(
          speaker="sut", transcript="Rolled a 7.", audio_pcm=_pcm()
      ),
  ])
  monkeypatch.setattr(
      user_turn_transport, "_LiveAgentSession", lambda *a, **k: fake_sut
  )

  transport = user_turn_transport.TtsUserTurnTransport(
      sut_agent=object(), app_name="app", tts_client=object()
  )

  async def _fake_synthesize(text, voice_profile):
    return _pcm()

  monkeypatch.setattr(transport, "_synthesize", _fake_synthesize)
  simulator = StaticUserSimulator(
      static_conversation=[_user_invocation("Roll a die.")]
  )

  await transport.run(
      user_simulator=simulator,
      scenario=None,
      voice_profile=VoiceProfile(transport=LiveTransport.TTS),
      max_turns=3,
  )

  # The queue was drained before the (single) turn's audio was sent, and the
  # send time was threaded into collect_turn for the latency anchor.
  assert fake_sut.drain_calls == 1
  assert fake_sut.last_input_sent_time is not None


def _make_collect_turn_session():
  import asyncio

  session = user_turn_transport._LiveAgentSession.__new__(
      user_turn_transport._LiveAgentSession
  )
  session._speaker = "sut"
  session._event_queue = asyncio.Queue()
  session._live_finished = asyncio.Event()
  return session


def _audio_content():
  from google.genai import types

  return types.Content(
      parts=[
          types.Part(
              inline_data=types.Blob(
                  data=_pcm(), mime_type="audio/pcm;rate=24000"
              )
          )
      ]
  )


class _FakeLiveEvent:

  def __init__(self, content, *, turn_complete=False):
    self.content = content
    self.author = "sut"
    self.partial = False
    self.output_transcription = None
    self.turn_complete = turn_complete

  def get_function_calls(self):
    return []

  def get_function_responses(self):
    return []


@pytest.mark.asyncio
async def test_collect_turn_skips_audio_before_input_sent_time():
  """Audio dequeued before input_sent_time is treated as stale and skipped."""
  import time

  session = _make_collect_turn_session()
  # The only audio is stale (dequeued well before a far-future input_sent_time),
  # so it is skipped and no latency anchor is set.
  await session._event_queue.put(_FakeLiveEvent(_audio_content()))
  await session._event_queue.put(_FakeLiveEvent(None, turn_complete=True))
  session._live_finished.set()

  result = await session.collect_turn(input_sent_time=time.time() + 100)

  assert result.first_audio_time is None
  assert result.audio_pcm is None


@pytest.mark.asyncio
async def test_collect_turn_keeps_audio_after_input_sent_time():
  """Audio dequeued at/after input_sent_time sets the latency anchor."""
  import time

  session = _make_collect_turn_session()
  await session._event_queue.put(_FakeLiveEvent(_audio_content()))
  await session._event_queue.put(_FakeLiveEvent(None, turn_complete=True))
  session._live_finished.set()

  # input_sent_time is in the past, so the audio is fresh and counts.
  sent = time.time() - 1
  result = await session.collect_turn(input_sent_time=sent)

  assert result.first_audio_time is not None
  assert result.first_audio_time >= sent
  assert result.audio_pcm is not None


def test_drain_pending_events_empties_queue():
  """drain_pending_events discards all currently-queued events."""
  import asyncio

  session = user_turn_transport._LiveAgentSession.__new__(
      user_turn_transport._LiveAgentSession
  )
  session._event_queue = asyncio.Queue()
  session._event_queue.put_nowait("a")
  session._event_queue.put_nowait("b")

  session.drain_pending_events()

  assert session._event_queue.empty()


@pytest.mark.asyncio
async def test_native_audio_transport_requires_scenario():
  transport = user_turn_transport.NativeAudioPersonaTransport(
      sut_agent=object(),
      app_name="app",
  )
  simulator = StaticUserSimulator(static_conversation=[])

  with pytest.raises(ValueError, match="requires a conversation_scenario"):
    await transport.run(
        user_simulator=simulator,
        scenario=None,
        voice_profile=VoiceProfile(),
        max_turns=3,
    )
