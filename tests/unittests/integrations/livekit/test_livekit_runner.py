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

"""Tests for the LiveKit connector's two frame bridges.

Verifies that `LiveKitRunner` forwards inbound room media into the
`LiveRequestQueue` in the formats ADK's live contract expects, and pushes
outbound `run_live` events back to the room as audio frames and data-track
payloads.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.genai import types
import pytest

pytest.importorskip("livekit")

from google.adk.integrations.livekit import _livekit_runner
from google.adk.integrations.livekit import LiveKitRunner
from livekit import rtc
import numpy as np

# --- Fixtures (minimal, one purpose each) ---


def _make_runner(events: list[Event]) -> Runner:
  """A Runner whose run_live yields the given events then finishes."""
  runner = MagicMock(spec=Runner)

  async def run_live(**kwargs):
    for event in events:
      yield event

  runner.run_live = run_live
  return runner


def _make_room():
  """A connected LiveKit room with async publish methods and no tracks."""
  room = MagicMock()
  room.remote_participants = {}
  local = room.local_participant
  local.publish_track = AsyncMock()
  local.publish_data = AsyncMock()
  return room


def _make_lk_runner(runner, room) -> LiveKitRunner:
  """Builds a runner with the outbound audio track stubbed out.

  `rtc.AudioSource` reaches into the LiveKit FFI, which needs a live worker;
  the bridge only ever calls `capture_frame` and `clear_queue` on it.
  """
  with (
      patch.object(_livekit_runner.rtc, "AudioSource"),
      patch.object(_livekit_runner.rtc, "LocalAudioTrack"),
  ):
    lk_runner = LiveKitRunner(
        runner=runner,
        room=room,
        user_id="u1",
        session_id="s1",
    )
  lk_runner._audio_source = MagicMock()
  lk_runner._audio_source.capture_frame = AsyncMock()
  return lk_runner


def _audio_event(data: bytes) -> Event:
  return Event(
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  inline_data=types.Blob(mime_type="audio/pcm", data=data)
              )
          ],
      ),
  )


def _function_call_event(name: str, args: dict) -> Event:
  return Event(
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(function_call=types.FunctionCall(name=name, args=args))
          ],
      ),
  )


def _published(room) -> list[dict]:
  """Decodes every payload published on the room data track."""
  return [
      json.loads(call.args[0])
      for call in room.local_participant.publish_data.await_args_list
  ]


class _FakeStream:
  """An async-iterable stand-in for rtc.AudioStream / rtc.VideoStream."""

  def __init__(self, events):
    self._events = events

  def __call__(self, *args, **kwargs):
    return self

  def __aiter__(self):
    async def gen():
      for event in self._events:
        yield event

    return gen()


# --- Outbound bridge: Event stream -> room ---


def test_defaults_to_audio_response_modality():
  """A runner built without a run_config responds in the AUDIO modality."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())

  assert lk_runner._run_config.response_modalities == [types.Modality.AUDIO]


def test_run_config_is_used_when_provided():
  """A caller-supplied run_config is passed through untouched."""
  run_config = RunConfig(
      response_modalities=[types.Modality.AUDIO],
      output_audio_transcription=types.AudioTranscriptionConfig(),
  )
  with (
      patch.object(_livekit_runner.rtc, "AudioSource"),
      patch.object(_livekit_runner.rtc, "LocalAudioTrack"),
  ):
    lk_runner = LiveKitRunner(
        runner=_make_runner([]),
        room=_make_room(),
        user_id="u1",
        session_id="s1",
        run_config=run_config,
    )

  assert lk_runner._run_config is run_config


@pytest.mark.asyncio
async def test_output_audio_captured_to_room_track():
  """Audio inline_data on an event is captured onto the room audio source."""
  event = _audio_event(b"\x01\x02\x03\x04")
  lk_runner = _make_lk_runner(_make_runner([event]), _make_room())

  await lk_runner._forward_events()

  lk_runner._audio_source.capture_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupted_event_clears_pending_output_audio():
  """Barge-in drops audio already queued for playback.

  Without this the agent keeps talking over the user for as long as the
  playback buffer lasts.
  """
  events = [
      _audio_event(b"\x01\x02"),
      Event(author="agent", interrupted=True),
  ]
  lk_runner = _make_lk_runner(_make_runner(events), _make_room())

  await lk_runner._forward_events()

  lk_runner._audio_source.clear_queue.assert_called_once()


@pytest.mark.asyncio
async def test_uninterrupted_session_never_clears_output_audio():
  """A normal turn does not drop buffered audio."""
  lk_runner = _make_lk_runner(
      _make_runner([_audio_event(b"\x01\x02")]), _make_room()
  )

  await lk_runner._forward_events()

  lk_runner._audio_source.clear_queue.assert_not_called()


@pytest.mark.asyncio
async def test_function_call_published_to_data_track():
  """A function_call event is published as JSON on the room data track."""
  room = _make_room()
  lk_runner = _make_lk_runner(
      _make_runner([_function_call_event("roll_die", {"sides": 6})]), room
  )

  await lk_runner._forward_events()

  (payload,) = _published(room)
  assert payload["type"] == "function_call"
  assert payload["name"] == "roll_die"
  assert payload["args"] == {"sides": 6}


@pytest.mark.asyncio
async def test_data_track_payloads_carry_the_adk_topic():
  """Clients filter on the topic, so every payload must set it."""
  room = _make_room()
  lk_runner = _make_lk_runner(
      _make_runner([_function_call_event("roll_die", {"sides": 6})]), room
  )

  await lk_runner._forward_events()

  _, kwargs = room.local_participant.publish_data.await_args
  assert kwargs["topic"] == _livekit_runner.DATA_TOPIC


@pytest.mark.asyncio
async def test_final_transcripts_published_to_data_track():
  """Both sides of the conversation reach the client as transcripts."""
  events = [
      Event(
          author="user",
          input_transcription=types.Transcription(text="roll a die"),
      ),
      Event(
          author="agent",
          output_transcription=types.Transcription(text="you rolled a four"),
      ),
  ]
  room = _make_room()
  lk_runner = _make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  assert _published(room) == [
      {"type": "transcript", "role": "user", "text": "roll a die"},
      {"type": "transcript", "role": "agent", "text": "you rolled a four"},
  ]


@pytest.mark.asyncio
async def test_partial_transcripts_are_not_published():
  """Partials arrive token by token and would re-render the same utterance."""
  events = [
      Event(
          author="agent",
          partial=True,
          output_transcription=types.Transcription(text="you rolled"),
      ),
      Event(
          author="agent",
          output_transcription=types.Transcription(text="you rolled a four"),
      ),
  ]
  room = _make_room()
  lk_runner = _make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  assert _published(room) == [
      {"type": "transcript", "role": "agent", "text": "you rolled a four"}
  ]


# --- Inbound bridge: room -> LiveRequestQueue ---


@pytest.mark.asyncio
async def test_inbound_audio_track_forwarded_as_pcm_blob():
  """Frames from a room audio track land on the queue as 16kHz PCM blobs."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  captured: list[types.Blob] = []
  lk_runner._queue.send_realtime = captured.append

  frame_event = MagicMock()
  frame_event.frame.data = b"\x10\x20"
  with patch.object(
      _livekit_runner.rtc, "AudioStream", _FakeStream([frame_event])
  ):
    await lk_runner._forward_audio(MagicMock())

  assert len(captured) == 1
  # The rate belongs in the mime type; a bare `audio/pcm` leaves the model
  # guessing at the sample rate.
  assert captured[0].mime_type == "audio/pcm;rate=16000"
  assert captured[0].data == b"\x10\x20"


@pytest.mark.asyncio
async def test_audio_stream_end_signalled_when_track_ends():
  """A muted or unpublished track flushes the model's audio buffer.

  Without the flush a server-VAD turn hangs waiting for input that will never
  arrive.
  """
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  lk_runner._queue.send_audio_stream_end = MagicMock()

  with patch.object(_livekit_runner.rtc, "AudioStream", _FakeStream([])):
    await lk_runner._forward_audio(MagicMock())

  lk_runner._queue.send_audio_stream_end.assert_called_once()


@pytest.mark.asyncio
async def test_inbound_video_track_forwarded_as_real_jpeg():
  """Video frames are JPEG-encoded, not raw buffers labelled image/jpeg."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  captured: list[types.Blob] = []
  lk_runner._queue.send_realtime = captured.append

  width, height = 64, 48
  frame_event = MagicMock()
  frame_event.frame = rtc.VideoFrame(
      width=width,
      height=height,
      type=rtc.VideoBufferType.RGBA,
      data=bytearray(np.zeros((height, width, 4), dtype=np.uint8).tobytes()),
  )
  with patch.object(
      _livekit_runner.rtc, "VideoStream", _FakeStream([frame_event])
  ):
    await lk_runner._forward_video(MagicMock())

  assert len(captured) == 1
  assert captured[0].mime_type == "image/jpeg"
  assert captured[0].data.startswith(b"\xff\xd8")  # JPEG SOI marker.


@pytest.mark.asyncio
async def test_video_frames_are_rate_limited():
  """Live models sample video; forwarding at capture rate floods the queue."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  captured: list[types.Blob] = []
  lk_runner._queue.send_realtime = captured.append

  width, height = 16, 16
  frame = rtc.VideoFrame(
      width=width,
      height=height,
      type=rtc.VideoBufferType.RGBA,
      data=bytearray(np.zeros((height, width, 4), dtype=np.uint8).tobytes()),
  )
  frame_events = []
  for _ in range(30):  # One second of 30fps capture.
    event = MagicMock()
    event.frame = frame
    frame_events.append(event)

  with patch.object(
      _livekit_runner.rtc, "VideoStream", _FakeStream(frame_events)
  ):
    await lk_runner._forward_video(MagicMock())

  assert len(captured) == 1


@pytest.mark.asyncio
async def test_inbound_text_message_becomes_a_user_turn():
  """A typed message on the data track is sent as user content."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  captured: list[types.Content] = []
  lk_runner._queue.send_content = lambda content: captured.append(content)

  lk_runner._on_data_received(
      _data_packet({"type": "text", "text": "roll a die"})
  )

  assert len(captured) == 1
  assert captured[0].role == "user"
  assert captured[0].parts[0].text == "roll a die"


@pytest.mark.asyncio
async def test_data_on_another_topic_is_ignored():
  """The bridge only claims its own topic; the room is shared."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  lk_runner._queue.send_content = MagicMock()

  lk_runner._on_data_received(
      _data_packet({"type": "text", "text": "not for us"}, topic="other-app")
  )

  lk_runner._queue.send_content.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_data_message_is_ignored():
  """A non-JSON payload must not take down the session."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  lk_runner._queue.send_content = MagicMock()

  packet = MagicMock()
  packet.topic = _livekit_runner.DATA_TOPIC
  packet.data = b"\xff\xfe not json"

  lk_runner._on_data_received(packet)

  lk_runner._queue.send_content.assert_not_called()


def _data_packet(message: dict, topic: str | None = None):
  packet = MagicMock()
  packet.topic = topic or _livekit_runner.DATA_TOPIC
  packet.data = json.dumps(message).encode("utf-8")
  return packet


# --- Lifecycle ---


@pytest.mark.asyncio
async def test_start_closes_queue_when_session_ends():
  """When run_live finishes, the live request queue is closed."""
  lk_runner = _make_lk_runner(_make_runner([]), _make_room())
  lk_runner._queue.close = MagicMock()

  await lk_runner.start()

  lk_runner._queue.close.assert_called_once()
