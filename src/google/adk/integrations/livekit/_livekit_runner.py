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

"""LiveKit connector for ADK live agents.

`LiveKitRunner` is the realtime analog of `SlackRunner`: it wraps an unmodified
ADK `Runner` and bridges a LiveKit room to the transport-agnostic
`LiveRequestQueue` -> `run_live()` -> `Event` contract. It contains no agent
logic, no codecs, and no signaling -- just two frame bridges:

  * Bridge 1 (inbound): room media tracks -> `LiveRequestQueue.send_realtime`,
    and room data messages -> `LiveRequestQueue.send_content`.
  * Bridge 2 (outbound): the `Event` stream -> room audio / data tracks.

LiveKit owns the hard parts (WebRTC, SIP, jitter, echo, resampling on
subscribe, track lifecycle), so this bridge is thin. Unlike the long-lived
`SlackRunner` singleton, a `LiveKitRunner` is dispatched *per call* -- one
worker per room.

This module lazily imports the LiveKit SDK; install it with::

    pip install "google-adk[livekit]"
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any
from typing import Optional

from google.genai import types

from ...agents.live_request_queue import LiveRequestQueue
from ...agents.run_config import RunConfig
from ...events.event import Event
from ...runners import Runner

try:
  from livekit import rtc
  from livekit.agents.utils import images
except ImportError as e:
  raise ImportError(
      "livekit is not installed. Please install it with "
      '`pip install "google-adk[livekit]"`.'
  ) from e

logger = logging.getLogger("google_adk." + __name__)

# ADK's live media contract. LiveKit resamples inbound audio on subscribe, so
# the bridge just requests these rates and wraps the raw bytes in a Blob.
_INPUT_SAMPLE_RATE = 16000
_OUTPUT_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1
_BYTES_PER_SAMPLE = 2  # 16-bit PCM.
# The rate belongs in the mime type: ADK identifies live input as
# `audio/pcm;rate=16000` (see `evaluation._audio_utils.LIVE_INPUT_MIME_TYPE`).
_AUDIO_MIME_TYPE = f"audio/pcm;rate={_INPUT_SAMPLE_RATE}"
_VIDEO_MIME_TYPE = "image/jpeg"
_OUTPUT_AUDIO_TRACK_NAME = "adk-agent-audio"

# Live models sample video, they do not consume it at capture frame rate.
# Forwarding every frame floods the queue for no gain, so sample at 1 fps and
# downscale before JPEG encoding.
_VIDEO_FRAMES_PER_SECOND = 1.0
_VIDEO_MAX_WIDTH = 1024
_VIDEO_MAX_HEIGHT = 1024
_VIDEO_JPEG_QUALITY = 75

# Topic for the agent's outbound data messages (transcripts, tool activity), so
# clients can subscribe to ADK traffic without parsing every room message.
DATA_TOPIC = "adk"


class LiveKitRunner:
  """Bridges a LiveKit room to an ADK `Runner.run_live()` session.

  One instance drives exactly one live session (one room / one call). Construct
  it after the worker has joined the room (`ctx.connect()`), then `await
  start()` to run until the session ends.

  The ADK session must exist before the call, or the runner must be built with
  `auto_create_session=True` -- a dispatched room is usually a brand new
  session, so `livekit_server` enables that for the runners it builds.

  Example (custom entrypoint)::

      from livekit.agents import AgentServer, JobContext, cli

      runner = InMemoryRunner(agent=root_agent, app_name="my_app")
      runner.auto_create_session = True
      server = AgentServer()

      @server.rtc_session(agent_name="my_app")
      async def entrypoint(ctx: JobContext) -> None:
        await ctx.connect()
        meta = json.loads(ctx.job.metadata or "{}")
        await LiveKitRunner(
            runner=runner,
            room=ctx.room,
            user_id=meta.get("user_id", "live-user"),
            session_id=meta.get("session_id", ctx.room.name),
        ).start()

      cli.run_app(server)  # the developer owns the process
  """

  def __init__(
      self,
      runner: Runner,
      room: rtc.Room,
      *,
      user_id: str,
      session_id: str,
      run_config: Optional[RunConfig] = None,
  ):
    """Initializes the runner.

    Args:
      runner: An unmodified ADK `Runner` (e.g. `InMemoryRunner`).
      room: The LiveKit room delivered by dispatch via `JobContext`, already
        connected.
      user_id: The ADK user id for the session.
      session_id: The ADK session id for the session.
      run_config: Optional run config. Defaults to AUDIO response modality,
        matching a voice session. Set `input_audio_transcription` /
        `output_audio_transcription` here to have transcripts published on the
        room data track.
    """
    self._runner = runner
    self._room = room
    self._user_id = user_id
    self._session_id = session_id
    self._run_config = run_config or RunConfig(
        response_modalities=[types.Modality.AUDIO]
    )
    self._queue = LiveRequestQueue()

    # Outbound audio: a local track the agent's voice is captured onto.
    self._audio_source = rtc.AudioSource(_OUTPUT_SAMPLE_RATE, _NUM_CHANNELS)
    self._audio_track = rtc.LocalAudioTrack.create_audio_track(
        _OUTPUT_AUDIO_TRACK_NAME, self._audio_source
    )
    self._forward_tasks: set[asyncio.Task[None]] = set()

  async def start(self) -> None:
    """Runs the live session until the room disconnects or `run_live` ends."""
    await self._publish_output_audio_track()
    self._subscribe_existing_tracks()
    self._room.on("track_subscribed", self._on_track_subscribed)
    self._room.on("data_received", self._on_data_received)

    try:
      await self._forward_events()
    finally:
      for task in self._forward_tasks:
        task.cancel()
      self._queue.close()

  # -- Bridge 1: inbound (room media track -> LiveRequestQueue) --------------

  def _on_track_subscribed(
      self,
      track: rtc.Track,
      publication: rtc.TrackPublication,
      participant: rtc.RemoteParticipant,
  ) -> None:
    """Spawns a forwarder when a remote participant publishes a track."""
    del publication, participant  # Unused; identity comes from dispatch.
    self._spawn_forwarder(track)

  def _subscribe_existing_tracks(self) -> None:
    """Forwards tracks already present when the worker joined the room."""
    for participant in self._room.remote_participants.values():
      for publication in participant.track_publications.values():
        if publication.track is not None:
          self._spawn_forwarder(publication.track)

  def _spawn_forwarder(self, track: rtc.Track) -> None:
    if track.kind == rtc.TrackKind.KIND_AUDIO:
      task = asyncio.create_task(self._forward_audio(track))
    elif track.kind == rtc.TrackKind.KIND_VIDEO:
      task = asyncio.create_task(self._forward_video(track))
    else:
      logger.debug("Ignoring track of unsupported kind: %s", track.kind)
      return
    self._forward_tasks.add(task)
    task.add_done_callback(self._forward_tasks.discard)

  async def _forward_audio(self, track: rtc.Track) -> None:
    """Streams a room audio track into the queue as 16kHz PCM blobs."""
    audio_stream = rtc.AudioStream(
        track, sample_rate=_INPUT_SAMPLE_RATE, num_channels=_NUM_CHANNELS
    )
    try:
      async for event in audio_stream:
        self._queue.send_realtime(
            types.Blob(
                mime_type=_AUDIO_MIME_TYPE,
                data=bytes(event.frame.data),
            )
        )
    finally:
      # The track ended (participant muted, unpublished, or left). Flush the
      # model's audio buffer so a server-VAD turn does not hang waiting for
      # input that will never arrive.
      self._queue.send_audio_stream_end()

  async def _forward_video(self, track: rtc.Track) -> None:
    """Streams a room video track into the queue as JPEG image blobs.

    Frames are sampled at `_VIDEO_FRAMES_PER_SECOND` and downscaled before
    encoding: live models sample video rather than consume every frame, so
    forwarding at capture rate only floods the queue.
    """
    video_stream = rtc.VideoStream(track)
    encode_options = images.EncodeOptions(
        format="JPEG",
        quality=_VIDEO_JPEG_QUALITY,
        resize_options=images.ResizeOptions(
            width=_VIDEO_MAX_WIDTH,
            height=_VIDEO_MAX_HEIGHT,
            strategy="scale_aspect_fit",
        ),
    )
    min_interval = 1.0 / _VIDEO_FRAMES_PER_SECOND
    next_frame_at = 0.0
    async for event in video_stream:
      now = time.monotonic()
      if now < next_frame_at:
        continue
      next_frame_at = now + min_interval
      # `images.encode` is CPU-bound (Pillow); keep it off the event loop so it
      # cannot stall audio forwarding.
      jpeg = await asyncio.to_thread(images.encode, event.frame, encode_options)
      self._queue.send_realtime(
          types.Blob(mime_type=_VIDEO_MIME_TYPE, data=jpeg)
      )

  def _on_data_received(self, packet: rtc.DataPacket) -> None:
    """Forwards a text message from the room into the session as a user turn.

    Gives clients a typed-input path alongside the microphone, using the same
    data track the agent publishes transcripts on.
    """
    if packet.topic != DATA_TOPIC:
      return
    text = _inbound_text(packet.data)
    if not text:
      return
    self._queue.send_content(
        types.Content(role="user", parts=[types.Part(text=text)])
    )

  # -- Bridge 2: outbound (Event stream -> room) ----------------------------

  async def _forward_events(self) -> None:
    """Drives `run_live` and pushes agent output back into the room."""
    async for event in self._runner.run_live(
        user_id=self._user_id,
        session_id=self._session_id,
        live_request_queue=self._queue,
        run_config=self._run_config,
    ):
      # Barge-in: the model was cut off, so drop the audio still queued for
      # playback. Without this the agent keeps talking over the user for as
      # long as the buffer lasts.
      if event.interrupted:
        self._audio_source.clear_queue()

      for audio in _audio_out(event):
        await self._audio_source.capture_frame(
            rtc.AudioFrame(
                data=audio,
                sample_rate=_OUTPUT_SAMPLE_RATE,
                num_channels=_NUM_CHANNELS,
                samples_per_channel=len(audio) // _BYTES_PER_SAMPLE,
            )
        )
      for payload in _data_out(event):
        await self._room.local_participant.publish_data(
            payload, topic=DATA_TOPIC
        )

  async def _publish_output_audio_track(self) -> None:
    await self._room.local_participant.publish_track(
        self._audio_track, rtc.TrackPublishOptions()
    )


def _inbound_text(data: bytes) -> Optional[str]:
  """Extracts the text of an inbound `{"type": "text", "text": ...}` message."""
  with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError):
    message = json.loads(data.decode("utf-8"))
    if isinstance(message, dict) and message.get("type") == "text":
      text = message.get("text")
      if isinstance(text, str) and text:
        return text
  logger.debug("Ignoring unrecognized inbound data message.")
  return None


def _audio_out(event: Event) -> list[bytes]:
  """Extracts raw output audio (24kHz PCM) from an event, if any."""
  if not (event.content and event.content.parts):
    return []
  blobs: list[bytes] = []
  for part in event.content.parts:
    inline_data = part.inline_data
    if (
        inline_data is not None
        and inline_data.data
        and (inline_data.mime_type or "").startswith("audio/")
    ):
      blobs.append(inline_data.data)
  return blobs


def _data_out(event: Event) -> list[bytes]:
  """Extracts data-track payloads (transcripts, tool activity) from an event.

  Partial transcripts are skipped: they arrive token by token and would make a
  client render the same utterance many times over.
  """
  payloads: list[bytes] = []

  if not event.partial:
    for transcription, role in (
        (event.input_transcription, "user"),
        (event.output_transcription, "agent"),
    ):
      if transcription is not None and transcription.text:
        payloads.append(
            _encode({
                "type": "transcript",
                "role": role,
                "text": transcription.text,
            })
        )

  if not (event.content and event.content.parts):
    return payloads

  for part in event.content.parts:
    if part.function_call is not None:
      payloads.append(
          _encode({
              "type": "function_call",
              "name": part.function_call.name,
              "args": part.function_call.args,
          })
      )
    elif part.function_response is not None:
      payloads.append(
          _encode({
              "type": "function_response",
              "name": part.function_response.name,
              "response": part.function_response.response,
          })
      )
  return payloads


def _encode(payload: dict[str, Any]) -> bytes:
  return json.dumps(payload).encode("utf-8")
