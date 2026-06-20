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

"""The audio I/O seam for live evaluation.

A `UserTurnTransport` carries a conversation produced by user simulation to the
agent under test *as audio*, and captures the agent's spoken reply. It is the
only live-specific machinery: the conversation source (a fixed script via
`StaticUserSimulator`, or a simulated user via `LlmBackedUserSimulator`) and the
downstream materialization and scoring are all shared with non-live eval.

Two audio-generation approaches are provided:

* `TtsUserTurnTransport` — synthesizes each simulated user turn to audio with a
  TTS voice and streams it to the agent. Works with any simulator (including a
  fixed script) and supports timed barge-in.
* `NativeAudioPersonaTransport` — drives a native-audio persona Live agent that
  hears the agent's audio and replies in kind, supporting reactive barge-in.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import asyncio
import logging
import random
import time
from typing import Awaitable
from typing import Callable
from typing import Optional

from google.genai import Client
from google.genai import types

from ...agents.base_agent import BaseAgent
from ...agents.live_request_queue import LiveRequestQueue
from ...agents.llm_agent import Agent
from ...agents.run_config import RunConfig
from ...agents.run_config import StreamingMode
from ...artifacts.base_artifact_service import BaseArtifactService
from ...artifacts.in_memory_artifact_service import InMemoryArtifactService
from ...runners import Runner
from ...sessions.base_session_service import BaseSessionService
from ...sessions.in_memory_session_service import InMemorySessionService
from ...utils.context_utils import Aclosing
from ...utils.feature_decorator import experimental
from ..conversation_scenarios import ConversationScenario
from ..eval_case import AudioReference
from .audio_realism import build_audio_realism_transform
from .audio_utils import LIVE_INPUT_MIME_TYPE
from .audio_utils import LIVE_OUTPUT_RATE_HZ
from .audio_utils import parse_sample_rate
from .audio_utils import to_live_input
from .live_app_details import build_app_details
from .live_conversation_types import CapturedUtterance
from .live_conversation_types import ConversationTurn
from .live_conversation_types import LiveConversation
from .persona_customer_agent import PersonaCustomerAgentFactory
from .user_simulator import Status as UserSimulatorStatus
from .user_simulator import UserSimulator
from .voice_profile import BargeInConfig
from .voice_profile import VoiceProfile

logger = logging.getLogger("google_adk." + __name__)

_USER_AUTHOR = "user"
_SUT_SPEAKER = "sut"
_USER_SPEAKER = "persona"
# 0.5s of 16kHz 16-bit mono PCM, a proven realtime chunk cadence.
_AUDIO_CHUNK_BYTES = 16000
# Trailing silence appended to an utterance in auto-VAD mode so the server-side
# voice-activity detector sees the speaker stop and emits `turn_complete`.
# 0.8s of 16kHz 16-bit mono PCM (16000 samples/s * 2 bytes * 0.8s).
_AUTO_VAD_TRAILING_SILENCE_BYTES = 16000 * 2 * 8 // 10
# Absolute cap on waiting for a turn to start/finish.
_TURN_TIMEOUT_SECONDS = 30
# Give up only after this many consecutive turns where the SUT said nothing.
_MAX_CONSECUTIVE_SILENT_TURNS = 2
# A tool call produces a turn_complete with no speech; the spoken result follows
# in a subsequent turn. Continue collecting across at most this many such
# tool-only completions so the spoken reply lands in the same utterance.
_MAX_TOOL_CONTINUATION_ROUNDS = 3
# Default TTS model used to synthesize simulated user turns.
_DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"
# TTS output is 24kHz 16-bit mono PCM.
_TTS_OUTPUT_MIME_TYPE = "audio/pcm;rate=24000"

# Progress events streamed to observers (Watch mode / CLI).
ProgressCallback = Callable[[dict], Awaitable[None]]


@experimental
class UserTurnTransport(ABC):
  """Carries a user-simulated conversation to the agent under test as audio."""

  @abstractmethod
  async def run(
      self,
      *,
      user_simulator: UserSimulator,
      scenario: Optional[ConversationScenario],
      voice_profile: VoiceProfile,
      max_turns: int,
      session_id: Optional[str] = None,
      progress_callback: Optional[ProgressCallback] = None,
  ) -> LiveConversation:
    """Runs the conversation over audio and returns the captured turns.

    Args:
      user_simulator: Produces the next user turn (text) given history. Either a
        fixed script or a simulated user.
      scenario: The conversation scenario, when the conversation came from one.
        Required by transports that build a persona agent; may be None for a
        fixed-script run over TTS.
      voice_profile: The voice and realism settings for the user's audio.
      max_turns: Safety cap on the number of turns.
      session_id: Optional id for the agent-under-test session.
      progress_callback: Optional async callback for live observation.

    Returns:
      The captured `LiveConversation`, ready for materialization and scoring.
    """


class _LiveAgentSession:
  """Runs one Live participant and collects a turn's audio + transcript.

  A single `run_live` stream is consumed in a background task, pushing events
  onto a queue. Automatic VAD is disabled by default; input audio is sent
  bracketed by manual activity markers, so the model produces a reliable
  `turn_complete`. `collect_turn` drains events until that `turn_complete`,
  aggregating audio bytes, transcript text, and tool activity.
  """

  def __init__(
      self,
      *,
      runner: Runner,
      session,
      speaker: str,
      automatic_vad: bool = False,
  ):
    self._runner = runner
    self._session = session
    self._speaker = speaker
    self._automatic_vad = automatic_vad
    self._live_request_queue = LiveRequestQueue()
    self._event_queue: asyncio.Queue = asyncio.Queue()
    self._live_finished = asyncio.Event()
    self._consume_task: Optional[asyncio.Task] = None

  async def __aenter__(self) -> _LiveAgentSession:
    self._consume_task = asyncio.create_task(self._consume_events())
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    from google.genai import errors
    from websockets.exceptions import ConnectionClosed
    from websockets.exceptions import ConnectionClosedOK

    self._live_request_queue.close()
    if self._consume_task is None:
      return
    try:
      await asyncio.wait_for(self._consume_task, timeout=30)
    except asyncio.TimeoutError:
      logger.warning(
          "Timed out waiting for %s run_live to finish.", self._speaker
      )
      self._consume_task.cancel()
      try:
        await self._consume_task
      except asyncio.CancelledError:
        pass
    except (ConnectionClosed, errors.APIError) as e:
      is_normal_closure = isinstance(e, ConnectionClosedOK) or (
          isinstance(e, errors.APIError) and e.code == 1000
      )
      if not is_normal_closure:
        raise

  def _build_run_config(self) -> RunConfig:
    """Builds the RunConfig for a relay participant.

    Automatic voice-activity detection is disabled by default so turn
    boundaries are controlled explicitly via manual activity markers; this
    avoids the model's turn detection getting confused by relayed audio. The
    native-audio persona enables auto-VAD so it can react (and barge in) to the
    agent's audio as it arrives.
    """
    realtime_input_config = None
    if not self._automatic_vad:
      realtime_input_config = types.RealtimeInputConfig(
          automatic_activity_detection=types.AutomaticActivityDetection(
              disabled=True
          )
      )
    return RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=realtime_input_config,
    )

  async def _consume_events(self) -> None:
    run_config = self._build_run_config()
    try:
      async with Aclosing(
          self._runner.run_live(
              session=self._session,
              live_request_queue=self._live_request_queue,
              run_config=run_config,
          )
      ) as agen:
        async for event in agen:
          await self._event_queue.put(event)
    finally:
      self._live_finished.set()

  def send_audio(
      self, pcm: bytes, *, mime_type: str = LIVE_INPUT_MIME_TYPE
  ) -> None:
    """Streams input audio to this participant.

    Manual activity markers (`activity_start`/`activity_end`) are only valid
    when automatic (server-side) VAD is disabled; sending them with auto-VAD
    enabled is rejected by the Live API and leaves the turn without a clean
    boundary. So we bracket the audio with markers only in manual-VAD mode and
    let the server detect speech boundaries otherwise.
    """
    manual_vad = not self._automatic_vad
    if manual_vad:
      self._live_request_queue.send_activity_start()
    else:
      # Auto-VAD detects end-of-speech from trailing silence, so pad the
      # utterance with a short silent tail; without it the server keeps waiting
      # for more speech and never emits `turn_complete`.
      pcm = pcm + b"\x00" * _AUTO_VAD_TRAILING_SILENCE_BYTES
    for start in range(0, len(pcm), _AUDIO_CHUNK_BYTES):
      chunk = pcm[start : start + _AUDIO_CHUNK_BYTES]
      if chunk:
        self._live_request_queue.send_realtime(
            types.Blob(data=chunk, mime_type=mime_type)
        )
    if manual_vad:
      self._live_request_queue.send_activity_end()

  def send_text(self, text: str) -> None:
    """Sends a text message to this participant (used to start the persona)."""
    self._live_request_queue.send_content(
        types.Content(role="user", parts=[types.Part(text=text)])
    )

  def drain_pending_events(self) -> None:
    """Discards any events already queued from a previous turn.

    With auto-VAD (and after a barge-in cuts a turn short), audio and
    transcription events from the prior turn can remain queued. Draining them
    before the next turn's input is sent ensures the next `collect_turn` only
    observes events caused by that input, which keeps the latency anchor and the
    captured audio aligned with the turn being measured.
    """
    while True:
      try:
        self._event_queue.get_nowait()
      except asyncio.QueueEmpty:
        break

  async def collect_turn(
      self,
      *,
      timeout_seconds: int = _TURN_TIMEOUT_SECONDS,
      listen_window_seconds: Optional[float] = None,
      input_sent_time: Optional[float] = None,
  ) -> CapturedUtterance:
    """Drains events until this participant produces a spoken reply.

    Args:
      timeout_seconds: Hard cap on waiting for the turn to complete.
      listen_window_seconds: If set, stop collecting this many seconds after the
        first audio is heard and return the partial reply flagged as
        `was_interrupted`. Used to barge in on the agent mid-turn.
      input_sent_time: Epoch seconds when the input that prompted this turn was
        sent. When provided, audio that arrives before this time is treated as
        stale (left over from a prior turn) and does not set the latency anchor,
        so `first_audio_time` reflects this turn's true response onset.

    Returns:
      The captured utterance. `was_interrupted` is True when a listen window cut
      the turn short before the speaker finished.
    """
    audio_chunks: list[bytes] = []
    partial_transcript_parts: list[str] = []
    final_transcript: Optional[str] = None
    tool_calls: list[types.FunctionCall] = []
    tool_responses: list[types.FunctionResponse] = []
    mime_type = _TTS_OUTPUT_MIME_TYPE
    start_time = time.time()
    first_audio_time: Optional[float] = None
    tool_continuations = 0
    was_interrupted = False

    deadline = start_time + timeout_seconds
    while True:
      remaining = deadline - time.time()
      # Once the speaker has started, a listen window caps how long we let it
      # speak before barging in.
      if listen_window_seconds is not None and first_audio_time is not None:
        window_remaining = (
            first_audio_time + listen_window_seconds - time.time()
        )
        if window_remaining <= 0:
          was_interrupted = True
          break
        remaining = min(remaining, window_remaining)
      if remaining <= 0:
        logger.warning("Timed out collecting %s turn.", self._speaker)
        break
      try:
        event = await asyncio.wait_for(
            self._event_queue.get(), timeout=remaining
        )
      except asyncio.TimeoutError:
        if listen_window_seconds is not None and first_audio_time is not None:
          was_interrupted = True
          break
        logger.warning("Timed out collecting %s turn.", self._speaker)
        break

      if event.content and event.content.parts:
        for part in event.content.parts:
          if (
              part.inline_data
              and part.inline_data.data
              and (part.inline_data.mime_type or "").startswith("audio/")
          ):
            now = time.time()
            # Ignore audio that predates the input being sent: it is a leftover
            # chunk from a prior turn and would make the latency anchor (and the
            # captured-audio duration) misrepresent this turn.
            is_stale = input_sent_time is not None and now < input_sent_time
            if is_stale:
              continue
            if first_audio_time is None:
              first_audio_time = now
            audio_chunks.append(part.inline_data.data)
            mime_type = part.inline_data.mime_type or mime_type

      if event.output_transcription and event.output_transcription.text:
        if event.partial:
          partial_transcript_parts.append(event.output_transcription.text)
        else:
          final_transcript = event.output_transcription.text

      for fc in event.get_function_calls():
        tool_calls.append(fc)
      for fr in event.get_function_responses():
        tool_responses.append(fr)

      if event.turn_complete and event.author != _USER_AUTHOR:
        has_speech = bool(audio_chunks) or bool(
            final_transcript or partial_transcript_parts
        )
        if (
            not has_speech
            and (tool_calls or tool_responses)
            and tool_continuations < _MAX_TOOL_CONTINUATION_ROUNDS
        ):
          tool_continuations += 1
          continue
        break

      if self._live_finished.is_set() and self._event_queue.empty():
        break

    transcript = (
        final_transcript
        if final_transcript is not None
        else "".join(partial_transcript_parts)
    )
    return CapturedUtterance(
        speaker=self._speaker,
        transcript=transcript,
        audio_pcm=b"".join(audio_chunks) if audio_chunks else None,
        mime_type=mime_type,
        start_time=start_time,
        end_time=time.time(),
        first_audio_time=first_audio_time,
        tool_calls=tool_calls,
        tool_responses=tool_responses,
        was_interrupted=was_interrupted,
    )

  @property
  def is_finished(self) -> bool:
    """Whether this participant's live session has fully closed."""
    return self._live_finished.is_set()


class _BaseAudioTransport(UserTurnTransport):
  """Shared plumbing for audio transports: sessions, persistence, recording."""

  def __init__(
      self,
      *,
      sut_agent: Agent,
      app_name: str,
      user_id: str = "live_eval_user",
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
  ):
    self._sut_agent = sut_agent
    self._app_name = app_name
    self._user_id = user_id
    self._session_service = session_service or InMemorySessionService()
    self._artifact_service = artifact_service or InMemoryArtifactService()
    self._session_id: Optional[str] = None

  def _new_sut_runner(self) -> Runner:
    return Runner(
        app_name=self._app_name,
        agent=self._sut_agent,
        session_service=self._session_service,
        artifact_service=self._artifact_service,
    )

  async def _persist_and_record(
      self,
      *,
      conversation: LiveConversation,
      turn_index: int,
      user_utterance: Optional[CapturedUtterance],
      sut_utterance: Optional[CapturedUtterance],
      progress_callback: Optional[ProgressCallback],
  ) -> None:
    """Persists audio for an utterance and records it on the turn."""
    turn = self._get_or_create_turn(conversation, turn_index)

    if user_utterance is not None:
      user_utterance.audio_reference = await self._persist_utterance_audio(
          user_utterance, turn_index
      )
      turn.persona_utterance = user_utterance
      if progress_callback and user_utterance.transcript:
        await progress_callback({
            "type": "transcript_update",
            "speaker": "persona",
            "turn_index": turn_index,
            "text": user_utterance.transcript,
        })

    if sut_utterance is not None:
      sut_utterance.audio_reference = await self._persist_utterance_audio(
          sut_utterance, turn_index
      )
      turn.sut_utterance = sut_utterance
      if progress_callback and sut_utterance.transcript:
        await progress_callback({
            "type": "transcript_update",
            "speaker": "sut",
            "turn_index": turn_index,
            "text": sut_utterance.transcript,
        })

  @staticmethod
  def _get_or_create_turn(
      conversation: LiveConversation, turn_index: int
  ) -> ConversationTurn:
    for turn in conversation.turns:
      if turn.turn_index == turn_index:
        return turn
    turn = ConversationTurn(turn_index=turn_index)
    conversation.turns.append(turn)
    return turn

  async def _persist_utterance_audio(
      self, utterance: CapturedUtterance, turn_index: int
  ) -> Optional[AudioReference]:
    """Saves an utterance's audio to the artifact service, if present."""
    if not utterance.audio_pcm:
      return None
    filename = f"turn_{turn_index}_{utterance.speaker}.pcm"
    version = await self._artifact_service.save_artifact(
        app_name=self._app_name,
        user_id=self._user_id,
        session_id=self._session_id,
        filename=filename,
        artifact=types.Part.from_bytes(
            data=utterance.audio_pcm, mime_type=utterance.mime_type
        ),
    )
    return AudioReference(
        artifact_filename=filename,
        version=version,
        mime_type=utterance.mime_type,
        sample_rate_hz=parse_sample_rate(
            utterance.mime_type, default=LIVE_OUTPUT_RATE_HZ
        ),
        # 16-bit PCM: two bytes per sample.
        num_samples=len(utterance.audio_pcm) // 2,
    )


@experimental
class TtsUserTurnTransport(_BaseAudioTransport):
  """Carries each simulated user turn to the agent as synthesized speech.

  The user simulator yields the next turn as text; this transport renders it to
  audio with a TTS voice, optionally applies realism, streams it to the agent
  under test, and captures the spoken reply. It works with any simulator,
  including a fixed script, and supports timed barge-in.
  """

  def __init__(
      self,
      *,
      sut_agent: Agent,
      app_name: str,
      user_id: str = "live_eval_user",
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      tts_model: str = _DEFAULT_TTS_MODEL,
      tts_client: Optional[Client] = None,
  ):
    super().__init__(
        sut_agent=sut_agent,
        app_name=app_name,
        user_id=user_id,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    self._tts_model = tts_model
    self._tts_client = tts_client or Client()

  async def run(
      self,
      *,
      user_simulator: UserSimulator,
      scenario: Optional[ConversationScenario],
      voice_profile: VoiceProfile,
      max_turns: int,
      session_id: Optional[str] = None,
      progress_callback: Optional[ProgressCallback] = None,
  ) -> LiveConversation:
    realism = build_audio_realism_transform(voice_profile.audio_realism)
    conversation = LiveConversation()
    # Capture the agent-under-test's instruction + tool declarations so managed
    # metrics can judge the trajectory with the same agent context the non-live
    # path provides.
    conversation.app_details = await build_app_details(self._sut_agent)

    sut_session = await self._session_service.create_session(
        app_name=self._app_name, user_id=self._user_id, session_id=session_id
    )
    self._session_id = sut_session.id
    conversation.session_id = sut_session.id

    # The simulator drives turn count via its own stop signal / limit; max_turns
    # is a hard safety cap.
    history: list = []
    async with _LiveAgentSession(
        runner=self._new_sut_runner(), session=sut_session, speaker=_SUT_SPEAKER
    ) as sut:
      conversation.termination_reason = await self._drive(
          user_simulator=user_simulator,
          voice_profile=voice_profile,
          realism=realism,
          max_turns=max_turns,
          sut=sut,
          history=history,
          conversation=conversation,
          progress_callback=progress_callback,
      )

    if progress_callback:
      await progress_callback({
          "type": "conversation_complete",
          "termination_reason": conversation.termination_reason,
          "turns": len(conversation.turns),
      })
    return conversation

  async def _drive(
      self,
      *,
      user_simulator: UserSimulator,
      voice_profile: VoiceProfile,
      realism,
      max_turns: int,
      sut: _LiveAgentSession,
      history: list,
      conversation: LiveConversation,
      progress_callback: Optional[ProgressCallback],
  ) -> str:
    """Runs the alternating turn loop; returns the termination reason."""
    from ...events.event import Event

    termination_reason = "completed"
    barge_in_count = 0
    for turn_index in range(max_turns):
      next_message = await user_simulator.get_next_user_message(list(history))
      if next_message.status != UserSimulatorStatus.SUCCESS:
        termination_reason = (
            "max_turns"
            if next_message.status == UserSimulatorStatus.TURN_LIMIT_REACHED
            else "completed"
        )
        break

      user_text = _content_text(next_message.user_message)
      if progress_callback:
        await progress_callback(
            {"type": "turn_started", "turn_index": turn_index}
        )

      user_audio = await self._synthesize(user_text, voice_profile)
      user_audio = await realism.apply(
          user_audio, mime_type=_TTS_OUTPUT_MIME_TYPE
      )
      user_utterance = CapturedUtterance(
          speaker=_USER_SPEAKER,
          transcript=user_text,
          audio_pcm=user_audio,
          mime_type=_TTS_OUTPUT_MIME_TYPE,
          start_time=time.time(),
      )
      await self._persist_and_record(
          conversation=conversation,
          turn_index=turn_index,
          user_utterance=user_utterance,
          sut_utterance=None,
          progress_callback=progress_callback,
      )

      # Discard any audio still queued from the previous turn so this turn's
      # latency and captured audio are measured cleanly.
      sut.drain_pending_events()
      sut.send_audio(
          to_live_input(user_audio, source_mime_type=_TTS_OUTPUT_MIME_TYPE),
          mime_type=LIVE_INPUT_MIME_TYPE,
      )
      input_sent_time = time.time()
      # Decide whether the user barges in on this reply, and if so cut the
      # agent's turn short after a randomized listen window. A per-conversation
      # cap (max_barge_ins) prevents interrupting on every turn.
      barge_in_now = _maybe_barge_in(voice_profile.barge_in, barge_in_count)
      listen_window = (
          _barge_in_listen_seconds(voice_profile.barge_in)
          if barge_in_now
          else None
      )
      sut_utterance = await sut.collect_turn(
          listen_window_seconds=listen_window,
          input_sent_time=input_sent_time,
      )
      sut_utterance.input_sent_time = input_sent_time
      if sut_utterance.was_interrupted:
        barge_in_count += 1
        if progress_callback:
          await progress_callback({
              "type": "barge_in",
              "turn_index": turn_index,
              "listen_seconds": (
                  round(listen_window, 2) if listen_window else None
              ),
          })
      await self._persist_and_record(
          conversation=conversation,
          turn_index=turn_index,
          user_utterance=None,
          sut_utterance=sut_utterance,
          progress_callback=progress_callback,
      )

      if progress_callback:
        await progress_callback({
            "type": "turn_complete",
            "turn_index": turn_index,
            "persona_transcript": user_utterance.transcript,
            "sut_transcript": sut_utterance.transcript,
        })

      # Feed both turns into the simulator history so the next user turn is
      # conditioned on the agent's reply.
      history.append(
          Event(author=_USER_AUTHOR, content=next_message.user_message)
      )
      if sut_utterance.transcript:
        history.append(
            Event(
                author=_SUT_SPEAKER,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=sut_utterance.transcript)],
                ),
            )
        )

      if sut.is_finished:
        termination_reason = "completed"
        break
    else:
      termination_reason = "max_turns"

    return termination_reason

  async def _synthesize(self, text: str, voice_profile: VoiceProfile) -> bytes:
    """Synthesizes `text` to PCM audio with the configured TTS voice."""
    if not text:
      return b""
    response = await self._tts_client.aio.models.generate_content(
        model=self._tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_profile.voice_name,
                    )
                ),
                language_code=voice_profile.language_code,
            ),
        ),
    )
    for candidate in response.candidates or []:
      if not candidate.content or not candidate.content.parts:
        continue
      for part in candidate.content.parts:
        if part.inline_data and part.inline_data.data:
          return part.inline_data.data
    logger.warning("TTS returned no audio for text: %r", text[:60])
    return b""


@experimental
class NativeAudioPersonaTransport(_BaseAudioTransport):
  """Drives a native-audio persona agent that hears and speaks to the agent.

  The simulated user is itself a Live agent built from the conversation
  scenario; it consumes the agent-under-test's audio and replies in its own
  voice, supporting reactive barge-in. This transport requires a scenario (it
  cannot perform a fixed script).
  """

  def __init__(
      self,
      *,
      sut_agent: Agent,
      app_name: str,
      user_id: str = "live_eval_user",
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      persona_agent_factory: Optional[PersonaCustomerAgentFactory] = None,
  ):
    super().__init__(
        sut_agent=sut_agent,
        app_name=app_name,
        user_id=user_id,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    self._persona_agent_factory = (
        persona_agent_factory or PersonaCustomerAgentFactory()
    )

  async def run(
      self,
      *,
      user_simulator: UserSimulator,
      scenario: Optional[ConversationScenario],
      voice_profile: VoiceProfile,
      max_turns: int,
      session_id: Optional[str] = None,
      progress_callback: Optional[ProgressCallback] = None,
  ) -> LiveConversation:
    if scenario is None:
      raise ValueError(
          "NativeAudioPersonaTransport requires a conversation_scenario; a"
          " fixed script can only be run over the TTS transport."
      )
    persona_agent = self._persona_agent_factory.build(
        scenario, voice_profile=voice_profile
    )
    realism = build_audio_realism_transform(voice_profile.audio_realism)
    conversation = LiveConversation()
    # Capture the agent-under-test's instruction + tool declarations so managed
    # metrics can judge the trajectory with the same agent context the non-live
    # path provides.
    conversation.app_details = await build_app_details(self._sut_agent)

    sut_session = await self._session_service.create_session(
        app_name=self._app_name, user_id=self._user_id, session_id=session_id
    )
    self._session_id = sut_session.id
    conversation.session_id = sut_session.id
    persona_session = await self._session_service.create_session(
        app_name="simulated_user", user_id=self._user_id
    )

    persona_runner = Runner(
        app_name="simulated_user",
        agent=persona_agent,
        session_service=self._session_service,
        artifact_service=self._artifact_service,
    )

    barge_in_enabled = bool(
        voice_profile.barge_in and voice_profile.barge_in.enabled
    )
    async with (
        _LiveAgentSession(
            runner=self._new_sut_runner(),
            session=sut_session,
            speaker=_SUT_SPEAKER,
            # Auto-VAD lets the SUT detect the persona talking over it.
            automatic_vad=barge_in_enabled,
        ) as sut,
        _LiveAgentSession(
            runner=persona_runner,
            session=persona_session,
            speaker=_USER_SPEAKER,
        ) as persona_live,
    ):
      conversation.termination_reason = await self._drive(
          voice_profile=voice_profile,
          realism=realism,
          max_turns=max_turns,
          sut=sut,
          persona_live=persona_live,
          conversation=conversation,
          progress_callback=progress_callback,
      )

    if progress_callback:
      await progress_callback({
          "type": "conversation_complete",
          "termination_reason": conversation.termination_reason,
          "turns": len(conversation.turns),
      })
    return conversation

  async def _drive(
      self,
      *,
      voice_profile: VoiceProfile,
      realism,
      max_turns: int,
      sut: _LiveAgentSession,
      persona_live: _LiveAgentSession,
      conversation: LiveConversation,
      progress_callback: Optional[ProgressCallback],
  ) -> str:
    """Runs the audio-to-audio relay loop; returns the termination reason."""
    persona_live.send_text(
        "Begin the conversation now. Speak your opening line out loud."
    )
    persona_utterance = await persona_live.collect_turn()
    termination_reason = "max_turns"
    consecutive_silent_turns = 0
    barge_in_count = 0

    for turn_index in range(max_turns):
      if progress_callback:
        await progress_callback(
            {"type": "turn_started", "turn_index": turn_index}
        )

      if not persona_utterance.audio_pcm:
        termination_reason = "completed"
        break

      persona_audio = await realism.apply(
          persona_utterance.audio_pcm, mime_type=persona_utterance.mime_type
      )
      await self._persist_and_record(
          conversation=conversation,
          turn_index=turn_index,
          user_utterance=persona_utterance,
          sut_utterance=None,
          progress_callback=progress_callback,
      )

      # Discard any audio still queued from the previous turn so this turn's
      # latency and captured audio are measured cleanly. This matters most here
      # because auto-VAD streams the SUT's audio more continuously.
      sut.drain_pending_events()
      sut.send_audio(
          to_live_input(
              persona_audio, source_mime_type=persona_utterance.mime_type
          ),
          mime_type=LIVE_INPUT_MIME_TYPE,
      )
      input_sent_time = time.time()
      # Reactive barge-in: let the agent speak for a randomized window, then cut
      # its turn short. With auto-VAD on the SUT, relaying the persona's next
      # reply mid-turn interrupts it. A per-conversation cap (max_barge_ins)
      # prevents interrupting on every turn.
      barge_in_now = _maybe_barge_in(voice_profile.barge_in, barge_in_count)
      listen_window = (
          _barge_in_listen_seconds(voice_profile.barge_in)
          if barge_in_now
          else None
      )
      sut_utterance = await sut.collect_turn(
          listen_window_seconds=listen_window,
          input_sent_time=input_sent_time,
      )
      sut_utterance.input_sent_time = input_sent_time
      if sut_utterance.was_interrupted:
        barge_in_count += 1
        if progress_callback:
          await progress_callback({
              "type": "barge_in",
              "turn_index": turn_index,
              "listen_seconds": (
                  round(listen_window, 2) if listen_window else None
              ),
          })
      await self._persist_and_record(
          conversation=conversation,
          turn_index=turn_index,
          user_utterance=None,
          sut_utterance=sut_utterance,
          progress_callback=progress_callback,
      )

      if progress_callback:
        await progress_callback({
            "type": "turn_complete",
            "turn_index": turn_index,
            "persona_transcript": persona_utterance.transcript,
            "sut_transcript": sut_utterance.transcript,
        })

      if sut.is_finished or persona_live.is_finished:
        termination_reason = "completed"
        break

      sut_silent = not sut_utterance.audio_pcm and not sut_utterance.transcript
      if sut_silent:
        consecutive_silent_turns += 1
        if consecutive_silent_turns >= _MAX_CONSECUTIVE_SILENT_TURNS:
          termination_reason = "completed"
          break
        persona_live.send_text(
            "The other speaker did not respond. Please continue the"
            " conversation toward your goal."
        )
        persona_utterance = await persona_live.collect_turn()
        continue

      consecutive_silent_turns = 0
      persona_live.send_audio(
          to_live_input(
              sut_utterance.audio_pcm,
              source_mime_type=sut_utterance.mime_type,
          ),
          mime_type=LIVE_INPUT_MIME_TYPE,
      )
      persona_utterance = await persona_live.collect_turn()

    return termination_reason


def _content_text(content: Optional[types.Content]) -> str:
  """Returns the concatenated text of a Content's parts."""
  if content is None or not content.parts:
    return ""
  return "".join(part.text for part in content.parts if part.text)


def _maybe_barge_in(
    barge_in: Optional[BargeInConfig], barge_in_count: int = 0
) -> bool:
  """Returns whether a barge-in should occur this turn, per config.

  Args:
    barge_in: The barge-in configuration, or None.
    barge_in_count: How many barge-ins have already happened this conversation.
      Used to enforce `max_barge_ins`.
  """
  if not barge_in or not barge_in.enabled:
    return False
  if (
      barge_in.max_barge_ins is not None
      and barge_in_count >= barge_in.max_barge_ins
  ):
    return False
  return random.random() < barge_in.probability


def _barge_in_listen_seconds(barge_in: Optional[BargeInConfig]) -> float:
  """Returns how long (seconds) to let the agent speak before interrupting.

  Drawn uniformly from `[min_listen_ms, max_listen_ms]` so the cut-in point
  varies turn to turn. Falls back to a short default if config is missing.
  """
  if barge_in is None:
    return 0.5
  low = max(0, barge_in.min_listen_ms)
  high = max(low, barge_in.max_listen_ms)
  return random.uniform(low, high) / 1000.0
