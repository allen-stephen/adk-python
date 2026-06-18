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

"""Persona-driven audio-to-audio live conversation runner.

This is the live-native heart of voice eval: two independent Gemini Live
sessions — a persona/customer agent and the agent under test (SUT) — exchange
*audio* turn by turn. Neither side is fed text; the persona listens to the SUT's
audio and replies with its own audio, and vice versa.

The runner captures each turn's audio and transcript, optionally applies
barge-in and audio-realism effects, persists audio to the artifact service, and
streams progress events so a UI or CLI can watch the conversation unfold.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import AsyncGenerator
from typing import Awaitable
from typing import Callable
from typing import Optional

from google.genai import types

from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.live_request_queue import LiveRequestQueue
from ...agents.llm_agent import Agent
from ...agents.readonly_context import ReadonlyContext
from ...agents.run_config import RunConfig
from ...agents.run_config import StreamingMode
from ...artifacts.base_artifact_service import BaseArtifactService
from ...artifacts.in_memory_artifact_service import InMemoryArtifactService
from ...runners import Runner
from ...sessions.base_session_service import BaseSessionService
from ...sessions.in_memory_session_service import InMemorySessionService
from ...sessions.session import Session
from ...tools.agent_tool import AgentTool
from ...tools.base_tool import BaseTool
from ...utils.context_utils import Aclosing
from ...utils.feature_decorator import experimental
from ..app_details import AgentDetails
from ..app_details import AppDetails
from ..eval_case import AudioReference
from .audio_realism import build_audio_realism_transform
from .audio_utils import LIVE_INPUT_MIME_TYPE
from .audio_utils import LIVE_OUTPUT_RATE_HZ
from .audio_utils import parse_sample_rate
from .audio_utils import to_live_input
from .live_conversation_scenario import LiveConversationScenario
from .live_conversation_types import CapturedUtterance
from .live_conversation_types import ConversationTurn
from .live_conversation_types import LiveConversation
from .persona import Persona
from .persona_customer_agent import PersonaCustomerAgentFactory

logger = logging.getLogger("google_adk." + __name__)

_USER_AUTHOR = "user"
# 0.5s of 16kHz 16-bit mono PCM, matching the proven CSEE chunk cadence.
_AUDIO_CHUNK_BYTES = 16000
# Absolute cap on waiting for a turn to start/finish.
_TURN_TIMEOUT_SECONDS = 30
# Give up only after this many consecutive turns where the SUT said nothing.
_MAX_CONSECUTIVE_SILENT_TURNS = 2
# A tool call produces a turn_complete with no speech; the spoken result follows
# in a subsequent turn. Continue collecting across at most this many such
# tool-only completions so the spoken reply lands in the same utterance.
_MAX_TOOL_CONTINUATION_ROUNDS = 3

# Progress events streamed to observers (Watch mode / CLI).
ProgressCallback = Callable[[dict], Awaitable[None]]


def _collect_agent_tree(root_agent: BaseAgent) -> list[BaseAgent]:
  """Returns the root agent and all of its declared descendants.

  Walks `sub_agents` recursively and also descends into agents wrapped by an
  `AgentTool`. De-dupes by agent name so each agent appears once. Captures the
  whole declared tree (a superset of agents that actually run), which is what the
  managed metrics want for rubric generation.
  """
  collected: dict[str, BaseAgent] = {}

  def _visit(agent: BaseAgent) -> None:
    if agent is None or agent.name in collected:
      return
    collected[agent.name] = agent
    for sub_agent in getattr(agent, "sub_agents", None) or []:
      _visit(sub_agent)
    # AgentTool-wrapped agents are not in `sub_agents`; descend into them too.
    for tool in getattr(agent, "tools", None) or []:
      if isinstance(tool, AgentTool) and getattr(tool, "agent", None):
        _visit(tool.agent)

  _visit(root_agent)
  return list(collected.values())


async def _resolve_agent_details(
    agent: BaseAgent,
    *,
    session_service: BaseSessionService,
    session: Session,
) -> AgentDetails:
  """Resolves an agent's instructions and tool declarations (best-effort).

  Uses a minimal `ReadonlyContext` to resolve provider-based instructions and
  dynamic tools, falling back to the raw instruction string / no tools when
  resolution is not possible (e.g. non-LlmAgent agents or resolution errors).
  """
  instructions = ""
  tool_declarations: list[object] = []

  readonly_context: Optional[ReadonlyContext] = None
  try:
    invocation_context = InvocationContext(
        session_service=session_service,
        invocation_id=f"___app_details___{agent.name}",
        agent=agent,
        session=session,
    )
    readonly_context = ReadonlyContext(invocation_context)
  except Exception as e:  # pylint: disable=broad-except
    logger.debug("Could not build context for agent %s: %s", agent.name, e)

  # Instructions.
  canonical_instruction = getattr(agent, "canonical_instruction", None)
  raw_instruction = getattr(agent, "instruction", None)
  if callable(canonical_instruction) and readonly_context is not None:
    try:
      instructions = (await canonical_instruction(readonly_context))[0]
    except Exception as e:  # pylint: disable=broad-except
      logger.debug("Could not resolve instruction for %s: %s", agent.name, e)
      instructions = raw_instruction if isinstance(raw_instruction, str) else ""
  elif isinstance(raw_instruction, str):
    instructions = raw_instruction

  # Tool declarations.
  canonical_tools = getattr(agent, "canonical_tools", None)
  if callable(canonical_tools):
    try:
      tools = await canonical_tools(readonly_context)
      declarations = []
      for tool in tools:
        if not isinstance(tool, BaseTool):
          continue
        declaration = tool._get_declaration()
        if declaration is not None:
          declarations.append(declaration)
      if declarations:
        tool_declarations = [types.Tool(function_declarations=declarations)]
    except Exception as e:  # pylint: disable=broad-except
      logger.debug("Could not resolve tools for %s: %s", agent.name, e)

  return AgentDetails(
      name=agent.name,
      instructions=instructions,
      tool_declarations=tool_declarations,
  )


async def _build_app_details(
    root_agent: BaseAgent,
    *,
    session_service: BaseSessionService,
    session: Session,
) -> AppDetails:
  """Builds AppDetails for the agent under test (root + sub-agents)."""
  agent_details: dict[str, AgentDetails] = {}
  for agent in _collect_agent_tree(root_agent):
    details = await _resolve_agent_details(
        agent, session_service=session_service, session=session
    )
    agent_details[agent.name] = details
  return AppDetails(agent_details=agent_details)


class _LiveAgentSession:
  """Runs one Live participant and collects a turn's audio + transcript.

  A single `run_live` stream is consumed in a background task, pushing events
  onto a queue. Automatic VAD is disabled; input audio is sent bracketed by
  manual activity markers (`send_audio`), so the model produces a reliable
  `turn_complete`. `collect_turn` drains events until that `turn_complete`,
  aggregating audio bytes, transcript text, and tool activity.
  """

  def __init__(
      self,
      *,
      runner: Runner,
      session,
      speaker: str,
  ):
    self._runner = runner
    self._session = session
    self._speaker = speaker
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

  @staticmethod
  def _build_run_config() -> RunConfig:
    """Builds the RunConfig for a relay participant.

    Automatic voice-activity detection is disabled so that turn boundaries are
    controlled explicitly via manual activity markers (`send_activity_start` /
    `send_activity_end`). This mirrors the proven CSEE mechanics: with auto-VAD
    the model's turn detection gets confused by relayed audio after the first
    turn and intermittently never responds.
    """
    return RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            )
        ),
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
    """Streams input audio to this participant bracketed by activity markers.

    Because automatic VAD is disabled (see `_build_run_config`), the turn is
    delimited explicitly: an `activity_start` signal, the audio sent as a
    sequence of realtime chunks, then an `activity_end` signal that tells the
    model the speaker is done and it should generate a response. This is the
    proven CSEE mechanic and gives reliable multi-turn replies.
    """
    self._live_request_queue.send_activity_start()
    for start in range(0, len(pcm), _AUDIO_CHUNK_BYTES):
      chunk = pcm[start : start + _AUDIO_CHUNK_BYTES]
      if chunk:
        self._live_request_queue.send_realtime(
            types.Blob(data=chunk, mime_type=mime_type)
        )
    self._live_request_queue.send_activity_end()

  def send_text(self, text: str) -> None:
    """Sends a text message to this participant (used to start the persona)."""
    self._live_request_queue.send_content(
        types.Content(role="user", parts=[types.Part(text=text)])
    )

  async def collect_turn(
      self,
      *,
      timeout_seconds: int = _TURN_TIMEOUT_SECONDS,
  ) -> CapturedUtterance:
    """Drains events until this participant produces a spoken reply.

    Aggregates output audio bytes, output transcript text, and any tool
    calls/responses the participant produced.

    A single logical reply can span multiple `turn_complete` signals: when the
    model calls a tool, it emits a `turn_complete` carrying only the tool
    call/response (no audio or transcript), executes the tool, then *speaks* the
    result in a following turn. We therefore continue past a tool-only
    completion (up to `_MAX_TOOL_CONTINUATION_ROUNDS`) so the spoken result is
    captured in the same `CapturedUtterance`, mirroring the proven CSEE
    tool-continuation loop.

    Returns:
      A `CapturedUtterance` for this participant's reply. `tool_calls` /
      `tool_responses` and a non-empty transcript/audio indicate a real reply;
      an empty utterance indicates the turn produced nothing (e.g. the session
      ended).
    """
    audio_chunks: list[bytes] = []
    # Incremental (partial) transcription fragments, used only as a fallback if
    # the model never emits a final flushed transcription.
    partial_transcript_parts: list[str] = []
    # The final, fully-accumulated transcription text the model flushes on
    # turn_complete/generation_complete. This is the source of truth.
    final_transcript: Optional[str] = None
    tool_calls: list[types.FunctionCall] = []
    tool_responses: list[types.FunctionResponse] = []
    mime_type = "audio/pcm;rate=24000"
    start_time = time.time()
    # Wall-clock of the first audio chunk: the moment this speaker actually
    # started responding. This (not start_time) is the correct latency anchor.
    first_audio_time: Optional[float] = None
    # Number of tool-only turn completions we have continued past so far.
    tool_continuations = 0

    deadline = start_time + timeout_seconds
    while True:
      remaining = deadline - time.time()
      if remaining <= 0:
        logger.warning("Timed out collecting %s turn.", self._speaker)
        break
      # Manual activity markers make the model emit a reliable turn_complete, so
      # we drain until that signal (or the absolute deadline) rather than using
      # an idle-timeout heuristic that can race delayed audio.
      try:
        event = await asyncio.wait_for(
            self._event_queue.get(), timeout=remaining
        )
      except asyncio.TimeoutError:
        logger.warning("Timed out collecting %s turn.", self._speaker)
        break

      logger.debug(
          "[%s] event author=%s partial=%s turn_complete=%s has_content=%s"
          " out_tx=%s in_tx=%s parts=%s",
          self._speaker,
          event.author,
          event.partial,
          event.turn_complete,
          bool(event.content and event.content.parts),
          bool(event.output_transcription and event.output_transcription.text),
          bool(event.input_transcription and event.input_transcription.text),
          [
              (
                  "audio"
                  if (p.inline_data and p.inline_data.data)
                  else "text"
                  if p.text
                  else "other"
              )
              for p in event.content.parts or []
          ]
          if event.content
          else [],
      )

      # Output audio lives on inline_data parts.
      if event.content and event.content.parts:
        for part in event.content.parts:
          if (
              part.inline_data
              and part.inline_data.data
              and (part.inline_data.mime_type or "").startswith("audio/")
          ):
            if first_audio_time is None:
              first_audio_time = time.time()
            audio_chunks.append(part.inline_data.data)
            mime_type = part.inline_data.mime_type or mime_type

      # ADK emits incremental partial transcriptions while the model speaks, then
      # a single final (partial=False) transcription carrying the full
      # accumulated text when the turn completes. Prefer the final flush; keep
      # the partials only as a fallback in case it is never sent.
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
        # A tool-only completion: the spoken result is coming in a follow-up
        # turn. Keep collecting so it lands in this same utterance.
        if (
            not has_speech
            and (tool_calls or tool_responses)
            and tool_continuations < _MAX_TOOL_CONTINUATION_ROUNDS
        ):
          tool_continuations += 1
          logger.debug(
              "[%s] tool-only turn_complete; continuing for spoken result"
              " (round %d).",
              self._speaker,
              tool_continuations,
          )
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
    )

  @property
  def is_finished(self) -> bool:
    """Whether this participant's live session has fully closed."""
    return self._live_finished.is_set()


@experimental
class PersonaLiveConversationRunner:
  """Orchestrates an audio-to-audio conversation between a persona and a SUT."""

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
    self._sut_agent = sut_agent
    self._app_name = app_name
    self._user_id = user_id
    self._session_service = session_service or InMemorySessionService()
    self._artifact_service = artifact_service or InMemoryArtifactService()
    self._persona_agent_factory = (
        persona_agent_factory or PersonaCustomerAgentFactory()
    )
    self._session_id: Optional[str] = None

  async def run(
      self,
      scenario: LiveConversationScenario,
      *,
      session_id: Optional[str] = None,
      progress_callback: Optional[ProgressCallback] = None,
  ) -> LiveConversation:
    """Runs the persona<->SUT audio conversation and returns captured turns.

    Args:
      scenario: The persona + config describing the conversation to run.
      session_id: Optional id for the SUT session (one is generated if unset).
      progress_callback: Optional async callback invoked with progress events
        (turn_started, transcript_update, turn_complete, conversation_complete)
        for live observation.

    Returns:
      The captured `LiveConversation`.
    """
    persona = scenario.persona
    persona_agent = self._persona_agent_factory.build(
        persona, model=persona.model or self._sut_model_name()
    )
    realism = build_audio_realism_transform(scenario.audio_realism)

    conversation = LiveConversation()

    sut_session = await self._session_service.create_session(
        app_name=self._app_name, user_id=self._user_id, session_id=session_id
    )
    self._session_id = sut_session.id
    conversation.session_id = sut_session.id
    # Capture the SUT agent tree's instructions + tool declarations so managed
    # metrics (trajectory/tool-use/task success) can use the agent configuration.
    conversation.app_details = await _build_app_details(
        self._sut_agent,
        session_service=self._session_service,
        session=sut_session,
    )
    persona_session = await self._session_service.create_session(
        app_name=f"persona_{persona.id}", user_id=self._user_id
    )

    sut_runner = Runner(
        app_name=self._app_name,
        agent=self._sut_agent,
        session_service=self._session_service,
        artifact_service=self._artifact_service,
    )
    persona_runner = Runner(
        app_name=f"persona_{persona.id}",
        agent=persona_agent,
        session_service=self._session_service,
        artifact_service=self._artifact_service,
    )

    async with (
        _LiveAgentSession(
            runner=sut_runner, session=sut_session, speaker="sut"
        ) as sut,
        _LiveAgentSession(
            runner=persona_runner, session=persona_session, speaker="persona"
        ) as persona_live,
    ):
      conversation.termination_reason = await self._drive(
          scenario=scenario,
          sut=sut,
          persona_live=persona_live,
          realism=realism,
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
      scenario: LiveConversationScenario,
      sut: _LiveAgentSession,
      persona_live: _LiveAgentSession,
      realism,
      conversation: LiveConversation,
      progress_callback: Optional[ProgressCallback],
  ) -> str:
    """Runs the alternating turn loop; returns the termination reason."""
    # The persona opens the conversation (it has the goal/intent). A Live agent
    # only speaks in response to input, so we nudge it with a short text prompt
    # to begin; it then speaks its opening line based on its persona/goal.
    persona_live.send_text(
        "Begin the conversation now. Speak your opening line out loud."
    )
    persona_utterance = await persona_live.collect_turn()
    termination_reason = "max_turns"
    consecutive_silent_turns = 0

    for turn_index in range(scenario.max_turns):
      if progress_callback:
        await progress_callback({
            "type": "turn_started",
            "turn_index": turn_index,
        })

      if not persona_utterance.audio_pcm:
        termination_reason = "completed"
        break

      # Optional realism degradation + barge-in shaping of persona audio.
      persona_audio = await realism.apply(
          persona_utterance.audio_pcm, mime_type=persona_utterance.mime_type
      )
      persona_utterance.was_interrupted = self._maybe_barge_in(scenario)

      await self._persist_and_record(
          conversation=conversation,
          turn_index=turn_index,
          persona_utterance=persona_utterance,
          sut_utterance=None,
          progress_callback=progress_callback,
      )

      # Note: we intentionally do NOT prime the SUT with a text message here.
      # ADK's send_content sends with turn_complete=True, which would make the
      # model respond to the prime text immediately (before the audio), so the
      # prime would be consumed as a spurious turn. The manual activity markers
      # in send_audio (activity_start -> audio -> activity_end) are what
      # reliably trigger a response, so priming is unnecessary.

      # SUT hears persona audio and replies. The persona's output audio is at
      # the Live output rate (24 kHz); resample to 16 kHz for Live input.
      sut.send_audio(
          to_live_input(
              persona_audio, source_mime_type=persona_utterance.mime_type
          ),
          mime_type=LIVE_INPUT_MIME_TYPE,
      )
      # Latency baseline: the moment the persona's audio finished being sent to
      # the SUT. The SUT's response latency is measured from here to its first
      # audio chunk.
      input_sent_time = time.time()
      sut_utterance = await sut.collect_turn()
      sut_utterance.input_sent_time = input_sent_time

      # Latency breakdown for observability: the measured latency is
      # first_audio_time - input_sent_time. Log the components (and whether the
      # turn involved tool calls, which add tool-execution time before the first
      # spoken audio) so high latencies are auditable rather than a black box.
      first_audio_time = sut_utterance.first_audio_time
      had_tool_calls = bool(
          sut_utterance.tool_calls or sut_utterance.tool_responses
      )
      measured_latency = (
          first_audio_time - input_sent_time
          if first_audio_time is not None
          else None
      )
      logger.info(
          "Turn %d latency: %s (input_sent_time=%.3f, first_audio_time=%s,"
          " had_tool_calls=%s)",
          turn_index,
          (
              f"{measured_latency:.3f}s"
              if measured_latency is not None
              else "n/a (no SUT audio)"
          ),
          input_sent_time,
          (
              f"{first_audio_time:.3f}"
              if first_audio_time is not None
              else "None"
          ),
          had_tool_calls,
      )

      await self._persist_and_record(
          conversation=conversation,
          turn_index=turn_index,
          persona_utterance=None,
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

      logger.info(
          "Turn %d complete. persona=%r sut=%r (sut_finished=%s,"
          " persona_finished=%s)",
          turn_index,
          persona_utterance.transcript[:60],
          sut_utterance.transcript[:60],
          sut.is_finished,
          persona_live.is_finished,
      )

      # If a live session truly closed, stop.
      if sut.is_finished or persona_live.is_finished:
        termination_reason = "completed"
        break

      # Tolerate the occasional silent SUT turn (the model sometimes does not
      # respond immediately). Only give up after several consecutive silent
      # turns; otherwise nudge the persona to continue.
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

      # Persona hears SUT audio and replies for the next turn. Resample the
      # SUT's output audio (24 kHz) to 16 kHz for Live input.
      persona_live.send_audio(
          to_live_input(
              sut_utterance.audio_pcm,
              source_mime_type=sut_utterance.mime_type,
          ),
          mime_type=LIVE_INPUT_MIME_TYPE,
      )
      persona_utterance = await persona_live.collect_turn()

    return termination_reason

  def _maybe_barge_in(self, scenario: LiveConversationScenario) -> bool:
    barge_in = scenario.barge_in
    if not barge_in or not barge_in.enabled:
      return False
    return random.random() < barge_in.probability

  async def _persist_and_record(
      self,
      *,
      conversation: LiveConversation,
      turn_index: int,
      persona_utterance: Optional[CapturedUtterance],
      sut_utterance: Optional[CapturedUtterance],
      progress_callback: Optional[ProgressCallback],
  ) -> None:
    """Persists audio for an utterance and records it on the turn."""
    turn = self._get_or_create_turn(conversation, turn_index)

    if persona_utterance is not None:
      persona_utterance.audio_reference = await self._persist_utterance_audio(
          persona_utterance, turn_index
      )
      turn.persona_utterance = persona_utterance
      if progress_callback and persona_utterance.transcript:
        await progress_callback({
            "type": "transcript_update",
            "speaker": "persona",
            "turn_index": turn_index,
            "text": persona_utterance.transcript,
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
    # Use a flat filename (no directory prefix): the web UI references audio
    # artifacts by their bare name, and artifacts are already namespaced by
    # session, so a prefix only causes lookup mismatches.
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
    )

  def _sut_model_name(self) -> Optional[str]:
    model = getattr(self._sut_agent, "model", None)
    return getattr(model, "model", None) if model is not None else None
