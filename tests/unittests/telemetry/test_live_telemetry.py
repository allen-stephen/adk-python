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

"""Tests for voice/live (speech-to-speech) telemetry.

Verifies that a live session produces a per-turn span tree with voice-specific
signals: time-to-first-token, transcripts, audio-token breakdowns, and audio
references, and that content redaction is honored.
"""

from __future__ import annotations

import json
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry._token_usage import TokenUsage
from google.adk.telemetry.context import ContentCapturingMode
from google.adk.telemetry.context import TelemetryConfig
from google.adk.telemetry.live_turn_tracing import LiveTurnTracer
from google.genai import types
import pytest

from .. import testing_utils
from ..flows.llm_flows.test_base_llm_flow import BaseLlmFlowForTesting


class _StopReceiveLoop(Exception):
  """Sentinel used to break the infinite `_receive_from_model` loop."""


# --- In-memory tracing harness --------------------------------------------


def _make_in_memory_tracer():
  """Wires an InMemorySpanExporter to a real TracerProvider."""
  from opentelemetry.sdk import trace as trace_sdk
  from opentelemetry.sdk.trace import export as trace_export
  from opentelemetry.sdk.trace.export import in_memory_span_exporter

  provider = trace_sdk.TracerProvider()
  exporter = in_memory_span_exporter.InMemorySpanExporter()
  provider.add_span_processor(trace_export.SimpleSpanProcessor(exporter))
  return provider.get_tracer('test_tracer'), exporter, provider


async def _drain_receive(
    flow, mock_connection, invocation_context, llm_request
):
  """Drives `_receive_from_model` to completion under a patched tracer.

  Patches both the `base_llm_flow` and `live_turn_tracing` module tracers so the
  `call_llm` and `live_turn` spans share one provider and nest correctly.

  Returns (events, exporter).
  """
  tracer, exporter, provider = _make_in_memory_tracer()
  events = []
  try:
    with (
        mock.patch('google.adk.flows.llm_flows.base_llm_flow.tracer', tracer),
        mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer),
    ):
      turn_tracer = LiveTurnTracer(invocation_context)
      try:
        async for event in flow._receive_from_model(
            mock_connection,
            'seed_event_id',
            invocation_context,
            llm_request,
            turn_tracer,
        ):
          events.append(event)
      except _StopReceiveLoop:
        pass
      finally:
        turn_tracer.close()
  finally:
    provider.shutdown()
  return events, exporter


def _spans_named(exporter, name):
  return [s for s in exporter.get_finished_spans() if s.name == name]


def _mock_connection(responses):
  async def mock_receive():
    for response in responses:
      yield response
    raise _StopReceiveLoop()

  conn = mock.AsyncMock()
  conn.receive = mock.Mock(side_effect=mock_receive)
  return conn


@pytest.fixture(autouse=True)
def _semconv_aligned_schema(monkeypatch):
  """Forces the semconv-aligned schema, under which the span tree is emitted.

  The live-turn span tree is gated on the OTel-semconv-aligned telemetry schema
  (the default on Agent Engine). Local test runs default to the legacy schema,
  which is exercised explicitly by `test_legacy_schema_emits_no_live_spans`.
  """
  monkeypatch.setenv('ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN', '2')


def _transcript_text(messages_json):
  """Extracts the transcript text from a `gen_ai.*.messages` JSON attribute."""
  messages = json.loads(messages_json)
  return messages[0]['parts'][0]['content']


# --- Token breakdown (Phase 1a) -------------------------------------------


def test_audio_token_breakdown_emitted_for_live_usage():
  """Per-modality audio/text token counts appear as span attributes."""
  usage = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=100,
      candidates_token_count=40,
      total_token_count=140,
      prompt_tokens_details=[
          types.ModalityTokenCount(
              modality=types.MediaModality.AUDIO, token_count=90
          ),
          types.ModalityTokenCount(
              modality=types.MediaModality.TEXT, token_count=10
          ),
      ],
      candidates_tokens_details=[
          types.ModalityTokenCount(
              modality=types.MediaModality.AUDIO, token_count=35
          ),
          types.ModalityTokenCount(
              modality=types.MediaModality.TEXT, token_count=5
          ),
      ],
  )

  attrs = TokenUsage(usage).to_attributes()

  assert attrs['gen_ai.usage.experimental.input_audio_tokens'] == 90
  assert attrs['gen_ai.usage.experimental.input_text_tokens'] == 10
  assert attrs['gen_ai.usage.experimental.output_audio_tokens'] == 35
  assert attrs['gen_ai.usage.experimental.output_text_tokens'] == 5
  # Modality counts are a breakdown of, not additive to, the totals.
  assert attrs['gen_ai.usage.input_tokens'] == 100
  assert attrs['gen_ai.usage.output_tokens'] == 40


def test_video_token_breakdown_emitted_for_live_usage():
  """Per-modality video token counts appear alongside audio/text."""
  usage = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=100,
      candidates_token_count=40,
      total_token_count=140,
      prompt_tokens_details=[
          types.ModalityTokenCount(
              modality=types.MediaModality.VIDEO, token_count=60
          ),
          types.ModalityTokenCount(
              modality=types.MediaModality.TEXT, token_count=40
          ),
      ],
      candidates_tokens_details=[
          types.ModalityTokenCount(
              modality=types.MediaModality.VIDEO, token_count=25
          ),
      ],
  )

  attrs = TokenUsage(usage).to_attributes()

  assert attrs['gen_ai.usage.experimental.input_video_tokens'] == 60
  assert attrs['gen_ai.usage.experimental.output_video_tokens'] == 25
  # Video counts are a breakdown of, not additive to, the totals.
  assert attrs['gen_ai.usage.input_tokens'] == 100
  assert attrs['gen_ai.usage.output_tokens'] == 40


def test_audio_token_breakdown_absent_without_details():
  """No per-modality attributes when the response has no breakdown."""
  usage = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=10, candidates_token_count=5, total_token_count=15
  )

  attrs = TokenUsage(usage).to_attributes()

  assert 'gen_ai.usage.experimental.input_audio_tokens' not in attrs
  assert 'gen_ai.usage.experimental.output_audio_tokens' not in attrs


# --- Helpers for building live responses ----------------------------------


def _audio_chunk(data=b'\x00\xff'):
  """A streamed model audio chunk (partial=False, as the live API emits)."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  inline_data=types.Blob(data=data, mime_type='audio/pcm')
              )
          ],
      )
  )


def _usage_only(input_tokens=10, output_tokens=5):
  """A trailing usage-only response, as the live API sends after a turn."""
  return LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=input_tokens,
          candidates_token_count=output_tokens,
          total_token_count=input_tokens + output_tokens,
      )
  )


def _function_call(name='roll_die'):
  """A model response carrying a function call (the tool-call generation)."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_function_call(name=name, args={})],
      )
  )


# --- Integration through _receive_from_model ------------------------------


@pytest.mark.asyncio
async def test_many_audio_chunks_produce_one_assistant_span():
  """A turn of many audio chunks aggregates into a single assistant span.

  This is the core regression guard: the live API streams audio chunks with
  partial=False, so a naive per-response span would emit one span per chunk.
  """
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  responses = [_audio_chunk() for _ in range(50)]
  responses.append(LlmResponse(turn_complete=True))
  responses.append(_usage_only())
  conn = _mock_connection(responses)

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assert len(_spans_named(exporter, 'live_turn')) == 1
  assert len(_spans_named(exporter, 'assistant')) == 1
  # The obsolete per-response span name must not appear on the live path.
  assert not _spans_named(exporter, 'call_llm')


@pytest.mark.asyncio
async def test_assistant_span_nests_under_turn_span():
  """The assistant span is a child of the live_turn span."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([_audio_chunk(), LlmResponse(turn_complete=True)])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  turn = _spans_named(exporter, 'live_turn')[0]
  assistant = _spans_named(exporter, 'assistant')[0]
  assert assistant.parent is not None
  assert assistant.parent.span_id == turn.context.span_id


@pytest.mark.asyncio
async def test_transcripts_split_across_user_and_assistant_spans():
  """Input transcript lands on the user span, output on the assistant span."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.SPAN_AND_EVENT
          )
      ),
  )
  flow = BaseLlmFlowForTesting()

  input_tx = LlmResponse(
      input_transcription=types.Transcription(
          text='what is the weather', finished=True
      )
  )
  output_tx = LlmResponse(
      output_transcription=types.Transcription(
          text='it is sunny', finished=True
      )
  )
  conn = _mock_connection(
      [input_tx, output_tx, LlmResponse(turn_complete=True)]
  )

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  user = _spans_named(exporter, 'user')[0]
  assistant = _spans_named(exporter, 'assistant')[0]
  assert (
      _transcript_text(user.attributes['gen_ai.input.messages'])
      == 'what is the weather'
  )
  assert (
      _transcript_text(assistant.attributes['gen_ai.output.messages'])
      == 'it is sunny'
  )


@pytest.mark.asyncio
async def test_usage_after_turn_complete_lands_on_assistant_span():
  """Token usage from a trailing usage-only response lands on the assistant."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([
      _audio_chunk(),
      LlmResponse(turn_complete=True),
      _usage_only(input_tokens=120, output_tokens=45),
  ])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assistant = _spans_named(exporter, 'assistant')[0]
  assert assistant.attributes['gen_ai.usage.input_tokens'] == 120
  assert assistant.attributes['gen_ai.usage.output_tokens'] == 45


@pytest.mark.asyncio
async def test_assistant_span_carries_event_id_for_correlation():
  """The assistant span is stamped with the streamed event id + invocation id.

  Tooling (e.g. the ADK web inspector) associates a span with the selected
  event via `gcp.vertex.agent.event_id`, mirroring the non-live `call_llm` path.
  """
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  output_tx = LlmResponse(
      output_transcription=types.Transcription(text='hello', finished=True)
  )
  conn = _mock_connection([output_tx, LlmResponse(turn_complete=True)])

  events, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assistant = _spans_named(exporter, 'assistant')[0]
  # The stamped event id must match an event actually streamed to the client.
  streamed_ids = {e.id for e in events}
  assert 'gcp.vertex.agent.event_id' in assistant.attributes
  assert assistant.attributes['gcp.vertex.agent.event_id'] in streamed_ids
  assert (
      assistant.attributes['gcp.vertex.agent.invocation_id']
      == ctx.invocation_id
  )


@pytest.mark.asyncio
async def test_ttft_recorded_on_assistant_span_after_user_audio():
  """TTFT is recorded on the assistant span when user audio precedes output."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([_audio_chunk(), LlmResponse(turn_complete=True)])

  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with (
        mock.patch('google.adk.flows.llm_flows.base_llm_flow.tracer', tracer),
        mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer),
    ):
      turn_tracer = LiveTurnTracer(ctx)
      # Simulate user speaking before the model responds.
      turn_tracer.on_user_audio()
      try:
        async for _ in flow._receive_from_model(
            conn, 'seed_event_id', ctx, LlmRequest(), turn_tracer
        ):
          pass
      except _StopReceiveLoop:
        pass
      turn_tracer.close()
  finally:
    provider.shutdown()

  assistant = _spans_named(exporter, 'assistant')[0]
  assert 'gen_ai.response.time_to_first_chunk' in assistant.attributes


@pytest.mark.asyncio
async def test_assistant_span_marks_duration_as_generation():
  """The assistant span labels its duration as server-side generation time.

  The backend cannot observe client-side audio playback, so the span duration
  covers generation only. The `assistant_duration_kind` attribute makes this
  explicit so consumers surface time-to-first-chunk as the headline latency and
  do not misread the duration as end-to-end perceived latency.
  """
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  def drive(t):
    t.on_user_audio()
    t.on_model_output(_audio_chunk())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    t.on_usage(_usage_only())

  exporter = _run_tracer(ctx, drive)

  assistant = _spans_named(exporter, 'assistant')[0]
  assert (
      assistant.attributes['gcp.vertex.agent.live.assistant_duration_kind']
      == 'generation'
  )


@pytest.mark.asyncio
async def test_transcripts_redacted_when_content_capture_disabled():
  """Transcripts are dropped when content capture is off."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.NO_CONTENT
          )
      ),
  )
  flow = BaseLlmFlowForTesting()

  input_tx = LlmResponse(
      input_transcription=types.Transcription(
          text='secret question', finished=True
      )
  )
  conn = _mock_connection([input_tx, LlmResponse(turn_complete=True)])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  user = _spans_named(exporter, 'user')[0]
  assert 'gen_ai.input.messages' not in user.attributes


@pytest.mark.asyncio
async def test_partial_transcript_does_not_land_on_span():
  """Only finished transcripts are attached; partial deltas are skipped."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  partial_tx = LlmResponse(
      input_transcription=types.Transcription(text='what is', finished=False),
      partial=True,
  )
  conn = _mock_connection([partial_tx, LlmResponse(turn_complete=True)])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  for span in exporter.get_finished_spans():
    assert 'gen_ai.input.messages' not in span.attributes


@pytest.mark.asyncio
async def test_standalone_turn_complete_opens_no_empty_turn_span():
  """A pure turn_complete with no output does not create an empty turn span."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([LlmResponse(turn_complete=True)])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assert not _spans_named(exporter, 'live_turn')


@pytest.mark.asyncio
async def test_output_audio_reference_lands_on_assistant_span():
  """A flushed output audio reference is attached to the assistant span."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  audio_ref_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  file_data=types.FileData(
                      file_uri=(
                          'artifact://app/user/session/_adk_live/out.pcm#1'
                      ),
                      mime_type='audio/pcm',
                  )
              )
          ],
      )
  )
  conn = _mock_connection([audio_ref_response, LlmResponse(turn_complete=True)])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assistant = _spans_named(exporter, 'assistant')[0]
  assert (
      assistant.attributes['gen_ai.output.experimental.audio_ref']
      == 'artifact://app/user/session/_adk_live/out.pcm#1'
  )


@pytest.mark.asyncio
async def test_legacy_schema_emits_no_live_spans(monkeypatch):
  """Under the legacy schema the live-turn tree is not emitted (no-op tracer)."""
  monkeypatch.setenv('ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN', '1')
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  responses = [_audio_chunk() for _ in range(5)]
  responses.append(LlmResponse(turn_complete=True))
  responses.append(_usage_only())
  conn = _mock_connection(responses)

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assert not _spans_named(exporter, 'live_turn')
  assert not _spans_named(exporter, 'user')
  assert not _spans_named(exporter, 'assistant')


def test_run_live_opens_root_invocation_span():
  """`run_live` wraps the session in the top-level invocation span.

  Setup: a runner with a mock live model that completes one turn.
  Act: consume run_live to completion.
  Assert: `record_invocation` was entered for the live session, matching the
    non-live path.
  """
  from google.adk.runners import _instrumentation

  response = LlmResponse(turn_complete=True)
  mock_model = testing_utils.MockModel.create([response])
  root_agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )
  live_request_queue = testing_utils.LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xff', mime_type='audio/pcm')
  )

  with mock.patch.object(
      _instrumentation,
      'record_invocation',
      wraps=_instrumentation.record_invocation,
  ) as spy:
    runner.run_live(live_request_queue)

  spy.assert_called()


@pytest.mark.asyncio
async def test_two_turns_produce_two_turn_spans():
  """Each turn boundary finalizes a turn and the next begins a fresh one."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([
      _audio_chunk(),
      LlmResponse(turn_complete=True),
      _usage_only(),
      _audio_chunk(),
      LlmResponse(turn_complete=True),
      _usage_only(),
  ])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  assert len(_spans_named(exporter, 'live_turn')) == 2
  assert len(_spans_named(exporter, 'assistant')) == 2


# --- Span duration semantics ----------------------------------------------


class _FakeClock:
  """A controllable clock for both time.time_ns and time.monotonic.

  Advancing the clock lets tests assert span durations deterministically. Values
  are in seconds; time_ns() converts to integer nanoseconds.
  """

  def __init__(self):
    self.now = 1_000.0

  def advance(self, seconds):
    self.now += seconds

  def monotonic(self):
    return self.now

  def time_ns(self):
    return int(self.now * 1e9)


@pytest.mark.asyncio
async def test_user_span_duration_matches_utterance_length():
  """The user span spans from first user audio to first model output.

  Setup: user speaks for 13s, then the model starts responding 0.4s later and
    responds for 3s.
  Assert:
    - user span duration ~ 13.4s (utterance + time-to-first-token window).
    - live_turn starts at the same time as the user span (first user audio).
    - assistant span starts after the user span ends.
  """
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  clock = _FakeClock()
  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with (
        mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer),
        mock.patch(
            'google.adk.telemetry.live_turn_tracing.time.monotonic',
            clock.monotonic,
        ),
        mock.patch(
            'google.adk.telemetry.live_turn_tracing.time.time_ns',
            clock.time_ns,
        ),
    ):
      turn_tracer = LiveTurnTracer(ctx)

      # User starts speaking, opening the user span at t=0.
      turn_tracer.on_user_audio()
      turn_tracer.on_input_transcript('a thirteen second question')

      # 13.4s later the model produces its first output (closes the user span).
      clock.advance(13.4)
      turn_tracer.on_model_output(LlmResponse(turn_complete=False))

      # Model responds for 3s, then the turn finalizes.
      clock.advance(3.0)
      turn_tracer.on_turn_boundary(LlmResponse(turn_complete=True))
      turn_tracer.on_usage(_usage_only())
  finally:
    provider.shutdown()

  user = _spans_named(exporter, 'user')[0]
  assistant = _spans_named(exporter, 'assistant')[0]
  live_turn = _spans_named(exporter, 'live_turn')[0]

  user_duration_s = (user.end_time - user.start_time) / 1e9
  assert user_duration_s == pytest.approx(13.4, abs=0.05)
  # The turn starts when the user starts speaking.
  assert live_turn.start_time == user.start_time
  # The assistant span begins only after the user finishes.
  assert assistant.start_time >= user.end_time


@pytest.mark.asyncio
async def test_assistant_span_duration_matches_response_length():
  """The assistant span spans from first model output to turn finalize."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  clock = _FakeClock()
  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with (
        mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer),
        mock.patch(
            'google.adk.telemetry.live_turn_tracing.time.monotonic',
            clock.monotonic,
        ),
        mock.patch(
            'google.adk.telemetry.live_turn_tracing.time.time_ns',
            clock.time_ns,
        ),
    ):
      turn_tracer = LiveTurnTracer(ctx)
      turn_tracer.on_user_audio()
      clock.advance(2.0)
      turn_tracer.on_model_output(LlmResponse(turn_complete=False))
      clock.advance(4.5)
      turn_tracer.on_turn_boundary(LlmResponse(turn_complete=True))
      turn_tracer.on_usage(_usage_only())
  finally:
    provider.shutdown()

  assistant = _spans_named(exporter, 'assistant')[0]
  response_duration_s = (assistant.end_time - assistant.start_time) / 1e9
  assert response_duration_s == pytest.approx(4.5, abs=0.05)


@pytest.mark.asyncio
async def test_late_input_transcript_still_opens_user_span():
  """An input transcript arriving after first model output opens a user span.

  Transcription can settle slower than the first model output. The user turn
  must still get a `user` span (not have the transcript merged into the turn
  span), so trace consumers see a consistent user/assistant tree regardless of
  transcription timing.
  """
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.SPAN_AND_EVENT
          )
      ),
  )
  flow = BaseLlmFlowForTesting()

  # Model output arrives before the finished input transcription.
  conn = _mock_connection([
      _audio_chunk(),
      LlmResponse(
          input_transcription=types.Transcription(
              text='late question', finished=True
          )
      ),
      LlmResponse(turn_complete=True),
  ])

  _, exporter = await _drain_receive(flow, conn, ctx, LlmRequest())

  # The late transcript lands on a user span, not the live_turn span.
  user_spans = _spans_named(exporter, 'user')
  assert len(user_spans) == 1
  assert (
      _transcript_text(user_spans[0].attributes['gen_ai.input.messages'])
      == 'late question'
  )
  live_turn = _spans_named(exporter, 'live_turn')[0]
  assert 'gen_ai.input.messages' not in live_turn.attributes
  # The user span nests under the live_turn.
  assert user_spans[0].parent is not None
  assert user_spans[0].parent.span_id == live_turn.context.span_id


# --- Tool round-trips within a single turn --------------------------------


def _run_tracer(ctx, drive):
  """Runs `drive(turn_tracer)` under a patched tracer, returning the exporter.

  `drive` receives a LiveTurnTracer and simulates the sequence of tracer calls
  the flow makes for a turn (mirroring `_receive_from_model` + the `run_live`
  function-response send).
  """
  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer):
      turn_tracer = LiveTurnTracer(ctx)
      drive(turn_tracer)
      turn_tracer.close()
  finally:
    provider.shutdown()
  return exporter


@pytest.mark.asyncio
async def test_tool_call_turn_stays_in_one_live_turn():
  """A tool call and its follow-up answer share a single live_turn.

  Setup: user speaks, model calls a tool (turn_complete), tool response is sent
    back, model speaks the answer (turn_complete), then usage arrives.
  Assert:
    - exactly one live_turn span,
    - two assistant spans (tool-call generation + answer), both under it,
    - no second live_turn for the follow-up answer.
  """
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  def drive(t):
    t.on_user_audio()
    # First generation: model emits a function call, then turn_complete. The
    # function call in the output marks the following turn_complete as a handoff.
    t.on_model_output(_function_call())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    # Second generation: the spoken answer, then the real turn_complete.
    t.on_model_output(_audio_chunk())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    t.on_usage(_usage_only())

  exporter = _run_tracer(ctx, drive)

  live_turns = _spans_named(exporter, 'live_turn')
  assistants = _spans_named(exporter, 'assistant')
  assert len(live_turns) == 1
  assert len(assistants) == 2
  for assistant in assistants:
    assert assistant.parent is not None
    assert assistant.parent.span_id == live_turns[0].context.span_id


@pytest.mark.asyncio
async def test_tool_call_turn_sums_usage_across_generations():
  """Token usage is summed across the tool-using turn's two generations."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  def drive(t):
    t.on_user_audio()
    t.on_model_output(_function_call())
    # First generation usage on its (handoff) turn_complete.
    tc1 = _usage_only(input_tokens=30, output_tokens=10)
    tc1.turn_complete = True
    t.on_turn_boundary(tc1)
    t.on_model_output(_audio_chunk())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    # Second generation usage.
    t.on_usage(_usage_only(input_tokens=50, output_tokens=25))

  exporter = _run_tracer(ctx, drive)

  # The finalized (answer) assistant span carries the combined total.
  finalized = [
      s
      for s in _spans_named(exporter, 'assistant')
      if 'gen_ai.usage.input_tokens' in s.attributes
  ]
  assert len(finalized) == 1
  assert finalized[0].attributes['gen_ai.usage.input_tokens'] == 80
  assert finalized[0].attributes['gen_ai.usage.output_tokens'] == 35


@pytest.mark.asyncio
async def test_interrupt_during_tool_round_trip_finalizes_turn():
  """A barge-in interrupt ends the turn even while a tool round-trip pends."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  def drive(t):
    t.on_user_audio()
    t.on_model_output(_function_call())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    # User barges in before the follow-up answer.
    t.on_turn_boundary(LlmResponse(interrupted=True))
    # A new turn begins.
    t.on_user_audio()
    t.on_model_output(_audio_chunk())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    t.on_usage(_usage_only())

  exporter = _run_tracer(ctx, drive)

  # The interrupted tool turn finalizes on its own, so the new turn is separate.
  assert len(_spans_named(exporter, 'live_turn')) == 2


@pytest.mark.asyncio
async def test_sequential_tool_calls_stay_in_one_live_turn():
  """Two back-to-back tool calls plus the answer share one live_turn."""
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  def drive(t):
    t.on_user_audio()
    # Tool call 1.
    t.on_model_output(_function_call())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    # Tool call 2.
    t.on_model_output(_function_call())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    # Final spoken answer.
    t.on_model_output(_audio_chunk())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    t.on_usage(_usage_only())

  exporter = _run_tracer(ctx, drive)

  assert len(_spans_named(exporter, 'live_turn')) == 1
  assert len(_spans_named(exporter, 'assistant')) == 3


def test_tool_call_turn_end_to_end_produces_one_live_turn():
  """Through the real run_live path, a tool turn yields a single live_turn.

  Drives the full `run_live` loop (including the function-response send back to
  the model), so the real call ordering — where the tool-call generation's
  turn_complete can be processed before the response send — is exercised. This
  is the direct regression guard for the separate-live_turn-after-tool-call bug.
  """
  import asyncio

  from google.adk.agents.run_config import RunConfig as _RunConfig
  from google.adk.utils.context_utils import Aclosing

  function_call = types.Part.from_function_call(
      name='roll_die', args={'sides': 6}
  )
  responses = [
      _audio_chunk(),
      # First generation: the model calls the tool, then turn_complete + usage.
      LlmResponse(content=types.Content(role='model', parts=[function_call])),
      LlmResponse(turn_complete=True),
      _usage_only(),
      # Second generation: the spoken answer, then the real turn_complete.
      _audio_chunk(),
      LlmResponse(turn_complete=True),
      _usage_only(),
  ]
  mock_model = testing_utils.MockModel.create(responses)

  def roll_die(sides: int) -> int:
    return 4

  root_agent = Agent(name='root_agent', model=mock_model, tools=[roll_die])
  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )
  live_request_queue = testing_utils.LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xff', mime_type='audio/pcm')
  )

  async def consume():
    run_res = runner.runner.run_live(
        session=runner.session,
        live_request_queue=live_request_queue,
        run_config=_RunConfig(response_modalities=['AUDIO']),
    )
    collected = []
    async with Aclosing(run_res) as agen:
      async for event in agen:
        collected.append(event)
        if len(collected) >= len(responses):
          break

  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer):
      try:
        asyncio.run(asyncio.wait_for(consume(), timeout=5.0))
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
  finally:
    provider.shutdown()

  # The follow-up answer must share the tool call's live_turn, not spawn a new
  # one (regression guard for the separate-live_turn-after-tool-call bug).
  assert len(_spans_named(exporter, 'live_turn')) == 1


# --- Multi-agent & workflow voice sessions --------------------------------
#
# A live session can fan out across agents: an LlmAgent can transfer to a
# sub-agent, a SequentialAgent/LoopAgent runs sub-agents in turn, and a
# workflow drives LlmAgent nodes. Each such agent/node runs its own
# `run_live`, which owns its own LiveTurnTracer. The tracer snapshots the OTel
# context active in `run_live` (the agent's `invoke_agent` span, or a workflow
# node span) so every `live_turn` nests under the agent/node that produced it,
# rather than floating to the invocation root. These tests verify that
# parenting contract at the tracer level (ParallelAgent live is unsupported by
# the runtime today, so concurrent live turns are out of scope).


def _drive_one_turn(t):
  """Simulates the tracer calls for a single simple (no-tool) turn."""
  t.on_user_audio()
  t.on_model_output(_audio_chunk())
  t.on_turn_boundary(LlmResponse(turn_complete=True))
  t.on_usage(_usage_only())


@pytest.mark.asyncio
async def test_live_turn_nests_under_explicit_parent_span():
  """A turn nests under the parent span supplied to the tracer."""
  from opentelemetry import trace as _trace

  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer):
      parent_span = tracer.start_span('invoke_agent test_agent')
      parent_context = _trace.set_span_in_context(parent_span)
      turn_tracer = LiveTurnTracer(ctx, parent_context=parent_context)
      _drive_one_turn(turn_tracer)
      turn_tracer.close()
      parent_span.end()
  finally:
    provider.shutdown()

  parent = [
      s
      for s in exporter.get_finished_spans()
      if s.name == 'invoke_agent test_agent'
  ][0]
  live_turn = _spans_named(exporter, 'live_turn')[0]
  assert live_turn.parent is not None
  assert live_turn.parent.span_id == parent.context.span_id
  # user/assistant spans stay nested under the turn.
  for child in _spans_named(exporter, 'user') + _spans_named(
      exporter, 'assistant'
  ):
    assert child.parent.span_id == live_turn.context.span_id


@pytest.mark.asyncio
async def test_live_turn_stamps_agent_name():
  """The producing agent's name is stamped on the live_turn span."""
  agent = Agent(name='billing_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  exporter = _run_tracer(ctx, _drive_one_turn)

  live_turn = _spans_named(exporter, 'live_turn')[0]
  assert live_turn.attributes['gen_ai.agent.name'] == 'billing_agent'


@pytest.mark.asyncio
async def test_multi_agent_turns_nest_under_their_own_agent_spans():
  """Transfer/sequential handoff: each agent's turn nests under its own span.

  Mirrors how agent transfer and SequentialAgent/LoopAgent drive live: the
  first agent's `run_live` (with its `invoke_agent` span) finishes its turn and
  closes its tracer before the second agent's `run_live` begins under a
  separate `invoke_agent` span. The two agents' turns must land under different
  parents.
  """
  from opentelemetry import trace as _trace

  agent_a = Agent(name='agent_a')
  agent_b = Agent(name='agent_b')
  ctx_a = await testing_utils.create_invocation_context(agent=agent_a)
  ctx_b = await testing_utils.create_invocation_context(agent=agent_b)

  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer):
      # Agent A speaks a turn, then hands off (its tracer closes first).
      span_a = tracer.start_span('invoke_agent agent_a')
      tracer_a = LiveTurnTracer(
          ctx_a, parent_context=_trace.set_span_in_context(span_a)
      )
      _drive_one_turn(tracer_a)
      tracer_a.close()
      span_a.end()
      # Agent B takes over under its own span.
      span_b = tracer.start_span('invoke_agent agent_b')
      tracer_b = LiveTurnTracer(
          ctx_b, parent_context=_trace.set_span_in_context(span_b)
      )
      _drive_one_turn(tracer_b)
      tracer_b.close()
      span_b.end()
  finally:
    provider.shutdown()

  spans_by_name = {s.name: s for s in exporter.get_finished_spans()}
  live_turns = _spans_named(exporter, 'live_turn')
  assert len(live_turns) == 2
  by_agent = {lt.attributes['gen_ai.agent.name']: lt for lt in live_turns}
  assert (
      by_agent['agent_a'].parent.span_id
      == spans_by_name['invoke_agent agent_a'].context.span_id
  )
  assert (
      by_agent['agent_b'].parent.span_id
      == spans_by_name['invoke_agent agent_b'].context.span_id
  )


def test_run_live_nests_live_turn_under_invoke_agent_span():
  """Through the real run_live path, live_turn nests under invoke_agent.

  Patches both the invocation-level tracer (which opens `invoke_agent`) and the
  live-turn tracer so both spans export to one in-memory provider, then asserts
  the parenting captured from the OTel context in `run_live` is correct.
  """
  import asyncio

  from google.adk.agents.run_config import RunConfig as _RunConfig
  from google.adk.utils.context_utils import Aclosing

  responses = [
      _audio_chunk(),
      LlmResponse(turn_complete=True),
      _usage_only(),
  ]
  mock_model = testing_utils.MockModel.create(responses)
  root_agent = Agent(name='voice_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )
  live_request_queue = testing_utils.LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xff', mime_type='audio/pcm')
  )

  async def consume():
    run_res = runner.runner.run_live(
        session=runner.session,
        live_request_queue=live_request_queue,
        run_config=_RunConfig(response_modalities=['AUDIO']),
    )
    collected = []
    async with Aclosing(run_res) as agen:
      async for event in agen:
        collected.append(event)
        if len(collected) >= len(responses):
          break

  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with (
        mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer),
        mock.patch('google.adk.telemetry.tracing.tracer', tracer),
    ):
      try:
        asyncio.run(asyncio.wait_for(consume(), timeout=5.0))
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
  finally:
    provider.shutdown()

  invoke_agent = [
      s
      for s in exporter.get_finished_spans()
      if s.name == 'invoke_agent voice_agent'
  ]
  live_turns = _spans_named(exporter, 'live_turn')
  assert len(invoke_agent) == 1
  assert len(live_turns) == 1
  assert live_turns[0].parent is not None
  assert live_turns[0].parent.span_id == invoke_agent[0].context.span_id


@pytest.mark.asyncio
async def test_live_turn_without_parent_still_produces_turn():
  """With no explicit parent (single-agent default), a turn is still emitted."""
  agent = Agent(name='solo_agent')
  ctx = await testing_utils.create_invocation_context(agent=agent)

  # _run_tracer constructs LiveTurnTracer with no parent_context.
  exporter = _run_tracer(ctx, _drive_one_turn)

  assert len(_spans_named(exporter, 'live_turn')) == 1


# --- Offline-eval inference completion details -----------------------------
#
# Offline evaluation on the Gemini Enterprise Agent Platform reads the
# `gen_ai.client.inference.operation.details` event (not span attributes) for
# the required inference signals. These tests assert the live path emits that
# event, with the same shape as the non-live path.
# See docs/proposals/live-voice-telemetry.md and
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/optimize/evaluation/evaluate-offline

COMPLETION_DETAILS_EVENT = 'gen_ai.client.inference.operation.details'


def _event_telemetry(
    content=ContentCapturingMode.SPAN_AND_EVENT,
):
  """A TelemetryConfig that opts in to the inference completion-details event.

  The event is gated on the experimental GenAI semconv opt-in (matching the
  non-live path), so it is off by default; these tests opt in explicitly.
  """
  return TelemetryConfig(
      genai_semconv_stability_opt_in='experimental',
      capture_message_content=content,
  )


def _capture_completion_events(monkeypatch):
  """Captures `gen_ai.client.inference.operation.details` events.

  Patches the shared `otel_logger.emit` (the same object the live tracer holds
  a reference to) and returns a list that collects emitted LogRecords.
  """
  from google.adk.telemetry import tracing

  records = []
  monkeypatch.setattr(
      tracing.otel_logger, 'emit', lambda record: records.append(record)
  )
  return records


def _completion_events(records):
  return [r for r in records if r.event_name == COMPLETION_DETAILS_EVENT]


async def _drain_receive_with_request(
    flow, mock_connection, invocation_context, llm_request
):
  """Like `_drain_receive`, but threads `llm_request` into the turn tracer."""
  tracer, exporter, provider = _make_in_memory_tracer()
  try:
    with (
        mock.patch('google.adk.flows.llm_flows.base_llm_flow.tracer', tracer),
        mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer),
    ):
      turn_tracer = LiveTurnTracer(invocation_context, llm_request=llm_request)
      try:
        async for _ in flow._receive_from_model(
            mock_connection,
            'seed_event_id',
            invocation_context,
            llm_request,
            turn_tracer,
        ):
          pass
      except _StopReceiveLoop:
        pass
      finally:
        turn_tracer.close()
  finally:
    provider.shutdown()
  return exporter


def _first_message_text(messages):
  return messages[0]['parts'][0]['content']


@pytest.mark.asyncio
async def test_live_turn_emits_inference_completion_details_event(monkeypatch):
  """A live turn emits the offline-eval inference event with all four fields."""
  records = _capture_completion_events(monkeypatch)
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(telemetry=_event_telemetry()),
  )
  flow = BaseLlmFlowForTesting()

  llm_request = LlmRequest(
      config=types.GenerateContentConfig(
          system_instruction='you are a helpful weather agent',
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name='get_weather',
                          description='Look up the weather.',
                      )
                  ]
              )
          ],
      )
  )
  conn = _mock_connection([
      LlmResponse(
          input_transcription=types.Transcription(
              text='what is the weather', finished=True
          )
      ),
      LlmResponse(
          output_transcription=types.Transcription(
              text='it is sunny', finished=True
          )
      ),
      LlmResponse(turn_complete=True),
  ])

  await _drain_receive_with_request(flow, conn, ctx, llm_request)

  events = _completion_events(records)
  assert len(events) == 1
  attrs = events[0].attributes
  assert _first_message_text(attrs['gen_ai.input.messages']) == (
      'what is the weather'
  )
  assert _first_message_text(attrs['gen_ai.output.messages']) == 'it is sunny'
  # system_instructions is a flat list of parts (not messages).
  assert attrs['gen_ai.system_instructions'][0]['content'] == (
      'you are a helpful weather agent'
  )
  tool_defs = attrs['gen_ai.tool.definitions']
  assert tool_defs[0]['name'] == 'get_weather'
  # Common attributes correlate the event to the conversation/agent.
  assert attrs['gen_ai.agent.name'] == 'test_agent'
  assert attrs['gen_ai.conversation.id'] == ctx.session.id


@pytest.mark.asyncio
async def test_live_completion_details_stamped_on_assistant_span(monkeypatch):
  """The same details land on the assistant span (single source of truth)."""
  _capture_completion_events(monkeypatch)
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(telemetry=_event_telemetry()),
  )
  flow = BaseLlmFlowForTesting()

  llm_request = LlmRequest(
      config=types.GenerateContentConfig(
          system_instruction='sys instr',
      )
  )
  conn = _mock_connection([
      LlmResponse(
          output_transcription=types.Transcription(
              text='hello there', finished=True
          )
      ),
      LlmResponse(turn_complete=True),
  ])

  exporter = await _drain_receive_with_request(flow, conn, ctx, llm_request)

  assistant = _spans_named(exporter, 'assistant')[0]
  assert (
      _transcript_text(assistant.attributes['gen_ai.output.messages'])
      == 'hello there'
  )
  # On the span, attributes are JSON strings; system_instructions is a flat
  # list of parts.
  sys_instr = json.loads(assistant.attributes['gen_ai.system_instructions'])
  assert sys_instr[0]['content'] == 'sys instr'


@pytest.mark.asyncio
async def test_live_completion_details_redacted_when_content_disabled(
    monkeypatch,
):
  """Message content is dropped from the event when content capture is off."""
  records = _capture_completion_events(monkeypatch)
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(
          telemetry=_event_telemetry(content=ContentCapturingMode.NO_CONTENT)
      ),
  )
  flow = BaseLlmFlowForTesting()

  llm_request = LlmRequest(
      config=types.GenerateContentConfig(system_instruction='secret sys instr')
  )
  conn = _mock_connection([
      LlmResponse(
          input_transcription=types.Transcription(
              text='secret question', finished=True
          )
      ),
      LlmResponse(
          output_transcription=types.Transcription(
              text='secret answer', finished=True
          )
      ),
      LlmResponse(turn_complete=True),
  ])

  await _drain_receive_with_request(flow, conn, ctx, llm_request)

  events = _completion_events(records)
  assert len(events) == 1
  attrs = events[0].attributes
  # Content is elided; the event still fires (structure without content).
  assert 'gen_ai.input.messages' not in attrs
  assert 'gen_ai.output.messages' not in attrs


@pytest.mark.asyncio
async def test_legacy_schema_emits_no_completion_details(monkeypatch):
  """Under the legacy schema the live path emits no inference event."""
  monkeypatch.setenv('ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN', '1')
  records = _capture_completion_events(monkeypatch)
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.SPAN_AND_EVENT
          )
      ),
  )
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([
      LlmResponse(
          output_transcription=types.Transcription(text='hi', finished=True)
      ),
      LlmResponse(turn_complete=True),
  ])

  await _drain_receive_with_request(flow, conn, ctx, LlmRequest())

  assert not _completion_events(records)


@pytest.mark.asyncio
async def test_no_completion_details_without_semconv_opt_in(monkeypatch):
  """The event is off by default: no semconv opt-in means no inference event.

  The live span tree still renders (it is gated on schema v2, forced on by the
  autouse fixture); only the completion-details event is suppressed, so
  developers who do not need offline eval pay nothing for it.
  """
  records = _capture_completion_events(monkeypatch)
  agent = Agent(name='test_agent')
  # Content capture is on, but the experimental semconv opt-in is absent.
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(
          telemetry=TelemetryConfig(
              capture_message_content=ContentCapturingMode.SPAN_AND_EVENT
          )
      ),
  )
  flow = BaseLlmFlowForTesting()

  conn = _mock_connection([
      LlmResponse(
          output_transcription=types.Transcription(text='hello', finished=True)
      ),
      LlmResponse(turn_complete=True),
  ])

  exporter = await _drain_receive_with_request(flow, conn, ctx, LlmRequest())

  # The span tree is still emitted...
  assert _spans_named(exporter, 'live_turn')
  # ...but the inference completion-details event is not.
  assert not _completion_events(records)


@pytest.mark.asyncio
async def test_tool_call_turn_emits_single_completion_details_event(
    monkeypatch,
):
  """A tool round-trip (two generations) emits one per-turn inference event."""
  records = _capture_completion_events(monkeypatch)
  agent = Agent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(telemetry=_event_telemetry()),
  )

  def drive(t):
    t.on_user_audio()
    # First generation: function call, ended by a handoff turn_complete.
    t.on_model_output(_function_call())
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    # Second generation: the spoken answer, ended by the real turn_complete.
    t.on_model_output(_audio_chunk())
    t.on_output_transcript('it is sunny')
    t.on_turn_boundary(LlmResponse(turn_complete=True))
    t.on_usage(_usage_only())

  tracer, _, provider = _make_in_memory_tracer()
  try:
    with mock.patch('google.adk.telemetry.live_turn_tracing.tracer', tracer):
      turn_tracer = LiveTurnTracer(ctx)
      drive(turn_tracer)
      turn_tracer.close()
  finally:
    provider.shutdown()

  # One conversational turn -> exactly one inference event, despite two
  # generations.
  assert len(_completion_events(records)) == 1
