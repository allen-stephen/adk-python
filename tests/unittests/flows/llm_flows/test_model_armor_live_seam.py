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

"""Tests for the live model-call guardrail seam in BaseLlmFlow.

Verifies that ``run_live``'s seams screen input and output with the
ModelArmorPlugin (marked as a live call): output is blocked early by
suppressing the in-flight output and ending the turn; *typed* live input is
screened in parallel by default (forwarded immediately, stopped on a dirty
verdict) or before forwarding in blocking mode; *spoken* live input is
transcribed by the model and so is screened through the after-model callback at
the transcription boundary (a model result, on ``input_transcription``); and
clean turns pass through unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest import mock

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.integrations.model_armor.client import ModelArmorVerdict
from google.adk.integrations.model_armor.config import ModelArmorConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.model_armor_plugin import ModelArmorPlugin
from google.genai import types
import pytest

from ... import testing_utils


class _Flow(BaseLlmFlow):
  pass


class _KeywordClient:
  """A fake ModelArmorClient that matches when text contains a keyword."""

  def __init__(self, *, keyword: str):
    self._keyword = keyword
    self.live_output_calls = 0
    self.stream_input_calls = 0
    self.output_screened_texts: list[str] = []

  def _match(self, text: str) -> bool:
    return self._keyword in text

  async def sanitize_user_prompt(self, text):
    return ModelArmorVerdict(match_found=self._match(text))

  async def sanitize_model_response(self, text):
    return ModelArmorVerdict(match_found=self._match(text))

  async def stream_sanitize_user_prompt(self, chunks):
    self.stream_input_calls += 1
    buffered = ''
    async for chunk in chunks:
      buffered += chunk
    yield ModelArmorVerdict(match_found=self._match(buffered))

  async def stream_sanitize_model_response(self, chunks):
    self.live_output_calls += 1
    buffered = ''
    async for chunk in chunks:
      buffered += chunk
    self.output_screened_texts.append(buffered)
    yield ModelArmorVerdict(match_found=self._match(buffered))


class _StopReceive(Exception):
  """Sentinel to break the _receive_from_model outer while-loop in tests."""


def _fake_connection(responses: list[LlmResponse]):
  """Builds a fake BaseLlmConnection whose receive() yields the responses.

  After the responses are exhausted, raises a sentinel to terminate the
  flow's ``while True`` receive loop.
  """

  async def _receive():
    for response in responses:
      yield response
    raise _StopReceive()

  connection = mock.AsyncMock()
  connection.receive = mock.Mock(side_effect=_receive)
  return connection


def _output_chunk(text: str, *, turn_complete: bool = False) -> LlmResponse:
  return LlmResponse(
      output_transcription=types.Transcription(text=text),
      partial=not turn_complete,
      turn_complete=turn_complete,
  )


async def _make_context(client: _KeywordClient, *, input_screening='parallel'):
  config = ModelArmorConfig(
      prompt_template_name='pt',
      response_template_name='rt',
      input_screening=input_screening,
  )
  plugin = ModelArmorPlugin(config=config, client=client)
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(name='test_agent', model=model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )
  invocation_context.live_request_queue = LiveRequestQueue()
  return invocation_context


@pytest.mark.asyncio
async def test_clean_live_output_is_delivered():
  """A clean live turn streams its output events through unchanged."""
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _output_chunk('Hello '),
      _output_chunk('there!', turn_complete=True),
  ])

  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass

  transcripts = [
      e.output_transcription.text for e in events if e.output_transcription
  ]
  assert 'Hello ' in transcripts
  assert 'there!' in transcripts


@pytest.mark.asyncio
async def test_blocked_live_output_is_suppressed_and_turn_ends():
  """A live output match suppresses the in-flight output and ends the turn.

  Setup: the model streams a clean chunk then a chunk containing the forbidden
    keyword.
  Act: run the live receive loop.
  Assert: no event carries the forbidden text; a blocked event with the safe
    message and turn_complete is emitted; the live request queue is closed.
  """
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _output_chunk('This is fine. '),
      _output_chunk('Now something FORBIDDEN.', turn_complete=True),
  ])

  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass

  all_text = ' '.join(
      (e.output_transcription.text if e.output_transcription else '')
      + (
          ' '.join(p.text for p in e.content.parts if p.text)
          if e.content and e.content.parts
          else ''
      )
      for e in events
  )
  assert 'FORBIDDEN' not in all_text

  blocked_message = ModelArmorConfig(
      prompt_template_name='pt', response_template_name='rt'
  ).blocked_message
  blocked_events = [
      e
      for e in events
      if e.content
      and e.content.parts
      and any(p.text == blocked_message for p in e.content.parts)
  ]
  assert len(blocked_events) == 1
  assert blocked_events[0].turn_complete is True
  assert client.live_output_calls >= 1
  # The turn was ended by closing the live request queue.
  queued = await invocation_context.live_request_queue.get()
  assert queued.close is True


@pytest.mark.asyncio
async def test_live_output_callback_is_marked_live():
  """The output screening runs with the live transport (per-call marker).

  A live match must be detected via the streaming transport, proving the
  callback context was marked as a live call.
  """
  client = _KeywordClient(keyword='BAD')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _output_chunk('totally BAD content', turn_complete=True),
  ])

  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass

  # The streaming transport (live) was used, not the sync one.
  assert client.live_output_calls >= 1


# --- Input screening (parallel vs. blocking) ---


class _DelayedClient(_KeywordClient):
  """A keyword client whose input screening blocks until released."""

  def __init__(self, *, keyword: str):
    super().__init__(keyword=keyword)
    self.release = asyncio.Event()
    self.input_started = asyncio.Event()

  async def stream_sanitize_user_prompt(self, chunks):
    self.input_started.set()
    buffered = ''
    async for chunk in chunks:
      buffered += chunk
    await self.release.wait()
    yield ModelArmorVerdict(match_found=self._match(buffered))


def _content_request(text: str) -> LiveRequest:
  return LiveRequest(
      content=types.Content(role='user', parts=[types.Part(text=text)])
  )


@pytest.mark.asyncio
async def test_parallel_input_forwarded_before_screening_completes():
  """In parallel mode, clean input reaches the model without waiting on screening.

  Setup: an input-screening client that blocks until explicitly released.
  Act: run _send_to_model with one content request, then close the queue.
  Assert: the model received the content while screening was still pending
    (i.e. no added latency on the input path).
  """
  client = _DelayedClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client, input_screening='parallel')
  flow = _Flow()
  connection = mock.AsyncMock()
  connection.send_content = mock.AsyncMock()

  invocation_context.live_request_queue.send(_content_request('hello'))
  invocation_context.live_request_queue.close()

  await flow._send_to_model(connection, invocation_context)

  # The model received the input even though screening has not been released.
  connection.send_content.assert_awaited_once()
  assert client.input_started.is_set()
  assert invocation_context._live_callback_blocked_response is None

  # Release screening and let the background task settle.
  client.release.set()
  await flow._cleanup_live_input_screen_tasks(invocation_context)


@pytest.mark.asyncio
async def test_parallel_dirty_input_sets_block_signal():
  """A dirty parallel input verdict records a pending live guardrail block."""
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client, input_screening='parallel')
  flow = _Flow()
  connection = mock.AsyncMock()
  connection.send_content = mock.AsyncMock()

  invocation_context.live_request_queue.send(
      _content_request('this is FORBIDDEN')
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(connection, invocation_context)
  # Let the background screening task complete.
  for task in invocation_context._live_input_screen_tasks:
    await task

  connection.send_content.assert_awaited_once()  # still forwarded (parallel)
  assert invocation_context._live_callback_blocked_response is not None


@pytest.mark.asyncio
async def test_blocking_dirty_input_is_not_forwarded():
  """In blocking mode, dirty input is screened before send and never forwarded."""
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client, input_screening='blocking')
  flow = _Flow()
  connection = mock.AsyncMock()
  connection.send_content = mock.AsyncMock()

  invocation_context.live_request_queue.send(
      _content_request('this is FORBIDDEN')
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(connection, invocation_context)

  connection.send_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocking_clean_input_is_forwarded():
  """In blocking mode, clean input passes screening and is forwarded."""
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client, input_screening='blocking')
  flow = _Flow()
  connection = mock.AsyncMock()
  connection.send_content = mock.AsyncMock()

  invocation_context.live_request_queue.send(_content_request('hello there'))
  invocation_context.live_request_queue.close()

  await flow._send_to_model(connection, invocation_context)

  connection.send_content.assert_awaited_once()


# --- Spoken (audio) input screening via the after-model callback ---
# Spoken input is transcribed by the model, so it reaches the plugin as a model
# result (input_transcription) and is screened through after_model_callback,
# not before_model_callback. These tests assert that end-to-end behavior.


def _input_transcription(text: str, *, finished: bool) -> LlmResponse:
  """Builds a live LlmResponse carrying a user input transcription chunk."""
  return LlmResponse(
      input_transcription=types.Transcription(text=text, finished=finished),
      partial=not finished,
  )


@pytest.mark.asyncio
async def test_dirty_spoken_input_is_blocked_and_turn_ends():
  """A dirty finished input transcription ends the turn with the safe message.

  Setup: the model streams a finished input transcription containing the
    forbidden keyword, then a clean output chunk.
  Act: run the live receive loop.
  Assert: the spoken input is screened (streaming transport), a blocked event
    with the safe message and turn_complete is emitted, the model output is
    suppressed, and the live request queue is closed.
  """
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _input_transcription('please do FORBIDDEN things', finished=True),
      _output_chunk('Sure, here you go.', turn_complete=True),
  ])

  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass

  blocked_message = ModelArmorConfig(
      prompt_template_name='pt', response_template_name='rt'
  ).blocked_message
  blocked_events = [
      e
      for e in events
      if e.content
      and e.content.parts
      and any(p.text == blocked_message for p in e.content.parts)
  ]
  assert len(blocked_events) == 1
  assert blocked_events[0].turn_complete is True
  # Spoken input was screened via the streaming (live) input transport.
  assert client.stream_input_calls >= 1
  # The clean model output never reached the user.
  all_text = ' '.join(
      e.output_transcription.text if e.output_transcription else ''
      for e in events
  )
  assert 'Sure, here you go.' not in all_text
  queued = await invocation_context.live_request_queue.get()
  assert queued.close is True


@pytest.mark.asyncio
async def test_clean_spoken_input_passes_through():
  """A clean finished input transcription does not block the turn."""
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _input_transcription('what is the weather today', finished=True),
      _output_chunk('It is sunny.', turn_complete=True),
  ])

  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass

  assert client.stream_input_calls >= 1
  assert invocation_context._live_callback_blocked_response is None
  transcripts = [
      e.output_transcription.text for e in events if e.output_transcription
  ]
  assert 'It is sunny.' in transcripts


@pytest.mark.asyncio
async def test_partial_spoken_input_is_not_screened():
  """Partial (unfinished) input transcriptions are not screened.

  Only the consolidated utterance (finished=True) is screened, so a partial
  chunk — even one containing the keyword — must not trigger a screening call.
  """
  client = _KeywordClient(keyword='FORBIDDEN')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _input_transcription('FORBIDDEN', finished=False),
  ])

  events = []
  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for event in agen:
        events.append(event)
  except _StopReceive:
    pass

  assert client.stream_input_calls == 0
  assert invocation_context._live_callback_blocked_response is None


# --- Output buffer must not double the consolidated transcription ---


def _output_transcription(
    text: str, *, finished: bool, turn_complete: bool = False
) -> LlmResponse:
  """Builds a live output transcription chunk with explicit finished state."""
  return LlmResponse(
      output_transcription=types.Transcription(text=text, finished=finished),
      partial=not finished,
      turn_complete=turn_complete,
  )


@pytest.mark.asyncio
async def test_consolidated_output_transcription_is_not_doubled():
  """A finished output transcription replaces (not appends to) the buffer.

  The connection emits incremental partial deltas plus a consolidated
  ``finished`` transcription carrying the full utterance. The guardrail must
  screen the utterance once, not twice.
  """
  client = _KeywordClient(keyword='ZZZNOPE')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _output_transcription('Yes, I', finished=False),
      _output_transcription(' can hear', finished=False),
      _output_transcription(' you.', finished=False),
      # Consolidated finished transcription restates the full utterance.
      _output_transcription(
          'Yes, I can hear you.', finished=True, turn_complete=True
      ),
  ])

  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for _ in agen:
        pass
  except _StopReceive:
    pass

  # The consolidated screen must see the utterance exactly once (no doubling).
  assert client.output_screened_texts[-1] == 'Yes, I can hear you.'
  assert all(
      'Yes, I can hear you.Yes, I can hear you.' not in t
      for t in client.output_screened_texts
  )


@pytest.mark.asyncio
async def test_consolidated_only_output_transcription_screened_once():
  """A model that emits only a finished transcription is screened once."""
  client = _KeywordClient(keyword='ZZZNOPE')
  invocation_context = await _make_context(client)
  flow = _Flow()
  connection = _fake_connection([
      _output_transcription(
          'Hello there, friend.', finished=True, turn_complete=True
      ),
  ])

  try:
    async with testing_utils.Aclosing(
        flow._receive_from_model(
            connection, 'e1', invocation_context, LlmRequest()
        )
    ) as agen:
      async for _ in agen:
        pass
  except _StopReceive:
    pass

  assert client.output_screened_texts
  assert all(t == 'Hello there, friend.' for t in client.output_screened_texts)


# --- A guardrail block ends the live session GRACEFULLY (no crash) ---


class _GuardrailEndConnection:
  """A fake live connection that mimics a guardrail-initiated session end.

  ``receive`` yields a dirty input transcription (which triggers a block and
  closes the session), then raises ``APIError(1000)`` to emulate the websocket
  closing with code 1000 after the queue is closed.
  """

  def __init__(self, responses, error):
    self._responses = responses
    self._error = error

  async def send_history(self, history):
    pass

  async def send_content(self, content):
    pass

  async def send_realtime(self, blob):
    pass

  async def receive(self):
    for response in self._responses:
      await asyncio.sleep(0)
      yield response
    raise self._error

  async def close(self):
    pass


class _GuardrailEndModel(testing_utils.MockModel):
  """A MockModel whose live connection raises after a guardrail block."""

  error_to_raise: object = None

  @contextlib.asynccontextmanager
  async def connect(self, llm_request):
    self.requests.append(llm_request)
    yield _GuardrailEndConnection(self.responses, self.error_to_raise)


async def _run_live_collect(flow, invocation_context):
  events = []
  async with testing_utils.Aclosing(flow.run_live(invocation_context)) as agen:
    async for event in agen:
      events.append(event)
  return events


@pytest.mark.asyncio
async def test_guardrail_block_ends_live_session_gracefully():
  """An APIError(1000) after a guardrail block exits run_live cleanly.

  Setup: the live connection yields a dirty input transcription (triggering a
    block that closes the session), then raises APIError(1000).
  Act: run the full live flow.
  Assert: run_live returns without raising, and the safe blocked message was
    surfaced.
  """
  from google.genai import errors

  client = _KeywordClient(keyword='FORBIDDEN')
  config = ModelArmorConfig(
      prompt_template_name='pt', response_template_name='rt'
  )
  plugin = ModelArmorPlugin(config=config, client=client)
  model = _GuardrailEndModel.create(
      responses=[
          _input_transcription('do FORBIDDEN things', finished=True),
      ]
  )
  model.error_to_raise = errors.APIError(
      1000, {'error': {'message': 'closed'}}, None
  )
  agent = Agent(name='live_agent', model=model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Must not raise.
  events = await _run_live_collect(
      flow=_Flow(), invocation_context=invocation_context
  )

  assert invocation_context._live_callback_session_ended is True
  blocked_message = config.blocked_message
  assert any(
      e.content
      and e.content.parts
      and any(p.text == blocked_message for p in e.content.parts)
      for e in events
  )


@pytest.mark.asyncio
async def test_non_guardrail_api_error_still_raises():
  """An APIError(1000) WITHOUT a guardrail block is not swallowed.

  Regression guard: the graceful-exit path must only apply to guardrail-
  initiated session ends, not to genuine connection errors.
  """
  from google.genai import errors

  client = _KeywordClient(keyword='FORBIDDEN')
  config = ModelArmorConfig(
      prompt_template_name='pt', response_template_name='rt'
  )
  plugin = ModelArmorPlugin(config=config, client=client)
  # Clean input -> no block -> flag stays False; the raised error must surface.
  model = _GuardrailEndModel.create(
      responses=[
          _input_transcription('hello there', finished=True),
      ]
  )
  model.error_to_raise = errors.APIError(
      1011, {'error': {'message': 'server error'}}, None
  )
  agent = Agent(name='live_agent', model=model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  with pytest.raises(errors.APIError):
    await _run_live_collect(flow=_Flow(), invocation_context=invocation_context)

  assert invocation_context._live_callback_session_ended is False
