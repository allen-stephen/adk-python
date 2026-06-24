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

"""Tests for generic model callbacks in the Live API flow.

Verifies that ``run_live``'s model-call seam fires the same
``before_model_callback``/``after_model_callback`` contract as non-live, for
both plain agent callbacks and generic plugins (no Model Armor involved):

- before-model callbacks fire only for genuinely pre-model input (typed live
  input, the unary path) and receive a fully-populated ``LlmRequest``;
- spoken input is transcribed by the model, so it is screened through the
  *after*-model callback (the transcription is a model result carried on
  ``LlmResponse.input_transcription``), keeping ``before_model_callback`` a
  true pre-model contract;
- after-model callbacks can replace/suppress live output (and block on
  transcribed input) and receive a full-fidelity ``LlmResponse``;
- ``CallbackContext.is_live`` reflects the transport;
- clean turns pass through unchanged.
"""

from __future__ import annotations

from typing import Optional
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
import pytest

from ... import testing_utils


class _Flow(BaseLlmFlow):
  pass


class _StopReceive(Exception):
  """Sentinel to break the _receive_from_model outer while-loop in tests."""


def _fake_connection(responses: list[LlmResponse]):
  """Builds a fake connection whose receive() yields then stops the loop."""

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


def _content_request(text: str) -> LiveRequest:
  return LiveRequest(
      content=types.Content(role='user', parts=[types.Part(text=text)])
  )


async def _make_context(*, plugins=None, before=None, after=None):
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(
      name='test_agent',
      model=model,
      before_model_callback=before,
      after_model_callback=after,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins or []
  )
  invocation_context.live_request_queue = LiveRequestQueue()
  return invocation_context


async def _collect_receive(flow, invocation_context, connection):
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
  return events


# --- Spoken input is screened via the after-model callback, not before ---


@pytest.mark.asyncio
async def test_before_model_callback_does_not_fire_for_spoken_input():
  """The before-model callback never fires for transcribed spoken input.

  Spoken input becomes text only after the model transcribes it, so it is a
  model result screened through after_model_callback. before_model_callback
  stays a true pre-model contract and must not see it.
  """
  before_seen: list[bool] = []
  after_inputs: list[Optional[str]] = []

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    before_seen.append(callback_context.is_live)
    return None

  def after(*, callback_context: CallbackContext, llm_response: LlmResponse):
    if llm_response.input_transcription:
      after_inputs.append(llm_response.input_transcription.text)
    return None

  invocation_context = await _make_context(before=before, after=after)
  connection = _fake_connection([
      LlmResponse(
          input_transcription=types.Transcription(
              text='hello there', finished=True
          ),
          partial=False,
      ),
      _output_chunk('Hi!', turn_complete=True),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  # before never fired (no genuinely pre-model input on the receive path);
  # after saw the transcribed utterance as a model result.
  assert before_seen == []
  assert 'hello there' in after_inputs


@pytest.mark.asyncio
async def test_after_model_callback_blocks_spoken_input():
  """An after_model_callback replacement on transcribed input ends the turn.

  Blocking on the input transcription suppresses the model's in-flight reply to
  the unsafe utterance and surfaces the safe replacement.
  """

  def after(*, callback_context: CallbackContext, llm_response: LlmResponse):
    if llm_response.input_transcription:
      return LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part(text='blocked-reply')]
          )
      )
    return None

  invocation_context = await _make_context(after=after)
  connection = _fake_connection([
      LlmResponse(
          input_transcription=types.Transcription(
              text='do something', finished=True
          ),
          partial=False,
      ),
      _output_chunk('model output', turn_complete=True),
  ])

  events = await _collect_receive(_Flow(), invocation_context, connection)

  texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert 'blocked-reply' in texts
  assert 'model output' not in ' '.join(texts)


# --- Agent before-model callback over live (genuinely pre-model input) ---


@pytest.mark.asyncio
async def test_live_before_model_callback_receives_populated_request():
  """The live before_model_callback sees the model and tools, not just content.

  Asserts the request seeded from the live connection carries the agent's
  configured model so a generic callback can inspect/modify it as in non-live.
  """
  captured: list[LlmRequest] = []

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    captured.append(llm_request)
    return None

  invocation_context = await _make_context(before=before)
  # Record a base request as run_live would, so the seam can seed from it.
  base = LlmRequest(model='gemini-live-model')
  invocation_context._live_llm_request = base
  flow = _Flow()

  invocation_context.live_request_queue.send(_content_request('hello'))
  invocation_context.live_request_queue.close()
  connection = mock.AsyncMock()
  connection.send_content = mock.AsyncMock()

  await flow._send_to_model(connection, invocation_context)
  for task in invocation_context._live_input_screen_tasks:
    await task

  assert captured
  assert captured[0].model == 'gemini-live-model'
  # The current turn's content is appended to the seeded request.
  assert any(
      part.text == 'hello'
      for content in captured[0].contents or []
      for part in content.parts or []
  )


# --- Agent after-model callback over live ---


@pytest.mark.asyncio
async def test_agent_after_model_callback_replaces_live_output():
  """An after_model_callback replacement suppresses output and ends the turn."""

  def after(*, callback_context: CallbackContext, llm_response: LlmResponse):
    return LlmResponse(
        content=types.Content(
            role='model', parts=[types.Part(text='sanitized')]
        )
    )

  invocation_context = await _make_context(after=after)
  connection = _fake_connection([
      _output_chunk('raw model words', turn_complete=True),
  ])

  events = await _collect_receive(_Flow(), invocation_context, connection)

  texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert 'sanitized' in texts
  transcripts = [
      e.output_transcription.text for e in events if e.output_transcription
  ]
  assert 'raw model words' not in transcripts


@pytest.mark.asyncio
async def test_live_after_model_callback_receives_full_fidelity_response():
  """The after_model_callback sees real content parts, not text-only.

  A turn whose model output carries a function call must be passed to the
  callback with that function call intact.
  """
  captured: list[LlmResponse] = []

  def after(*, callback_context: CallbackContext, llm_response: LlmResponse):
    captured.append(llm_response)
    # Return a replacement so the turn ends here (avoids executing the
    # function call downstream); the point is what the callback received.
    return LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text='done')])
    )

  invocation_context = await _make_context(after=after)
  function_call = types.Part(
      function_call=types.FunctionCall(name='do_it', args={'x': 1})
  )
  connection = _fake_connection([
      LlmResponse(
          content=types.Content(role='model', parts=[function_call]),
          turn_complete=True,
      ),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert captured
  parts = captured[-1].content.parts
  assert any(p.function_call and p.function_call.name == 'do_it' for p in parts)


@pytest.mark.asyncio
async def test_after_model_callback_fires_for_input_and_output_in_live():
  """The after-callback fires up to twice per live turn: input then output.

  Transcribed spoken input and the model's own output are both model results,
  each surfaced to after_model_callback (distinguished by which field is set).
  """
  kinds: list[str] = []

  def after(*, callback_context: CallbackContext, llm_response: LlmResponse):
    if llm_response.input_transcription:
      kinds.append('input')
    else:
      kinds.append('output')
    return None

  invocation_context = await _make_context(after=after)
  connection = _fake_connection([
      LlmResponse(
          input_transcription=types.Transcription(
              text='a question', finished=True
          ),
          partial=False,
      ),
      _output_chunk('an answer', turn_complete=True),
  ])

  await _collect_receive(_Flow(), invocation_context, connection)

  assert kinds == ['input', 'output']


@pytest.mark.asyncio
async def test_clean_live_turn_passes_through():
  """With no callbacks, live output streams through unchanged."""
  invocation_context = await _make_context()
  connection = _fake_connection([
      _output_chunk('Hello ', turn_complete=False),
      _output_chunk('world', turn_complete=True),
  ])

  events = await _collect_receive(_Flow(), invocation_context, connection)

  transcripts = [
      e.output_transcription.text for e in events if e.output_transcription
  ]
  assert 'Hello ' in transcripts
  assert 'world' in transcripts


# --- Generic (non-Model-Armor) plugin over live ---


@pytest.mark.asyncio
async def test_generic_plugin_callbacks_fire_in_live():
  """A plain BasePlugin's model callbacks run in run_live.

  Setup: a plugin that records is_live and blocks output on a keyword.
  Act: stream output containing the keyword.
  Assert: the plugin's after-callback fired as live and replaced the output.
  """

  class _RecordingPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name='recording')
      self.after_live: Optional[bool] = None
      self.saw_input_transcription = False

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
      self.after_live = callback_context.is_live
      if llm_response.input_transcription:
        self.saw_input_transcription = True
        return None
      text = ''
      if llm_response.content and llm_response.content.parts:
        text = ''.join(p.text or '' for p in llm_response.content.parts)
      if 'NOPE' in text:
        return LlmResponse(
            content=types.Content(
                role='model', parts=[types.Part(text='replaced')]
            )
        )
      return None

  plugin = _RecordingPlugin()
  invocation_context = await _make_context(plugins=[plugin])
  connection = _fake_connection([
      LlmResponse(
          input_transcription=types.Transcription(text='say it', finished=True),
          partial=False,
      ),
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part(text='NOPE bad')]
          ),
          turn_complete=True,
      ),
  ])

  events = await _collect_receive(_Flow(), invocation_context, connection)

  # The after-callback fired as live, screened the transcribed input, and
  # replaced the offending output.
  assert plugin.after_live is True
  assert plugin.saw_input_transcription is True
  texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert 'replaced' in texts
  assert 'NOPE bad' not in ' '.join(texts)


# --- is_live signal correctness ---


@pytest.mark.asyncio
async def test_is_live_false_for_non_live_callback():
  """CallbackContext.is_live defaults to False (non-live)."""
  invocation_context = await _make_context()
  ctx = CallbackContext(invocation_context)

  assert ctx.is_live is False


# --- Voice (audio) input fires before_model_callback once per turn ---------


def _blob_request() -> LiveRequest:
  return LiveRequest(blob=types.Blob(data=b'\x00\x01', mime_type='audio/pcm'))


def _fake_send_connection():
  connection = mock.AsyncMock()
  connection.send_realtime = mock.AsyncMock()
  connection.send_content = mock.AsyncMock()
  connection.close = mock.AsyncMock()
  return connection


async def _drive_send(flow, invocation_context, requests):
  """Enqueues requests (then a close) and runs _send_to_model to completion."""
  for request in requests:
    invocation_context.live_request_queue.send(request)
  invocation_context.live_request_queue.close()
  connection = _fake_send_connection()
  await flow._send_to_model(connection, invocation_context)
  return connection


@pytest.mark.asyncio
async def test_voice_before_model_callback_fires_once_per_activity_auto_vad():
  """Automatic VAD: the first audio blob fires before_model_callback once.

  Subsequent blobs in the same activity (no turn boundary) do not re-fire.
  """
  calls: list[bool] = []

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    calls.append(callback_context.is_live)
    return None

  invocation_context = await _make_context(before=before)
  await _drive_send(
      _Flow(),
      invocation_context,
      [_blob_request(), _blob_request(), _blob_request()],
  )

  assert calls == [True]  # fired exactly once, marked live


@pytest.mark.asyncio
async def test_voice_before_model_callback_has_no_user_text():
  """The voice before-model request carries no user text (audio, not text)."""
  captured: list[LlmRequest] = []

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    captured.append(llm_request)
    return None

  invocation_context = await _make_context(before=before)
  await _drive_send(_Flow(), invocation_context, [_blob_request()])

  assert captured
  texts = [
      part.text
      for content in captured[0].contents or []
      for part in content.parts or []
      if part.text
  ]
  assert not texts


@pytest.mark.asyncio
async def test_voice_before_model_callback_refires_after_turn_boundary():
  """A turn boundary re-arms voice firing so the next activity fires once."""
  calls: list[bool] = []

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    calls.append(callback_context.is_live)
    return None

  invocation_context = await _make_context(before=before)
  flow = _Flow()

  # First voice activity: one firing across several blobs.
  await _drive_send(flow, invocation_context, [_blob_request(), _blob_request()])
  assert calls == [True]

  # A receive-side turn boundary re-arms the gate.
  connection = _fake_connection([_output_chunk('answer', turn_complete=True)])
  await _collect_receive(flow, invocation_context, connection)
  assert invocation_context._live_voice_before_model_armed is True

  # Second voice activity fires once more.
  invocation_context.live_request_queue = LiveRequestQueue()
  await _drive_send(flow, invocation_context, [_blob_request(), _blob_request()])
  assert calls == [True, True]


@pytest.mark.asyncio
async def test_voice_before_model_callback_manual_vad_activity_signals():
  """Manual VAD: activity_start fires once; activity_end re-arms."""
  calls: list[bool] = []

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    calls.append(callback_context.is_live)
    return None

  invocation_context = await _make_context(before=before)
  await _drive_send(
      _Flow(),
      invocation_context,
      [
          LiveRequest(activity_start=types.ActivityStart()),
          _blob_request(),  # same activity -> no re-fire
          LiveRequest(activity_end=types.ActivityEnd()),  # re-arm
          LiveRequest(activity_start=types.ActivityStart()),  # next turn fires
          _blob_request(),
      ],
  )

  assert calls == [True, True]


@pytest.mark.asyncio
async def test_voice_before_model_callback_return_value_ignored():
  """Voice is observe-only: a returned replacement does not block forwarding."""

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    # Try to block; this must be ignored for voice (audio already streaming).
    return LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text='blocked')])
    )

  invocation_context = await _make_context(before=before)
  connection = await _drive_send(
      _Flow(), invocation_context, [_blob_request()]
  )

  # The audio was still forwarded to the model; no pending block was recorded.
  assert connection.send_realtime.await_count >= 1
  assert invocation_context._live_callback_blocked_response is None


@pytest.mark.asyncio
async def test_voice_before_model_callback_exception_does_not_break_send():
  """A failing voice before-model callback is logged, not raised."""

  def before(*, callback_context: CallbackContext, llm_request: LlmRequest):
    raise RuntimeError('boom')

  invocation_context = await _make_context(before=before)
  # Should not raise; the send loop swallows callback errors.
  connection = await _drive_send(
      _Flow(), invocation_context, [_blob_request()]
  )
  assert connection.send_realtime.await_count >= 1
