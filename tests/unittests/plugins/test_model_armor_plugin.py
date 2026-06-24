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

"""Tests for ModelArmorPlugin.

Verifies enforcement behavior (BLOCK short-circuits, OBSERVE passes through)
and that the transport is selected by the per-call live marker, so CFC turns
are screened with the unary (sync) transport.
"""

from __future__ import annotations

import logging
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.model_armor.client import ModelArmorVerdict
from google.adk.integrations.model_armor.config import EnforcementMode
from google.adk.integrations.model_armor.config import ModelArmorConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.model_armor_plugin import ModelArmorPlugin
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


class _FakeClient:
  """A fake ModelArmorClient recording which transport was used."""

  def __init__(self, *, match: bool):
    self._match = match
    self.calls: list[str] = []

  async def sanitize_user_prompt(self, text):
    self.calls.append('unary_input')
    return ModelArmorVerdict(match_found=self._match)

  async def sanitize_model_response(self, text):
    self.calls.append('unary_output')
    return ModelArmorVerdict(match_found=self._match)

  async def stream_sanitize_user_prompt(self, chunks):
    self.calls.append('stream_input')
    async for _ in chunks:
      pass
    yield ModelArmorVerdict(match_found=self._match)

  async def stream_sanitize_model_response(self, chunks):
    self.calls.append('stream_output')
    async for _ in chunks:
      pass
    yield ModelArmorVerdict(match_found=self._match)


def _config(**overrides):
  defaults = dict(
      project='p',
      location='us-central1',
      prompt_template_name='pt',
      response_template_name='rt',
  )
  defaults.update(overrides)
  return ModelArmorConfig(**defaults)


async def _callback_context(*, live: bool = False) -> CallbackContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='a', user_id='u')
  agent = LlmAgent(name='test_agent', model='gemini-2.0-flash')
  invocation_context = InvocationContext(
      session_service=session_service,
      invocation_id='i1',
      session=session,
      agent=agent,
  )
  ctx = CallbackContext(invocation_context)
  if live:
    ctx.is_live = True
  return ctx


def _user_request(text: str) -> LlmRequest:
  return LlmRequest(
      contents=[types.Content(role='user', parts=[types.Part(text=text)])]
  )


def _model_response(text: str) -> LlmResponse:
  return LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text=text)])
  )


def _input_transcription_response(text: str) -> LlmResponse:
  """A live model result carrying the user's transcribed spoken input."""
  return LlmResponse(
      input_transcription=types.Transcription(text=text, finished=True)
  )


@pytest.mark.asyncio
async def test_block_input_short_circuits_with_safe_message():
  """A blocked user prompt returns a safe replacement response."""
  client = _FakeClient(match=True)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  result = await plugin.before_model_callback(
      callback_context=await _callback_context(),
      llm_request=_user_request('bad input'),
  )

  assert result is not None
  assert result.content.parts[0].text == _config().blocked_message
  assert result.custom_metadata['model_armor_blocked'] is True


@pytest.mark.asyncio
async def test_clean_input_passes_through():
  """A clean user prompt is not blocked."""
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  result = await plugin.before_model_callback(
      callback_context=await _callback_context(),
      llm_request=_user_request('hello'),
  )

  assert result is None


@pytest.mark.asyncio
async def test_observe_mode_does_not_block_on_match():
  """OBSERVE mode passes matched content through (logs only)."""
  client = _FakeClient(match=True)
  plugin = ModelArmorPlugin(
      config=_config(enforcement=EnforcementMode.OBSERVE), client=client
  )

  result = await plugin.after_model_callback(
      callback_context=await _callback_context(),
      llm_response=_model_response('flagged output'),
  )

  assert result is None


@pytest.mark.asyncio
async def test_block_output_replaces_response():
  """A blocked model response is replaced with a safe message."""
  client = _FakeClient(match=True)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  result = await plugin.after_model_callback(
      callback_context=await _callback_context(),
      llm_response=_model_response('harmful output'),
  )

  assert result is not None
  assert result.content.parts[0].text == _config().blocked_message


# --- Spoken input is screened via after_model_callback (a model result) ---


@pytest.mark.asyncio
async def test_spoken_input_is_screened_as_input_in_after_callback():
  """A transcribed utterance is screened as input via after_model_callback.

  Spoken input arrives as a model result (input_transcription). The plugin must
  screen it against the *input* template (streaming transport in live) and
  block on a match — keeping before_model_callback pre-model-only.
  """
  client = _FakeClient(match=True)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  result = await plugin.after_model_callback(
      callback_context=await _callback_context(live=True),
      llm_response=_input_transcription_response('bad utterance'),
  )

  assert result is not None
  assert result.content.parts[0].text == _config().blocked_message
  # Screened as input (not output), via the streaming transport.
  assert client.calls == ['stream_input']


@pytest.mark.asyncio
async def test_clean_spoken_input_passes_through_after_callback():
  """A clean transcribed utterance is not blocked."""
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  result = await plugin.after_model_callback(
      callback_context=await _callback_context(live=True),
      llm_response=_input_transcription_response('hello there'),
  )

  assert result is None
  assert client.calls == ['stream_input']


@pytest.mark.asyncio
async def test_before_callback_ignores_spoken_input_path():
  """before_model_callback screens pre-model input only, never transcription.

  An empty live request (spoken turns carry no pre-model text) must not screen
  and must not warn — the utterance is handled in after_model_callback.
  """
  client = _FakeClient(match=True)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  result = await plugin.before_model_callback(
      callback_context=await _callback_context(live=True),
      llm_request=LlmRequest(),
  )

  assert result is None
  assert client.calls == []


@pytest.mark.asyncio
async def test_unary_call_uses_sync_transport():
  """A non-live call screens with the synchronous transport."""
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  await plugin.before_model_callback(
      callback_context=await _callback_context(live=False),
      llm_request=_user_request('hello'),
  )

  assert client.calls == ['unary_input']


@pytest.mark.asyncio
async def test_live_call_uses_streaming_transport():
  """A live-marked call screens with the streaming transport."""
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  await plugin.before_model_callback(
      callback_context=await _callback_context(live=True),
      llm_request=_user_request('hello'),
  )

  assert client.calls == ['stream_input']


@pytest.mark.asyncio
async def test_cfc_turn_is_screened_as_unary():
  """A CFC turn (no live marker) screens with the sync transport.

  CFC sets live_request_queue and runs on the live machinery, but is
  semantically unary. Because the live marker is absent on the callback
  context, the plugin must select the synchronous transport.
  """
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  ctx = await _callback_context(live=False)
  # Simulate CFC: a live_request_queue is present on the invocation context.
  ctx.get_invocation_context().live_request_queue = object()

  await plugin.before_model_callback(
      callback_context=ctx, llm_request=_user_request('hello')
  )

  assert client.calls == ['unary_input']


@pytest.mark.asyncio
async def test_streaming_disabled_falls_back_to_sync_in_live():
  """With streaming disabled, even live calls use the sync transport."""
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(streaming=False), client=client)

  await plugin.before_model_callback(
      callback_context=await _callback_context(live=True),
      llm_request=_user_request('hello'),
  )

  assert client.calls == ['unary_input']


@pytest.mark.asyncio
async def test_live_turn_without_text_warns_once(caplog):
  """A live output turn with no screenable text warns once.

  Output screening is text-based; with transcription off a live output turn
  carries no text and the guardrail covers nothing, so it warns once. (The
  before-callback no longer warns: spoken input legitimately carries no
  pre-model text and is screened in after_model_callback.)

  Setup: a live call whose request/response carry no text.
  Act: invoke the input and output callbacks twice each.
  Assert: no screening occurs, and exactly one warning is emitted (output).
  """
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)
  empty_request = LlmRequest()
  empty_response = LlmResponse()

  with caplog.at_level(logging.WARNING):
    for _ in range(2):
      assert (
          await plugin.before_model_callback(
              callback_context=await _callback_context(live=True),
              llm_request=empty_request,
          )
          is None
      )
      assert (
          await plugin.after_model_callback(
              callback_context=await _callback_context(live=True),
              llm_response=empty_response,
          )
          is None
      )

  assert client.calls == []
  warnings = [
      r
      for r in caplog.records
      if r.levelno == logging.WARNING and 'no screenable text' in r.message
  ]
  assert len(warnings) == 1


@pytest.mark.asyncio
async def test_unary_turn_without_text_does_not_warn(caplog):
  """A non-live turn with no text does not emit the live-coverage warning."""
  client = _FakeClient(match=False)
  plugin = ModelArmorPlugin(config=_config(), client=client)

  with caplog.at_level(logging.WARNING):
    await plugin.before_model_callback(
        callback_context=await _callback_context(live=False),
        llm_request=LlmRequest(),
    )

  warnings = [r for r in caplog.records if 'no screenable text' in r.message]
  assert not warnings
