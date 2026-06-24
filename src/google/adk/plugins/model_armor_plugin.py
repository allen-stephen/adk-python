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

"""Model Armor guardrail plugin.

Screens user input and model output with Google Cloud Model Armor in both
unary (``run_async``) and live (``run_live``) modes. The plugin reuses the
existing model callbacks; the transport (sync vs. bidi streaming) is selected
by the per-call live signal conveyed on the callback context, so CFC turns
(``support_cfc=True``) — which internally run on the live machinery but are
semantically unary — are screened with the synchronous API.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..agents.invocation_context import InvocationContext
from ..integrations.model_armor.client import ModelArmorClient
from ..integrations.model_armor.client import ModelArmorVerdict
from ..integrations.model_armor.config import EnforcementMode
from ..integrations.model_armor.config import ModelArmorConfig
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from .base_plugin import BasePlugin

logger = logging.getLogger('google_adk.' + __name__)


class ModelArmorPlugin(BasePlugin):
  """A plugin that screens input and output with Google Cloud Model Armor.

  Example::

      from google.adk.apps import App
      from google.adk.plugins import ModelArmorPlugin
      from google.adk.integrations.model_armor import ModelArmorConfig

      app = App(
          name='my_app',
          root_agent=my_agent,
          plugins=[
              ModelArmorPlugin(
                  config=ModelArmorConfig(
                      project='my-project',
                      location='us-central1',
                      prompt_template_name='my-prompt-template',
                      response_template_name='my-response-template',
                  )
              )
          ],
      )
  """

  def __init__(
      self,
      *,
      config: ModelArmorConfig,
      name: str = 'model_armor_plugin',
      client: Optional[ModelArmorClient] = None,
  ):
    """Initializes the Model Armor plugin.

    Args:
      config: The Model Armor configuration.
      name: A unique identifier for this plugin instance.
      client: An optional pre-constructed client (e.g. for testing). If not
        provided, one is constructed from ``config`` on first use.
    """
    super().__init__(name)
    self._config = config
    self._client = client or ModelArmorClient(config=config)
    # Whether a one-time "no screenable text in a live turn" warning has fired.
    self._warned_live_no_text = False

  def _warn_live_no_screenable_text(self, stage: str) -> None:
    """Warns once when a live turn yields no screenable text.

    Model Armor screening is text-based: live turns are screened on their
    transcription. If transcription is disabled (or a turn carries only
    non-text data), there is no text to screen and the guardrail silently
    covers nothing. Surface that once so the gap is visible.
    """
    if self._warned_live_no_text:
      return
    self._warned_live_no_text = True
    logger.warning(
        'Model Armor: a live %s turn produced no screenable text; the'
        ' guardrail cannot screen it. Ensure audio transcription is enabled'
        ' (RunConfig.input_audio_transcription / output_audio_transcription)'
        ' so live %s can be screened.',
        stage,
        stage,
    )

  @property
  def live_input_blocking(self) -> bool:
    """Whether live input should be screened before forwarding to the model.

    Read by the live flow seam to choose blocking vs. parallel input
    screening. Derived from ``ModelArmorConfig.input_screening``.
    """
    return self._config.input_screening == 'blocking'

  # --- Input screening -----------------------------------------------------

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Screens genuinely pre-model user input before the model call.

    Covers the unary path and *typed* live input — the input the model has not
    yet seen. Spoken live input cannot be screened here: it is transcribed by
    the model, so it only exists after the model call and is screened in
    ``after_model_callback`` instead (see that method). Keeping this seam
    pre-model-only preserves the integrity of the ``before_model_callback``
    contract across modes.
    """
    if not self._config.prompt_template_name:
      return None

    live = callback_context.is_live
    text = _extract_request_text(llm_request)
    if not text:
      # No warning here for live: spoken input legitimately carries no
      # pre-model text and is screened in after_model_callback instead.
      return None

    verdict = await self._screen_input(text, live=live)
    return self._handle_input_verdict(callback_context, verdict)

  # --- Output / spoken-input screening -------------------------------------

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Screens model results: model output, and transcribed spoken input.

    Two kinds of model result arrive here in live mode, distinguished by the
    populated field on ``llm_response``:

    - ``input_transcription`` — the user's transcribed speech. Screened as
      *input* (``prompt_template_name``); blocking suppresses the model's reply
      to the unsafe utterance. This is the convenience layer that keeps "screen
      input in one place" for developers even though the engine routes spoken
      input through the after-model seam.
    - otherwise — the model's own output. Screened as *output*
      (``response_template_name``).
    """
    live = callback_context.is_live

    # Spoken input (a model result): screen as input.
    input_text = _extract_input_transcription_text(llm_response)
    if input_text is not None:
      if not self._config.prompt_template_name:
        return None
      verdict = await self._screen_input(input_text, live=live)
      return self._handle_input_verdict(callback_context, verdict)

    # Model output: screen as output.
    if not self._config.response_template_name:
      return None

    text = _extract_response_text(llm_response)
    if not text:
      if live:
        self._warn_live_no_screenable_text('output')
      return None

    verdict = await self._screen_output(text, live=live)
    return self._handle_output_verdict(callback_context, llm_response, verdict)

  # --- Transport selection -------------------------------------------------

  async def _screen_input(self, text: str, *, live: bool) -> ModelArmorVerdict:
    """Screens input text using the transport for the call mode."""
    if live and self._config.streaming:
      return await _consume_stream(
          self._client.stream_sanitize_user_prompt(_single_chunk(text))
      )
    return await self._client.sanitize_user_prompt(text)

  async def _screen_output(self, text: str, *, live: bool) -> ModelArmorVerdict:
    """Screens output text using the transport for the call mode."""
    if live and self._config.streaming:
      return await _consume_stream(
          self._client.stream_sanitize_model_response(_single_chunk(text))
      )
    return await self._client.sanitize_model_response(text)

  # --- Verdict handling ----------------------------------------------------

  def _handle_input_verdict(
      self, callback_context: CallbackContext, verdict: ModelArmorVerdict
  ) -> Optional[LlmResponse]:
    """Acts on an input verdict per the enforcement mode."""
    if not verdict.match_found:
      return None

    self._log_verdict('input', verdict)
    if self._config.enforcement == EnforcementMode.OBSERVE:
      return None

    # BLOCK: short-circuit the model call with a safe replacement response.
    return self._blocked_response()

  def _handle_output_verdict(
      self,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
      verdict: ModelArmorVerdict,
  ) -> Optional[LlmResponse]:
    """Acts on an output verdict per the enforcement mode."""
    if not verdict.match_found:
      return None

    self._log_verdict('output', verdict)
    if self._config.enforcement == EnforcementMode.OBSERVE:
      return None

    # BLOCK: replace the model response with a safe message.
    return self._blocked_response()

  def _blocked_response(self) -> LlmResponse:
    """Builds the safe replacement response used when blocking."""
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[types.Part(text=self._config.blocked_message)],
        ),
        custom_metadata={'model_armor_blocked': True},
    )

  def _log_verdict(self, stage: str, verdict: ModelArmorVerdict) -> None:
    """Logs a guardrail verdict (in both BLOCK and OBSERVE modes)."""
    logger.warning(
        'Model Armor %s match found (enforcement=%s): %s',
        stage,
        self._config.enforcement.value,
        verdict.raw_result,
    )


# --- Helpers ---------------------------------------------------------------


def _extract_request_text(llm_request: LlmRequest) -> Optional[str]:
  """Extracts the latest user text from an LlmRequest."""
  if not llm_request.contents:
    return None
  # Screen the most recent user turn.
  for content in reversed(llm_request.contents):
    if content.role and content.role != 'user':
      continue
    text = _content_text(content)
    if text:
      return text
  return None


def _extract_input_transcription_text(
    llm_response: LlmResponse,
) -> Optional[str]:
  """Extracts transcribed spoken-input text from a live LlmResponse.

  Returns the utterance text when ``input_transcription`` is populated (the
  marker that this model result is the user's transcribed speech), else
  ``None`` so the response is treated as model output.
  """
  transcription = llm_response.input_transcription
  if transcription and transcription.text:
    return transcription.text
  return None


def _extract_response_text(llm_response: LlmResponse) -> Optional[str]:
  """Extracts text from an LlmResponse."""
  if not llm_response.content:
    return None
  return _content_text(llm_response.content)


def _content_text(content: Optional[types.Content]) -> Optional[str]:
  """Joins all text parts of a Content into a single string."""
  if not content or not content.parts:
    return None
  texts = [part.text for part in content.parts if part.text]
  if not texts:
    return None
  return '\n'.join(texts)


async def _single_chunk(text: str) -> AsyncIterator[str]:
  """Yields a single text chunk as an async iterator."""
  yield text


async def _consume_stream(stream) -> ModelArmorVerdict:
  """Consumes a verdict stream, returning the first match or the last verdict.

  Returning on the first match enables early blocking; otherwise the final
  verdict is used.
  """
  last: Optional[ModelArmorVerdict] = None
  async for verdict in stream:
    last = verdict
    if verdict.match_found:
      return verdict
  return last or ModelArmorVerdict(match_found=False)
