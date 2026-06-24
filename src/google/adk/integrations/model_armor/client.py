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

"""Mode-aware client wrapper around the Model Armor service.

This module isolates the ``google-cloud-modelarmor`` SDK behind a small,
normalized surface so the plugin never depends on raw SDK types. It exposes
both transports:

- Unary (sync ``sanitize_*``) for ``run_async`` / SSE / CFC turns.
- Bidi streaming (``stream_sanitize_*``) for ``run_live`` turns.
"""

from __future__ import annotations

import dataclasses
from typing import AsyncIterable
from typing import AsyncIterator
from typing import Optional

from google.auth.credentials import Credentials

from ... import version
from .config import ModelArmorConfig
from .config import StreamingMode

USER_AGENT = f'adk-model-armor-plugin google-adk/{version.__version__}'

_INSTALL_HINT = (
    'ModelArmorPlugin requires google-cloud-modelarmor. '
    "Install it with: pip install 'google-adk[gcp]'."
)


@dataclasses.dataclass
class ModelArmorVerdict:
  """A normalized Model Armor verdict, decoupled from raw SDK types."""

  match_found: bool
  """Whether any filter reported ``MATCH_FOUND``."""

  sanitized_text: Optional[str] = None
  """De-identified/redacted text, if the template produced a transform."""

  raw_result: object = None
  """The raw ``SanitizationResult`` proto, for logging/diagnostics."""


def _import_modelarmor():
  """Deferred import of the Model Armor SDK with a friendly install hint."""
  try:
    from google.cloud import modelarmor_v1beta
  except ImportError as e:
    raise ImportError(_INSTALL_HINT) from e
  return modelarmor_v1beta


def _regional_endpoint(location: Optional[str]) -> Optional[str]:
  """Builds the Model Armor regional endpoint for a location."""
  if not location or location == 'global':
    return None
  return f'modelarmor.{location}.rep.googleapis.com'


def get_model_armor_client(
    *,
    location: Optional[str],
    credentials: Optional[Credentials] = None,
    user_agent: Optional[str] = None,
):
  """Creates a Model Armor async client.

  Args:
    location: The regional location of the templates. Drives the regional
      endpoint. ``None``/``global`` uses the default endpoint.
    credentials: Optional credentials. If ``None``, Application Default
      Credentials are used.
    user_agent: Optional user agent override.

  Returns:
    A ``modelarmor_v1beta.ModelArmorAsyncClient``.
  """
  modelarmor_v1beta = _import_modelarmor()

  from google.api_core.client_options import ClientOptions
  from google.api_core.gapic_v1.client_info import ClientInfo

  client_info = ClientInfo(user_agent=user_agent or USER_AGENT)

  client_options = None
  endpoint = _regional_endpoint(location)
  if endpoint:
    client_options = ClientOptions(api_endpoint=endpoint)

  return modelarmor_v1beta.ModelArmorAsyncClient(
      credentials=credentials,
      client_info=client_info,
      client_options=client_options,
  )


def _streaming_mode(config: ModelArmorConfig):
  """Maps the plugin's StreamingMode to the SDK enum."""
  modelarmor_v1beta = _import_modelarmor()
  if config.streaming_mode == StreamingMode.BUFFERED:
    return modelarmor_v1beta.StreamingMode.STREAMING_MODE_BUFFERED
  return modelarmor_v1beta.StreamingMode.STREAMING_MODE_REALTIME


def _verdict_from_result(result) -> ModelArmorVerdict:
  """Normalizes a ``SanitizationResult`` proto into a ModelArmorVerdict."""
  modelarmor_v1beta = _import_modelarmor()
  match_found = (
      result.filter_match_state
      == modelarmor_v1beta.FilterMatchState.MATCH_FOUND
  )
  return ModelArmorVerdict(
      match_found=match_found,
      sanitized_text=_extract_sanitized_text(result),
      raw_result=result,
  )


def _extract_sanitized_text(result) -> Optional[str]:
  """Best-effort extraction of de-identified text from SDP results."""
  try:
    for filter_result in result.filter_results.values():
      sdp = getattr(filter_result, 'sdp_filter_result', None)
      if not sdp:
        continue
      deidentify = getattr(sdp, 'deidentify_result', None)
      if deidentify and getattr(deidentify, 'data', None):
        text = getattr(deidentify.data, 'text', None)
        if text:
          return text
  except (AttributeError, TypeError):
    return None
  return None


class ModelArmorClient:
  """A thin, mode-aware wrapper exposing normalized sanitize operations."""

  def __init__(
      self,
      *,
      config: ModelArmorConfig,
      client=None,
      credentials: Optional[Credentials] = None,
  ):
    """Initializes the wrapper.

    Args:
      config: The Model Armor configuration.
      client: An optional pre-constructed async client (e.g. for testing). If
        ``None``, one is lazily constructed on first use.
      credentials: Optional credentials used when constructing the client.
    """
    self._config = config
    self._client = client
    self._credentials = credentials

  @property
  def client(self):
    """Returns the underlying async client, constructing it lazily."""
    if self._client is None:
      self._client = get_model_armor_client(
          location=self._config.location,
          credentials=self._credentials,
      )
    return self._client

  def _template_path(self, template_name: str) -> str:
    """Resolves a template name to a fully-qualified resource path."""
    if template_name.startswith('projects/'):
      return template_name
    if not self._config.project or not self._config.location:
      raise ValueError(
          'project and location are required to resolve the short template '
          f'name {template_name!r}. Provide them in ModelArmorConfig or pass '
          'a fully-qualified template resource name.'
      )
    return (
        f'projects/{self._config.project}'
        f'/locations/{self._config.location}'
        f'/templates/{template_name}'
    )

  # --- Unary transport -----------------------------------------------------

  async def sanitize_user_prompt(self, text: str) -> ModelArmorVerdict:
    """Screens user input text using the synchronous API."""
    modelarmor_v1beta = _import_modelarmor()
    request = modelarmor_v1beta.SanitizeUserPromptRequest(
        name=self._template_path(self._config.prompt_template_name),
        user_prompt_data=modelarmor_v1beta.DataItem(text=text),
    )
    response = await self.client.sanitize_user_prompt(request=request)
    return _verdict_from_result(response.sanitization_result)

  async def sanitize_model_response(self, text: str) -> ModelArmorVerdict:
    """Screens model output text using the synchronous API."""
    modelarmor_v1beta = _import_modelarmor()
    request = modelarmor_v1beta.SanitizeModelResponseRequest(
        name=self._template_path(self._config.response_template_name),
        model_response_data=modelarmor_v1beta.DataItem(text=text),
    )
    response = await self.client.sanitize_model_response(request=request)
    return _verdict_from_result(response.sanitization_result)

  # --- Bidi streaming transport --------------------------------------------

  async def stream_sanitize_user_prompt(
      self, chunks: AsyncIterator[str]
  ) -> AsyncIterable[ModelArmorVerdict]:
    """Screens streamed user input, yielding a verdict per server response."""
    modelarmor_v1beta = _import_modelarmor()
    name = self._template_path(self._config.prompt_template_name)
    streaming_mode = _streaming_mode(self._config)

    async def _requests():
      async for chunk in chunks:
        yield modelarmor_v1beta.SanitizeUserPromptRequest(
            name=name,
            user_prompt_data=modelarmor_v1beta.DataItem(text=chunk),
            streaming_mode=streaming_mode,
        )

    stream = await self.client.stream_sanitize_user_prompt(requests=_requests())
    async for response in stream:
      yield _verdict_from_result(response.sanitization_result)

  async def stream_sanitize_model_response(
      self, chunks: AsyncIterator[str]
  ) -> AsyncIterable[ModelArmorVerdict]:
    """Screens streamed model output, yielding a verdict per server response."""
    modelarmor_v1beta = _import_modelarmor()
    name = self._template_path(self._config.response_template_name)
    streaming_mode = _streaming_mode(self._config)

    async def _requests():
      async for chunk in chunks:
        yield modelarmor_v1beta.SanitizeModelResponseRequest(
            name=name,
            model_response_data=modelarmor_v1beta.DataItem(text=chunk),
            streaming_mode=streaming_mode,
        )

    stream = await self.client.stream_sanitize_model_response(
        requests=_requests()
    )
    async for response in stream:
      yield _verdict_from_result(response.sanitization_result)
