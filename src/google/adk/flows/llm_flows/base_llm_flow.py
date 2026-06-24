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

from __future__ import annotations

from abc import ABC
import asyncio
import inspect
import logging
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from opentelemetry import trace
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import ConnectionClosedOK

from . import _output_schema_processor
from . import functions
from ...agents.base_agent import BaseAgent
from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.live_request_queue import LiveRequestQueue
from ...agents.readonly_context import ReadonlyContext
from ...agents.run_config import StreamingMode
from ...auth.auth_tool import AuthConfig
from ...events.event import Event
from ...models.base_llm_connection import BaseLlmConnection
from ...models.google_llm import Gemini
from ...models.google_llm import GoogleLLMVariant
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry import _instrumentation
from ...telemetry.tracing import trace_call_llm
from ...telemetry.tracing import trace_send_data
from ...telemetry.tracing import tracer
from ...tools.base_toolset import BaseToolset
from ...tools.tool_context import ToolContext
from ...utils.context_utils import Aclosing
from .audio_cache_manager import AudioCacheManager
from .functions import build_auth_request_event

# Prefix used by toolset auth credential IDs
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent
  from ...models.base_llm import BaseLlm
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger('google_adk.' + __name__)

_ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'


def _live_input_callback_is_blocking(
    invocation_context: InvocationContext,
) -> bool:
  """Whether any registered plugin requests pre-send (blocking) live input.

  A plugin may opt into running its before-model callback on live input
  *before* the input is forwarded to the model by exposing a truthy
  ``live_input_blocking`` attribute. This keeps the flow generic — it does not
  depend on any specific plugin type. Defaults to False (parallel,
  latency-optimized).
  """
  plugin_manager = getattr(invocation_context, 'plugin_manager', None)
  if not plugin_manager:
    return False
  for plugin in getattr(plugin_manager, 'plugins', []):
    if getattr(plugin, 'live_input_blocking', False):
      return True
  return False


def _live_output_text(llm_response: LlmResponse) -> Optional[str]:
  """Extracts model output text from a live LlmResponse.

  Prefers output transcription (the text rendering of audio output) and falls
  back to model content text. Returns None for user/input or non-text chunks.
  """
  transcription = llm_response.output_transcription
  if transcription and transcription.text:
    return transcription.text
  content = llm_response.content
  if content and content.role != 'user' and content.parts:
    texts = [part.text for part in content.parts if part.text]
    if texts:
      return ''.join(texts)
  return None


def _live_output_is_finished_transcription(llm_response: LlmResponse) -> bool:
  """Whether a live LlmResponse is a consolidated (finished) output transcript.

  The live connection yields output transcription twice over a turn:
  incremental ``partial`` deltas, then a single ``finished`` response carrying
  the full accumulated utterance. The output buffer must *replace* itself with
  this authoritative text rather than append it, so the after-model callback
  sees the utterance once instead of twice.
  """
  transcription = llm_response.output_transcription
  return bool(transcription and transcription.finished and transcription.text)


def _build_live_after_model_response(
    last_output_response: Optional[LlmResponse],
    buffered_text: str,
) -> LlmResponse:
  """Builds the full-fidelity LlmResponse for the live after-model callback.

  Gives the after-model callback the same fidelity it has in non-live mode:
  when the model emitted real content (text, audio, function calls), that
  content is preserved. Output produced only as transcription (the text
  rendering of audio) has no content parts, so a text content is synthesized
  from the consolidated transcript instead. Either way, the callback sees the
  consolidated turn output, screened once.
  """
  if (
      last_output_response is not None
      and last_output_response.content is not None
      and last_output_response.content.parts
  ):
    response = last_output_response.model_copy(deep=True)
    return response
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text=buffered_text)],
      )
  )


def _live_input_transcription_text(llm_response: LlmResponse) -> Optional[str]:
  """Extracts spoken-input text from a live LlmResponse.

  Returns the consolidated user utterance only when the input transcription is
  ``finished`` (the per-utterance boundary). Partial transcriptions return
  ``None`` so input is processed once per utterance rather than per fragment.

  Spoken input is transcribed *by the model*, so the transcription is itself a
  product of a model invocation: it arrives on the receive side, in the model's
  output stream, after the audio has already reached the model. It is therefore
  screened through the *after*-model callback (which inspects model results),
  not the before-model callback. This keeps ``before_model_callback`` a true
  pre-model contract: only genuinely pre-model input (typed live input, and the
  unary path) ever reaches it.
  """
  transcription = llm_response.input_transcription
  if transcription and transcription.finished and transcription.text:
    return transcription.text
  return None


def _build_live_input_transcription_response(text: str) -> LlmResponse:
  """Builds the after-model LlmResponse carrying transcribed spoken input.

  The ``input_transcription`` field is populated (and ``content``/
  ``output_transcription`` left empty) so an after-model callback can recognize
  this response as the user's transcribed utterance — a model result to screen
  as *input* — and distinguish it from the model's own *output*.
  """
  return LlmResponse(
      input_transcription=types.Transcription(text=text, finished=True)
  )


# Timing configuration
DEFAULT_TRANSFER_AGENT_DELAY = 1.0
DEFAULT_TASK_COMPLETION_DELAY = 1.0

DEFAULT_MAX_RECONNECT_ATTEMPTS = 5

# Statistics configuration
DEFAULT_ENABLE_CACHE_STATISTICS = False


def _finalize_model_response_event(
    llm_request: LlmRequest,
    llm_response: LlmResponse,
    model_response_event: Event,
) -> Event:
  """Finalize and build the model response event from LLM response.

  Merges the LLM response data into the model response event and
  populates function call IDs and long-running tool information.

  Args:
    llm_request: The original LLM request.
    llm_response: The LLM response from the model.
    model_response_event: The base event to populate.

  Returns:
    The finalized Event with LLM response data merged in.
  """
  finalized_event = Event.model_validate({
      **model_response_event.model_dump(exclude_none=True),
      **llm_response.model_dump(exclude_none=True),
  })

  if finalized_event.content:
    function_calls = finalized_event.get_function_calls()
    if function_calls:
      functions.populate_client_function_call_id(finalized_event)
      finalized_event.long_running_tool_ids = (
          functions.get_long_running_function_calls(
              function_calls, llm_request.tools_dict
          )
      )

  return finalized_event


async def _resolve_toolset_auth(
    invocation_context: InvocationContext,
    agent: LlmAgent,
) -> AsyncGenerator[Event, None]:
  """Resolves authentication for toolsets before tool listing.

  For each toolset with auth configured via get_auth_config():
  - If credential is available, populate auth_config.exchanged_auth_credential
  - If credential is not available, yield auth request event and interrupt

  Args:
    invocation_context: The invocation context.
    agent: The LLM agent.

  Yields:
    Auth request events if any toolset needs authentication.
  """
  if not agent.tools:
    return

  pending_auth_requests: dict[str, AuthConfig] = {}
  callback_context = CallbackContext(invocation_context)

  for tool_union in agent.tools:
    if not isinstance(tool_union, BaseToolset):
      continue

    auth_config = tool_union.get_auth_config()
    if not auth_config:
      continue

    auth_config_copy = auth_config.model_copy(deep=True)
    from ...auth.credential_manager import CredentialManager

    try:
      credential = await CredentialManager(
          auth_config_copy
      ).get_auth_credential(callback_context)
    except ValueError as e:
      # Validation errors from CredentialManager should be logged but not
      # block the flow - the toolset may still work without auth
      logger.warning(
          'Failed to get auth credential for toolset %s: %s',
          type(tool_union).__name__,
          e,
      )
      credential = None

    if credential:
      # Store in invocation context to avoid data leakage and race conditions
      invocation_context.credential_by_key[auth_config.credential_key] = (
          credential
      )
    else:
      # Need auth - will interrupt
      toolset_id = (
          f'{TOOLSET_AUTH_CREDENTIAL_ID_PREFIX}{type(tool_union).__name__}'
      )
      pending_auth_requests[toolset_id] = auth_config_copy

  if not pending_auth_requests:
    return

  from ...auth.auth_handler import AuthHandler

  auth_requests = {
      credential_id: AuthHandler(auth_config).generate_auth_request()
      for credential_id, auth_config in pending_auth_requests.items()
  }

  # Yield event with auth requests using the shared helper
  yield build_auth_request_event(
      invocation_context,
      auth_requests,
      author=agent.name,
  )

  # Interrupt invocation
  invocation_context.end_invocation = True


async def _handle_before_model_callback(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
    *,
    is_live_call: bool = False,
) -> Optional[LlmResponse]:
  """Runs before-model callbacks (plugins then agent callbacks).

  Args:
    invocation_context: The invocation context.
    llm_request: The LLM request being built.
    model_response_event: The model response event for callback context.
    is_live_call: Whether this model call is a real live (bidi) call. Conveyed
      per-call on the callback context (``CallbackContext.is_live``) so
      callbacks can branch on transport. CFC turns are not live and leave this
      False.

  Returns:
    An LlmResponse if a callback short-circuits the LLM call, else None.
  """
  agent = invocation_context.agent

  callback_context = CallbackContext(
      invocation_context, event_actions=model_response_event.actions
  )
  if is_live_call:
    callback_context.is_live = True

  # First run callbacks from the plugins.
  callback_response = (
      await invocation_context.plugin_manager.run_before_model_callback(
          callback_context=callback_context,
          llm_request=llm_request,
      )
  )
  if callback_response:
    return callback_response

  # If no overrides are provided from the plugins, further run the canonical
  # callbacks.
  if not agent.canonical_before_model_callbacks:
    return
  for callback in agent.canonical_before_model_callbacks:
    callback_response = callback(
        callback_context=callback_context, llm_request=llm_request
    )
    if inspect.isawaitable(callback_response):
      callback_response = await callback_response
    if callback_response:
      return callback_response


async def _handle_after_model_callback(
    invocation_context: InvocationContext,
    llm_response: LlmResponse,
    model_response_event: Event,
    *,
    is_live_call: bool = False,
) -> Optional[LlmResponse]:
  """Runs after-model callbacks (plugins then agent callbacks).

  Also handles grounding metadata injection when google_search_agent is
  among the agent's tools.

  Args:
    invocation_context: The invocation context.
    llm_response: The LLM response to process.
    model_response_event: The model response event for callback context.
    is_live_call: Whether this model call is a real live (bidi) call. Conveyed
      per-call on the callback context (``CallbackContext.is_live``) so
      callbacks can branch on transport. CFC turns are not live and leave this
      False.

  Returns:
    An altered LlmResponse if a callback modifies it, else None.
  """
  agent = invocation_context.agent

  # Add grounding metadata to the response if needed.
  # TODO(b/448114567): Remove this function once the workaround is no longer needed.
  async def _maybe_add_grounding_metadata(
      response: Optional[LlmResponse] = None,
  ) -> Optional[LlmResponse]:
    readonly_context = ReadonlyContext(invocation_context)
    if (tools := invocation_context.canonical_tools_cache) is None:
      tools = await agent.canonical_tools(readonly_context)
      invocation_context.canonical_tools_cache = tools

    if not any(tool.name == 'google_search_agent' for tool in tools):
      return response
    ground_metadata = invocation_context.session.state.get(
        'temp:_adk_grounding_metadata', None
    )
    if not ground_metadata:
      return response

    if not response:
      response = llm_response
    response.grounding_metadata = ground_metadata
    return response

  callback_context = CallbackContext(
      invocation_context, event_actions=model_response_event.actions
  )
  if is_live_call:
    callback_context.is_live = True

  # First run callbacks from the plugins.
  callback_response = (
      await invocation_context.plugin_manager.run_after_model_callback(
          callback_context=callback_context,
          llm_response=llm_response,
      )
  )
  if callback_response:
    return await _maybe_add_grounding_metadata(callback_response)

  # If no overrides are provided from the plugins, further run the canonical
  # callbacks.
  if not agent.canonical_after_model_callbacks:
    return await _maybe_add_grounding_metadata()
  for callback in agent.canonical_after_model_callbacks:
    callback_response = callback(
        callback_context=callback_context, llm_response=llm_response
    )
    if inspect.isawaitable(callback_response):
      callback_response = await callback_response
    if callback_response:
      return await _maybe_add_grounding_metadata(callback_response)
  return await _maybe_add_grounding_metadata()


async def _run_and_handle_error(
    response_generator: AsyncGenerator[LlmResponse, None],
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
    call_llm_span: Optional[trace.Span] = None,
) -> AsyncGenerator[LlmResponse, None]:
  """Wraps an LLM response generator with error callback handling.

  Runs the response generator within a tracing span. If an error occurs,
  runs on-model-error callbacks (plugins then agent callbacks). If a
  callback returns a response, that response is yielded instead of
  re-raising the error.

  Args:
    response_generator: The async generator producing LLM responses.
    invocation_context: The invocation context.
    llm_request: The LLM request.
    model_response_event: The model response event.
    call_llm_span: The call_llm span to rebind error callbacks to. When
      provided, on_model_error callbacks run under this span so plugins observe
      the same span as before/after model callbacks.

  Yields:
    LlmResponse objects from the generator.

  Raises:
    The original model error if no error callback handles it.
  """
  agent = invocation_context.agent
  if not hasattr(agent, 'canonical_on_model_error_callbacks'):
    raise TypeError(
        'Expected agent to have canonical_on_model_error_callbacks'
        f' attribute, but got {type(agent)}'
    )

  async def _run_on_model_error_callbacks(
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    error_response = (
        await invocation_context.plugin_manager.run_on_model_error_callback(
            callback_context=callback_context,
            llm_request=llm_request,
            error=error,
        )
    )
    if error_response is not None:
      return error_response

    for callback in agent.canonical_on_model_error_callbacks:
      error_response = callback(
          callback_context=callback_context,
          llm_request=llm_request,
          error=error,
      )
      if inspect.isawaitable(error_response):
        error_response = await error_response
      if error_response is not None:
        return error_response

    return None

  try:
    async with _instrumentation.record_inference_telemetry(
        llm_request,
        invocation_context,
        model_response_event,
    ) as tel_ctx:
      async with Aclosing(response_generator) as agen:
        async for llm_response in agen:
          tel_ctx.record_llm_response(invocation_context, llm_response)
          yield llm_response
  except Exception as model_error:
    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )
    if call_llm_span is not None:
      with trace.use_span(call_llm_span, end_on_exit=False):
        error_response = await _run_on_model_error_callbacks(
            callback_context=callback_context,
            llm_request=llm_request,
            error=model_error,
        )
    else:
      error_response = await _run_on_model_error_callbacks(
          callback_context=callback_context,
          llm_request=llm_request,
          error=model_error,
      )
    if error_response is not None:
      yield error_response
    else:
      raise model_error


async def _process_agent_tools(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Process the agent's tools and populate ``llm_request.tools_dict``.

  Iterates over the agent's ``tools`` list, converts each tool union
  (callable, BaseTool, or BaseToolset) into resolved ``BaseTool``
  instances, and calls ``process_llm_request`` on each to register
  tool declarations in the request.

  Tool-union resolution is dispatched concurrently via ``asyncio.gather``
  to overlap I/O-bound listings (e.g. MCP ``list_tools`` over the
  network). The subsequent ``process_llm_request`` calls are kept
  serial in the original ``agent.tools`` order: some tools read/write
  ``llm_request`` state (e.g. ``GoogleSearchTool`` writes
  ``llm_request.model``; ``ComputerUseToolset`` performs an idempotency
  check on ``llm_request.config.tools``) and rely on observing the
  post-state of earlier tools.

  After this function returns, ``llm_request.tools_dict`` maps tool
  names to ``BaseTool`` instances ready for function call dispatch.

  Args:
    invocation_context: The invocation context (``agent`` is read from
      ``invocation_context.agent``).
    llm_request: The LLM request to populate with tool declarations.
  """
  agent = invocation_context.agent
  if agent is None or not hasattr(agent, 'tools') or not agent.tools:
    return

  multiple_tools = len(agent.tools) > 1
  model = agent.canonical_model

  from ...agents.llm_agent import _convert_tool_union_to_tools

  # Resolve tool_unions in parallel. ``asyncio.gather`` preserves
  # input order in the returned list, so the serial commit phase below
  # still observes ``agent.tools`` order. If any resolution raises,
  # gather cancels the siblings and propagates -- same observable
  # behavior as the previous serial loop, which would propagate the
  # first exception and abandon the rest.
  resolved_tools_per_union = await asyncio.gather(*(
      _convert_tool_union_to_tools(
          tool_union,
          ReadonlyContext(invocation_context),
          model,
          multiple_tools,
      )
      for tool_union in agent.tools
  ))

  # Serial commit phase, in original ``agent.tools`` order. Mutations
  # to ``llm_request`` and reads of its state (model, config.tools,
  # tools_dict) preserve today's ordering semantics exactly.
  for tool_union, tools in zip(agent.tools, resolved_tools_per_union):
    tool_context = ToolContext(invocation_context)

    # If it's a toolset, process it first
    if isinstance(tool_union, BaseToolset):
      await tool_union.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

    # Then process all tools from this tool union
    for tool in tools:
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  if invocation_context.live_request_queue is not None:
    _mark_live_async_tools_non_blocking(llm_request)


def _mark_live_async_tools_non_blocking(llm_request: LlmRequest) -> None:
  """Marks live streaming and response-scheduling tools as NON_BLOCKING.

  These tools emit asynchronous FunctionResponses, which the Live API only
  accepts for NON_BLOCKING declarations.
  """
  if not llm_request.config.tools:
    return
  for gemini_tool in llm_request.config.tools:
    for declaration in gemini_tool.function_declarations or []:
      tool = llm_request.tools_dict.get(declaration.name)
      if tool is None:
        continue
      is_streaming_tool = hasattr(tool, 'func') and inspect.isasyncgenfunction(
          tool.func
      )
      if tool.response_scheduling is not None or is_streaming_tool:
        declaration.behavior = types.Behavior.NON_BLOCKING


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfers to another agent.
  """

  def __init__(self):
    self.request_processors: list[BaseLlmRequestProcessor] = []
    self.response_processors: list[BaseLlmResponseProcessor] = []

    # Initialize configuration and managers
    self.audio_cache_manager = AudioCacheManager()

  async def run_live(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow using live api."""
    from google.genai import errors

    llm_request = LlmRequest()
    event_id = Event.new_id()

    # Preprocess before calling the LLM.
    async with Aclosing(
        self._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event
    if invocation_context.end_invocation:
      return

    agent = invocation_context.agent
    llm_request.model = agent.canonical_live_model.model

    # Record the base request so the live before-model callback seam can seed
    # per-call requests with the full model/config/tools (non-live fidelity).
    invocation_context._live_llm_request = llm_request

    llm = self.__get_llm(invocation_context)
    logger.debug(
        'Establishing live connection for agent: %s with llm request: %s',
        invocation_context.agent.name,
        llm_request,
    )

    attempt = 1
    while True:
      try:
        # On subsequent attempts, use the saved token to reconnect
        if invocation_context.live_session_resumption_handle:
          logger.info('Attempting to reconnect (Attempt %s)...', attempt)
          attempt += 1
          if not llm_request.live_connect_config:
            llm_request.live_connect_config = types.LiveConnectConfig()
          if not llm_request.live_connect_config.session_resumption:
            llm_request.live_connect_config.session_resumption = (
                types.SessionResumptionConfig()
            )
          llm_request.live_connect_config.session_resumption.handle = (
              invocation_context.live_session_resumption_handle
          )

          # Only set transparent=True for Vertex AI backend, as the Gemini API
          # backend explicitly rejects it.
          if (
              isinstance(llm, Gemini)
              and llm._api_backend == GoogleLLMVariant.VERTEX_AI  # pylint: disable=protected-access
          ):
            session_resumption = (
                llm_request.live_connect_config.session_resumption
            )
            if session_resumption.transparent is None:
              session_resumption.transparent = True

        # When seeding a fresh connection with prior conversation history, set
        # initial_history_in_client_content to True. This tells the Live server
        # that the provided history already includes the model's past responses,
        # preventing the server from generating duplicate responses for those replayed turns.
        #
        # ``history_config`` is only supported by the Gemini Developer API
        # backend; the Vertex AI / Gemini Enterprise Agent Platform backend has
        # no such field on its live setup message and rejects it. On Vertex,
        # history is instead seeded by ``send_history`` below
        # (``send_client_content`` with prior turns), so we skip
        # ``history_config`` for that backend.
        if (
            llm_request.contents
            and not invocation_context.live_session_resumption_handle
            and isinstance(llm, Gemini)
            and llm._api_backend == GoogleLLMVariant.GEMINI_API  # pylint: disable=protected-access
        ):
          if not llm_request.live_connect_config:
            llm_request.live_connect_config = types.LiveConnectConfig()
          if not llm_request.live_connect_config.history_config:
            llm_request.live_connect_config.history_config = (
                types.HistoryConfig()
            )
          if (
              llm_request.live_connect_config.history_config.initial_history_in_client_content
              is None
          ):
            llm_request.live_connect_config.history_config.initial_history_in_client_content = (
                True
            )

        logger.info(
            'Establishing live connection for agent: %s',
            invocation_context.agent.name,
        )
        async with llm.connect(llm_request) as llm_connection:
          # Reset retry count to allow the maximum reconnect attempts for
          # subsequent connection drops.
          attempt = 1
          # Skip sending history if we are resuming a session. The server
          # already has the state associated with the resumption handle.
          if (
              llm_request.contents
              and not invocation_context.live_session_resumption_handle
          ):
            # Sends the conversation history to the model.
            with tracer.start_as_current_span('send_data'):
              # Combine regular contents with audio/transcription from session
              logger.debug('Sending history to model: %s', llm_request.contents)
              await llm_connection.send_history(llm_request.contents)
              trace_send_data(
                  invocation_context, event_id, llm_request.contents
              )

          send_task = asyncio.create_task(
              self._send_to_model(llm_connection, invocation_context)
          )

          try:
            async with Aclosing(
                self._receive_from_model(
                    llm_connection,
                    event_id,
                    invocation_context,
                    llm_request,
                )
            ) as agen:
              async for event in agen:
                # Empty event means the queue is closed.
                if not event:
                  break
                logger.debug('Receive new event: %s', event)
                yield event
                # send back the function response to models
                if event.get_function_responses():
                  logger.debug(
                      'Sending back last function response event: %s', event
                  )
                  invocation_context.live_request_queue.send_content(
                      event.content
                  )
                # We handle agent transfer here in `run_live` rather than
                # in `_postprocess_live` to prevent duplication of function
                # response processing. If agent transfer were handled in
                # `_postprocess_live`, events yielded from child agent's
                # `run_live` would bubble up to parent agent's `run_live`,
                # causing `event.get_function_responses()` to be true in both
                # child and parent, and `send_content()` to be called twice for
                # the same function response. By handling agent transfer here,
                # we ensure that only child agent processes its own function
                # responses after the transfer.
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].function_response
                    and event.content.parts[0].function_response.name
                    == 'transfer_to_agent'
                ):
                  await asyncio.sleep(DEFAULT_TRANSFER_AGENT_DELAY)
                  # cancel the tasks that belongs to the closed connection.
                  send_task.cancel()
                  logger.debug('Closing live connection')
                  await llm_connection.close()
                  logger.debug('Live connection closed.')
                  # transfer to the sub agent.
                  transfer_to_agent = event.actions.transfer_to_agent
                  if transfer_to_agent:
                    logger.debug('Transferring to agent: %s', transfer_to_agent)
                    agent_to_run = self._get_agent_to_run(
                        invocation_context, transfer_to_agent
                    )
                    child_ctx = invocation_context.model_copy()
                    # Child Live agent should start a new Live session.
                    # Do not reuse the parent session's resumption handle.
                    child_ctx.live_session_resumption_handle = None

                    if child_ctx.run_config:
                      child_ctx.run_config = child_ctx.run_config.model_copy(
                          deep=True
                      )
                      if child_ctx.run_config.session_resumption:
                        child_ctx.run_config.session_resumption.handle = None

                    async with Aclosing(
                        agent_to_run.run_live(child_ctx)
                    ) as agen:
                      async for item in agen:
                        yield item
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].function_response
                    and event.content.parts[0].function_response.name
                    == 'task_completed'
                ):
                  # this is used for sequential agent to signal the end of the agent.
                  await asyncio.sleep(DEFAULT_TASK_COMPLETION_DELAY)
                  # cancel the tasks that belongs to the closed connection.
                  send_task.cancel()
                  return
          finally:
            # Clean up
            if not send_task.done():
              send_task.cancel()
            try:
              await send_task
            except asyncio.CancelledError:
              pass
            # Clean up any parallel live before-model-callback tasks.
            await self._cleanup_live_input_screen_tasks(invocation_context)
      except (ConnectionClosed, ConnectionClosedOK) as e:
        # A before/after-model callback block intentionally closed the session:
        # this connection close is expected, so end the live flow gracefully
        # (no reconnect, no error) regardless of whether session resumption is
        # configured.
        if invocation_context._live_callback_session_ended:
          logger.info('Live session ended by a model-callback block.')
          return
        # If we have a session resumption handle, we attempt to reconnect.
        # This handle is updated dynamically during the session.
        if invocation_context.live_session_resumption_handle:
          if attempt > DEFAULT_MAX_RECONNECT_ATTEMPTS:
            logger.error('Max reconnection attempts reached (%s).', e)
            raise
          logger.info(
              'Connection closed (%s), reconnecting with session handle.', e
          )
          continue
        logger.error('Connection closed: %s.', e)
        raise
      except errors.APIError as e:
        # A model-callback block intentionally closed the session; the
        # resulting close (typically code 1000) is expected, so exit gracefully.
        if invocation_context._live_callback_session_ended:
          logger.info('Live session ended by a model-callback block.')
          return
        # Error code 1000, 1006 and 1011 indicates a recoverable connection drop.
        # In that case, we attempt to reconnect with session handle if available.
        if e.code in [1000, 1006, 1011]:
          if invocation_context.live_session_resumption_handle:
            if attempt > DEFAULT_MAX_RECONNECT_ATTEMPTS:
              logger.error('Max reconnection attempts reached (%s).', e)
              raise
            logger.info(
                'Connection lost (%s), reconnecting with session handle.', e
            )
            continue

        logger.error('APIError in live flow: %s', e)
        raise
      except Exception as e:
        logger.error(
            'An unexpected error occurred in live flow: %s', e, exc_info=True
        )
        raise

  def _build_live_before_model_request(
      self,
      invocation_context: InvocationContext,
      content: types.Content,
  ) -> LlmRequest:
    """Builds the LlmRequest passed to before-model callbacks in live mode.

    To give callbacks the same fidelity they have in non-live mode, the request
    is seeded from the live turn's base request (model, config, system
    instruction, tools) when available, with the current turn's content
    appended. Falls back to a content-only request if no base request was
    recorded.
    """
    base_request = invocation_context._live_llm_request
    if base_request is not None:
      llm_request = base_request.model_copy(deep=True)
      if llm_request.contents is None:
        llm_request.contents = []
      llm_request.contents.append(content)
      return llm_request
    return LlmRequest(contents=[content])

  async def _handle_live_before_model_callback(
      self,
      invocation_context: InvocationContext,
      content: types.Content,
  ) -> Optional[LlmResponse]:
    """Runs before-model callbacks for live input (marked as a live call).

    Reuses the unary callback machinery but marks the per-call callback context
    as live (``CallbackContext.is_live``), so callbacks can branch on
    transport. The request is seeded from the live turn's base request so
    callbacks see the full model/config/tools, matching non-live fidelity.

    Args:
      invocation_context: The invocation context.
      content: The user content being sent to the live model.

    Returns:
      An LlmResponse if a callback returns a replacement (block), else None.
    """
    llm_request = self._build_live_before_model_request(
        invocation_context, content
    )
    model_response_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
    )
    return await _handle_before_model_callback(
        invocation_context,
        llm_request,
        model_response_event,
        is_live_call=True,
    )

  async def _fire_live_voice_before_model_callback(
      self,
      invocation_context: InvocationContext,
  ) -> None:
    """Fires before-model callbacks once at the start of a voice activity.

    Voice (audio) input streams as blobs with no pre-model text, so the
    content-path before-model seam never runs for spoken turns. To still fire
    before-model callbacks once per spoken turn for observe-only use cases
    (logging, metrics, request inspection), this runs the canonical before-model
    callbacks (plugins then agent callbacks) with the seeded base request and no
    user text. It is **observe-only**: any returned replacement is ignored,
    because the audio is already streaming to the model and cannot be blocked
    pre-model (spoken input is screened post-model on its transcription). Fires
    on the send side, independent of the model's response. Exceptions are logged
    and swallowed so the send loop is never disrupted.
    """
    # No user text for voice: seed the request from the base (model/config/
    # tools) with an empty user content so callbacks see full fidelity minus the
    # prompt text.
    llm_request = self._build_live_before_model_request(
        invocation_context,
        types.Content(role='user', parts=[]),
    )
    model_response_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
    )
    try:
      # Return value intentionally discarded: voice cannot be blocked pre-model.
      await _handle_before_model_callback(
          invocation_context,
          llm_request,
          model_response_event,
          is_live_call=True,
      )
    except Exception:  # pylint: disable=broad-except
      logger.exception('Live voice before-model callback failed.')

  async def _handle_live_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    """Runs after-model callbacks for live output (marked as a live call).

    Args:
      invocation_context: The invocation context.
      llm_response: The consolidated model response for the turn.
      model_response_event: The model response event for the callback context.

    Returns:
      An altered/replaced LlmResponse if a callback modifies it, else None.
    """
    return await _handle_after_model_callback(
        invocation_context,
        llm_response,
        model_response_event,
        is_live_call=True,
    )

  def _spawn_live_before_model_callback(
      self,
      invocation_context: InvocationContext,
      content: types.Content,
  ) -> None:
    """Runs the before-model callback concurrently with forwarding live input.

    Spawns a background task that runs the before-model callbacks (marked as a
    live call). If a callback returns a replacement response, the task records
    a pending block on the invocation context and appends the event; the
    receive loop honors the signal by stopping generation.
    """

    async def _screen() -> None:
      try:
        blocked_response = await self._handle_live_before_model_callback(
            invocation_context, content
        )
        if blocked_response is not None:
          invocation_context._live_callback_blocked_response = blocked_response
          await self._append_live_blocked_event(
              invocation_context, blocked_response
          )
      except asyncio.CancelledError:
        raise
      except Exception:  # pylint: disable=broad-except
        logger.exception('Live before-model callback failed.')

    task = asyncio.create_task(_screen())
    invocation_context._live_input_screen_tasks.append(task)

  async def _cleanup_live_input_screen_tasks(
      self,
      invocation_context: InvocationContext,
  ) -> None:
    """Cancels and drains any pending parallel before-model-callback tasks."""
    tasks = invocation_context._live_input_screen_tasks
    if not tasks:
      return
    for task in tasks:
      if not task.done():
        task.cancel()
    for task in tasks:
      try:
        await task
      except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
        pass
    invocation_context._live_input_screen_tasks = []

  async def _append_live_blocked_event(
      self,
      invocation_context: InvocationContext,
      blocked_response: LlmResponse,
  ) -> None:
    """Appends a callback-block event to the session for a live turn."""
    blocked_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        content=blocked_response.content,
    )
    await invocation_context.session_service.append_event(
        session=invocation_context.session,
        event=blocked_event,
    )

  async def _send_to_model(
      self,
      llm_connection: BaseLlmConnection,
      invocation_context: InvocationContext,
  ):
    """Sends data to model."""
    while True:
      live_request_queue = invocation_context.live_request_queue
      live_request = await live_request_queue.get()
      # duplicate the live_request to all the active streams
      logger.debug(
          'Sending live request %s to active streams: %s',
          live_request,
          invocation_context.active_streaming_tools,
      )
      if invocation_context.active_streaming_tools:
        for active_streaming_tool in (
            invocation_context.active_streaming_tools
        ).values():
          if active_streaming_tool.stream:
            active_streaming_tool.stream.send(live_request)
      # Yield to event loop for cooperative multitasking
      await asyncio.sleep(0)

      if live_request.close:
        await llm_connection.close()
        return

      if live_request.activity_start:
        # Start of a voice activity (manual VAD): fire before-model callbacks
        # once for this spoken turn (observe-only), then disarm so audio chunks
        # in the same activity do not re-fire.
        if invocation_context._live_voice_before_model_armed:
          invocation_context._live_voice_before_model_armed = False
          await self._fire_live_voice_before_model_callback(invocation_context)
        await llm_connection.send_realtime(types.ActivityStart())
      elif live_request.activity_end:
        # End of a voice activity: re-arm so the next spoken turn fires once.
        invocation_context._live_voice_before_model_armed = True
        await llm_connection.send_realtime(types.ActivityEnd())
      elif live_request.blob:
        # First audio blob of a voice activity (automatic VAD, no activity_start
        # on the send side): fire before-model callbacks once for this spoken
        # turn (observe-only), then disarm so subsequent chunks do not re-fire.
        # Re-armed at the turn boundary by the receive loop.
        if invocation_context._live_voice_before_model_armed:
          invocation_context._live_voice_before_model_armed = False
          await self._fire_live_voice_before_model_callback(invocation_context)
        # Cache input audio chunks before flushing
        self.audio_cache_manager.cache_audio(
            invocation_context, live_request.blob, cache_type='input'
        )

        await llm_connection.send_realtime(live_request.blob)

      if live_request.content:
        content = live_request.content
        # Persist user text content to session (similar to non-live mode)
        # Skip function responses - they are already handled separately
        is_function_response = content.parts and any(
            part.function_response for part in content.parts
        )
        if not is_function_response:
          if not content.role:
            content.role = 'user'
          # Live before-model callback seam. Two modes:
          #
          # - blocking: run the before-model callback BEFORE forwarding the
          #   input to the model. A returned replacement response prevents the
          #   model from ever seeing the input (no leading-output leak), at the
          #   cost of one round-trip of latency per turn.
          #
          # - parallel (default, latency-optimized): forward the input to the
          #   model immediately and run the callback concurrently. A returned
          #   replacement sets a cross-task block signal that the receive loop
          #   honors by stopping generation and suppressing output. Clean input
          #   adds no latency.
          if _live_input_callback_is_blocking(invocation_context):
            blocked_response = await self._handle_live_before_model_callback(
                invocation_context, content
            )
            if blocked_response is not None:
              await self._append_live_blocked_event(
                  invocation_context, blocked_response
              )
              continue
          else:
            self._spawn_live_before_model_callback(invocation_context, content)
          user_content_event = Event(
              id=Event.new_id(),
              invocation_id=invocation_context.invocation_id,
              author='user',
              content=content,
          )
          await invocation_context.session_service.append_event(
              session=invocation_context.session,
              event=user_content_event,
          )
        await llm_connection.send_content(live_request.content)

  async def _receive_from_model(
      self,
      llm_connection: BaseLlmConnection,
      event_id: str,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    """Receive data from model and process events using BaseLlmConnection."""

    def get_author_for_event(llm_response: LlmResponse) -> str:
      """Get the author of the event.

      When the model returns input transcription, the author is set to "user".
      Otherwise, the author is the agent name (not 'model').

      Args:
        llm_response: The LLM response from the LLM call.

      Returns:
        The author of the event as a string, either "user" or the agent's name.
      """
      if llm_response and (
          llm_response.input_transcription
          or (llm_response.content and llm_response.content.role == 'user')
      ):
        return 'user'
      else:
        return invocation_context.agent.name

    # Per-turn buffer accumulating model output text for the after-model
    # callback (run the running buffer per-chunk to allow an early decision,
    # plus a consolidated check at turn_complete). ``last_output_response``
    # holds the most recent full model response so the callback receives a
    # full-fidelity LlmResponse.
    output_buffer: list[str] = []
    output_blocked = False
    last_output_response: Optional[LlmResponse] = None

    while True:
      async with Aclosing(llm_connection.receive()) as agen:
        async for llm_response in agen:
          if llm_response.live_session_resumption_update:
            logger.info(
                'Update session resumption handle:'
                f' {llm_response.live_session_resumption_update}.'
            )
            invocation_context.live_session_resumption_handle = (
                llm_response.live_session_resumption_update.new_handle
            )
          if llm_response.go_away:
            logger.info(f'Received go away signal: {llm_response.go_away}')
            # The server signals that it will close the connection soon.
            # We proactively raise ConnectionClosed to trigger the reconnection
            # logic in run_live, which will use the latest session handle.
            raise ConnectionClosed(None, None)

          # A turn boundary (the model finished or the user interrupted) delimits
          # the current spoken turn. Re-arm the voice before-model callback so
          # the next voice activity fires it once. The firing itself happens on
          # the send side (activity_start / first audio blob); this only resets
          # the gate, so the before-model firing stays model-independent.
          if llm_response.turn_complete or llm_response.interrupted:
            invocation_context._live_voice_before_model_armed = True

          # Live spoken-input after-model callback seam. Spoken input is
          # transcribed *by the model*, so the transcription is itself a model
          # result: it arrives here, on the receive side, after the audio has
          # already reached the model. It is therefore screened through the
          # *after*-model callback (which inspects model results) — keeping
          # ``before_model_callback`` a true pre-model contract reserved for
          # genuinely pre-model input (typed live input, and the unary path).
          # The callback receives an LlmResponse with ``input_transcription``
          # populated so it can recognize this as the user's utterance and
          # screen it as input. A returned replacement sets the same cross-task
          # block signal the typed parallel path uses; the poll below honors it
          # by ending the turn and suppressing output. Only typed input can be
          # blocked before the model sees it.
          if not output_blocked:
            spoken_input = _live_input_transcription_text(llm_response)
            if spoken_input:
              screen_event = Event(
                  id=Event.new_id(),
                  invocation_id=invocation_context.invocation_id,
                  author='user',
              )
              blocked_response = await self._handle_live_after_model_callback(
                  invocation_context,
                  _build_live_input_transcription_response(spoken_input),
                  screen_event,
              )
              if blocked_response is not None:
                invocation_context._live_callback_blocked_response = (
                    blocked_response
                )

          # Input block: a typed parallel before-model callback or a spoken
          # after-model callback returned a replacement for the user input.
          # Stop generation, suppress in-flight output, surface the replacement
          # response, and end the turn.
          if (
              not output_blocked
              and invocation_context._live_callback_blocked_response is not None
          ):
            output_blocked = True
            input_block = invocation_context._live_callback_blocked_response
            invocation_context._live_callback_blocked_response = None
            blocked_event = Event(
                id=Event.new_id(),
                invocation_id=invocation_context.invocation_id,
                author=invocation_context.agent.name,
                content=input_block.content,
            )
            blocked_event.turn_complete = True
            yield blocked_event
            # A callback block ends the live session gracefully: surface the
            # replacement reply (above) then close. Mark it so run_live treats
            # the ensuing connection close as an intentional end, not an error.
            invocation_context._live_callback_session_ended = True
            if invocation_context.live_request_queue:
              invocation_context.live_request_queue.close()
            continue

          # Live after-model callback seam.
          if not output_blocked:
            chunk_text = _live_output_text(llm_response)
            # Track the most recent model response so the after-model callback
            # receives a full-fidelity LlmResponse (text + audio + function
            # calls), not a text-only reconstruction.
            if llm_response.content is not None:
              last_output_response = llm_response
            # The connection emits incremental `partial` deltas plus a final
            # `finished` transcription carrying the full accumulated utterance.
            # Append deltas (per-chunk early call), but *replace* the buffer
            # with the consolidated finished transcription so the utterance is
            # processed once rather than twice.
            if chunk_text:
              if _live_output_is_finished_transcription(llm_response):
                output_buffer = [chunk_text]
              else:
                output_buffer.append(chunk_text)
            is_turn_complete = bool(llm_response.turn_complete)
            # Run the after-model callback on each output chunk (early
            # decision) and again at turn_complete (consolidated check). At
            # turn completion, fire even without buffered text so non-text
            # turns (function calls, audio-only) still reach the callback with
            # full fidelity.
            buffered = ''.join(output_buffer)
            has_turn_content = (
                is_turn_complete and last_output_response is not None
            )
            if buffered or has_turn_content:
              if chunk_text or is_turn_complete:
                screen_event = Event(
                    id=Event.new_id(),
                    invocation_id=invocation_context.invocation_id,
                    author=invocation_context.agent.name,
                )
                callback_response = (
                    await self._handle_live_after_model_callback(
                        invocation_context,
                        _build_live_after_model_response(
                            last_output_response, buffered
                        ),
                        screen_event,
                    )
                )
                if callback_response is not None:
                  # Suppress the in-flight output, surface the replacement
                  # event, and end the turn.
                  output_blocked = True
                  await self._append_live_blocked_event(
                      invocation_context, callback_response
                  )
                  blocked_event = Event(
                      id=Event.new_id(),
                      invocation_id=invocation_context.invocation_id,
                      author=invocation_context.agent.name,
                      content=callback_response.content,
                  )
                  blocked_event.turn_complete = True
                  yield blocked_event
                  # Graceful callback-initiated session end (see input path).
                  invocation_context._live_callback_session_ended = True
                  if invocation_context.live_request_queue:
                    invocation_context.live_request_queue.close()
                  continue
          if output_blocked:
            # Drop any remaining in-flight output for this turn.
            if llm_response.turn_complete:
              output_buffer = []
              output_blocked = False
              last_output_response = None
            continue

          model_response_event = Event(
              id=Event.new_id(),
              invocation_id=invocation_context.invocation_id,
              author=get_author_for_event(llm_response),
          )

          async with Aclosing(
              self._postprocess_live(
                  invocation_context,
                  llm_request,
                  llm_response,
                  model_response_event,
              )
          ) as agen:
            async for event in agen:
              # Cache output audio chunks from model responses
              # TODO: support video data
              if (
                  invocation_context.run_config.save_live_blob
                  and event.content
                  and event.content.parts
                  and event.content.parts[0].inline_data
                  and event.content.parts[0].inline_data.mime_type.startswith(
                      'audio/'
                  )
              ):
                audio_blob = types.Blob(
                    data=event.content.parts[0].inline_data.data,
                    mime_type=event.content.parts[0].inline_data.mime_type,
                )
                self.audio_cache_manager.cache_audio(
                    invocation_context, audio_blob, cache_type='output'
                )

              yield event

          # Reset the callback buffers/signals at the end of a normal turn.
          if llm_response.turn_complete:
            output_buffer = []
            output_blocked = False
            last_output_response = None
            invocation_context._live_callback_blocked_response = None
      # Give opportunity for other tasks to run.
      await asyncio.sleep(0)

  async def run_async(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow."""
    while True:
      last_event = None
      async with Aclosing(self._run_one_step_async(invocation_context)) as agen:
        async for event in agen:
          last_event = event
          yield event
      if not last_event or last_event.is_final_response() or last_event.partial:
        if last_event and last_event.partial:
          logger.warning('The last event is partial, which is not expected.')
        break

  async def _run_one_step_async(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """One step means one LLM call."""
    llm_request = LlmRequest()

    # Preprocess before calling the LLM.
    async with Aclosing(
        self._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event
    if invocation_context.end_invocation:
      return

    # Resume the LLM agent based on the last event from the current branch.
    # 1. User content: continue the normal flow
    # 2. Function call: call the tool and get the response event.
    events = invocation_context._get_events(
        current_invocation=True, current_branch=True
    )

    # Long running tool calls should have been handled before this point.
    # If there are still long running tool calls, it means the agent is paused
    # before, and its branch hasn't been resumed yet.
    if invocation_context.is_resumable and events and len(events) > 1:
      pause = False
      if invocation_context.should_pause_invocation(events[-1]):
        pause = True
      elif invocation_context.should_pause_invocation(events[-2]):
        # NOTE: This only checks the last 2 events. If an LRO is followed by
        # multiple text responses, this check may not trigger correctly.
        # This is a known limitation of the current 2-event window.
        # Check if the function call in events[-2] is resolved by events[-1]
        fc_ids = {fc.id for fc in events[-2].get_function_calls()}
        fr_ids = {fr.id for fr in events[-1].get_function_responses()}
        if fc_ids and not fc_ids.issubset(fr_ids):
          pause = True

      if pause:
        return

    if (
        invocation_context.is_resumable
        and events
        and events[-1].get_function_calls()
    ):
      model_response_event = events[-1]
      async with Aclosing(
          self._postprocess_handle_function_calls_async(
              invocation_context, model_response_event, llm_request
          )
      ) as agen:
        async for event in agen:
          event.id = Event.new_id()
          yield event
        return

    # Calls the LLM.
    model_response_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        branch=invocation_context.branch,
    )
    async with Aclosing(
        self._call_llm_async(
            invocation_context, llm_request, model_response_event
        )
    ) as agen:
      async for llm_response in agen:
        # Postprocess after calling the LLM.
        async with Aclosing(
            self._postprocess_async(
                invocation_context,
                llm_request,
                llm_response,
                model_response_event,
            )
        ) as agen:
          async for event in agen:
            # Update the mutable event id to avoid conflict
            model_response_event.id = Event.new_id()
            model_response_event.timestamp = platform_time.get_time()
            yield event

  async def _preprocess_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent
    if not hasattr(agent, 'tools') or not hasattr(agent, 'canonical_model'):
      raise TypeError(
          'Expected agent to have tools and canonical_model attributes,'
          f' but got {type(agent)}'
      )

    # Runs processors.
    for processor in self.request_processors:
      async with Aclosing(
          processor.run_async(invocation_context, llm_request)
      ) as agen:
        async for event in agen:
          yield event

    # Resolve toolset authentication before tool listing.
    # This ensures credentials are ready before get_tools() is called.
    async with Aclosing(
        self._resolve_toolset_auth(invocation_context, agent)
    ) as agen:
      async for event in agen:
        yield event

    if invocation_context.end_invocation:
      return

    # Run processors for tools.
    await _process_agent_tools(invocation_context, llm_request)

  async def _postprocess_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    """Postprocess after calling the LLM.

    Args:
      invocation_context: The invocation context.
      llm_request: The original LLM request.
      llm_response: The LLM response from the LLM call.
      model_response_event: A mutable event for the LLM response.

    Yields:
      A generator of events.
    """

    # Runs processors.
    async with Aclosing(
        self._postprocess_run_processors_async(invocation_context, llm_response)
    ) as agen:
      async for event in agen:
        yield event

    # Skip the model response event if there is no content and no error code.
    # This is needed for the code executor to trigger another loop.
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
        and not llm_response.grounding_metadata
    ):
      return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():

      # Skip partial function call events - they should not trigger execution
      # since partial events are not saved to session (see runners.py).
      # Only execute function calls in the non-partial events.
      if model_response_event.partial:
        return

      async with Aclosing(
          self._postprocess_handle_function_calls_async(
              invocation_context, model_response_event, llm_request
          )
      ) as agen:
        async for event in agen:
          yield event

  async def _postprocess_live(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    """Postprocess after calling the LLM asynchronously.

    Args:
      invocation_context: The invocation context.
      llm_request: The original LLM request.
      llm_response: The LLM response from the LLM call.
      model_response_event: A mutable event for the LLM response.

    Yields:
      A generator of events.
    """

    # Runs processors.
    async with Aclosing(
        self._postprocess_run_processors_async(invocation_context, llm_response)
    ) as agen:
      async for event in agen:
        yield event

    # Skip the model response event if there is no content and no error code.
    # This is needed for the code executor to trigger another loop.
    # But don't skip control events like turn_complete or transcription events.
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
        and not llm_response.turn_complete
        and not llm_response.input_transcription
        and not llm_response.output_transcription
        and not llm_response.usage_metadata
        and not llm_response.live_session_resumption_update
        and not llm_response.grounding_metadata
    ):
      return

    # Handle session resumption updates for cross-connection resumption
    if llm_response.live_session_resumption_update:
      model_response_event.live_session_resumption_update = (
          llm_response.live_session_resumption_update
      )
      yield model_response_event
      return

    # Handle transcription events ONCE per llm_response, outside the event loop
    if llm_response.input_transcription:
      model_response_event.input_transcription = (
          llm_response.input_transcription
      )
      model_response_event.partial = llm_response.partial
      yield model_response_event
      return

    if llm_response.output_transcription:
      model_response_event.output_transcription = (
          llm_response.output_transcription
      )
      model_response_event.partial = llm_response.partial
      yield model_response_event
      return

    # Flush audio caches based on control events using configurable settings
    if invocation_context.run_config.save_live_blob:
      flushed_events = await self._handle_control_event_flush(
          invocation_context, llm_response
      )
      for event in flushed_events:
        yield event
      if flushed_events:
        # NOTE below return is O.K. for now, because currently we only flush
        # events on interrupted or turn_complete. turn_complete is a pure
        # control event and interrupted is not with content but those content
        # is ignorable because model is already interrupted. If we have other
        # case to flush events in the future that are not pure control events,
        # we should not return here.
        return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():
      function_response_event = await functions.handle_function_calls_live(
          invocation_context, model_response_event, llm_request.tools_dict
      )
      # Always yield the function response event first
      yield function_response_event

      # Check if this is a set_model_response function response
      if json_response := _output_schema_processor.get_structured_model_response(
          function_response_event
      ):
        # Create and yield a final model response event
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                invocation_context, json_response
            )
        )
        yield final_event

  async def _postprocess_run_processors_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    for processor in self.response_processors:
      async with Aclosing(
          processor.run_async(invocation_context, llm_response)
      ) as agen:
        async for event in agen:
          yield event

  async def _postprocess_handle_function_calls_async(
      self,
      invocation_context: InvocationContext,
      function_call_event: Event,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    if function_response_event := await functions.handle_function_calls_async(
        invocation_context, function_call_event, llm_request.tools_dict
    ):
      auth_event = functions.generate_auth_event(
          invocation_context, function_response_event
      )
      if auth_event:
        yield auth_event

        # Interrupt invocation (mirrors _resolve_toolset_auth behavior)
        invocation_context.end_invocation = True

      tool_confirmation_event = functions.generate_request_confirmation_event(
          invocation_context, function_call_event, function_response_event
      )
      if tool_confirmation_event:
        yield tool_confirmation_event

      # Always yield the function response event first
      yield function_response_event

      # Check if this is a set_model_response function response
      if json_response := _output_schema_processor.get_structured_model_response(
          function_response_event
      ):
        # Create and yield a final model response event
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                invocation_context, json_response
            )
        )
        yield final_event

      # NOTE: This recursive nested execution block is preserved as a backward-compatible
      # fallback for deprecated execution paths (such as legacy `SequentialAgent`) that
      # do not run under the modern ADK 2.0 `DynamicNodeScheduler`.
      #
      # In modern resumable workflow environments, this block is safely bypassed
      # because the scheduler wrapper (e.g., `_llm_agent_wrapper.py`) intercepts the
      # `transfer_to_agent` action at the outer execution frame and exits, returning
      # control to the top-level coordinator.
      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async with Aclosing(agent_to_run.run_async(invocation_context)) as agen:
          async for event in agen:
            yield event

  def _get_agent_to_run(
      self, invocation_context: InvocationContext, agent_name: str
  ) -> BaseAgent:
    root_agent = invocation_context.agent.root_agent
    agent_to_run = root_agent.find_agent(agent_name)
    if not agent_to_run:
      raise ValueError(f'Agent {agent_name} not found in the agent tree.')

    from google.adk.agents.llm_agent import LlmAgent

    if (
        isinstance(invocation_context.agent, LlmAgent)
        and invocation_context.agent.disallow_transfer_to_peers
        and agent_to_run.parent_agent == invocation_context.agent.parent_agent
        and agent_to_run != invocation_context.agent
    ):
      raise ValueError(f'Transfer to sibling agent {agent_name} is disallowed.')
    return agent_to_run

  async def _call_llm_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:

    async def _call_llm_with_tracing() -> AsyncGenerator[LlmResponse, None]:
      with tracer.start_as_current_span('call_llm') as span:
        # Runs before_model_callback inside the call_llm span so
        # plugins observe the same span as after/error callbacks.
        if response := await self._handle_before_model_callback(
            invocation_context, llm_request, model_response_event
        ):
          yield response
          return

        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.labels = llm_request.config.labels or {}

        # Add agent name as a label to the llm_request. This will help
        # with slicing billing reports on a per-agent basis.
        if _ADK_AGENT_NAME_LABEL_KEY not in llm_request.config.labels:
          llm_request.config.labels[_ADK_AGENT_NAME_LABEL_KEY] = (
              invocation_context.agent.name
          )

        # Calls the LLM.
        llm = self.__get_llm(invocation_context)

        if invocation_context.run_config.support_cfc:
          invocation_context.live_request_queue = LiveRequestQueue()
          responses_generator = self.run_live(invocation_context)
          async with Aclosing(
              self._run_and_handle_error(
                  responses_generator,
                  invocation_context,
                  llm_request,
                  model_response_event,
                  call_llm_span=span,
              )
          ) as agen:
            async for llm_response in agen:
              # Rebind to call_llm span for after_model_callback.
              with trace.use_span(span, end_on_exit=False):
                if altered := (
                    await self._handle_after_model_callback(
                        invocation_context,
                        llm_response,
                        model_response_event,
                    )
                ):
                  llm_response = altered
              # only yield partial response in SSE streaming mode
              if (
                  invocation_context.run_config.streaming_mode
                  == StreamingMode.SSE
                  or not llm_response.partial
              ):
                yield llm_response
              if llm_response.turn_complete:
                invocation_context.live_request_queue.close()
        else:
          # Check if we can make this llm call or not. If the current
          # call pushes the counter beyond the max set value, then the
          # execution is stopped right here, and exception is thrown.
          invocation_context.increment_llm_call_count()
          responses_generator = llm.generate_content_async(
              llm_request,
              stream=invocation_context.run_config.streaming_mode
              == StreamingMode.SSE,
          )
          async with Aclosing(
              self._run_and_handle_error(
                  responses_generator,
                  invocation_context,
                  llm_request,
                  model_response_event,
                  call_llm_span=span,
              )
          ) as agen:
            async for llm_response in agen:
              trace_call_llm(
                  invocation_context,
                  model_response_event.id,
                  llm_request,
                  llm_response,
                  span,
              )
              # Rebind to call_llm span for after_model_callback.
              with trace.use_span(span, end_on_exit=False):
                if altered := (
                    await self._handle_after_model_callback(
                        invocation_context,
                        llm_response,
                        model_response_event,
                    )
                ):
                  llm_response = altered

              yield llm_response

    async with Aclosing(_call_llm_with_tracing()) as agen:
      async for event in agen:
        yield event

  def _finalize_model_response_event(
      self,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Event:
    return _finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )

  async def _resolve_toolset_auth(
      self,
      invocation_context: InvocationContext,
      agent: LlmAgent,
  ) -> AsyncGenerator[Event, None]:
    async with Aclosing(
        _resolve_toolset_auth(invocation_context, agent)
    ) as agen:
      async for event in agen:
        yield event

  async def _handle_before_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    return await _handle_before_model_callback(
        invocation_context, llm_request, model_response_event
    )

  async def _handle_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    return await _handle_after_model_callback(
        invocation_context, llm_response, model_response_event
    )

  async def _run_and_handle_error(
      self,
      response_generator: AsyncGenerator[LlmResponse, None],
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
      call_llm_span: Optional[trace.Span] = None,
  ) -> AsyncGenerator[LlmResponse, None]:
    async with Aclosing(
        _run_and_handle_error(
            response_generator,
            invocation_context,
            llm_request,
            model_response_event,
            call_llm_span=call_llm_span,
        )
    ) as agen:
      async for response in agen:
        yield response

  async def _handle_control_event_flush(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> list[Event]:
    """Handle audio cache flushing based on control events.

    Args:
      invocation_context: The invocation context containing audio caches.
      llm_response: The LLM response containing control event information.

    Returns:
      A list of Event objects created from the flushed caches.
    """

    # Log cache statistics if enabled
    if DEFAULT_ENABLE_CACHE_STATISTICS:
      stats = self.audio_cache_manager.get_cache_stats(invocation_context)
      logger.debug('Audio cache stats: %s', stats)

    if llm_response.interrupted:
      # user interrupts so the model will stop. we can flush model audio here
      return await self.audio_cache_manager.flush_caches(
          invocation_context,
          flush_user_audio=False,
          flush_model_audio=True,
      )
    elif llm_response.turn_complete:
      # turn completes so we can flush both user and model
      return await self.audio_cache_manager.flush_caches(
          invocation_context,
          flush_user_audio=True,
          flush_model_audio=True,
      )
    # TODO: Once generation_complete is surfaced on LlmResponse, we can flush
    # model audio here (flush_user_audio=False, flush_model_audio=True).
    return []

  def __get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    agent = invocation_context.agent

    # Check for conformance test replay mode
    if config := invocation_context.session.state.get('_adk_replay_config'):
      from ...cli.conformance._conformance_test_google_llm import _ConformanceTestGemini

      # Models are stateless, so the current replay state is cached in the
      # session state to maintain the state across model calls
      # key: (agent_name, user_message_index)
      # value: replay index
      user_message_index = config.get('user_message_index')
      replay_indexes = config.get('_adk_replay_indexes', {})
      if (agent.name, user_message_index) not in replay_indexes:
        replay_indexes[(agent.name, user_message_index)] = 0
      current_replay_index = replay_indexes[(agent.name, user_message_index)]

      config['current_replay_index'] = current_replay_index
      config['agent_name'] = agent.name
      model = _ConformanceTestGemini(
          config=config,
      )

      replay_indexes[(agent.name, user_message_index)] = (
          current_replay_index + 1
      )
      config['_adk_replay_indexes'] = replay_indexes
      return model

    if invocation_context.live_request_queue is not None:
      return agent.canonical_live_model

    if not hasattr(agent, 'canonical_model'):
      raise TypeError(
          'Expected agent to have canonical_model attribute,'
          f' but got {type(agent)}'
      )
    return agent.canonical_model
