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

"""Per-turn OpenTelemetry tracing for live (speech-to-speech) sessions.

A live session is a single long-lived bidirectional connection over which many
conversational turns flow, each as a stream of small audio/text chunks. Emitting
one span per chunk floods the trace; :class:`LiveTurnTracer` instead aggregates
each turn's chunks into a small, readable span tree:

    live_turn                       one per conversational turn
    ├─ user                         input transcript + input audio reference
    └─ assistant                    output transcript + output audio reference,
       └─ execute_tool {tool}       token usage, time-to-first-chunk, tool calls

Only signals that appear are recorded: the ``user`` span is opened only when
there is user-side input (a transcript or captured audio), and the ``assistant``
span is opened on the first model output of the turn.

Span durations are meaningful: the ``live_turn`` and ``user`` spans start at the
user's first audio chunk (so the ``user`` span duration ~ the length of the
user's utterance), the ``user`` span closes when the model starts responding,
and the ``assistant`` span spans the model's response. Time-to-first-chunk is
recorded as an attribute on the ``assistant`` span.

Schema version:

  The live-turn span tree is emitted only under the OTel-semconv-aligned
  telemetry schema (``ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=2``; the default on
  Agent Engine). Under the legacy schema the tracer is a no-op, so the live
  path behaves exactly as before. See :mod:`...telemetry._schema_version`.

Turn boundaries and deferred finalize:

  The model signals the end of a turn with ``turn_complete`` / ``interrupted``,
  but the turn's token usage often arrives in a trailing usage-only response
  *after* that signal. To capture accurate tokens, a boundary marks the turn
  *pending finalize* rather than closing it immediately. The turn is finalized
  (usage stamped, spans closed) when the trailing usage arrives, when the next
  turn begins, or when the session ends — whichever comes first.

Tool calls:

  A tool call spans two model generations: the model pauses to call the tool
  (ending that generation with ``turn_complete``), then speaks the answer as a
  second generation. Both carry ``turn_complete``, so the tracer detects the
  function call in the first generation's output and treats the following
  ``turn_complete`` as a handoff — keeping the ``live_turn`` open so the answer
  becomes a second ``assistant`` span under the same turn. A barge-in interrupt
  ends the turn even mid-round-trip.

The tracer is intentionally not attached to :class:`InvocationContext`: its
state is scoped to a single ``run_live`` call and must not leak across
invocations, so ``run_live`` owns one instance for the session's lifetime.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from google.genai import types
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Span

from . import _metrics
from ._schema_version import resolve_schema_version
from ._schema_version import SCHEMA_VERSION_SEMCONV_ALIGNED
from .tracing import set_live_assistant_span_attributes
from .tracing import set_live_turn_span_attributes
from .tracing import set_live_user_span_attributes
from .tracing import tracer

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..models.llm_response import LlmResponse

# Span names. Kept ADK-native and aligned with the roles in a voice turn; the
# OTel `gen_ai.operation.name` attribute is stamped by the tracing setters.
LIVE_TURN_SPAN_NAME = 'live_turn'
USER_SPAN_NAME = 'user'
ASSISTANT_SPAN_NAME = 'assistant'

# Scalar token-count fields summed when combining per-generation usage.
_USAGE_SCALAR_FIELDS = (
    'prompt_token_count',
    'cached_content_token_count',
    'candidates_token_count',
    'thoughts_token_count',
    'tool_use_prompt_token_count',
    'total_token_count',
)
# Per-modality detail lists merged (by modality) when combining usage.
_USAGE_DETAIL_FIELDS = (
    'prompt_tokens_details',
    'candidates_tokens_details',
    'cache_tokens_details',
    'tool_use_prompt_tokens_details',
)


def _has_function_call(llm_response: LlmResponse) -> bool:
  """Whether an LlmResponse's content contains any function call parts."""
  content = llm_response.content
  if content is None or not content.parts:
    return False
  return any(part.function_call is not None for part in content.parts)


def _sum_optional(a: int | None, b: int | None) -> int | None:
  if a is None and b is None:
    return None
  return (a or 0) + (b or 0)


def _merge_modality_details(
    a: list[types.ModalityTokenCount] | None,
    b: list[types.ModalityTokenCount] | None,
) -> list[types.ModalityTokenCount] | None:
  """Merges two lists of ModalityTokenCount, summing counts per modality."""
  if not a and not b:
    return None
  totals: dict[types.MediaModality, int] = {}
  for entry in list(a or []) + list(b or []):
    if entry.token_count is None:
      continue
    totals[entry.modality] = totals.get(entry.modality, 0) + entry.token_count
  return [
      types.ModalityTokenCount(modality=modality, token_count=count)
      for modality, count in totals.items()
  ]


def _merge_usage_metadata(
    a: types.GenerateContentResponseUsageMetadata | None,
    b: types.GenerateContentResponseUsageMetadata | None,
) -> types.GenerateContentResponseUsageMetadata | None:
  """Sums two usage-metadata objects (scalars and per-modality details).

  Used to combine the token counts of a tool-using turn's separate model
  generations into a single per-turn total. Returns the non-None argument when
  the other is None.
  """
  if a is None:
    return b
  if b is None:
    return a
  merged = types.GenerateContentResponseUsageMetadata()
  for field in _USAGE_SCALAR_FIELDS:
    value = _sum_optional(getattr(a, field), getattr(b, field))
    if value is not None:
      setattr(merged, field, value)
  for field in _USAGE_DETAIL_FIELDS:
    details = _merge_modality_details(getattr(a, field), getattr(b, field))
    if details is not None:
      setattr(merged, field, details)
  # Preserve traffic_type if either side carries it.
  merged.traffic_type = a.traffic_type or b.traffic_type
  return merged


class LiveTurnTracer:
  """Aggregates a live session's chunk stream into a per-turn span tree.

  Under the legacy telemetry schema the tracer is a no-op: every method returns
  without creating spans, so the live path is unchanged. The span tree is
  emitted only under the OTel-semconv-aligned schema.
  """

  def __init__(
      self,
      invocation_context: InvocationContext,
      parent_context: context_api.Context | None = None,
  ):
    self._invocation_context = invocation_context
    # Explicit parent OTel context for the `live_turn` span. In a concurrent
    # asyncio runtime the implicit "current span" is unreliable, and across a
    # multi-agent live session (agent transfer, sequential/loop sub-agents,
    # workflow nodes) each agent's turns must nest under that agent's/node's
    # span rather than float to the invocation root. `run_live` resolves the
    # right parent (the per-agent or node span) and passes it here. When None,
    # OTel's current context is used (single-agent case).
    self._parent_context = parent_context
    # The span tree is only emitted under the semconv-aligned schema; when
    # disabled every handler short-circuits so the live path is unchanged.
    self._enabled = resolve_schema_version() == SCHEMA_VERSION_SEMCONV_ALIGNED
    self._turn_span: Span | None = None
    self._user_span: Span | None = None
    self._assistant_span: Span | None = None
    # Turn awaiting its trailing usage-only response before it can close.
    self._pending_finalize = False
    # Accumulators stamped onto the assistant span at finalize.
    # Usage summed across the turn's generations (folded in at each generation
    # boundary), plus the current generation's latest (cumulative, last-wins)
    # usage that has not yet been folded in.
    self._usage_total: types.GenerateContentResponseUsageMetadata | None = None
    self._current_gen_usage: (
        types.GenerateContentResponseUsageMetadata | None
    ) = None
    self._finish_reason: types.FinishReason | None = None
    # Monotonic timestamp of the user's committed audio for the pending turn,
    # used to compute time-to-first-chunk. None when no turn is awaiting its
    # first model output.
    self._ttft_started_at: float | None = None
    self._ttft_recorded = False
    # Epoch-ns timestamp of the user's first audio chunk this turn. Used to
    # backdate the `live_turn` and `user` span start times so their durations
    # reflect the real turn / utterance window (not when telemetry first
    # observed a transcript). None until the user starts speaking.
    self._turn_started_at_ns: int | None = None
    # Whether the user span has already been closed at first model output.
    self._user_span_closed = False
    # Whether the model has produced output this turn. Once true, the user's
    # turn is over: late input transcripts/audio route to the turn span rather
    # than (re)opening a user span.
    self._model_responded = False
    # Whether a tool round-trip is in progress this turn. The Live API ends a
    # generation (turn_complete) to call a tool, then generates again after the
    # tool result; both carry turn_complete. When a tool response has been sent
    # back, the next turn_complete is a handoff, not the end of the turn, so we
    # keep the `live_turn` open and let the follow-up answer add a second
    # `assistant` span under it.
    self._tool_call_pending = False

  @property
  def assistant_span(self) -> Span | None:
    """The open assistant span, so callers can nest tool spans under it."""
    return self._assistant_span

  def on_user_audio(self) -> None:
    """Arms the TTFT timer for the next turn on user speech.

    Called for each user audio chunk sent to the model. Only the first chunk
    after a turn boundary arms the timer, so the measured latency starts from
    when the user *began* the utterance the model will respond to. If the user
    starts speaking again while a previous turn is pending finalize, that turn
    is finalized first.
    """
    if not self._enabled:
      return
    if self._pending_finalize:
      self._finalize_turn()
    if self._ttft_started_at is None:
      self._ttft_started_at = time.monotonic()
    if self._turn_started_at_ns is None:
      self._turn_started_at_ns = time.time_ns()

  def on_model_output(self, llm_response: LlmResponse) -> Span | None:
    """Ensures the turn is open, records TTFT, and accumulates turn signals.

    Args:
      llm_response: The model response for this chunk. Its usage metadata and
        finish reason (when present) are accumulated for the turn.

    Returns:
      The open assistant span (so the caller can nest tool spans under it), or
      None when the tracer is disabled.
    """
    if not self._enabled:
      return None
    # A new turn's output after a pending finalize means the trailing usage of
    # the previous turn never arrived; finalize it before starting the new one.
    if self._pending_finalize:
      self._finalize_turn()

    self._ensure_turn_span()

    # If this generation issued a function call, the upcoming turn_complete is a
    # tool handoff, not the turn's end. Detecting it here (the tool-call content
    # always precedes its turn_complete) keeps the logic independent of when the
    # flow signals the function-response send.
    if _has_function_call(llm_response):
      self._tool_call_pending = True

    now_ns = time.time_ns()
    # A fresh assistant span is needed either at the first model output of the
    # turn or after a tool handoff closed the previous one. Backdate its start
    # to now so its duration ~ this generation's response time.
    new_assistant_span = self._assistant_span is None

    # The user's turn ends when the model starts responding: close the user
    # span (if open) at first model output so its duration ~ utterance length.
    if self._user_span is not None and not self._user_span_closed:
      self._user_span.end(end_time=now_ns)
      self._user_span_closed = True
    self._model_responded = True

    span = self._ensure_assistant_span(
        start_time=now_ns if new_assistant_span else None
    )

    if not self._ttft_recorded and self._ttft_started_at is not None:
      elapsed_s = time.monotonic() - self._ttft_started_at
      set_live_assistant_span_attributes(
          span,
          self._invocation_context,
          time_to_first_token_s=elapsed_s,
      )
      _metrics.record_live_time_to_first_token(
          agent_name=self._invocation_context.agent.name,
          model=self._model_name(),
          elapsed_s=elapsed_s,
      )
      self._ttft_recorded = True

    self._accumulate_usage(llm_response)
    return span

  def on_usage(self, llm_response: LlmResponse) -> None:
    """Accumulates a usage-only response, finalizing a pending turn.

    The trailing usage-only response is what lets us stamp accurate per-turn
    token counts on the assistant span; receiving it completes a pending turn.
    """
    if not self._enabled:
      return
    self._accumulate_usage(llm_response)
    if self._pending_finalize:
      self._finalize_turn()

  def on_input_transcript(self, transcript: str) -> None:
    """Attaches the user's (final) input transcript to the user span.

    Input transcription usually arrives before the model starts responding, so
    it lands on the (still-open) user span. If it arrives *after* the user span
    has already closed at first model output, it falls back to the `live_turn`
    span, since attributes cannot be added to an ended span.
    """
    if not self._enabled or not transcript:
      return
    if self._pending_finalize:
      self._finalize_turn()
    self._ensure_turn_span()
    if self._model_responded:
      # The user's turn is over (late transcript) — fall back to the turn span,
      # since the user span has closed and attributes can't be added after end.
      set_live_turn_span_attributes(
          self._turn_span,
          self._invocation_context,
          model=self._model_name(),
      )
      set_live_user_span_attributes(
          self._turn_span, self._invocation_context, transcript=transcript
      )
    else:
      set_live_user_span_attributes(
          self._ensure_user_span(),
          self._invocation_context,
          transcript=transcript,
      )

  def on_output_transcript(self, transcript: str) -> None:
    """Attaches the model's (final) output transcript to the assistant span."""
    if not self._enabled or not transcript:
      return
    self._ensure_turn_span()
    set_live_assistant_span_attributes(
        self._ensure_assistant_span(),
        self._invocation_context,
        model=self._model_name(),
        transcript=transcript,
    )

  def on_audio_reference(self, *, audio_ref: str, is_input: bool) -> None:
    """Attaches an input/output audio artifact reference to the right span."""
    if not self._enabled or not audio_ref:
      return
    self._ensure_turn_span()
    if is_input:
      # Route to the user span; if the model already responded (late flush),
      # fall back to the turn span rather than reopening a new user span.
      target = (
          self._turn_span if self._model_responded else self._ensure_user_span()
      )
      set_live_user_span_attributes(
          target, self._invocation_context, audio_ref=audio_ref
      )
    else:
      set_live_assistant_span_attributes(
          self._ensure_assistant_span(),
          self._invocation_context,
          audio_ref=audio_ref,
      )

  def on_turn_boundary(self, llm_response: LlmResponse) -> None:
    """Handles a ``turn_complete`` / ``interrupted`` boundary.

    A barge-in (``interrupted``) ends the turn immediately. A ``turn_complete``
    that follows a tool response is a generation handoff, not the turn's end, so
    the turn stays open for the model's follow-up answer. Otherwise the turn is
    marked pending finalize; its spans close once the trailing usage arrives
    (see :meth:`on_usage`), the next turn begins, or the session ends.
    """
    if not self._enabled:
      return
    self._accumulate_usage(llm_response)
    if self._turn_span is None:
      return

    interrupted = bool(llm_response.interrupted)
    if self._tool_call_pending and not interrupted:
      # Tool handoff: this turn_complete ends the tool-call generation, not the
      # conversational turn. Fold this generation's usage, close the tool-call
      # assistant span (the execute_tool span already nested under it), and keep
      # the live_turn open so the model's follow-up answer opens a fresh
      # assistant span under the same turn.
      self._fold_generation_usage()
      self._tool_call_pending = False
      if self._assistant_span is not None:
        self._assistant_span.end(end_time=time.time_ns())
        self._assistant_span = None
      return

    self._pending_finalize = True

  def close(self) -> None:
    """Finalizes any open or pending turn when the session ends."""
    if not self._enabled:
      return
    self._finalize_turn()

  # --- internals -----------------------------------------------------------

  def _accumulate_usage(self, llm_response: LlmResponse) -> None:
    # Within a generation, usage updates are cumulative, so keep the latest
    # (last-wins). Generations are summed via `_fold_generation_usage`.
    if llm_response.usage_metadata is not None:
      self._current_gen_usage = llm_response.usage_metadata
    if llm_response.finish_reason is not None:
      self._finish_reason = llm_response.finish_reason

  def _fold_generation_usage(self) -> None:
    """Folds the current generation's usage into the turn total.

    Called at each generation boundary (tool handoff) and at finalize, so a
    tool-using turn's tokens are the sum of its generations rather than only the
    last generation's counts.
    """
    if self._current_gen_usage is None:
      return
    self._usage_total = _merge_usage_metadata(
        self._usage_total, self._current_gen_usage
    )
    self._current_gen_usage = None

  def _ensure_turn_span(self) -> Span:
    if self._turn_span is None:
      # Backdate the turn start to the user's first audio chunk so the span
      # duration covers the whole turn (falls back to now for text-initiated
      # turns where no user audio was seen).
      self._turn_span = tracer.start_span(
          LIVE_TURN_SPAN_NAME,
          context=self._parent_context,
          start_time=self._turn_started_at_ns,
      )
      set_live_turn_span_attributes(
          self._turn_span,
          self._invocation_context,
          model=self._model_name(),
      )
    return self._turn_span

  def _ensure_user_span(self) -> Span:
    if self._user_span is None:
      turn_context = trace.set_span_in_context(self._ensure_turn_span())
      # Backdate the user span start to the first user audio chunk so its
      # duration ~ the length of the user's utterance.
      self._user_span = tracer.start_span(
          USER_SPAN_NAME,
          context=turn_context,
          start_time=self._turn_started_at_ns,
      )
      set_live_user_span_attributes(self._user_span, self._invocation_context)
    return self._user_span

  def _ensure_assistant_span(self, start_time: int | None = None) -> Span:
    if self._assistant_span is None:
      turn_context = trace.set_span_in_context(self._ensure_turn_span())
      self._assistant_span = tracer.start_span(
          ASSISTANT_SPAN_NAME, context=turn_context, start_time=start_time
      )
      set_live_assistant_span_attributes(
          self._assistant_span,
          self._invocation_context,
          model=self._model_name(),
      )
    return self._assistant_span

  def _finalize_turn(self) -> None:
    # A common end timestamp so the assistant / turn span durations line up.
    end_ns = time.time_ns()
    # Fold the final generation's usage into the turn total (summing across a
    # tool-using turn's generations).
    self._fold_generation_usage()
    # Stamp aggregated usage + finish reason on the assistant span, then close
    # user, assistant, and turn spans (children before parent).
    if self._assistant_span is not None:
      set_live_assistant_span_attributes(
          self._assistant_span,
          self._invocation_context,
          usage_metadata=self._usage_total,
          finish_reason=self._finish_reason,
      )
      self._assistant_span.end(end_time=end_ns)
      self._assistant_span = None
    # The user span normally closes at first model output; close it here as a
    # fallback (e.g. a turn with user input but no model response).
    if self._user_span is not None and not self._user_span_closed:
      self._user_span.end(end_time=end_ns)
    self._user_span = None
    if self._turn_span is not None:
      self._turn_span.end(end_time=end_ns)
      self._turn_span = None

    self._pending_finalize = False
    self._usage_total = None
    self._current_gen_usage = None
    self._finish_reason = None
    self._ttft_started_at = None
    self._ttft_recorded = False
    self._turn_started_at_ns = None
    self._user_span_closed = False
    self._model_responded = False
    self._tool_call_pending = False

  def _model_name(self) -> str | None:
    live_model = getattr(
        self._invocation_context.agent, 'canonical_live_model', None
    )
    return getattr(live_model, 'model', None)
