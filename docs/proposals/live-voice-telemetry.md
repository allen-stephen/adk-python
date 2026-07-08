# Voice Observability for Live (Speech-to-Speech) Sessions

# Summary

This proposal adds the signals that voice-agent operators actually need: **how
fast the agent starts talking** (time-to-first-chunk), **what was heard and
said** (transcripts on the trace), **how much audio/video vs. text cost**
(per-modality token breakdown), and **a coherent per-turn span tree** rooted
under a proper invocation span. All of it is expressed in the **OpenTelemetry
GenAI semantic conventions** ADK is migrating toward, so it lights up in any
OTLP backend (Cloud Trace, Arize, Phoenix, Jaeger) with no new SDK.

The result: open a live session's trace and see, for each turn, a `live_turn`
span with a `user` and `assistant` child — carrying the latency the user felt,
the transcript of the exchange, a reference to the captured audio, and token
usage broken down into audio, video, and text — instead of a flood of one span
per streamed audio chunk.

# The Problem

A live voice session is a single long-lived WebSocket over which many
conversational turns stream as audio chunks. That shape is invisible to the
tracing we currently have instrumented:

- **No sense of responsiveness:** lacking native support for tracking
  *time-to-first-chunk*.
- **No transcripts on the trace:** ADK saves transcripts as session events, but
  they never reached the trace UI. Debugging a mishearing ("why did it answer
  about *weather* when I asked about *whether*?") meant leaving the trace and
  cross-referencing the session log by hand.
- **Media cost is opaque:** live responses report token usage, but audio and
  video tokens dominate cost and were folded into a single input/output total.
- **A flat, rootless span tree:** The non-live path wraps every run in a
  top-level invocation span; the live path did not. Live `execute_tool` spans
  appeared as loose siblings with no per-turn grouping and no root — awkward to
  read and inconsistent with `run_async`.

Per-turn voice tracing is also becoming a baseline expectation in the ecosystem
(e.g. Arize's [OpenAI Realtime tracing
guide](https://github.com/Arize-ai/tutorials/blob/main/python/llm/tracing/openai/openai-realtime-api-tracing.ipynb)),
signaling real market demand for this capability.

# The Proposal

Reconstruct a **per-turn span tree** from the live event stream and annotate it
with voice-specific attributes drawn from the OTel GenAI semantic conventions.
For each conversational turn:

```
invocation / invoke_workflow            ← root span for the whole live session
└─ invoke_agent {agent}
   └─ live_turn                          ← one per conversational turn
      ├─ user                            ← input transcript + input audio ref
      ├─ assistant                       ← the tool-call generation
      │  └─ execute_tool {tool}          ← tool call within the turn
      └─ assistant                       ← the spoken answer after the tool
            gen_ai.response.time_to_first_chunk
            gen_ai.output.messages       (transcript, opt-in)
            gen_ai.output.type = speech
            gen_ai.usage.experimental.input_audio_tokens (audio/video/text)
```

(A turn with no tool call has a single `assistant` child.)

A single new component, a per-session **turn tracer**, watches the model's
event stream and aggregates each turn's many streamed chunks into one
`live_turn` span with `user` and `assistant` children — recording the voice
attributes as the relevant events flow by and nesting any `execute_tool` spans
under the assistant. A tool call spans two model generations (the model pauses
to call the tool, then speaks the answer); the turn stays open across the tool
round-trip so the follow-up answer stays in the same `live_turn` as a second
`assistant` span. No new configuration is required to get the defaults.

## Semantic-convention alignment

Every attribute is a stable OTel GenAI semconv attribute where one exists, and
falls back to ADK's documented `*.experimental.*` namespace only where the
convention has no equivalent yet. This matches ADK's telemetry migration (which
is retiring the legacy `gcp.vertex.agent.*` span attributes) and the Agent
Platform observability requirement that agents emit OTel-format telemetry.

| Signal | Attribute | Source |
| :- | :- | :- |
| Time-to-first-chunk | `gen_ai.response.time_to_first_chunk` (seconds) on `assistant`; plus a `gen_ai.live.time_to_first_token` histogram metric | stable semconv |
| Transcripts | `gen_ai.input.messages` on `user`, `gen_ai.output.messages` on `assistant` (JSON message payloads) | stable semconv (opt-in) |
| Output modality | `gen_ai.output.type = speech` on `assistant` | stable semconv |
| Media token breakdown | `gen_ai.usage.experimental.{input,output}_{audio,video,text}_tokens` on `assistant` | ADK experimental |
| Audio references | `gen_ai.input.experimental.audio_ref` on `user`, `gen_ai.output.experimental.audio_ref` on `assistant` (artifact URIs) | ADK experimental |
| Provider / model | `gen_ai.provider.name` (`gcp.vertex_ai` / `gcp.gemini`), `gen_ai.request.model` | stable semconv |
| Finish reason | `gen_ai.response.finish_reasons` on `assistant` | stable semconv |
| Conversation / agent | `gen_ai.conversation.id`, `gen_ai.agent.name` | stable semconv |
| Operation name | `gen_ai.operation.name` per span (`live_turn`, `live_turn.user`, `generate_content`) | stable semconv |
| Root span parity | live sessions wrapped in the same top-level invocation span as `run_async` | — |

The `live_turn` span is an **ADK-native aggregate**: the OTel operation
vocabulary has no value for a speech-to-speech conversational turn, so it is
stamped with the custom `gen_ai.operation.name = live_turn`. The `assistant`
child reuses the semconv `generate_content` operation; the `user` child uses
`live_turn.user`.

### Gating

- **Transcripts** are message content and follow the OTel content-capture
  contract: they are recorded only in the span-bearing content modes
  (`SPAN_ONLY` / `SPAN_AND_EVENT`) via
  `TelemetryConfig.should_add_content_to_experimental_spans`. Per the semconv
  `Opt-In` requirement level for `gen_ai.*.messages`, they are **off by
  default**.
- **Audio references** are non-content pointers (artifact URIs) and are recorded
  unconditionally when supplied — which the flow only does when the existing
  `save_live_blob` audio switch is on.
- **The span tree** is emitted only under the OTel-semconv-aligned telemetry
  schema (`ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=2`, the default on Agent Engine).
  Under the legacy schema the turn tracer is a no-op and the live path is
  unchanged — no new spans, no legacy `gcp.vertex.agent.*` attributes.

## Why references, not inline audio bytes

ADK puts an `artifact://…` reference on the span rather than the audio bytes: it
already flushes live audio to the **artifact service**, so the reference is free,
spans stay small (audio can be megabytes), and user audio is not shipped to the
tracing backend by default. Backends that want playback can resolve the URI.

## Out of scope

- **Audio evaluation.** Scoring the captured audio (tone, quality, etc.) belongs
  in ADK's evaluation framework, not the tracing layer.
- **Legacy-schema live tracing.** The span tree targets the semconv-aligned
  schema only; there is no legacy `gcp.vertex.agent.*` rendering of live turns.

# What it looks like

Each turn becomes a self-describing unit. A representative `live_turn` span:

```jsonc
{
  "name": "live_turn",
  "attributes": {
    "gen_ai.operation.name": "live_turn",
    "gen_ai.provider.name": "gcp.vertex_ai",
    "gen_ai.request.model": "gemini-live-2.5-flash",
    "gen_ai.agent.name": "triage"
  },
  "children": [
    {
      "name": "user",
      "attributes": {
        "gen_ai.input.messages": "[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"what's the weather in London?\"}]}]",
        "gen_ai.input.experimental.audio_ref": "artifact://app/user/sess/_adk_live/in.pcm#3"
      }
    },
    {
      "name": "assistant",                 // the tool-call generation
      "children": [ { "name": "execute_tool {get_weather}" } ]
    },
    {
      "name": "assistant",                 // the spoken answer after the tool
      "attributes": {
        "gen_ai.operation.name": "generate_content",
        "gen_ai.output.type": "speech",
        "gen_ai.response.time_to_first_chunk": 0.612,
        "gen_ai.output.messages": "[{\"role\":\"assistant\",\"parts\":[{\"type\":\"text\",\"content\":\"It's 12 degrees and cloudy in London.\"}]}]",
        "gen_ai.output.experimental.audio_ref": "artifact://app/user/sess/_adk_live/out.pcm#3",
        "gen_ai.usage.input_tokens": 320,       // summed across both generations
        "gen_ai.usage.output_tokens": 210,
        "gen_ai.usage.experimental.input_audio_tokens": 300,
        "gen_ai.usage.experimental.output_audio_tokens": 200
      }
    }
  ]
}
```

**Span durations are meaningful**, not just decoration. The `live_turn` and
`user` spans start at the user's first audio chunk, and the `user` span closes
when the model starts responding — so its bar length ≈ how long the user spoke.
The `assistant` span spans the model's response, so its bar length ≈ the
response time. (Time-to-first-chunk is the responsiveness *attribute* on the
assistant span; it is distinct from either duration.) This mirrors what a viewer
expects from a voice trace: a long user bar for a long question, a shorter
assistant bar for a quick reply.

# How it works

A single per-session helper, `LiveTurnTracer`, is created once per `run_live`
call and threaded into the send/receive loops. It aggregates the turn's chunk
stream rather than tracing each chunk:

- On the **first user audio chunk** of a turn, it captures the turn's start
  time and arms the time-to-first-chunk timer.
- On the **first model output** of a turn, it opens the `live_turn` span
  (backdated to that start time) and its `assistant` child (so `execute_tool`
  spans nest under the assistant) and records time-to-first-chunk. The `user`
  child is opened only when there is user-side input, and closes here — so its
  duration ≈ the utterance length.
- As **final transcripts** and **audio references** arrive, it attaches them to
  the appropriate child span (a late input transcript that arrives after the
  user span has closed falls back to the `live_turn` span).
- On a **tool round-trip**, the Live API ends the first generation with
  `turn_complete` to call the tool, then generates the spoken answer as a second
  generation (also ending in `turn_complete`). The tracer detects the function
  call in the first generation's output and treats the following `turn_complete`
  as a handoff, keeping the `live_turn` open across the round-trip: the follow-up
  answer becomes a second `assistant` span under the same turn, and token usage
  is summed across both generations. A barge-in interrupt during the round-trip
  ends the turn immediately.
- On **`turn_complete` / `interrupted`** (the real turn end), it marks the turn
  *pending finalize* rather than closing immediately, because token usage
  typically arrives in a trailing usage-only response. The turn is finalized —
  usage stamped on the assistant span, spans closed — when that usage arrives,
  when the next turn begins, or when the session ends.

The turn tracer's state is scoped to one session and never touches
`InvocationContext`, so nothing leaks across invocations. The audio/video/text
token breakdown is a pure enrichment of the existing token-usage mapping, so it
also benefits the non-live path for free wherever a model reports per-modality
counts. `run_live` is wrapped in the same top-level invocation span mechanism
already used by `run_async`, closing the root-span gap.

## Multi-agent & workflow voice sessions

A live conversation is rarely one agent. It can hand off to a specialist agent
mid-conversation, march through a sequence of agents, or be driven as a graph of
agent nodes. The turn tracer plugs into ADK's existing span machinery so a voice
trace stays readable across all of these — and, importantly, it does **not**
introduce a second span-tree system. There are already two, split by concern:

- **`node_tracing`** emits the *workflow structure* spans (`invoke_workflow`,
  `invoke_node`). When a node is an agent, it deliberately emits **no** span.
- **`_instrumentation`** emits the **`invoke_agent {name}`** span for every
  agent, whether it runs standalone or as a workflow node.

So an agent gets exactly one `invoke_agent` span, from one place, in every
topology. The turn tracer touches neither system: it captures whichever span is
*current* when `run_live` runs, which is always the agent's own `invoke_agent`
span, and anchors its `live_turn` spans there. Two properties make this clean.
First, **each agent's live session owns its own turn tracer** — created when that
agent starts running and closed when it stops — so a turn can never straddle a
handoff and no state leaks between agents; the outgoing agent's in-flight turn is
finalized before the incoming agent begins. Second, **each turn nests under the
`invoke_agent` span of the agent that produced it**, not under a shared root.
Every `live_turn` also carries a `gen_ai.agent.name` attribute, so a backend can
group or filter turns by agent even in a flattened view.

### What live mode actually supports

Live mode is narrower than the async path. The tracer engages only where live
execution is implemented; the unsupported orchestrators raise
`NotImplementedError` in live mode, so the tracer never runs there.

| Topology | Live support | `live_turn` parent |
| :- | :- | :- |
| Single `LlmAgent` | ✅ | `invoke_agent {agent}` |
| `LlmAgent` as a node in a `Workflow` | ✅ | `invoke_agent {agent}` (under `invoke_workflow` / `invoke_node` ancestors) |
| Agent transfer A → B | ✅ | B's `invoke_agent {B}` (nested under A's `invoke_agent {A}`) |
| `SequentialAgent` sub-agents | ✅ | each child's `invoke_agent {child}` (under `invoke_agent {seq}`) |
| `LoopAgent` | ❌ `NotImplementedError` | n/a |
| `ParallelAgent` | ❌ `NotImplementedError` | n/a |
| `RemoteA2aAgent` | ❌ `NotImplementedError` | n/a |

The nesting for a session that transfers from a triage agent to a billing agent
(transfer nests the target inside the source's still-open span, rather than as a
sibling — parenting is always correct, never orphaned):

```
invocation / invoke_workflow            ← the whole live session
└─ invoke_agent {triage}
   ├─ live_turn   (gen_ai.agent.name = triage)
   │  ├─ user
   │  └─ assistant
   └─ invoke_agent {billing}            ← nested under triage (transfer)
      └─ live_turn   (gen_ai.agent.name = billing)
         ├─ user
         └─ assistant
```

Concurrent live turns from a `ParallelAgent` are out of scope: ADK does not run
`ParallelAgent` in live mode today, so simultaneous per-agent turns are not a
case the tracing needs to handle. If parallel live execution is introduced
later, the per-agent tracer and per-agent span anchoring already give each
branch an independent, correctly parented turn subtree.

## Footprint

- **New:** one module (`telemetry/live_turn_tracing.py`, alongside the sibling
  `telemetry/node_tracing.py`) and three attribute-setter helpers plus one
  metric in the telemetry package.
- **Changed:** the live receive/send loops thread the turn tracer through; the
  token-usage mapping gains per-modality attributes; `run_live` gains the root
  span; the turn tracer anchors each turn under the producing agent's / node's
  span for multi-agent and workflow sessions.
- **Config:** none required. Reuses the existing content-capture and
  `save_live_blob` switches and the telemetry schema-version opt-in.

# Testing / Validation

Covered by unit tests in `tests/unittests/telemetry/test_live_telemetry.py`,
which drive the live receive loop with a scripted fake connection and an
in-memory span exporter (no live API needed). An autouse fixture pins the
semconv-aligned schema for the span-tree tests:

- **Chunk aggregation** — a turn of 50 streamed audio chunks produces exactly
  one `assistant` span (the key regression guard), and no `call_llm` span
  appears on the live path.
- **Span tree** — the `assistant` span nests under `live_turn`; two turn
  boundaries produce two turn/assistant spans.
- **Time-to-first-chunk** — recorded as `gen_ai.response.time_to_first_chunk`
  on the assistant span when user audio precedes model output.
- **Transcripts** — input transcript lands on `user` as `gen_ai.input.messages`,
  output on `assistant` as `gen_ai.output.messages`; partial deltas do not; and
  they are dropped when span content capture is disabled.
- **Media token breakdown** — per-modality audio and video attributes appear on
  the assistant span and are a breakdown of (not additive to) the totals.
- **Deferred usage** — token usage arriving after `turn_complete` still lands on
  the assistant span.
- **Audio references** — a flushed output audio URI lands on the assistant span
  as `gen_ai.output.experimental.audio_ref`.
- **Root span** — `run_live` enters the top-level invocation span, matching
  `run_async`.
- **Multi-agent parenting** — a turn nests under its producing agent's span and
  carries `gen_ai.agent.name`; after a handoff, each agent's turns land under
  their own agent span (verified both directly and through the real `run_live`
  path).
- **Legacy schema** — under `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=1` the tracer
  emits no live-turn spans (no-op), leaving the live path unchanged.

All existing live/streaming/telemetry suites continue to pass. End-to-end
verification against the real Gemini Live API (span tree, time-to-first-chunk,
transcripts, token split in a live backend) is done manually.

# Open Questions

- **Metric surface.** Time-to-first-chunk is emitted both as a span attribute
  and a `gen_ai.live.time_to_first_token` histogram. If a dashboard-first
  workflow emerges we may want additional live metrics (turns per session,
  audio-token rate); deferred until there's demand.
- **Optional inline audio.** The default is artifact references. If a use case
  needs zero-storage inline audio on spans, it can be added later behind an
  explicit opt-in without changing the reference path.
- **Promotion of experimental keys.** `gen_ai.*.experimental.audio_ref` and the
  per-modality token keys drop the `.experimental.` segment if and when OTel
  standardizes equivalents.
```