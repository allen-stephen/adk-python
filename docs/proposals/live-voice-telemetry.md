# Voice Observability for Live (Speech-to-Speech) Sessions

# Summary

This proposal adds four signals for voice agents: **how fast the agent starts
talking** (time-to-first-chunk), **what was heard and said** (transcripts on the
trace), **how much audio/video vs. text cost** (per-modality token breakdown),
and **a coherent per-turn span tree** rooted under a proper invocation span. All
of it uses the **OpenTelemetry GenAI semantic conventions** ADK is migrating
toward, so it renders in any OTLP backend (Cloud Trace, Arize, Phoenix, Jaeger)
with no new SDK.

The result: for each turn, a `live_turn` span with a `user` and `assistant`
child — carrying the latency the user felt, the transcript, a reference to the
captured audio, and token usage split into audio, video, and text — instead of a
flood of one span per streamed audio chunk.

# The Problem

A live voice session is a single long-lived WebSocket over which many
conversational turns stream as audio chunks. That shape is invisible to ADK's
current tracing:

- **No sense of responsiveness:** no native support for tracking
  *time-to-first-chunk*.
- **No transcripts on the trace:** ADK saves transcripts as session events, but
  they never reach the trace UI. Debugging a mishearing means leaving the trace
  and cross-referencing the session log by hand.
- **Opaque media cost:** live responses report token usage, but audio and video
  tokens — which dominate cost — are folded into a single input/output total.
- **A flat, rootless span tree:** the non-live path wraps every run in a
  top-level invocation span; the live path does not. Live `execute_tool` spans
  appear as loose siblings with no per-turn grouping and no root, inconsistent
  with `run_async`.

Per-turn voice tracing is also a baseline expectation in the ecosystem — e.g.
Arize's [OpenAI Realtime tracing
guide](https://github.com/Arize-ai/tutorials/blob/main/python/llm/tracing/openai/openai-realtime-api-tracing.ipynb).

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
`live_turn` span with `user` and `assistant` children, nesting any
`execute_tool` spans under the assistant. It requires no new configuration. (See
[How it works](#how-it-works) for the aggregation mechanics, including how a turn
stays open across a tool round-trip.)

## Semantic-convention alignment

Every attribute is a stable OTel GenAI semconv attribute where one exists, and
falls back to ADK's documented `*.experimental.*` namespace only where the
convention has no equivalent yet. This matches ADK's telemetry migration (which
is retiring the legacy `gcp.vertex.agent.*` span attributes) and the Agent
Platform observability requirement that agents emit OTel-format telemetry.

| Signal | Attribute | Source |
| :- | :- | :- |
| Time-to-first-chunk (headline latency) | `gen_ai.response.time_to_first_chunk` (seconds) on `assistant`; plus a `gen_ai.live.time_to_first_token` histogram metric | stable semconv |
| Assistant duration semantics | `gcp.vertex.agent.live.assistant_duration_kind = generation` on `assistant` (marks the span duration as server-side generation, not client playback) | ADK-native |
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

## Where this lands: managed observability

Because the live spans use the OpenTelemetry GenAI semantic conventions — and
the semconv-aligned schema is the **default on Agent Engine** (see
[Gating](#gating)) — the per-turn voice tree flows straight into **Cloud Trace**
and the **Gemini Enterprise Agent Platform** observability surfaces with no
additional instrumentation and no client-side work. There is no separate export
path to build.

The signals this proposal emits feed the managed views directly:

- **Traces tab** — the `live_turn` / `user` / `assistant` / `execute_tool` tree
  renders as the per-session directed-acyclic-graph of spans, with inputs and
  outputs (transcripts, opt-in) inline.
- **Latency dashboards** — `gen_ai.response.time_to_first_chunk` and the
  `live_turn` / `assistant` span durations feed the p50/p95/p99 latency views.
- **Models & Usage** — the per-modality token breakdown
  (`gen_ai.usage.experimental.{input,output}_{audio,video,text}_tokens`) feeds
  the token-usage and per-model views, so audio vs. text cost is visible.
- **Tools** — the nested `execute_tool` spans feed the per-tool call-count /
  latency / error views.
- **Topology** — `gen_ai.agent.name` on each turn feeds the per-agent and
  multi-agent topology graphs.

Any OTLP backend (Cloud Trace, Arize, Phoenix, Jaeger) receives the same spans.
For the Google Cloud setup, see
[Instrument ADK applications with OpenTelemetry](https://docs.cloud.google.com/stackdriver/docs/instrumentation/ai-agent-adk)
and the
[OpenTelemetry Semantic Conventions for generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

# What it looks like

A single-agent live session, rendered from these spans:

![Single-agent live session trace: per-turn `live_turn` spans, each with a `user` and `assistant` child, showing transcripts, time-to-first-chunk, and token usage.](assets/live-single-agent-trace.png)

*Single-agent live session in adk-web: each conversational turn is one
`live_turn` span with `user` and `assistant` children — durations reflect the
utterance and the model's generation, with time-to-first-chunk as the headline
latency.*

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
        "gcp.vertex.agent.live.assistant_duration_kind": "generation",
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

Span durations carry meaning. The `live_turn` and `user` spans start at the
user's first audio chunk, and the `user` span closes when the model starts
responding — so its bar length ≈ how long the user spoke. The `assistant` span
covers the model's *generation* (first output chunk to turn end), so its bar
length ≈ how long the model took to produce the response.

One caveat: the assistant duration measures **generation time on the server, not
audio playback on the client** — the backend cannot see when the user's device
finishes playing the reply. So it is a secondary "generation cost" signal,
labeled as such so it is not misread as end-to-end perceived latency. The primary
responsiveness signal — how long before the agent *starts* talking — is
time-to-first-chunk, captured separately.

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
  the appropriate child span. Every user turn gets a `user` span regardless of
  transcription timing: if the input transcript settles *after* the model has
  already started responding, the `user` span is created retroactively, bounded
  by the utterance window, so the user/assistant tree stays consistent.
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

A live conversation can hand off to a specialist agent mid-conversation, run
through a sequence of agents, or be driven as a graph of agent nodes. The turn
tracer plugs into ADK's existing span machinery rather than introducing a second
span-tree system. Two systems already exist, split by concern:

- **`node_tracing`** emits the *workflow structure* spans (`invoke_workflow`,
  `invoke_node`). When a node is an agent, it deliberately emits **no** span.
- **`_instrumentation`** emits the **`invoke_agent {name}`** span for every
  agent, whether it runs standalone or as a workflow node.

An agent therefore gets exactly one `invoke_agent` span, in every topology. The
turn tracer touches neither system: it anchors its `live_turn` spans under
whichever span is *current* when `run_live` runs — always the agent's own
`invoke_agent` span. Two properties keep this correct:

- **Each agent's live session owns its own turn tracer**, created when the agent
  starts and closed when it stops. A turn can never straddle a handoff, and no
  state leaks between agents — the outgoing agent's in-flight turn is finalized
  before the incoming agent begins.
- **Each turn nests under the `invoke_agent` span of the agent that produced
  it**, not a shared root, and carries a `gen_ai.agent.name` attribute so a
  backend can group or filter turns by agent even in a flattened view.

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

![Multi-agent live session trace: turns from each agent nested under their own `invoke_agent` span after a transfer, each `live_turn` tagged with its producing agent.](assets/live-multi-agent-trace.png)

*Multi-agent live session in adk-web: each turn nests under its producing
agent's `invoke_agent` span, so a transfer reads as a correctly parented subtree
rather than orphaned siblings.*

Concurrent live turns from a `ParallelAgent` are out of scope, since ADK does not
run `ParallelAgent` in live mode today. If parallel live execution is added
later, the per-agent tracer and span anchoring already give each branch an
independent, correctly parented turn subtree.

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

# Client support (adk-web)

These spans are consumed two ways, which carry different amounts of work:

- **Managed / OTLP path — no client work.** Agent Engine + Cloud Trace and the
  Gemini Enterprise Agent Platform render the live spans as-is (see
  [Where this lands: managed observability](#where-this-lands-managed-observability)),
  delivered entirely by the backend changes in this proposal.
- **adk-web-native inspector — work on both sides.** The local inspector
  experience shown in the screenshots is not free from the backend change alone.
  It requires coordinated work in **both adk-python** (emit the per-turn span
  tree and voice attributes — this proposal) **and adk-web** (consume and render
  them).

At a requirements level, the adk-web side must:

- **Render the semconv live span tree resiliently.** The viewer validates and
  renders the `live_turn` / `user` / `assistant` / `execute_tool` tree, and
  degrades gracefully on an unexpected span shape (dropping only the offending
  span) so an evolving live/semconv shape can never blank the whole trace view.
- **Surface the voice signals.** Transcripts, time-to-first-chunk (as the
  headline responsiveness metric), and the assistant span's generation-duration
  semantics are shown in the span detail, distinct from the raw span duration.
- **Correlate spans to the selected event.** Live `assistant` spans carry
  ADK-native correlation ids so the inspector can associate a span with the
  event the user selected, mirroring the non-live path.

The frontend work is tracked and delivered separately from this backend
proposal.

# Open Questions

- **End-to-end perceived latency.** As noted above, the assistant span excludes
  client audio playback, which the backend cannot observe. Full perceived latency
  (including playback completion) would need a client-emitted signal fed back to
  the trace; deferred until there's demand.
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