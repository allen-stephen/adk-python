# Live Execution-Path Tracing in ADK

# Summary

This proposes to attach execution-path information to each live turn and forward
the same run metadata the non-live path already carries, so a live conversation
highlights its path through the agent graph like a text conversation does. It's
the companion to the [live voice tracing proposal](live-voice-telemetry.md),
which covers the other inspector tabs.

# Motivation

The Graph tab is one of the highest-value debugging surfaces — it shows which
agents and nodes a conversation actually traversed. For non-live sessions it
highlights the execution path as events stream in. For live sessions it has two
gaps:

* **No execution-path highlighting:** live turns don't carry the execution-path
  information the non-live path attaches, so the Graph tab can draw the base
  graph but can't show *which* nodes the conversation ran through.
* **Broken multi-agent view:** when a live conversation hands off to a
  specialist agent, the graph can't render or highlight the correct nested
  subtree, so a handoff doesn't read as a path through the graph.

There is a second, smaller gap: **custom run metadata** that the non-live path
forwards is dropped on the live path, so any inspector feature keyed on it is
blank for live sessions.

# Proposal

## Execution-path parity

Stamp each live turn with the same execution-path information the non-live path
already attaches, so a downstream surface can map a live turn to its node in the
agent graph. This must hold across a live handoff: when a conversation transfers
to a specialist agent mid-session, the transferred agent's turns should carry a
path that reflects the handoff chain, so the Graph tab highlights the specialist
as a nested subtree rather than leaving it unmapped.

The workflow-driven live case (a live session driven as a graph of agent nodes)
already attaches this information through ADK's existing node machinery and is
**unchanged** by this proposal. The gap is limited to the direct single-agent
and agent-transfer live paths, which bypass that machinery today.

## Metadata propagation

The live entry point should forward run metadata the same way the non-live path
already does, so any surface that reads it works identically for both.

# Scope & non-goals

* **Not the telemetry span tree.** The per-turn `live_turn` span tree and its
  transcripts, latency, and token signals are covered by the
  [live voice tracing proposal](live-voice-telemetry.md). This proposal only
  adds execution-path information and run metadata on top.
* **Not a graph-rendering redesign.** This reuses the existing graph and
  highlighting system; it only supplies the signal that system needs.
* **Workflow-driven live sessions are untouched.** They already attach
  execution-path information correctly.
* **Availability caveat (shared with the telemetry proposal).** Live
  observability is gated behind the semconv-aligned telemetry schema (the
  default on managed surfaces). Under the legacy schema — the current local-dev
  default — live tracing is inactive, so these Graph-tab improvements light up
  under the same conditions as the rest of the live-inspector work.

# Relationship to the live voice tracing proposal

Together these two proposals bring every high-value inspector tab to parity
between live and non-live sessions.

| Inspector tab | Unblocked by |
| :---- | :---- |
| Request / Response | [live voice tracing](live-voice-telemetry.md) |
| Trace tree | [live voice tracing](live-voice-telemetry.md) |
| Usage | [live voice tracing](live-voice-telemetry.md) |
| Graph | this proposal |
| Metadata-driven surfaces | this proposal |
