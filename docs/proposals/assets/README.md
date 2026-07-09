# Proposal assets

Screenshots referenced by `../live-voice-telemetry.md`. Drop the image files
here with these exact names so the relative Markdown links resolve:

- `live-single-agent-trace.png` — a single-agent live session trace in adk-web
  (per-turn `live_turn` spans with `user`/`assistant` children).
- `live-multi-agent-trace.png` — a multi-agent live session trace in adk-web
  (turns nested under each agent's `invoke_agent` span after a transfer).
