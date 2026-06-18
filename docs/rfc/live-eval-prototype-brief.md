# Live Eval in ADK — Engineering Prototype Brief

> **Read the RFC first:** `docs/rfc/live-eval-and-cli-strategy.md` covers the problem,
> the strategy, and the open question. This brief adds what the RFC leaves out — the
> **CSEE context** that inspired the work, the **code anchors** for the prototype that
> already exists on `feat/live-evals-v2`, and notes for extending it.

## The CSEE context (not in the RFC)

CSEE (Customer Simulation & Evaluation Engine, `~/repos/public/csee-platform`) is the bespoke platform that inspired this work. We are **not forking it** — it carries domain-specific drive-thru / Food AI baggage — but it proves the demand and demonstrates the full end-to-end voice-eval journey we want ADK to support natively. Its LLM-first design (all reasoning in agent prompts; tools do I/O only) is a good guide.

What CSEE does end-to-end, and where each piece lives in its repo:

- **Persona/scenario generation** — a `SequentialAgent` (persona → ordering intent → assemble) generates diverse personas, intents, voice assignment, avatars. *(`csee/agents/scenario_generator.py`)*
- **Audio-to-audio simulation** — *dual* Gemini Live sessions: a persona (voice) converses turn-by-turn with the system-under-test (voice) until order completion or max turns. *(`csee/services/simulation.py`, `csee/services/customer_live.py`)*
- **Audio realism** — FFmpeg degradation: noise, drive-thru effects, cross-talk. *(`csee/services/audio_realism.py`)*
- **LLM-as-judge scoring** — order accuracy, friction, quality → weighted composite + pass/fail + failure categories + feedback. *(`csee/agents/evaluation_agent.py`, `docs/prd.md` §4)*
- **Batch + scale** — fan-out to Cloud Run Jobs; results in Firestore; analysis dashboard (pass rate, score distributions, failure heatmap, batch comparison). *(`csee/services/cloud_run_jobs.py`, `csee/frontend/`)*
- **Live + human modes** — watch a bot↔bot conversation live, or speak as the customer via mic.

Best overview reads: `csee-platform/README.md` and `csee-platform/docs/prd.md`.

## Code anchors (what the prototype built)

ADK eval engine — `adk-python/src/google/adk/evaluation/`:
- `simulation/persona.py` — the `Persona` data model (voice, character prompt, goal) plus `BargeInConfig` / `AudioRealismConfig`.
- `simulation/persona_live_conversation.py`, `simulation/persona_customer_agent.py` — the true audio-to-audio runner: the persona is a Live agent that **speaks** via `send_realtime`, relayed turn-by-turn against the agent under test.
- `simulation/live_conversation_materializer.py` — converts a finished live conversation into scorable `Invocation`s (transcript + `AudioReference`s); wired into `local_eval_service.py`.
- `eval_case.py` — `AudioReference` plus `Invocation.user_audio` / `agent_audio` and `EvalCase.live_persona_scenario`. Audio is carried **by reference** (artifact), modality-generic via mime type.
- `latency_evaluator.py` — `response_latency_v1`, a purely local metric, registered in `metric_evaluator_registry.py`.
- `simulation/audio_realism.py` — the realism transform hook (additive Gaussian noise today), applied on the persona audio path.
- `vertex_ai_eval_facade.py`, `_vertex_ai_scenario_generation_facade.py` — the managed Gen AI Eval Service delegation (GCP-gated, graceful-optional); the multi-turn quality metrics reused for live runs route through here.
- `cli/cli_tools_click.py` — `adk eval --persona / --use_live / --watch / --managed_metrics / --live_timeout_seconds`.
- `cli/dev_server.py` — `RunEvalRequest.use_live` / `live_persona_scenario` and the `add-persona` endpoint.

Samples — `contributing/samples/live/`: `*.persona.json` (a `LiveConversationScenario`) and `*.evalset.json` showing a persona case alongside a text case.

Library-only / not yet wired into a product surface (good extension points): `_vertex_ai_scenario_generation_facade.generate_live_scenarios` (persona auto-generation) and `live_eval_conversion.convert_live_session_to_eval_invocations` (recorded human live session → eval case).

ADK Web — `adk-web/src/app/`:
- `core/services/stream-chat.service.ts`, `core/services/websocket.service.ts` — live path (`/run_live`, audio PCM, JPEG).
- `core/models/Eval.ts`, `components/eval-tab/`, `core/services/eval.service.ts` — eval path; the bridge surfaces here (save live session as eval case). In-line per-turn audio playback is the main remaining piece.
- Note the naming collision: eval-set `model_execution_mode: 'live'|'replay'` means re-run-vs-replay, **not** bidi audio.

agents-cli (context only) — `agents-cli/src/google/agents/cli/eval/`: v0.5.0 eval runs on the Vertex Gen AI Eval SDK and does not depend on `google-adk`. Per the RFC, `agents-cli` should reference the `adk` engine for live eval rather than re-implement it.

## Extending the prototype

The end-to-end path (persona speaks → audio-to-audio conversation → materialize → score → persist → view in ADK Web) works today. The natural next pieces:

1. **In-line audio playback** in ADK Web, per turn, beside the transcript and scores.
2. **Richer audio realism** — `AudioRealismConfig.background_noise` is defined but unused; add background-noise mixing and channel effects to the transform.
3. **Real barge-in** — today it only sets a `was_interrupted` flag; make it actually interrupt the agent's turn.
4. **Richer acoustic metrics** beyond latency (turn-taking, talk-ratio, intelligibility) — the managed-service growth area.
5. **Wire the two library-only converters** (persona generation, recorded-session → eval case) into the CLI/Web surfaces.

## Designing for video (out of scope now, don't design it out)

The Live API streams video as input, and the architecture already supports it — the contracts and transport are modality-generic, so video should stay a later extension rather than a refactor:

- **Already generic:** `EvalCase`/`Invocation` are `genai_types.Content` (carry image/video `inline_data`); `AudioReference` keys off `mime_type`; `LiveRequestQueue.send_realtime(blob)` and the connection layer route media by mime type (`image/` → video); `convert_events_to_eval_invocations` preserves any `inline_data` part.
- **Keep the media-reference and realtime-input paths mime-typed** rather than assuming `audio/pcm` — that is the whole cost of video-readiness now.
- **Two audio assumptions to relax when video lands (not now):** the hardcoded `response_modalities=["AUDIO"]` in the live runner, and the input-blob cache in `audio_cache_manager.py` (assumes audio; "video not supported yet"). Neither corrupts video sent to the model; they only matter once you persist/replay video blobs.

## Watch-outs

- Don't add scoring logic to `agents-cli`, and don't reimplement the engine in ADK Web — call into the `adk` engine and the managed service (see the RFC's "Where things live").
- Keep the local loop runnable without a GCP project; managed metrics degrade gracefully when absent (the facade already does this).
- Audio is carried by reference (artifact), not inline — keep that path mime-typed/modality-generic so video reuses the same field rather than forcing a second contract change.
