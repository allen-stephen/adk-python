# Run it locally

## 1. Set up

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it).

```bash
git checkout feat/live-telemetry
uv sync
```

## 2. Configure model access

Live requires a Gemini **Live API** model. Simplest is the Gemini API:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
export GOOGLE_API_KEY=<your-key>   # from https://ai.google.dev/
```

Then in the sample's `agent.py`, comment out the Vertex `model=...` line and
uncomment the `# Gemini API` one.

(For Vertex instead: `gcloud auth application-default login`, then set
`GOOGLE_GENAI_USE_VERTEXAI=TRUE`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`; the samples work as-is.)

You can also put these in a `.env` file inside the sample folder.

## 3. Turn the tracer on

**The tracer is a no-op without this** — under the legacy schema no `live_turn`
spans are emitted:

```bash
export ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=2
```

To also show transcripts on the spans (off by default):

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

## 4. Run and interact

```bash
cd contributing/samples/live
adk web
```

1. Open the printed URL (usually <http://127.0.0.1:8000>).
2. Select **`live_bidi_streaming_single_agent`** in the top-left dropdown.
3. Click the **Audio** icon by the chat input and talk to the agent, e.g.
   *"roll a six-sided die and tell me if it's prime"* (exercises a tool call).
4. Open the **Trace** view for the session to see the `live_turn` span tree. If no spans are showing, click the refresh button in the top right.

Other samples under `contributing/samples/live/` cover more scenarios —
`live_bidi_streaming_multi_agent` shows transfer nesting under `invoke_agent`.
