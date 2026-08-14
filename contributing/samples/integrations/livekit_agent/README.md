# LiveKit Voice Agent

Put an **unmodified** ADK live agent behind a real transport with the
[LiveKit integration](../../../../src/google/adk/integrations/livekit/README.md).
This sample proves the full round trip: a **browser with a live microphone**
summons the ADK worker into a LiveKit room via **explicit dispatch**, speaks to
it, and hears the reply — all over WebRTC.

```
Browser (mic + speaker) ──WebRTC──►  LiveKit room  ◄──dispatch──  ADK worker
        │                                                     (LiveKitRunner → run_live)
        └── GET /token ──► token server ── create_dispatch(room, "roll_dice", metadata) ──┘
```

The agent itself (`agent.py`) is a plain ADK live agent that rolls dice and
checks primes — it has no idea LiveKit exists. `adk web` still runs it locally.

## Files

| File                | Role                                                                                                                                                    |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `agent.py`          | The ADK live agent (`root_agent`). Unchanged by LiveKit.                                                                                                |
| `livekit_worker.py` | Thin consumer: builds a wired `AgentServer` with `livekit_server` and hands it to LiveKit's `cli.run_app()`. Registers under dispatch name `roll_dice`. |
| `client/main.py`    | The "separate app": a FastAPI backend that mints a join token **and** explicitly dispatches the worker. Serves the browser client.                      |
| `client/static/`    | The browser front end (LiveKit's own `livekit-client` via CDN — no ADK client code, no build step).                                                     |

The browser shows a live transcript and logs the agent's tool calls, both read
off the room data track, and lets you type instead of talk — the same track
carries text turns back to the agent.

## Prerequisites

```bash
pip install "google-adk[livekit]"
```

Set your LiveKit and model credentials:

```bash
export LIVEKIT_URL="wss://your-project.livekit.cloud"   # or ws://localhost:7880
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
# plus your Vertex/Gemini credentials for gemini-live-2.5-flash-native-audio
```

## Run it

You need a LiveKit server. Use a **free local dev server** (no cloud account) or
LiveKit Cloud — both work unchanged.

### Option A — local dev server (recommended for testing)

Install the LiveKit CLI/server (`brew install livekit` on macOS, or see
<https://docs.livekit.io/home/self-hosting/local/>), then in its own terminal:

```bash
livekit-server --dev
```

`--dev` prints a fixed dev key/secret. Point the sample at it:

```bash
export LIVEKIT_URL="ws://localhost:7880"
export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="secret"
```

### Option B — LiveKit Cloud

Set `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` from your project
in the [LiveKit Cloud dashboard](https://cloud.livekit.io/) instead.

### Start the sample

Run the two processes in separate terminals, from the repository root (each
inherits the env vars above).

**Terminal 1 — the ADK worker** (registers with LiveKit, then waits for
dispatch):

```bash
python -m contributing.samples.integrations.livekit_agent.livekit_worker dev
```

> The trailing `dev` subcommand is LiveKit's `cli.run_app` dev mode (auto-reload,
> verbose logs). Drop it, or use `start`, for production.

**Terminal 2 — the token + dispatch server** (also serves the browser client):

```bash
python -m contributing.samples.integrations.livekit_agent.client.main
```

Open <http://localhost:8080>, click **Start talking**, allow microphone access,
and speak — for example, *"roll a 20 sided die and tell me if it's prime."* You
should hear the dice agent answer, see the transcript fill in as you both speak,
and see the `roll_die` / `check_prime` tool calls in the log. Interrupt it
mid-sentence and it stops immediately.

> Live microphone capture requires a secure context. `http://localhost` counts
> as secure in all major browsers, so no HTTPS setup is needed for local dev.

### Headless smoke test (no browser)

To confirm the worker joins and `run_live` starts without opening a browser,
dispatch a job with the LiveKit CLI while the worker (Terminal 1) is running:

```bash
lk dispatch create --room smoke-test --agent-name roll_dice \
  --metadata '{"user_id":"tester","session_id":"smoke-test"}'
```

Watch Terminal 1: the worker should accept the job, join `smoke-test`, and log
the start of the live session.

## What each step proves

1. **Explicit dispatch:** `client/main.py` calls
   `AgentDispatchService.create_dispatch(room, "roll_dice", metadata=...)`. This
   is how a real caller summons the worker — the worker does nothing until
   dispatched.
1. **Identity passthrough:** the dispatch metadata carries `user_id` /
   `session_id`, which the worker's `resolve_ids` reads back off the
   `JobContext` (LiveKit has no such concepts of its own).
1. **Inbound bridge:** the browser mic becomes a room audio track that
   `LiveKitRunner` forwards into `LiveRequestQueue.send_realtime` as 16 kHz PCM.
   Typed messages travel the data track into `send_content` as user turns.
1. **Outbound bridge:** the agent's `run_live` audio events are captured onto a
   room audio track and played back in the browser, while transcripts and tool
   activity go out on the data track. Interrupting the agent clears its
   playback queue, so barge-in is immediate.

## Local alternative

To iterate on the agent without any LiveKit setup, just run `adk web` on this
folder and use the Audio button — same `root_agent`, no transport.
