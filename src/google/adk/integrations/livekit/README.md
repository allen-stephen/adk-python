# LiveKit Integration

The ADK LiveKit integration puts an **unmodified** ADK live agent behind a real
transport -- telephony (SIP/PSTN), WebRTC (web/mobile), and Unity/gaming -- via
[LiveKit](https://livekit.io/). It is the realtime analog of the `SlackRunner`:
a thin connector over the transport-agnostic
`LiveRequestQueue` -> `run_live()` -> `Event` contract. It contains no agent
logic, no codecs, and no signaling -- just two frame bridges:

- **Inbound:** room media tracks -> `LiveRequestQueue.send_realtime`
  (16 kHz mono `audio/pcm`, `image/jpeg` video), and room data messages ->
  `LiveRequestQueue.send_content`.
- **Outbound:** the `Event` stream -> room audio track (24 kHz) and a data track
  (transcripts and tool call/response payloads).

LiveKit owns the hard parts (WebRTC, SIP, jitter, echo, resampling on subscribe,
track lifecycle), and its dispatch owns the worker lifecycle -- ADK never owns
the worker process.

## Prerequisites

Install the ADK with LiveKit support:

```bash
pip install "google-adk[livekit]"
```

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, plus your model
credentials (e.g. Vertex/Gemini for a native-audio live model).

## Usage

There are two ways to wire an agent up. Neither owns the worker process --
LiveKit's dispatch owns lifecycle, and you always run the worker via
`cli.run_app()`, just like `SlackRunner.start()` hands off to Slack's own
runloop.

### Server factory

ADK builds a wired `AgentServer` from your `App`; you hand it to LiveKit's
`cli.run_app()`.

```python
from google.adk.apps.app import App
from google.adk.integrations.livekit import livekit_server
from livekit.agents import cli

app = App(name="my_app", root_agent=root_agent)

# Default identity mapping (job metadata, then room name). Set agent_name to
# require explicit dispatch (AgentDispatchService.create_dispatch):
server = livekit_server(app=app, agent_name="my_app")

# Or map a JobContext -> (user_id, session_id) yourself:
# server = livekit_server(app=app, resolve_ids=lambda ctx: (...))

if __name__ == "__main__":
    cli.run_app(server)  # the developer owns the process
```

### Custom entrypoint (`LiveKitRunner`)

For custom dispatch, metadata, or multi-agent routing, write the entrypoint
yourself and drive `LiveKitRunner` directly.

```python
import json

from google.adk.integrations.livekit import LiveKitRunner
from google.adk.runners import InMemoryRunner
from livekit.agents import AgentServer, JobContext, cli

runner = InMemoryRunner(agent=root_agent, app_name="my_app")
runner.auto_create_session = True  # a dispatched room is a new conversation
server = AgentServer()

@server.rtc_session(agent_name="my_app")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()  # worker joins the dispatched room
    meta = json.loads(ctx.job.metadata or "{}")
    await LiveKitRunner(
        runner=runner,
        room=ctx.room,
        user_id=meta.get("user_id", "live-user"),
        session_id=meta.get("session_id", ctx.room.name),
    ).start()

if __name__ == "__main__":
    cli.run_app(server)
```

## Sessions

LiveKit has no notion of `user_id` / `session_id` -- those are ADK concepts. The
only identity dispatch hands you is `ctx.room.name` and the free-form
`ctx.job.metadata` string, so `livekit_server` takes a `resolve_ids` callable
that maps a `JobContext` to `(user_id, session_id)` (defaulting to job metadata,
then room name).

A dispatched room is normally a brand new conversation, so the runner
`livekit_server` builds from an `App` has `auto_create_session` enabled. A
`Runner` you pass in is used exactly as configured -- enable
`auto_create_session` on it yourself, or create the session out of band, or
`run_live` raises `SessionNotFoundError`.

## Run config

Both surfaces accept a `run_config`. It defaults to audio-only output; pass the
transcription options to have transcripts published on the room data track:

```python
from google.adk.agents.run_config import RunConfig
from google.genai import types

run_config = RunConfig(
    response_modalities=[types.Modality.AUDIO],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)
server = livekit_server(app=app, run_config=run_config)
```

## Data track protocol

The connector publishes JSON messages on the `adk` topic, and reads typed user
input from the same topic. Clients filter on the topic, so the room stays usable
by other applications.

Published by the agent:

| Message | Shape |
| :- | :- |
| Transcript | `{"type": "transcript", "role": "user"\|"agent", "text": ...}` |
| Tool call | `{"type": "function_call", "name": ..., "args": {...}}` |
| Tool result | `{"type": "function_response", "name": ..., "response": {...}}` |

Accepted from clients:

| Message | Shape |
| :- | :- |
| Text turn | `{"type": "text", "text": ...}` |

Partial transcripts are not published: they arrive token by token and would make
a client re-render the same utterance repeatedly.

## Media handling

- **Inbound audio** is requested at 16 kHz mono; LiveKit resamples on subscribe,
  so the bridge only wraps the bytes in a `Blob`. When a track ends (the
  participant muted, unpublished, or left) the bridge signals
  `send_audio_stream_end()` so a server-VAD turn does not hang.
- **Inbound video** is sampled at 1 fps, downscaled, and JPEG-encoded off the
  event loop. Live models sample video rather than consume it at capture rate,
  so forwarding every frame only floods the queue.
- **Outbound audio** is captured onto a 24 kHz local track. On an `interrupted`
  event the bridge clears the playback queue, so barge-in stops the agent
  talking over the user instead of draining the buffer first.

## Client ecosystem

Because `LiveKitRunner` joins a standard LiveKit room as a participant, every
LiveKit client SDK is a working front end with no ADK client code -- browser
(`client-sdk-js`), Unity (`client-sdk-unity`), and iOS/Android. LiveKit
maintains those clients; ADK maintains none.

See
[`contributing/samples/integrations/livekit_agent`](../../../../../../contributing/samples/integrations/livekit_agent/README.md)
for a runnable browser-to-agent sample.
