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

"""Run the roll_dice agent over LiveKit (telephony / WebRTC / Unity).

This puts the SAME `root_agent` from `agent.py` behind a live transport with no
changes to the agent itself. This file is a thin consumer of
`google.adk.integrations.livekit`; `adk web` on `agent.py` still works exactly
as before.

    # 1. install the optional extra
    pip install "google-adk[livekit]"

    # 2. set LiveKit + model credentials (env)
    #    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, plus your Vertex/
    #    Gemini credentials for gemini-live-2.5-flash-native-audio.

    # 3. start the worker; it registers with LiveKit and waits for dispatch
    python -m contributing.samples.integrations.livekit_agent.livekit_worker

The worker registers under the dispatch name "roll_dice" and does nothing until
a job is *explicitly dispatched* into a room (see `client/main.py`). Once
dispatched, the same dice agent answers a browser mic, a phone call (SIP), or a
Unity NPC -- same agent, no code change.

ADK never owns the worker process: `livekit_server` builds a wired `AgentServer`
from your `App`, and this file hands it to LiveKit's own `cli.run_app()`, which
owns dispatch and lifecycle -- mirroring how `SlackRunner.start()` hands off to
Slack's runloop.
"""

from __future__ import annotations

import json

from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.integrations.livekit import livekit_server  # pip install "google-adk[livekit]"
from google.genai import types
from livekit.agents import cli

# The unchanged sample agent. This is the whole point: agent logic stays in ADK.
from .agent import root_agent

# The dispatch name callers target when summoning this worker into a room.
AGENT_NAME = "roll_dice"

app = App(name=AGENT_NAME, root_agent=root_agent)

# Audio out, plus transcription of both sides. The transcripts are what the
# browser renders as a live caption track -- the connector publishes them on the
# room data track, so no ADK client code is needed to read them.
run_config = RunConfig(
    response_modalities=[types.Modality.AUDIO],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)


# --- Server factory (recommended) ---------------------------------------------
# ADK builds a wired AgentServer from your App; you own the process by handing it
# to LiveKit's cli.run_app().
#
# LiveKit has no notion of user_id / session_id -- those are ADK concepts. The
# only identity LiveKit hands you on the JobContext is `ctx.room.name` and the
# free-form `ctx.job.metadata` string (set by whoever dispatched the job). The
# caller in `client/main.py` puts user_id / session_id in that metadata, and the
# resolver below reads it back out.


def resolve_ids(ctx) -> tuple[str, str]:
  meta = json.loads(ctx.job.metadata or "{}")
  user_id = meta.get("user_id", "live-user")
  session_id = meta.get("session_id", ctx.room.name)  # room name = call id
  return user_id, session_id


# agent_name enables explicit dispatch: the worker waits until a caller summons
# it by this name (see client/main.py).
server = livekit_server(
    app=app,
    resolve_ids=resolve_ids,
    run_config=run_config,
    agent_name=AGENT_NAME,
)

# For the simplest case you can omit resolve_ids and run_config entirely and
# accept the defaults (metadata -> room name; audio out, no transcripts):
#
#     server = livekit_server(app=app, agent_name=AGENT_NAME)


# --- Custom entrypoint (full control: custom dispatch / routing) --------------
# Equivalent to the factory above, written out. Uncomment to use instead.
#
# from google.adk.integrations.livekit import LiveKitRunner
# from google.adk.runners import InMemoryRunner
# from livekit.agents import AgentServer, JobContext
#
# runner = InMemoryRunner(agent=root_agent, app_name=AGENT_NAME)
# # A dispatched room is a brand new conversation, so let the runner create the
# # session. The factory above does this for you.
# runner.auto_create_session = True
# server = AgentServer()
#
# @server.rtc_session(agent_name=AGENT_NAME)
# async def entrypoint(ctx: JobContext) -> None:
#   await ctx.connect()                              # worker joins the room
#   meta = json.loads(ctx.job.metadata or "{}")      # session/user from dispatch
#   await LiveKitRunner(
#       runner=runner,
#       room=ctx.room,                               # room delivered by dispatch
#       user_id=meta.get("user_id", "live-user"),
#       session_id=meta.get("session_id", ctx.room.name),
#       run_config=run_config,
#   ).start()


if __name__ == "__main__":
  # LiveKit owns the process; ADK just supplied the wired server.
  cli.run_app(server)
