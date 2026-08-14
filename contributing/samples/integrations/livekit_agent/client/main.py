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

"""A tiny token + dispatch server for the LiveKit browser client.

This is the "separate app" that summons the ADK worker. A browser cannot hold
the LiveKit API secret, so this backend does two things the client can't:

  1. Mints a room-join access token for the browser participant.
  2. *Explicitly dispatches* the "roll_dice" agent into that room, passing
     user_id / session_id as job metadata (which the worker's `resolve_ids`
     reads back out).

The browser then connects with the token, publishes its mic, and plays the
agent's audio track -- exercising both bridges end to end.

Run it (after the worker is already running):

    pip install "google-adk[livekit]"
    # set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
    python -m contributing.samples.integrations.livekit_agent.client.main

Then open http://localhost:8080.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
  from livekit import api
except ImportError as e:
  raise ImportError(
      "livekit is not installed. Please install it with "
      '`pip install "google-adk[livekit]"`.'
  ) from e

# Must match `AGENT_NAME` in livekit_worker.py -- the dispatch name the worker
# registered under.
AGENT_NAME = "roll_dice"

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(
        f"{name} is not set. Set LIVEKIT_URL, LIVEKIT_API_KEY, and"
        " LIVEKIT_API_SECRET before starting the token server."
    )
  return value


@app.get("/")
async def index() -> FileResponse:
  """Serves the browser client."""
  return FileResponse(_STATIC_DIR / "index.html")


@app.get("/token")
async def token(room: str | None = None, identity: str | None = None) -> dict:
  """Mints a join token and dispatches the ADK worker into the room.

  Args:
    room: The room to join. Defaults to a fresh unique name (which also
      becomes the ADK session_id).
    identity: The participant identity, reused as the ADK user_id. Defaults to
      a fresh unique name.

  Returns:
    The LiveKit server URL and a join token for the browser to connect with.
  """
  livekit_url = _require_env("LIVEKIT_URL")
  api_key = _require_env("LIVEKIT_API_KEY")
  api_secret = _require_env("LIVEKIT_API_SECRET")

  room = room or f"roll-dice-{uuid.uuid4().hex[:8]}"
  identity = identity or f"caller-{uuid.uuid4().hex[:8]}"

  # LiveKit has no user_id / session_id -- pass ADK's identity as job metadata.
  metadata = json.dumps({"user_id": identity, "session_id": room})

  # Explicit dispatch: summon the worker into this specific room.
  async with api.LiveKitAPI(
      url=livekit_url, api_key=api_key, api_secret=api_secret
  ) as lkapi:
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name=AGENT_NAME, room=room, metadata=metadata
        )
    )

  join_token = (
      api.AccessToken(api_key, api_secret)
      .with_identity(identity)
      .with_grants(api.VideoGrants(room_join=True, room=room))
      .to_jwt()
  )

  return {"url": livekit_url, "token": join_token, "room": room}


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8080)
