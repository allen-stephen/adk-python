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

"""Server factory for the LiveKit connector.

`livekit_server` absorbs the `ctx.connect()` + metadata + wiring boilerplate and
returns a ready `AgentServer` built from an ADK `App` (or a preconfigured
`Runner`). This mirrors `SlackRunner`, which hands you a wired framework object:
the developer still owns the worker process by handing the returned server to
LiveKit's own `cli.run_app()`.

LiveKit has no notion of `user_id` / `session_id` -- those are ADK concepts. The
only identity dispatch hands you is `ctx.room.name` and the free-form
`ctx.job.metadata` string. So the factory takes an optional resolver mapping a
`JobContext` to `(user_id, session_id)`, defaulting to job metadata then room
name.

Setting `agent_name` enables *explicit dispatch*: jobs are not auto-dispatched to
rooms; a caller summons the worker via `AgentDispatchService.create_dispatch` (or
by naming the agent in a participant token).

This module lazily imports the LiveKit SDK; install it with::

    pip install "google-adk[livekit]"
"""

from __future__ import annotations

import json
import logging
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

from ...agents.run_config import RunConfig
from ...runners import InMemoryRunner
from ...runners import Runner
from ._livekit_runner import LiveKitRunner

try:
  from livekit.agents import AgentServer
  from livekit.agents import JobExecutorType
except ImportError as e:
  raise ImportError(
      "livekit is not installed. Please install it with "
      '`pip install "google-adk[livekit]"`.'
  ) from e

if TYPE_CHECKING:
  from livekit.agents import JobContext

  from ...apps.app import App

logger = logging.getLogger("google_adk." + __name__)

ResolveIds = Callable[["JobContext"], tuple[str, str]]


def _default_resolve_ids(ctx: JobContext) -> tuple[str, str]:
  """Resolves ids from job metadata, falling back to the room name."""
  metadata = json.loads(ctx.job.metadata or "{}")
  user_id = metadata.get("user_id", "live-user")
  session_id = metadata.get("session_id", ctx.room.name)
  return user_id, session_id


def livekit_server(
    *,
    app: Optional[App] = None,
    runner: Optional[Runner] = None,
    resolve_ids: Optional[ResolveIds] = None,
    run_config: Optional[RunConfig] = None,
    agent_name: str = "",
) -> AgentServer:
  """Builds a LiveKit `AgentServer` that drives an ADK live session.

  Exactly one of `app` or `runner` must be provided. Hand the returned server to
  LiveKit's `cli.run_app()`; the developer owns the process, just like
  `SlackRunner.start()`.

  A dispatched room is normally a brand new conversation, so the runner built
  from `app` has `auto_create_session` enabled. A `runner` you pass in is used
  exactly as configured -- enable `auto_create_session` on it yourself, or
  create the session out of band, or `run_live` raises `SessionNotFoundError`.

  Args:
    app: An ADK `App`; an `InMemoryRunner` is built from its `root_agent`.
    runner: A preconfigured ADK `Runner`, used as-is.
    resolve_ids: Optional resolver mapping a `JobContext` to
      `(user_id, session_id)`. Defaults to job metadata, then room name.
    run_config: Optional run config for every session this server serves.
      Defaults to AUDIO response modality. Set the audio transcription options
      here to have transcripts published on the room data track.
    agent_name: If set, enables explicit dispatch under this name. Callers
      summon the worker via `AgentDispatchService.create_dispatch`. When empty,
      jobs are auto-dispatched to every room.

  Returns:
    A LiveKit `AgentServer` ready for `cli.run_app()`.

  Raises:
    ValueError: If not exactly one of `app` / `runner` is provided.
  """
  if (app is None) == (runner is None):
    raise ValueError("Provide exactly one of `app` or `runner`.")

  if runner is not None:
    resolved_runner = runner
  else:
    assert app is not None  # Guaranteed by the check above; narrows for mypy.
    resolved_runner = InMemoryRunner(agent=app.root_agent, app_name=app.name)
    # `InMemoryRunner` does not forward this through its constructor, and a
    # dispatched room has no session yet.
    resolved_runner.auto_create_session = True

  resolve = resolve_ids or _default_resolve_ids

  async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()  # The worker joins the dispatched room.
    user_id, session_id = resolve(ctx)
    await LiveKitRunner(
        runner=resolved_runner,
        room=ctx.room,
        user_id=user_id,
        session_id=session_id,
        run_config=run_config,
    ).start()

  # Jobs run as threads, not subprocesses. LiveKit's default process executor
  # pickles the entrypoint and ships it to a spawned worker, and neither half of
  # what this factory closes over survives that: the entrypoint is a closure,
  # and an ADK `Runner` owns model clients, service handles and async state that
  # cannot be pickled. Threads also keep one `Runner` -- and so one session
  # service -- shared across every call the worker serves. ADK live sessions are
  # I/O-bound on the model socket, so the GIL is not the constraint here.
  #
  # To use process isolation instead, skip this factory: define a module-level
  # entrypoint that builds its own `Runner` inside the job, and drive
  # `LiveKitRunner` from it.
  server = AgentServer(job_executor_type=JobExecutorType.THREAD)
  server.rtc_session(entrypoint, agent_name=agent_name)
  return server
