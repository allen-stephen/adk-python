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

"""Tests for the LiveKit server factory.

Verifies id resolution, app/runner validation, explicit-dispatch wiring, and
that a dispatched job drives a real `Runner.run_live` end to end.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.genai import types
import pytest

pytest.importorskip("livekit")

from google.adk.integrations.livekit import _livekit_server
from google.adk.integrations.livekit import livekit_server

sys.path.append("tests")

from unittests.testing_utils import MockModel  # noqa: E402

# --- Fixtures (minimal, one purpose each) ---


def _make_ctx(metadata: str = "", room_name: str = "room-42"):
  ctx = MagicMock()
  ctx.connect = AsyncMock()
  ctx.job.metadata = metadata
  ctx.room.name = room_name
  ctx.room.remote_participants = {}
  ctx.room.local_participant.publish_track = AsyncMock()
  ctx.room.local_participant.publish_data = AsyncMock()
  return ctx


def _make_app(name: str = "dice") -> App:
  """An App whose agent answers over a mocked live connection."""
  model = MockModel.create(responses=["you rolled a four"])
  return App(name=name, root_agent=LlmAgent(name="dice_agent", model=model))


# --- Construction ---


def test_requires_exactly_one_of_app_or_runner():
  """Passing neither app nor runner is rejected."""
  with pytest.raises(ValueError):
    livekit_server()


def test_rejects_both_app_and_runner():
  """Passing both app and runner is rejected."""
  with pytest.raises(ValueError):
    livekit_server(app=MagicMock(), runner=MagicMock(spec=Runner))


def test_agent_name_registered_for_explicit_dispatch():
  """The factory registers the entrypoint under the given agent_name."""
  server = livekit_server(runner=MagicMock(spec=Runner), agent_name="roll_dice")

  assert server._agent_name == "roll_dice"
  assert server._entrypoint_fnc is not None


def test_omitting_agent_name_leaves_jobs_auto_dispatched():
  """With no agent_name, LiveKit dispatches jobs to every room."""
  server = livekit_server(runner=MagicMock(spec=Runner))

  assert server._agent_name == ""


# --- Identity resolution ---


def test_default_ids_read_from_metadata():
  """The default resolver reads user_id/session_id from job metadata."""
  ctx = _make_ctx(metadata='{"user_id": "alice", "session_id": "call-7"}')

  user_id, session_id = _livekit_server._default_resolve_ids(ctx)

  assert user_id == "alice"
  assert session_id == "call-7"


def test_default_ids_fall_back_to_room_name():
  """With no metadata, session_id defaults to the room name."""
  ctx = _make_ctx(metadata="", room_name="room-42")

  user_id, session_id = _livekit_server._default_resolve_ids(ctx)

  assert user_id == "live-user"
  assert session_id == "room-42"


@pytest.mark.asyncio
async def test_entrypoint_connects_and_starts_runner():
  """The registered entrypoint connects the worker, then drives a runner."""
  runner = MagicMock(spec=Runner)
  ctx = _make_ctx(metadata='{"user_id": "bob", "session_id": "call-9"}')

  server = livekit_server(runner=runner)
  entrypoint = _entrypoint_of(server)

  with patch.object(_livekit_server, "LiveKitRunner") as mock_runner_cls:
    mock_runner_cls.return_value.start = AsyncMock()
    await entrypoint(ctx)

  ctx.connect.assert_awaited_once()
  _, kwargs = mock_runner_cls.call_args
  assert kwargs["user_id"] == "bob"
  assert kwargs["session_id"] == "call-9"
  mock_runner_cls.return_value.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_resolver_overrides_the_default():
  """A caller-supplied resolver decides the ADK identity."""
  ctx = _make_ctx(room_name="room-42")
  server = livekit_server(
      runner=MagicMock(spec=Runner),
      resolve_ids=lambda ctx: ("caller", f"call-{ctx.room.name}"),
  )

  with patch.object(_livekit_server, "LiveKitRunner") as mock_runner_cls:
    mock_runner_cls.return_value.start = AsyncMock()
    await _entrypoint_of(server)(ctx)

  _, kwargs = mock_runner_cls.call_args
  assert kwargs["user_id"] == "caller"
  assert kwargs["session_id"] == "call-room-42"


@pytest.mark.asyncio
async def test_run_config_reaches_the_session():
  """A run_config given to the factory is applied to every session."""
  run_config = RunConfig(
      response_modalities=[types.Modality.AUDIO],
      output_audio_transcription=types.AudioTranscriptionConfig(),
  )
  server = livekit_server(runner=MagicMock(spec=Runner), run_config=run_config)

  with patch.object(_livekit_server, "LiveKitRunner") as mock_runner_cls:
    mock_runner_cls.return_value.start = AsyncMock()
    await _entrypoint_of(server)(_make_ctx())

  _, kwargs = mock_runner_cls.call_args
  assert kwargs["run_config"] is run_config


# --- Sessions ---


def test_runner_built_from_app_creates_sessions():
  """A dispatched room has no session yet, so the runner must create one."""
  with patch.object(_livekit_server, "InMemoryRunner") as mock_runner_cls:
    livekit_server(app=_make_app())

  assert mock_runner_cls.return_value.auto_create_session is True


def test_caller_supplied_runner_is_left_alone():
  """A Runner passed in is used exactly as configured."""
  runner = MagicMock(spec=Runner)
  runner.auto_create_session = False

  livekit_server(runner=runner)

  assert runner.auto_create_session is False


@pytest.mark.asyncio
async def test_dispatched_job_creates_the_session_it_was_given():
  """End to end: a dispatched room reaches `run_live` over a real Runner.

  This is the regression guard for the session lifecycle. An `InMemoryRunner`
  defaults to `auto_create_session=False`, so a brand new room used to raise
  `SessionNotFoundError` before a single frame moved -- which no amount of
  mocking the Runner could catch.
  """
  app = _make_app()
  ctx = _make_ctx(metadata='{"user_id": "alice", "session_id": "call-1"}')
  server, runner = _server_and_runner(app)

  # `run_live` stays open for the life of the call, so drive the entrypoint as
  # a task and stop it once the session is established.
  task = asyncio.create_task(_entrypoint_of(server)(ctx))
  try:
    session = await _await_session(runner, app.name, "alice", "call-1")
    assert session is not None, _failure_of(task) or (
        "run_live never created the dispatched session."
    )
  finally:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task

  ctx.connect.assert_awaited_once()
  # The outbound audio track is published before run_live is driven.
  ctx.room.local_participant.publish_track.assert_awaited_once()


def _server_and_runner(app: App):
  """Builds a server, capturing the real Runner the factory wired up."""
  built = {}
  real_cls = _livekit_server.InMemoryRunner

  def _capture(*args, **kwargs):
    built["runner"] = real_cls(*args, **kwargs)
    return built["runner"]

  with patch.object(_livekit_server, "InMemoryRunner", side_effect=_capture):
    server = livekit_server(app=app)
  return server, built["runner"]


async def _await_session(runner: Runner, app_name: str, user_id: str, sid: str):
  """Polls the runner's session service until the session shows up."""
  for _ in range(100):
    session = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=sid
    )
    if session is not None:
      return session
    await asyncio.sleep(0.05)
  return None


def _failure_of(task: asyncio.Task) -> str:
  """Renders a task's exception, so a crashed session reports its own cause."""
  if not task.done() or task.cancelled():
    return ""
  exc = task.exception()
  return f"Entrypoint raised {type(exc).__name__}: {exc}" if exc else ""


def _entrypoint_of(server):
  """Returns the rtc_session handler the factory registered."""
  entrypoint = server._entrypoint_fnc
  assert entrypoint is not None, "No entrypoint registered on the AgentServer."
  return entrypoint
