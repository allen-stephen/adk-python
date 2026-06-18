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

"""Tests for converting a recorded live session into eval invocations.

Verifies that per-turn audio is persisted to the artifact service and that the
resulting invocations carry audio references back to those artifacts.
"""

from __future__ import annotations

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.evaluation.live_eval_conversion import convert_live_session_to_eval_invocations
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest


def _audio_part(data: bytes) -> types.Part:
  return types.Part(
      inline_data=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
  )


def _make_live_session() -> Session:
  user_event = Event(
      author="user",
      content=types.Content(role="user", parts=[_audio_part(b"user-audio")]),
      invocation_id="inv1",
  )
  agent_event = Event(
      author="agent",
      content=types.Content(role="model", parts=[_audio_part(b"agent-audio")]),
      invocation_id="inv1",
  )
  return Session(
      id="sess1",
      app_name="my-voice-agent",
      user_id="demo_user",
      events=[user_event, agent_event],
  )


@pytest.mark.asyncio
async def test_user_audio_is_persisted_and_referenced():
  """User audio bytes are saved as an artifact and referenced on the invocation."""
  artifact_service = InMemoryArtifactService()
  session = _make_live_session()

  invocations = await convert_live_session_to_eval_invocations(
      session=session,
      artifact_service=artifact_service,
      app_name="my-voice-agent",
      user_id="demo_user",
  )

  assert len(invocations) == 1
  user_audio = invocations[0].user_audio
  assert user_audio is not None
  assert user_audio.mime_type == "audio/pcm;rate=16000"

  stored = await artifact_service.load_artifact(
      app_name="my-voice-agent",
      user_id="demo_user",
      session_id="sess1",
      filename=user_audio.artifact_filename,
  )
  assert stored.inline_data.data == b"user-audio"


@pytest.mark.asyncio
async def test_agent_audio_is_persisted_and_referenced():
  """Agent audio bytes are saved as an artifact and referenced on the invocation."""
  artifact_service = InMemoryArtifactService()
  session = _make_live_session()

  invocations = await convert_live_session_to_eval_invocations(
      session=session,
      artifact_service=artifact_service,
      app_name="my-voice-agent",
      user_id="demo_user",
  )

  agent_audio = invocations[0].agent_audio
  assert agent_audio is not None

  stored = await artifact_service.load_artifact(
      app_name="my-voice-agent",
      user_id="demo_user",
      session_id="sess1",
      filename=agent_audio.artifact_filename,
  )
  assert stored.inline_data.data == b"agent-audio"


@pytest.mark.asyncio
async def test_no_audio_yields_no_reference():
  """A text-only turn produces no audio reference."""
  artifact_service = InMemoryArtifactService()
  user_event = Event(
      author="user",
      content=types.Content(role="user", parts=[types.Part(text="hello")]),
      invocation_id="inv1",
  )
  agent_event = Event(
      author="agent",
      content=types.Content(role="model", parts=[types.Part(text="hi")]),
      invocation_id="inv1",
  )
  session = Session(
      id="sess1",
      app_name="my-voice-agent",
      user_id="demo_user",
      events=[user_event, agent_event],
  )

  invocations = await convert_live_session_to_eval_invocations(
      session=session,
      artifact_service=artifact_service,
      app_name="my-voice-agent",
      user_id="demo_user",
  )

  assert invocations[0].user_audio is None
  assert invocations[0].agent_audio is None
