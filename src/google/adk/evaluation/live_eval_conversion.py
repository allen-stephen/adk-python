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

"""Utilities for turning a recorded live (voice) session into an eval case.

A live session captures audio blobs and transcripts that the standard eval-case
conversion drops on the floor. This module persists the per-turn audio to the
ArtifactService and attaches `AudioReference`s to the produced invocations, so a
recorded voice conversation can be replayed and scored as a faithful eval case.
"""

from __future__ import annotations

from typing import Optional

from google.genai import types as genai_types

from ..artifacts.base_artifact_service import BaseArtifactService
from ..sessions.session import Session
from ..utils.feature_decorator import experimental
from .eval_case import AudioReference
from .eval_case import Invocation
from .evaluation_generator import EvaluationGenerator

_DEFAULT_AUDIO_MIME_TYPE = "audio/pcm;rate=16000"
_DEFAULT_SAMPLE_RATE_HZ = 16000


def _is_audio_part(part: genai_types.Part) -> bool:
  return (
      part.inline_data is not None
      and part.inline_data.mime_type is not None
      and part.inline_data.mime_type.startswith("audio/")
  )


def _extract_audio(content: Optional[genai_types.Content]) -> Optional[bytes]:
  """Concatenates the audio bytes from a content's parts, if any."""
  if content is None or not content.parts:
    return None
  chunks = [
      part.inline_data.data
      for part in content.parts
      if _is_audio_part(part) and part.inline_data.data
  ]
  if not chunks:
    return None
  return b"".join(chunks)


@experimental
async def convert_live_session_to_eval_invocations(
    *,
    session: Session,
    artifact_service: BaseArtifactService,
    app_name: str,
    user_id: str,
) -> list[Invocation]:
  """Converts a recorded live session into eval invocations with audio refs.

  This mirrors `convert_session_to_eval_invocations` but additionally persists
  the per-turn audio (user input and agent response) to the artifact service and
  attaches `AudioReference`s to each invocation.

  Args:
    session: The recorded live session.
    artifact_service: The artifact service used to persist audio bytes.
    app_name: The app name used as the artifact namespace.
    user_id: The user id used as the artifact namespace.

  Returns:
    The list of invocations, with audio references attached where audio was
    present.
  """
  events = session.events if session and session.events else []
  invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

  for index, invocation in enumerate(invocations):
    user_audio = _extract_audio(invocation.user_content)
    if user_audio:
      invocation.user_audio = await _persist_audio(
          artifact_service=artifact_service,
          app_name=app_name,
          user_id=user_id,
          session_id=session.id,
          filename=f"{invocation.invocation_id or index}_user_audio.pcm",
          audio=user_audio,
      )

    agent_audio = _extract_audio(invocation.final_response)
    if agent_audio:
      invocation.agent_audio = await _persist_audio(
          artifact_service=artifact_service,
          app_name=app_name,
          user_id=user_id,
          session_id=session.id,
          filename=f"{invocation.invocation_id or index}_agent_audio.pcm",
          audio=agent_audio,
      )

  return invocations


async def _persist_audio(
    *,
    artifact_service: BaseArtifactService,
    app_name: str,
    user_id: str,
    session_id: str,
    filename: str,
    audio: bytes,
) -> AudioReference:
  """Saves audio bytes as an artifact and returns a reference to it."""
  version = await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=genai_types.Part.from_bytes(
          data=audio, mime_type=_DEFAULT_AUDIO_MIME_TYPE
      ),
  )
  return AudioReference(
      artifact_filename=filename,
      version=version,
      mime_type=_DEFAULT_AUDIO_MIME_TYPE,
      sample_rate_hz=_DEFAULT_SAMPLE_RATE_HZ,
  )
