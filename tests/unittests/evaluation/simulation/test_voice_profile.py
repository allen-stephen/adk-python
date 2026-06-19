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

"""Tests for the voice/realism layer and audio-realism transforms."""

from __future__ import annotations

import array

from google.adk.evaluation.simulation.audio_realism import \
    build_audio_realism_transform
from google.adk.evaluation.simulation.audio_realism import \
    NoOpAudioRealismTransform
from google.adk.evaluation.simulation.voice_profile import AudioRealismConfig
from google.adk.evaluation.simulation.voice_profile import LiveTransport
from google.adk.evaluation.simulation.voice_profile import VoiceProfile
import pytest


def _pcm(num_samples: int, value: int = 1000) -> bytes:
  return array.array("h", [value] * num_samples).tobytes()


def test_voice_profile_defaults():
  profile = VoiceProfile()

  assert profile.voice_name == "Aoede"
  assert profile.language_code == "en-US"
  assert profile.transport is None
  assert profile.audio_realism is None
  assert profile.barge_in is None


def test_voice_profile_transport_override():
  profile = VoiceProfile(transport=LiveTransport.NATIVE_AUDIO)

  assert profile.transport == LiveTransport.NATIVE_AUDIO


def test_build_transform_disabled_is_noop():
  assert isinstance(
      build_audio_realism_transform(None), NoOpAudioRealismTransform
  )
  assert isinstance(
      build_audio_realism_transform(AudioRealismConfig(enabled=False)),
      NoOpAudioRealismTransform,
  )


@pytest.mark.asyncio
async def test_noise_changes_audio():
  transform = build_audio_realism_transform(
      AudioRealismConfig(enabled=True, intensity=0.5)
  )
  pcm = _pcm(200)

  out = await transform.apply(pcm, mime_type="audio/pcm;rate=16000")

  assert len(out) == len(pcm)
  assert out != pcm


@pytest.mark.asyncio
async def test_speaking_rate_changes_length():
  # Speaking faster (rate > 1) yields fewer samples.
  transform = build_audio_realism_transform(
      AudioRealismConfig(enabled=True, speaking_rate=2.0)
  )
  pcm = _pcm(400)

  out = await transform.apply(pcm, mime_type="audio/pcm;rate=16000")

  assert len(out) < len(pcm)


@pytest.mark.asyncio
async def test_background_noise_applies_floor_at_zero_intensity():
  transform = build_audio_realism_transform(
      AudioRealismConfig(enabled=True, intensity=0.0, background_noise=True)
  )
  pcm = _pcm(200)

  out = await transform.apply(pcm, mime_type="audio/pcm;rate=16000")

  assert out != pcm
