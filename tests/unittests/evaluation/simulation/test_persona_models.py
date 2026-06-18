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

"""Tests for persona-driven live eval typed models and realism transforms."""

from __future__ import annotations

import array

from google.adk.evaluation.simulation.audio_realism import build_audio_realism_transform
from google.adk.evaluation.simulation.audio_realism import GaussianNoiseRealismTransform
from google.adk.evaluation.simulation.audio_realism import NoOpAudioRealismTransform
from google.adk.evaluation.simulation.live_conversation_scenario import LiveConversationScenario
from google.adk.evaluation.simulation.persona import AudioRealismConfig
from google.adk.evaluation.simulation.persona import BargeInConfig
from google.adk.evaluation.simulation.persona import Persona
import pytest


def _make_persona() -> Persona:
  return Persona(
      id="hungry_customer",
      character_prompt="You are a hungry customer ordering lunch.",
      goal="Order a burger and fries.",
      voice_name="Kore",
  )


def test_persona_defaults():
  """A persona has sensible voice/language defaults."""
  persona = _make_persona()

  assert persona.voice_name == "Kore"
  assert persona.language_code == "en-US"
  assert persona.model is None


def test_scenario_requires_persona_and_defaults():
  """A scenario wraps a persona with a default max-turn cap and no realism."""
  scenario = LiveConversationScenario(persona=_make_persona())

  assert scenario.persona.id == "hungry_customer"
  assert scenario.max_turns == 10
  assert scenario.barge_in is None
  assert scenario.audio_realism is None


def test_scenario_serializes_camelcase():
  """Scenario serializes with camelCase aliases like other eval models."""
  scenario = LiveConversationScenario(
      persona=_make_persona(),
      barge_in=BargeInConfig(enabled=True, probability=0.3),
  )

  dumped = scenario.model_dump(by_alias=True)

  assert "maxTurns" in dumped
  assert dumped["bargeIn"]["enabled"] is True


def test_build_transform_returns_noop_when_disabled():
  """A disabled or absent realism config yields a no-op transform."""
  assert isinstance(
      build_audio_realism_transform(None), NoOpAudioRealismTransform
  )
  assert isinstance(
      build_audio_realism_transform(AudioRealismConfig(enabled=False)),
      NoOpAudioRealismTransform,
  )


def test_build_transform_returns_noise_when_enabled():
  """An enabled realism config yields the noise transform."""
  transform = build_audio_realism_transform(
      AudioRealismConfig(enabled=True, intensity=0.5)
  )

  assert isinstance(transform, GaussianNoiseRealismTransform)


@pytest.mark.asyncio
async def test_noop_transform_passes_audio_through():
  """The no-op transform returns the audio unchanged."""
  pcm = b"\x01\x00\x02\x00\x03\x00"

  result = await NoOpAudioRealismTransform().apply(
      pcm, mime_type="audio/pcm;rate=16000"
  )

  assert result == pcm


@pytest.mark.asyncio
async def test_noise_transform_changes_audio_but_keeps_length():
  """The noise transform alters samples while preserving byte length."""
  samples = array.array("h", [0] * 1000)
  pcm = samples.tobytes()

  result = await GaussianNoiseRealismTransform(
      AudioRealismConfig(enabled=True, intensity=1.0)
  ).apply(pcm, mime_type="audio/pcm;rate=16000")

  assert len(result) == len(pcm)
  assert result != pcm
