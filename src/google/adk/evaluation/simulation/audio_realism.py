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

"""Audio-realism transforms for persona-driven live evaluation.

A transform takes the persona's raw PCM audio and returns (possibly degraded)
PCM audio of the same format, simulating real-world conditions like background
noise. Transforms are PCM-in/PCM-out and mime-typed so they stay independent of
any particular audio container or modality.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import array
import random

from ...utils.feature_decorator import experimental
from .persona import AudioRealismConfig


@experimental
class AudioRealismTransform(ABC):
  """Applies a realism effect to PCM audio."""

  @abstractmethod
  async def apply(self, pcm: bytes, *, mime_type: str) -> bytes:
    """Returns transformed PCM audio of the same format as the input.

    Args:
      pcm: Raw little-endian 16-bit PCM audio bytes.
      mime_type: The MIME type of the audio (e.g. "audio/pcm;rate=16000").

    Returns:
      Transformed PCM bytes of the same format.
    """


@experimental
class NoOpAudioRealismTransform(AudioRealismTransform):
  """A transform that returns the audio unchanged."""

  async def apply(self, pcm: bytes, *, mime_type: str) -> bytes:
    return pcm


@experimental
class GaussianNoiseRealismTransform(AudioRealismTransform):
  """Mixes additive Gaussian noise into 16-bit PCM audio.

  This is a dependency-free default that stress-tests robustness to noisy
  audio. Richer transforms (channel effects, cross-talk) can be implemented
  against the `AudioRealismTransform` interface.
  """

  _MAX_INT16 = 32767
  _MIN_INT16 = -32768

  def __init__(self, config: AudioRealismConfig):
    self._config = config

  async def apply(self, pcm: bytes, *, mime_type: str) -> bytes:
    if not self._config.enabled or self._config.intensity <= 0.0 or not pcm:
      return pcm

    # 16-bit signed samples. The noise standard deviation scales with intensity.
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    noise_sd = self._config.intensity * 1500.0
    for i in range(len(samples)):
      noisy = samples[i] + int(random.gauss(0.0, noise_sd))
      samples[i] = max(self._MIN_INT16, min(self._MAX_INT16, noisy))
    return samples.tobytes()


@experimental
def build_audio_realism_transform(
    config: AudioRealismConfig | None,
) -> AudioRealismTransform:
  """Returns the transform implied by `config`, or a no-op when disabled."""
  if config is None or not config.enabled:
    return NoOpAudioRealismTransform()
  return GaussianNoiseRealismTransform(config)
