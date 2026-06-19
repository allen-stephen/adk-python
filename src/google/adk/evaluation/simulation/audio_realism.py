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

"""Audio-realism transforms applied to the user's audio in a live eval.

A transform takes the user's raw PCM audio and returns (possibly degraded) PCM
audio, simulating real-world conditions like background noise or a faster/slower
speaker. Transforms are PCM-in/PCM-out and mime-typed so they stay independent
of any particular audio container or modality.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import array
import random

from ...utils.feature_decorator import experimental
from .audio_utils import parse_sample_rate
from .audio_utils import resample_pcm16
from .voice_profile import AudioRealismConfig


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
class CompositeAudioRealismTransform(AudioRealismTransform):
  """Applies a configured set of realism effects to 16-bit PCM audio.

  Effects are dependency-free so the local loop runs without extra packages:
  additive Gaussian noise (optionally always-on as background noise) and a
  speaking-rate (time-stretch) change. Richer transforms (channel effects,
  cross-talk) can be implemented against `AudioRealismTransform`.
  """

  _MAX_INT16 = 32767
  _MIN_INT16 = -32768

  def __init__(self, config: AudioRealismConfig):
    self._config = config

  async def apply(self, pcm: bytes, *, mime_type: str) -> bytes:
    if not self._config.enabled or not pcm:
      return pcm

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])

    if self._config.intensity > 0.0 or self._config.background_noise:
      self._add_noise(samples)

    out = samples.tobytes()

    if self._config.speaking_rate != 1.0:
      out = self._change_rate(out, mime_type=mime_type)

    return out

  def _add_noise(self, samples: array.array) -> None:
    """Mixes additive Gaussian noise into the samples in place.

    A small floor is applied when `background_noise` is set so noise is present
    even at low intensity, modelling an always-on noisy channel.
    """
    floor = 300.0 if self._config.background_noise else 0.0
    noise_sd = max(floor, self._config.intensity * 1500.0)
    if noise_sd <= 0.0:
      return
    for i in range(len(samples)):
      noisy = samples[i] + int(random.gauss(0.0, noise_sd))
      samples[i] = max(self._MIN_INT16, min(self._MAX_INT16, noisy))

  def _change_rate(self, pcm: bytes, *, mime_type: str) -> bytes:
    """Time-stretches audio to change the speaking rate, preserving the rate.

    Resampling to a scaled rate and relabelling it as the original rate makes
    the speech play faster (rate > 1) or slower (rate < 1) without changing the
    sample rate the agent expects.
    """
    src_rate = parse_sample_rate(mime_type, default=16000)
    scaled_rate = max(1, int(src_rate / self._config.speaking_rate))
    return resample_pcm16(pcm, src_rate=src_rate, dst_rate=scaled_rate)


@experimental
def build_audio_realism_transform(
    config: AudioRealismConfig | None,
) -> AudioRealismTransform:
  """Returns the transform implied by `config`, or a no-op when disabled."""
  if config is None or not config.enabled:
    return NoOpAudioRealismTransform()
  return CompositeAudioRealismTransform(config)
