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

"""Tests for PCM resampling used to relay audio between Live sessions."""

from __future__ import annotations

import array

from google.adk.evaluation.simulation.audio_utils import LIVE_INPUT_RATE_HZ
from google.adk.evaluation.simulation.audio_utils import parse_sample_rate
from google.adk.evaluation.simulation.audio_utils import resample_pcm16
from google.adk.evaluation.simulation.audio_utils import to_live_input


def _pcm(num_samples: int) -> bytes:
  return array.array("h", [(i % 100) for i in range(num_samples)]).tobytes()


def test_parse_sample_rate_reads_rate():
  assert parse_sample_rate("audio/pcm;rate=24000", default=16000) == 24000


def test_parse_sample_rate_falls_back_to_default():
  assert parse_sample_rate("audio/pcm", default=16000) == 16000
  assert parse_sample_rate(None, default=16000) == 16000


def test_resample_24k_to_16k_reduces_sample_count():
  """Downsampling 24kHz to 16kHz yields ~2/3 the samples."""
  pcm = _pcm(2400)

  out = resample_pcm16(pcm, src_rate=24000, dst_rate=16000)

  assert len(out) // 2 == 1600


def test_resample_is_noop_when_rates_match():
  pcm = _pcm(100)

  assert resample_pcm16(pcm, src_rate=16000, dst_rate=16000) == pcm


def test_resample_handles_empty():
  assert resample_pcm16(b"", src_rate=24000, dst_rate=16000) == b""


def test_to_live_input_converts_24k_source_to_16k():
  """A 24kHz captured utterance is resampled to the 16kHz Live input rate."""
  pcm = _pcm(2400)

  out = to_live_input(pcm, source_mime_type="audio/pcm;rate=24000")

  # 2400 samples @24k -> 1600 @16k
  assert len(out) // 2 == 1600


def test_to_live_input_noop_when_already_16k():
  pcm = _pcm(1600)

  out = to_live_input(pcm, source_mime_type="audio/pcm;rate=16000")

  assert out == pcm
  assert LIVE_INPUT_RATE_HZ == 16000
