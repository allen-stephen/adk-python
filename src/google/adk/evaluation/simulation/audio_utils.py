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

"""Audio helpers for relaying PCM between two Live sessions.

The Live API consumes 16 kHz, 16-bit, mono PCM as input but emits 24 kHz PCM
as output. To relay one agent's spoken audio into the other agent we must
resample from the output rate to the input rate, otherwise the receiver hears
sped-up, mis-pitched audio it cannot transcribe.
"""

from __future__ import annotations

import array
import re

# The Live API input/output sample rates (16-bit mono PCM).
LIVE_INPUT_RATE_HZ = 16000
LIVE_OUTPUT_RATE_HZ = 24000
LIVE_INPUT_MIME_TYPE = "audio/pcm;rate=16000"

_RATE_RE = re.compile(r"rate=(\d+)")


def parse_sample_rate(mime_type: str | None, *, default: int) -> int:
  """Extracts the sample rate from a mime type like 'audio/pcm;rate=24000'."""
  if not mime_type:
    return default
  match = _RATE_RE.search(mime_type)
  return int(match.group(1)) if match else default


def resample_pcm16(pcm: bytes, *, src_rate: int, dst_rate: int) -> bytes:
  """Resamples 16-bit mono PCM from src_rate to dst_rate.

  Uses linear interpolation, which is sufficient for speech relay and avoids a
  heavy DSP dependency.

  Args:
    pcm: Raw little-endian signed 16-bit mono PCM bytes.
    src_rate: The source sample rate in Hz.
    dst_rate: The target sample rate in Hz.

  Returns:
    Resampled PCM bytes at dst_rate. Returns the input unchanged when the rates
    match or the input is empty.
  """
  if not pcm or src_rate == dst_rate:
    return pcm

  samples = array.array("h")
  samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
  if len(samples) < 2:
    return pcm

  ratio = src_rate / dst_rate
  out_len = max(1, int(len(samples) / ratio))
  out = array.array("h", bytes(2 * out_len))
  last_index = len(samples) - 1
  for i in range(out_len):
    src_pos = i * ratio
    left = int(src_pos)
    right = min(left + 1, last_index)
    frac = src_pos - left
    out[i] = int(samples[left] * (1.0 - frac) + samples[right] * frac)
  return out.tobytes()


def to_live_input(pcm: bytes, *, source_mime_type: str | None) -> bytes:
  """Converts captured output audio to 16 kHz PCM for Live API input."""
  src_rate = parse_sample_rate(source_mime_type, default=LIVE_OUTPUT_RATE_HZ)
  return resample_pcm16(pcm, src_rate=src_rate, dst_rate=LIVE_INPUT_RATE_HZ)
