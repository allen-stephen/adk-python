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

"""Tests for the local speaking-rate acoustic metric."""

from __future__ import annotations

from google.adk.evaluation.acoustic_evaluator import SpeakingRateV1Evaluator
from google.adk.evaluation.eval_case import AudioReference
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types as genai_types


def _invocation(
    *,
    words: int,
    num_samples: int,
    sample_rate_hz: int = 24000,
    was_interrupted: bool = False,
):
  text = " ".join(["word"] * words) if words else ""
  return Invocation(
      invocation_id="t",
      user_content=genai_types.Content(
          role="user", parts=[genai_types.Part(text="hi")]
      ),
      final_response=genai_types.Content(
          role="model", parts=[genai_types.Part(text=text)]
      ),
      agent_audio=AudioReference(
          artifact_filename="a.pcm",
          sample_rate_hz=sample_rate_hz,
          num_samples=num_samples,
      ),
      was_interrupted=was_interrupted,
  )


def test_speaking_rate_passes_under_threshold():
  evaluator = SpeakingRateV1Evaluator(threshold=5.0)
  invocation = _invocation(words=6, num_samples=48000)  # 2 seconds -> 3 wps

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.overall_score == 3.0


def test_speaking_rate_fails_over_threshold():
  evaluator = SpeakingRateV1Evaluator(threshold=5.0)
  invocation = _invocation(words=20, num_samples=48000)  # 2s -> 10 wps

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.FAILED


def test_not_evaluated_without_agent_audio():
  evaluator = SpeakingRateV1Evaluator(threshold=3.5)
  invocation = Invocation(
      invocation_id="t",
      user_content=genai_types.Content(
          role="user", parts=[genai_types.Part(text="hi")]
      ),
      final_response=genai_types.Content(
          role="model", parts=[genai_types.Part(text="some words here")]
      ),
  )

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert result.overall_score is None


def test_not_evaluated_without_words():
  evaluator = SpeakingRateV1Evaluator(threshold=3.5)
  invocation = _invocation(words=0, num_samples=48000)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED


def test_not_evaluated_when_interrupted():
  # An interrupted turn has only a fragment of audio; do not score its rate.
  evaluator = SpeakingRateV1Evaluator(threshold=3.5)
  invocation = _invocation(words=6, num_samples=48000, was_interrupted=True)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert result.overall_score is None
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
  )


def test_not_evaluated_when_audio_too_short():
  # 6 words over 6000 samples @ 24kHz = 0.25s, below the min duration guard, so
  # the implausible 24 wps is not scored.
  evaluator = SpeakingRateV1Evaluator(threshold=3.5)
  invocation = _invocation(words=6, num_samples=6000)

  result = evaluator.evaluate_invocations([invocation])

  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert result.overall_score is None


def test_interrupted_turn_excluded_from_average():
  # The interrupted turn is skipped; only the valid turn drives the score.
  evaluator = SpeakingRateV1Evaluator(threshold=3.5)
  valid = _invocation(words=6, num_samples=48000)  # 3 wps
  interrupted = _invocation(
      words=20, num_samples=6000, was_interrupted=True
  )  # would be 80 wps

  result = evaluator.evaluate_invocations([valid, interrupted])

  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.overall_score == 3.0
