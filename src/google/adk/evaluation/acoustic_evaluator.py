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

"""A local acoustic metric: speaking rate for live (voice) eval.

Beyond latency, *how* an agent sounds matters — a voice agent that talks too
fast or too slow is harder to follow. Speaking rate (words per second of spoken
audio) is a simple, local proxy for this. It is computed from the agent's
transcript and the duration of its captured audio, with no cloud project, and it
establishes the seam that richer acoustic metrics (turn-taking, talk-ratio,
intelligibility) will plug into.
"""

from __future__ import annotations

import logging
from typing import ClassVar
from typing import Optional

from typing_extensions import override

from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

logger = logging.getLogger("google_adk." + __name__)

# A comfortable conversational speaking rate is roughly 2-3 words/second. The
# default threshold passes runs at or below this upper bound.
_DEFAULT_SPEAKING_RATE_THRESHOLD_WPS = 5.0

# Speaking rate is words divided by audio duration. Below this duration the
# captured audio is too short for a meaningful rate (a fragment of audio paired
# with a full transcript yields an implausible value), so such turns are
# reported NOT_EVALUATED rather than scored.
_MIN_RATE_DURATION_SECONDS = 0.5


class SpeakingRateV1Evaluator(Evaluator):
  """Measures the agent's speaking rate (words per second) per invocation.

  For each invocation that has agent audio, the rate is the agent transcript's
  word count divided by the audio duration (derived from the `AudioReference`'s
  sample count and rate). Lower-or-equal-to-threshold passes; an invocation with
  no agent audio or duration is reported NOT_EVALUATED rather than failed.

  This metric runs entirely locally and requires no cloud project.
  """

  criterion_type: ClassVar[type[BaseCriterion]] = BaseCriterion

  def __init__(
      self,
      threshold: Optional[float] = None,
      eval_metric: Optional[EvalMetric] = None,
  ):
    if threshold is not None and eval_metric:
      raise ValueError(
          "Either eval_metric should be specified or threshold should be"
          " specified."
      )

    if eval_metric and eval_metric.criterion:
      self._threshold = eval_metric.criterion.threshold
    elif eval_metric and eval_metric.threshold is not None:
      self._threshold = eval_metric.threshold
    elif threshold is not None:
      self._threshold = threshold
    else:
      self._threshold = _DEFAULT_SPEAKING_RATE_THRESHOLD_WPS

  @staticmethod
  def _speaking_rate(invocation: Invocation) -> Optional[float]:
    """Returns the agent's words-per-second for an invocation, if derivable."""
    audio = invocation.agent_audio
    if (
        audio is None
        or not audio.num_samples
        or not audio.sample_rate_hz
        or invocation.final_response is None
        or not invocation.final_response.parts
    ):
      return None

    duration_seconds = audio.num_samples / audio.sample_rate_hz
    if duration_seconds < _MIN_RATE_DURATION_SECONDS:
      return None

    word_count = sum(
        len(part.text.split())
        for part in invocation.final_response.parts
        if part.text
    )
    if word_count == 0:
      return None

    return word_count / duration_seconds

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    per_invocation_results = []
    rates = []

    for actual in actual_invocations:
      # A barged-in turn was cut short, so its captured audio is only a fragment
      # and not a reliable basis for speaking rate.
      rate = None if actual.was_interrupted else self._speaking_rate(actual)
      if rate is None:
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=None,
                eval_status=EvalStatus.NOT_EVALUATED,
            )
        )
        continue

      rates.append(rate)
      eval_status = (
          EvalStatus.PASSED if rate <= self._threshold else EvalStatus.FAILED
      )
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              score=rate,
              eval_status=eval_status,
          )
      )

    if not rates:
      return EvaluationResult(
          overall_score=None,
          overall_eval_status=EvalStatus.NOT_EVALUATED,
          per_invocation_results=per_invocation_results,
      )

    overall_score = sum(rates) / len(rates)
    overall_eval_status = (
        EvalStatus.PASSED
        if overall_score <= self._threshold
        else EvalStatus.FAILED
    )
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=overall_eval_status,
        per_invocation_results=per_invocation_results,
    )
