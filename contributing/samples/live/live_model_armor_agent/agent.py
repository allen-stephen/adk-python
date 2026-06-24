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

"""Voice (live) agent guarded by the same Model Armor plugin.

This is the *same* ``ModelArmorPlugin`` and the *same* ``ModelArmorConfig`` as
the text sample — no live-specific developer code. In live mode the plugin
screens over the bidi streaming transport:

- **Typed input** is screened in parallel by default
  (``input_screening='parallel'``): the input is forwarded to the model
  immediately and screened concurrently, so clean turns add no latency. A
  policy match stops generation and ends the turn. Set
  ``input_screening='blocking'`` for a hard stop (screen before forwarding) at
  the cost of one screening round-trip per turn.
- **Spoken (audio) input** is transcribed by the model, so the transcription is
  itself a model result (``LlmResponse.input_transcription``) that arrives after
  the audio has reached the model. The plugin therefore screens it in
  ``after_model_callback`` (not ``before_model_callback``), ending the turn on a
  match. This keeps ``before_model_callback`` a true pre-model contract: only
  typed input can be blocked before the model sees it.
- **Output** is screened as it streams: the running output is checked per chunk
  to block as early as possible, plus a consolidated check at turn completion.
  On a match the in-flight output is suppressed and the turn is ended.

Note: live screening is text-based and depends on transcription. ``adk web``
enables audio transcription by default; if you disable it, there is no text to
screen and the plugin logs a one-time warning.

Prerequisites:
  - ``pip install 'google-adk[gcp]'``
  - A GCP project with Model Armor templates in a region (e.g. ``us-central1``).
  - Application Default Credentials with Model Armor access.

Usage (voice):
  adk web   # then select live_model_armor_agent and use the microphone.
"""

import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.apps import App
from google.adk.integrations.model_armor import EnforcementMode
from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ModelArmorPlugin
from google.genai import types

logger = logging.getLogger(__name__)


# --- demo: agent model callbacks fire in live mode ------------------------
# Plain *agent-level* before/after_model_callback that simply log. The point of
# this demo is that these callbacks ALSO run on the live (bidi) path, not only
# on run_async/SSE. ``callback_context.is_live`` is the per-call live signal set
# by the framework: it is True for real run_live turns and False for unary/CFC.
# Run the text sample to see ``is_live=False`` and this voice sample to see
# ``is_live=True`` from the *same* callback contract. These return None, so they
# observe-only and never alter the turn (Model Armor screening is separate, via
# the plugin below).
def log_before_model(
    *, callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
  """Logs that before_model_callback fired and on which transport."""
  last_user_text = ""
  for content in reversed(llm_request.contents or []):
    if content.role in (None, "user") and content.parts:
      last_user_text = " ".join(p.text for p in content.parts if p.text)
      if last_user_text:
        break
  logger.info(
      "[demo callback] before_model_callback FIRED (is_live=%s) agent=%s"
      " last_user_text=%r",
      callback_context.is_live,
      callback_context.agent_name,
      last_user_text,
  )
  return None


def log_after_model(
    *, callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
  """Logs that after_model_callback fired and what it observed."""
  logger.info(
      "[demo callback] after_model_callback FIRED (is_live=%s) agent=%s"
      " has_input_transcription=%s has_output_transcription=%s has_content=%s",
      callback_context.is_live,
      callback_context.agent_name,
      bool(llm_response.input_transcription),
      bool(llm_response.output_transcription),
      bool(llm_response.content),
  )
  return None


# --- end demo -------------------------------------------------------------


# The identical config drives both text and voice — there is no live-specific
# developer configuration. Replace with your own Model Armor resources.
model_armor_config = ModelArmorConfig(
    project="agents-cli-test-dev-qi5zi1",
    location="us-central1",
    prompt_template_name="live-example",
    response_template_name="live-example",
    enforcement=EnforcementMode.BLOCK,
    # 'parallel' (default) optimizes for latency; 'blocking' for a hard stop.
    input_screening="parallel",
)

root_agent = Agent(
    name="live_model_armor_agent",
    model="gemini-live-2.5-flash-native-audio",
    description="A voice assistant screened by Model Armor.",
    instruction="You are a helpful, concise voice assistant.",
    # Demo: these plain agent callbacks fire on the live (bidi) path too. Watch
    # the server logs for "[demo callback] ... FIRED (is_live=True)".
    before_model_callback=log_before_model,
    after_model_callback=log_after_model,
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)

app = App(
    name="live_model_armor_agent",
    root_agent=root_agent,
    plugins=[ModelArmorPlugin(config=model_armor_config)],
)
