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

"""Text (unary) agent guarded by the Model Armor plugin.

The same ``ModelArmorPlugin`` screens user input and model output through your
Model Armor templates. In text mode it uses Model Armor's synchronous
``sanitize_*`` API.

Prerequisites:
  - ``pip install 'google-adk[gcp]'``
  - A GCP project with Model Armor enabled and prompt/response templates
    created in a region (e.g. ``us-central1``).
  - Application Default Credentials (``gcloud auth application-default
    login``) or a service account with Model Armor access.

Usage:
  adk run contributing/samples/plugin/model_armor_unary

Note on the zero-code alternative:
  If you only need server-side text screening, the model platform can screen
  prompts/responses natively via ``generate_content_config.model_armor_config``
  — no plugin required. Use this plugin when you also need voice (live)
  coverage, custom blocked responses, or log-only (OBSERVE) monitoring.
"""

import logging

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.integrations.model_armor import EnforcementMode
from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ModelArmorPlugin

logger = logging.getLogger(__name__)


# --- demo: agent model callbacks (same code as the live sample) -----------
# Plain agent-level callbacks that log. On this unary (run_async) path
# ``callback_context.is_live`` is False; the *same* callbacks log
# ``is_live=True`` in the voice sample (live_model_armor_agent), demonstrating
# that the framework runs agent model callbacks on both transports. They return
# None (observe-only) and do not alter the turn.
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
      " has_content=%s",
      callback_context.is_live,
      callback_context.agent_name,
      bool(llm_response.content),
  )
  return None


# --- end demo -------------------------------------------------------------


# Replace project/location/template names with your own Model Armor resources.
model_armor_config = ModelArmorConfig(
    project="your-gcp-project",
    location="us-central1",
    prompt_template_name="your-prompt-template",
    response_template_name="your-response-template",
    enforcement=EnforcementMode.BLOCK,  # or OBSERVE to log-only.
)

root_agent = LlmAgent(
    name="model_armor_text_agent",
    model="gemini-2.5-flash",
    description="A text assistant screened by Model Armor.",
    instruction="You are a helpful, concise assistant.",
    # Demo: the same callbacks used by the voice sample. Here they log
    # "[demo callback] ... FIRED (is_live=False)".
    before_model_callback=log_before_model,
    after_model_callback=log_after_model,
)

app = App(
    name="model_armor_unary",
    root_agent=root_agent,
    plugins=[ModelArmorPlugin(config=model_armor_config)],
)
