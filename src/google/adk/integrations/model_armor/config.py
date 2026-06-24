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

from __future__ import annotations

from enum import Enum
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import model_validator


class EnforcementMode(Enum):
  """How the plugin acts on a Model Armor ``MATCH_FOUND`` verdict."""

  BLOCK = 'block'
  """Block the offending content.

  For input, the model call is short-circuited and a safe replacement
  response is returned. For output, the model response is replaced with a
  safe message. In live mode, the in-flight output is suppressed and the
  turn is ended.
  """

  OBSERVE = 'observe'
  """Do not block; deliver the content but emit a guardrail verdict event.

  This mirrors Model Armor's "Inspect only" behaviour and is useful for
  staged rollouts where teams want visibility before enforcing.
  """


class StreamingMode(Enum):
  """Streaming sanitization cadence for the live (bidi) transport.

  These mirror the native ``modelarmor_v1beta.StreamingMode`` values.
  """

  REALTIME = 'realtime'
  """Screen each chunk as it streams, blocking as early as possible."""

  BUFFERED = 'buffered'
  """Buffer the stream and screen the accumulated content."""


class ModelArmorConfig(BaseModel):
  """Configuration for the Model Armor guardrail plugin.

  The template field names intentionally mirror the native
  ``google.genai.types.ModelArmorConfig`` vocabulary
  (``prompt_template_name`` / ``response_template_name``) so that teams can
  converge on a single mental model across server-side (platform-native)
  screening and this client-side plugin.
  """

  # Forbid any fields not defined in the model.
  model_config = ConfigDict(extra='forbid')

  project: Optional[str] = None
  """The GCP project ID hosting the Model Armor templates.

  If not set, the project is inferred from Application Default Credentials.
  """

  location: Optional[str] = None
  """The regional location of the Model Armor templates (e.g. ``us-central1``).

  Model Armor is a regional service; this drives the regional endpoint
  ``modelarmor.<location>.rep.googleapis.com``. Required unless the templates
  are referenced by fully-qualified resource name.
  """

  prompt_template_name: Optional[str] = None
  """The Model Armor template used to screen user input (prompts).

  May be a short template ID (combined with ``project``/``location``) or a
  fully-qualified resource name
  ``projects/<p>/locations/<l>/templates/<t>``. If unset, input screening is
  skipped.
  """

  response_template_name: Optional[str] = None
  """The Model Armor template used to screen model output (responses).

  May be a short template ID (combined with ``project``/``location``) or a
  fully-qualified resource name
  ``projects/<p>/locations/<l>/templates/<t>``. If unset, output screening is
  skipped.
  """

  enforcement: EnforcementMode = EnforcementMode.BLOCK
  """What to do when Model Armor reports a match. Defaults to ``BLOCK``."""

  streaming: bool = True
  """Whether to use the bidi streaming transport in live mode.

  When ``True`` (default), live turns are screened with the streaming
  ``stream_sanitize_*`` API. When ``False``, live turns fall back to buffering
  the full turn and screening it with the synchronous API.
  """

  streaming_mode: StreamingMode = StreamingMode.REALTIME
  """The streaming cadence for the live transport. Defaults to ``REALTIME``."""

  input_screening: Literal['parallel', 'blocking'] = 'parallel'
  """How live user input is screened relative to forwarding it to the model.

  ``parallel`` (default, latency-optimized): the input is forwarded to the
  model immediately and screened concurrently. If the verdict is a match, the
  in-flight generation is stopped and the output is suppressed. Clean input
  incurs no added latency, at the cost of a brief window in which a small
  amount of leading output may reach the user before a block lands.

  ``blocking`` (hard-stop): the input is screened before it is forwarded to
  the model. A match prevents the model from ever seeing the input, with no
  leading-output leak, at the cost of one screening round-trip of latency on
  every turn.

  Only affects live (bidi) turns; unary turns are always screened before the
  model call.
  """

  blocked_message: str = "I'm sorry, but I can't help with that request."
  """The safe replacement text returned to the user when content is blocked."""

  @model_validator(mode='after')
  def _validate_templates(self) -> 'ModelArmorConfig':
    """Ensure at least one template is configured."""
    if not self.prompt_template_name and not self.response_template_name:
      raise ValueError(
          'At least one of prompt_template_name or response_template_name'
          ' must be set for ModelArmorConfig.'
      )
    return self
