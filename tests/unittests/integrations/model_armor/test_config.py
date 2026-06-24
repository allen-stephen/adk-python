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

"""Tests for ModelArmorConfig.

Verifies the config validates template requirements and exposes the expected
defaults.
"""

from __future__ import annotations

from google.adk.integrations.model_armor.config import EnforcementMode
from google.adk.integrations.model_armor.config import ModelArmorConfig
from google.adk.integrations.model_armor.config import StreamingMode
import pytest


def test_defaults_block_and_realtime():
  """A config defaults to BLOCK enforcement and REALTIME streaming."""
  config = ModelArmorConfig(prompt_template_name='t1')

  assert config.enforcement == EnforcementMode.BLOCK
  assert config.streaming is True
  assert config.streaming_mode == StreamingMode.REALTIME


def test_missing_both_templates_raises():
  """A config with neither template configured is rejected."""
  with pytest.raises(ValueError, match='At least one of'):
    ModelArmorConfig(project='p', location='us-central1')


def test_only_response_template_is_valid():
  """Configuring only the response template is allowed."""
  config = ModelArmorConfig(response_template_name='t2')

  assert config.response_template_name == 't2'
  assert config.prompt_template_name is None


def test_extra_fields_forbidden():
  """Unknown fields are rejected to catch typos."""
  with pytest.raises(ValueError):
    ModelArmorConfig(prompt_template_name='t1', unknown_field=True)
