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

"""Tests for the Model Armor client wrapper.

Verifies template-path resolution and that the normalized verdict reflects the
underlying SDK match state, using a mocked async SDK client.
"""

from __future__ import annotations

from unittest import mock

from google.adk.integrations.model_armor.client import ModelArmorClient
from google.adk.integrations.model_armor.client import ModelArmorVerdict
from google.adk.integrations.model_armor.config import ModelArmorConfig
import pytest


def _config(**overrides):
  defaults = dict(
      project='p',
      location='us-central1',
      prompt_template_name='pt',
      response_template_name='rt',
  )
  defaults.update(overrides)
  return ModelArmorConfig(**defaults)


def test_short_template_name_is_resolved_to_resource_path():
  """A short template name is expanded with project and location."""
  client = ModelArmorClient(config=_config(), client=mock.Mock())

  path = client._template_path('pt')

  assert path == 'projects/p/locations/us-central1/templates/pt'


def test_fully_qualified_template_name_is_passed_through():
  """A fully-qualified template resource name is left unchanged."""
  client = ModelArmorClient(config=_config(), client=mock.Mock())

  path = client._template_path('projects/x/locations/y/templates/z')

  assert path == 'projects/x/locations/y/templates/z'


def test_short_name_without_project_raises():
  """Resolving a short name needs project and location."""
  config = ModelArmorConfig(prompt_template_name='pt')
  client = ModelArmorClient(config=config, client=mock.Mock())

  with pytest.raises(ValueError, match='project and location are required'):
    client._template_path('pt')


@pytest.mark.asyncio
async def test_sanitize_user_prompt_returns_match_verdict():
  """A MATCH_FOUND sanitization result yields a match verdict."""
  import google.cloud.modelarmor_v1beta as m

  sdk_client = mock.Mock()
  result = m.SanitizationResult(
      filter_match_state=m.FilterMatchState.MATCH_FOUND
  )
  response = m.SanitizeUserPromptResponse(sanitization_result=result)
  sdk_client.sanitize_user_prompt = mock.AsyncMock(return_value=response)
  client = ModelArmorClient(config=_config(), client=sdk_client)

  verdict = await client.sanitize_user_prompt('hello')

  assert isinstance(verdict, ModelArmorVerdict)
  assert verdict.match_found is True


@pytest.mark.asyncio
async def test_sanitize_model_response_returns_no_match_verdict():
  """A NO_MATCH_FOUND result yields a clean verdict."""
  import google.cloud.modelarmor_v1beta as m

  sdk_client = mock.Mock()
  result = m.SanitizationResult(
      filter_match_state=m.FilterMatchState.NO_MATCH_FOUND
  )
  response = m.SanitizeModelResponseResponse(sanitization_result=result)
  sdk_client.sanitize_model_response = mock.AsyncMock(return_value=response)
  client = ModelArmorClient(config=_config(), client=sdk_client)

  verdict = await client.sanitize_model_response('hi there')

  assert verdict.match_found is False
