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

"""Tests for the eval-case authoring endpoints on the dev server.

Covers the two new authoring entry points used by the web UI's redesigned
"New eval case" modal:
- add-eval-case: save a fixed (scripted) conversation.
- generate-eval-cases: synthesize scenarios via the Vertex Gen AI Eval SDK.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
import pytest

_APP_NAME = "test_app"
_EVAL_SET_ID = "my_set"


def _write_minimal_agent(agents_dir):
  """Writes a minimal loadable agent package under agents_dir/_APP_NAME."""
  agent_dir = agents_dir / _APP_NAME
  agent_dir.mkdir(parents=True)
  (agent_dir / "__init__.py").write_text("from . import agent\n")
  (agent_dir / "agent.py").write_text(
      "from google.adk.agents.llm_agent import LlmAgent\n"
      "root_agent = LlmAgent(name='root_agent', model='gemini-2.0-flash')\n"
  )


@pytest.fixture
def test_client(tmp_path):
  """Client backed by a temp agents dir containing a minimal agent."""
  _write_minimal_agent(tmp_path)
  app = get_fast_api_app(
      agents_dir=str(tmp_path),
      web=True,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      allow_origins=["*"],
      a2a=False,
      host="127.0.0.1",
      port=8000,
  )
  return TestClient(app)


def _create_eval_set(client: TestClient) -> None:
  response = client.post(
      f"/dev/apps/{_APP_NAME}/eval-sets",
      json={"eval_set": {"eval_set_id": _EVAL_SET_ID, "eval_cases": []}},
  )
  assert response.status_code == 200, response.text


def test_add_eval_case_saves_scripted_conversation(test_client):
  _create_eval_set(test_client)

  payload = {
      "eval_id": "scripted_case",
      "conversation": [{
          "user_content": {
              "role": "user",
              "parts": [{"text": "Roll a 20-sided die."}],
          },
          "final_response": {
              "role": "model",
              "parts": [{"text": "You rolled a 17."}],
          },
      }],
  }

  response = test_client.post(
      f"/dev/apps/{_APP_NAME}/eval-sets/{_EVAL_SET_ID}/add-eval-case",
      json=payload,
  )
  assert response.status_code == 200, response.text

  # The case is persisted and retrievable with its scripted conversation.
  listed = test_client.get(
      f"/dev/apps/{_APP_NAME}/eval_sets/{_EVAL_SET_ID}/evals"
  )
  assert listed.status_code == 200, listed.text
  assert "scripted_case" in listed.json()


def test_add_eval_case_underscore_alias(test_client):
  _create_eval_set(test_client)

  payload = {
      "eval_id": "alias_case",
      "conversation": [
          {
              "user_content": {
                  "role": "user",
                  "parts": [{"text": "Hi"}],
              }
          }
      ],
  }

  response = test_client.post(
      f"/dev/apps/{_APP_NAME}/eval_sets/{_EVAL_SET_ID}/add_eval_case",
      json=payload,
  )
  assert response.status_code == 200, response.text


def test_add_eval_case_missing_set_returns_error(test_client):
  payload = {
      "eval_id": "case",
      "conversation": [
          {"user_content": {"role": "user", "parts": [{"text": "Hi"}]}}
      ],
  }
  response = test_client.post(
      f"/dev/apps/{_APP_NAME}/eval-sets/does_not_exist/add-eval-case",
      json=payload,
  )
  # A missing eval set surfaces as a not-found error.
  assert response.status_code == 404


def test_generate_eval_cases_returns_created_ids(test_client):
  _create_eval_set(test_client)

  with patch(
      "google.adk.evaluation._scenario_generation_helper"
      ".generate_and_add_eval_cases",
      return_value=["abc12345", "def67890"],
  ) as mock_generate:
    response = test_client.post(
        f"/dev/apps/{_APP_NAME}/eval-sets/{_EVAL_SET_ID}/generate-eval-cases",
        json={
            "count": 2,
            "generation_instruction": "Test scenarios.",
        },
    )

  assert response.status_code == 200, response.text
  # Responses serialize with camelCase aliases (see cli.utils.common.BaseModel).
  assert response.json() == {"evalIds": ["abc12345", "def67890"]}
  mock_generate.assert_called_once()
  _, kwargs = mock_generate.call_args
  assert kwargs["app_name"] == _APP_NAME
  assert kwargs["eval_set_id"] == _EVAL_SET_ID
  assert kwargs["config"].count == 2


def test_generate_eval_cases_uses_backend_default_model(test_client):
  """When no model is supplied, the backend-managed default is used."""
  _create_eval_set(test_client)

  with patch(
      "google.adk.evaluation._scenario_generation_helper"
      ".generate_and_add_eval_cases",
      return_value=[],
  ) as mock_generate:
    response = test_client.post(
        f"/dev/apps/{_APP_NAME}/eval-sets/{_EVAL_SET_ID}/generate-eval-cases",
        json={"count": 1},
    )

  assert response.status_code == 200, response.text
  _, kwargs = mock_generate.call_args
  assert kwargs["config"].model_name == "gemini-flash-latest"


def test_generate_eval_cases_surfaces_value_error_as_400(test_client):
  _create_eval_set(test_client)

  with patch(
      "google.adk.evaluation._scenario_generation_helper"
      ".generate_and_add_eval_cases",
      side_effect=ValueError("Missing project id."),
  ):
    response = test_client.post(
        f"/dev/apps/{_APP_NAME}/eval-sets/{_EVAL_SET_ID}/generate-eval-cases",
        json={"count": 1},
    )

  assert response.status_code == 400
  assert "Missing project id." in response.text
