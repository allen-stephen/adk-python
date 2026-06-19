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

"""Shared helper for generating eval cases from conversation scenarios.

Both the CLI (`adk eval_set generate_eval_cases`) and the dev server's
`generate-eval-cases` endpoint use this helper so the generation logic stays in
one place. Generation requires GCP credentials (see `ScenarioGenerator`).
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import TYPE_CHECKING

from ._vertex_ai_scenario_generation_facade import ScenarioGenerator
from .conversation_scenarios import ConversationGenerationConfig
from .eval_case import EvalCase
from .eval_case import SessionInput

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from .eval_sets_manager import EvalSetsManager


def generate_and_add_eval_cases(
    *,
    root_agent: BaseAgent,
    config: ConversationGenerationConfig,
    eval_sets_manager: EvalSetsManager,
    app_name: str,
    eval_set_id: str,
    initial_session_state: dict,
) -> list[str]:
  """Generates conversation scenarios and adds them as eval cases.

  The eval set is created if it does not already exist. Each generated scenario
  is keyed by a stable id derived from its content; scenarios whose id already
  exists in the set are skipped (idempotent).

  Args:
    root_agent: The root agent representing the system under test.
    config: Configuration controlling how many scenarios to generate and how.
    eval_sets_manager: Manager used to read/write the eval set.
    app_name: The name of the app the eval set belongs to.
    eval_set_id: The id of the eval set to add cases to.
    initial_session_state: Initial session state to seed each case with.

  Returns:
    The list of eval case ids that were newly added (excludes skipped cases).
  """
  if (
      eval_sets_manager.get_eval_set(app_name=app_name, eval_set_id=eval_set_id)
      is None
  ):
    eval_sets_manager.create_eval_set(
        app_name=app_name, eval_set_id=eval_set_id
    )

  generator = ScenarioGenerator()
  scenarios = generator.generate_scenarios(root_agent, config)

  # TODO: Expose initial session state when the simulation library supports it.
  session_input = SessionInput(
      app_name=app_name,
      user_id="test_user_id",
      state=initial_session_state,
  )

  added_eval_ids: list[str] = []
  for scenario in scenarios:
    scenario_str = json.dumps(
        scenario.model_dump(exclude_none=True), sort_keys=True
    )
    eval_id = hashlib.sha256(scenario_str.encode("utf-8")).hexdigest()[:8]
    if (
        eval_sets_manager.get_eval_case(
            app_name=app_name, eval_set_id=eval_set_id, eval_case_id=eval_id
        )
        is not None
    ):
      continue

    eval_case = EvalCase(
        eval_id=eval_id,
        conversation_scenario=scenario,
        session_input=session_input,
        creation_timestamp=datetime.now().timestamp(),
    )
    eval_sets_manager.add_eval_case(
        app_name=app_name, eval_set_id=eval_set_id, eval_case=eval_case
    )
    added_eval_ids.append(eval_id)

  return added_eval_ids
