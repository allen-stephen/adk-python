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

"""Tests for eval sets manager utilities.

Verifies that the persona eval-set shell is created and reused correctly so a
persona-driven live run does not require a pre-authored eval set.
"""

from google.adk.evaluation._eval_sets_manager_utils import get_or_create_persona_eval_shell
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.evaluation.simulation.persona import Persona
import pytest


@pytest.fixture
def manager():
  return InMemoryEvalSetsManager()


@pytest.fixture
def persona():
  return Persona(
      id="dice_player",
      character_prompt="curious dice gamer",
      goal="roll a d20 and check if it is prime",
  )


def test_creates_set_and_case_when_absent(manager, persona):
  """The shell is created with persona-derived ids when none exists."""
  eval_set_id, eval_case_id = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  assert eval_set_id == "dice_player"
  assert eval_case_id == "dice_player_goal"
  eval_set = manager.get_eval_set("app", eval_set_id)
  assert eval_set is not None
  assert [c.eval_id for c in eval_set.eval_cases] == [eval_case_id]


def test_is_idempotent_across_repeated_calls(manager, persona):
  """Re-running the same persona reuses the existing set/case (no error)."""
  first = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  second = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  assert first == second
  eval_set = manager.get_eval_set("app", first[0])
  assert len(eval_set.eval_cases) == 1


def test_synthetic_case_carries_persona_goal_and_prompt(manager, persona):
  """The placeholder case is seeded from the persona's goal and prompt."""
  _, eval_case_id = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  eval_case = manager.get_eval_case(
      "app", "dice_player", eval_case_id
  )
  assert eval_case.conversation_scenario.starting_prompt == (
      "roll a d20 and check if it is prime"
  )
  assert eval_case.conversation_scenario.conversation_plan == (
      "curious dice gamer"
  )


def test_sanitizes_unsafe_persona_id(manager):
  """A persona id with unsafe characters yields valid set/case ids."""
  persona = Persona(id="dice player!", character_prompt="x", goal="y")

  eval_set_id, eval_case_id = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  assert eval_set_id == "dice_player_"
  assert eval_case_id == "dice_player__goal"


def test_defaults_starting_prompt_when_goal_empty(manager):
  """An empty persona goal falls back to a default starting prompt."""
  persona = Persona(id="p", character_prompt="x", goal="")

  _, eval_case_id = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  eval_case = manager.get_eval_case("app", "p", eval_case_id)
  assert eval_case.conversation_scenario.starting_prompt == (
      "Start the conversation."
  )


def test_persists_to_disk_with_local_manager(tmp_path, persona):
  """The shell is persisted so a separate manager instance can read it back.

  This is the behavior the dev server relies on: results attach to a real,
  reviewable eval set on disk.
  """
  manager = LocalEvalSetsManager(agents_dir=str(tmp_path))

  eval_set_id, eval_case_id = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  reloaded = LocalEvalSetsManager(agents_dir=str(tmp_path))
  eval_case = reloaded.get_eval_case("app", eval_set_id, eval_case_id)
  assert eval_case is not None


def test_idempotent_with_local_manager(tmp_path, persona):
  """Repeated persona runs against a persistent manager do not raise."""
  manager = LocalEvalSetsManager(agents_dir=str(tmp_path))

  first = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )
  second = get_or_create_persona_eval_shell(
      manager, app_name="app", persona=persona
  )

  assert first == second
  eval_set = manager.get_eval_set("app", first[0])
  assert len(eval_set.eval_cases) == 1
