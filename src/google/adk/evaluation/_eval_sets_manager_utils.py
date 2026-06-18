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

import logging
import re
from typing import Optional
from typing import TYPE_CHECKING

from ..errors.not_found_error import NotFoundError
from .conversation_scenarios import ConversationScenario
from .eval_case import EvalCase
from .eval_set import EvalSet
from .eval_sets_manager import EvalSetsManager

if TYPE_CHECKING:
  from .simulation.persona import Persona

logger = logging.getLogger("google_adk." + __name__)

# Eval set/case ids must match this pattern (enforced by LocalEvalSetsManager).
_ID_SAFE_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def get_eval_set_from_app_and_id(
    eval_sets_manager: EvalSetsManager, app_name: str, eval_set_id: str
) -> EvalSet:
  """Returns an EvalSet if found; otherwise, raises NotFoundError."""
  eval_set = eval_sets_manager.get_eval_set(app_name, eval_set_id)
  if not eval_set:
    raise NotFoundError(f"Eval set `{eval_set_id}` not found.")
  return eval_set


def get_eval_case_from_eval_set(
    eval_set: EvalSet, eval_case_id: str
) -> Optional[EvalCase]:
  """Returns an EvalCase if found; otherwise, None."""
  eval_case_to_find = None

  # Look up the eval case by eval_case_id
  for eval_case in eval_set.eval_cases:
    if eval_case.eval_id == eval_case_id:
      eval_case_to_find = eval_case
      break

  return eval_case_to_find


def add_eval_case_to_eval_set(
    eval_set: EvalSet, eval_case: EvalCase
) -> EvalSet:
  """Adds an eval case to an eval set and returns the updated eval set."""
  eval_case_id = eval_case.eval_id

  if [x for x in eval_set.eval_cases if x.eval_id == eval_case_id]:
    raise ValueError(
        f"Eval id `{eval_case_id}` already exists in `{eval_set.eval_set_id}`"
        " eval set.",
    )

  eval_set.eval_cases.append(eval_case)
  return eval_set


def update_eval_case_in_eval_set(
    eval_set: EvalSet, updated_eval_case: EvalCase
) -> EvalSet:
  """Updates an eval case in an eval set and returns the updated eval set."""
  # Find the eval case to be updated.
  eval_case_id = updated_eval_case.eval_id
  eval_case_to_update = get_eval_case_from_eval_set(eval_set, eval_case_id)

  if not eval_case_to_update:
    raise NotFoundError(
        f"Eval case `{eval_case_id}` not found in eval set"
        f" `{eval_set.eval_set_id}`."
    )

  # Remove the existing eval case and add the updated eval case.
  eval_set.eval_cases.remove(eval_case_to_update)
  eval_set.eval_cases.append(updated_eval_case)
  return eval_set


def _sanitize_id(value: str) -> str:
  """Coerces an arbitrary string into a valid eval set/case id."""
  sanitized = _ID_SAFE_PATTERN.sub("_", value)
  return sanitized or "persona"


def get_or_create_persona_eval_shell(
    eval_sets_manager: EvalSetsManager,
    *,
    app_name: str,
    persona: Persona,
) -> tuple[str, str]:
  """Returns (eval_set_id, eval_case_id) for a persona, creating them if needed.

  A persona-driven live run does not need a pre-authored eval set: the
  conversation is generated fresh from the persona. This helper provides the
  minimal bookkeeping shell that the inference/evaluate pipeline requires — an
  eval set containing a single placeholder eval case — so callers (the CLI and
  the dev server) don't have to require the user to author one.

  The synthesized case content is never read for a persona run (inference uses
  the live scenario directly and metrics score only the generated conversation);
  it exists solely so results have a set/case to attach to. The operation is
  idempotent: re-running the same persona reuses the existing set/case.

  Args:
    eval_sets_manager: The manager used to look up / persist the shell.
    app_name: The app the eval set belongs to.
    persona: The persona the run is for; its id seeds the set/case ids.

  Returns:
    A tuple of (eval_set_id, eval_case_id).
  """
  # The eval set is named after the persona (it represents that persona), and
  # the case represents a goal. We treat persona sets like any other eval set,
  # so no special prefix is used.
  safe_id = _sanitize_id(persona.id)
  eval_set_id = safe_id
  eval_case_id = f"{safe_id}_goal"

  if eval_sets_manager.get_eval_set(app_name, eval_set_id) is None:
    eval_sets_manager.create_eval_set(
        app_name=app_name, eval_set_id=eval_set_id
    )

  if (
      eval_sets_manager.get_eval_case(app_name, eval_set_id, eval_case_id)
      is None
  ):
    eval_sets_manager.add_eval_case(
        app_name=app_name,
        eval_set_id=eval_set_id,
        eval_case=EvalCase(
            eval_id=eval_case_id,
            conversation_scenario=ConversationScenario(
                starting_prompt=persona.goal or "Start the conversation.",
                conversation_plan=persona.character_prompt,
            ),
        ),
    )

  return eval_set_id, eval_case_id


def delete_eval_case_from_eval_set(
    eval_set: EvalSet, eval_case_id: str
) -> EvalSet:
  """Deletes an eval case from an eval set and returns the updated eval set."""
  # Find the eval case to be deleted.
  eval_case_to_delete = get_eval_case_from_eval_set(eval_set, eval_case_id)

  if not eval_case_to_delete:
    raise NotFoundError(
        f"Eval case `{eval_case_id}` not found in eval set"
        f" `{eval_set.eval_set_id}`."
    )

  # Remove the existing eval case.
  logger.info(
      "EvalCase`%s` was found in the eval set. It will be removed permanently.",
      eval_case_id,
  )
  eval_set.eval_cases.remove(eval_case_to_delete)
  return eval_set
