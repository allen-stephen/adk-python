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

"""Builds `AppDetails` (agent instruction + tool declarations) for live runs.

The non-live eval path captures the agent under test's instruction and tool
declarations by intercepting its LLM requests (see
`EvaluationGenerator._get_app_details_by_invocation_id`). The live path has no
such interception, so without this helper the agent context would be missing
from materialized live invocations.

That context is required by managed (LLM-judged) metrics: the reference-free
trajectory/tool-use/hallucination autoraters read the agent's instruction and
tools to judge *why* the agent behaved as it did. Omitting it makes the autorater
infer its own rubric from the trace alone (e.g. penalizing correct tool use as if
the agent should have answered from internal knowledge).
"""

from __future__ import annotations

import logging

from google.genai import types as genai_types

from ...agents.base_agent import BaseAgent
from ..app_details import AgentDetails
from ..app_details import AppDetails

logger = logging.getLogger("google_adk." + __name__)


async def build_app_details(agent: BaseAgent) -> AppDetails:
  """Builds `AppDetails` for the agent under test in a live run.

  Captures, per the agent's author-declared definition:
    - `name`: the agent name.
    - `instructions`: the raw `instruction` string (matching the developer's
      authored intent). Non-string instructions (callables / instruction
      providers) are resolved at runtime against a context that does not exist
      here, so they are recorded as empty.
    - `tool_declarations`: the agent's tools as `genai_types.Tool`s (each
      wrapping the tools' function declarations), mirroring exactly what the
      non-live path records (`llm_request.config.tools`). The managed eval API
      requires `Tool` objects here, not bare `FunctionDeclaration`s.
      Canonicalizing first lets ADK strip injected params (e.g. `tool_context`)
      that the raw callables carry.

  Best-effort: any field that cannot be resolved is left empty rather than
  failing the run, since this metadata feeds scoring and must not block capture.

  Args:
    agent: The agent under test (the live SUT root agent).

  Returns:
    An `AppDetails` with a single entry for the agent. Sub-agents are not
    enumerated here; the live SUT is evaluated as the root agent it runs as.
  """
  name = getattr(agent, "name", "") or ""

  instruction = getattr(agent, "instruction", "")
  instructions = instruction if isinstance(instruction, str) else ""

  tool_declarations = await _resolve_tool_declarations(agent)

  return AppDetails(
      agent_details={
          name: AgentDetails(
              name=name,
              instructions=instructions,
              tool_declarations=tool_declarations,
          )
      }
  )


async def _resolve_tool_declarations(
    agent: BaseAgent,
) -> list[genai_types.Tool]:
  """Returns the agent's tools as `genai_types.Tool`s (best-effort).

  Uses `canonical_tools()` (which wraps bare callables into `FunctionTool`s and
  strips ADK-injected params) and each tool's `_get_declaration()`. The
  resulting function declarations are wrapped in a single `Tool`, matching the
  shape the managed eval API expects on `AgentConfig.tools` (a list of `Tool`s,
  each carrying `function_declarations`) and the shape the non-live path records
  from `llm_request.config.tools`. Tools that do not expose a declaration are
  skipped; an empty result yields no `Tool`.
  """
  canonical_tools = getattr(agent, "canonical_tools", None)
  if not callable(canonical_tools):
    return []
  try:
    tools = await canonical_tools()
  except Exception:  # pylint: disable=broad-except
    # Tool resolution is metadata-only; never fail the run over it.
    logger.warning(
        "Failed to resolve canonical tools for agent %r; live agent context"
        " will omit tool declarations.",
        getattr(agent, "name", agent),
        exc_info=True,
    )
    return []

  declarations = []
  for tool in tools:
    get_declaration = getattr(tool, "_get_declaration", None)
    if not callable(get_declaration):
      continue
    try:
      declaration = get_declaration()
    except Exception:  # pylint: disable=broad-except
      logger.warning(
          "Failed to get declaration for tool %r; skipping it in live agent"
          " context.",
          getattr(tool, "name", tool),
          exc_info=True,
      )
      continue
    if declaration is not None:
      declarations.append(declaration)

  if not declarations:
    return []
  return [genai_types.Tool(function_declarations=declarations)]
