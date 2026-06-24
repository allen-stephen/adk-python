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

"""Model Armor Integration.

This module provides client and configuration types for screening agent
input and output with Google Cloud Model Armor. See
``google.adk.plugins.ModelArmorPlugin`` for the plugin that uses them.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from .client import ModelArmorClient
  from .client import ModelArmorVerdict
  from .config import EnforcementMode
  from .config import ModelArmorConfig
  from .config import StreamingMode
  from .credentials import ModelArmorCredentialsConfig

# Map attribute names to relative module paths.
_lazy_imports = {
    'EnforcementMode': '.config',
    'ModelArmorConfig': '.config',
    'StreamingMode': '.config',
    'ModelArmorClient': '.client',
    'ModelArmorVerdict': '.client',
    'ModelArmorCredentialsConfig': '.credentials',
}


def __getattr__(name: str):
  if name in _lazy_imports:
    import importlib

    module_path = _lazy_imports[name]
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())
