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

"""LiveKit integration.

Bridges a LiveKit room to an ADK live agent, giving an unmodified agent
telephony (SIP/PSTN), WebRTC, and Unity/gaming ingress. Install with:
pip install "google-adk[livekit]"
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from ._livekit_runner import LiveKitRunner
  from ._livekit_server import livekit_server

_lazy_imports = {
    "LiveKitRunner": "._livekit_runner",
    "livekit_server": "._livekit_server",
}


def __getattr__(name: str) -> typing.Any:
  if name in _lazy_imports:
    import importlib

    module = importlib.import_module(_lazy_imports[name], __name__)
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())
