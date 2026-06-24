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

from ...tools._google_credentials import BaseGoogleCredentialsConfig

MODEL_ARMOR_TOKEN_CACHE_KEY = 'model_armor_token_cache'
MODEL_ARMOR_SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
]


class ModelArmorCredentialsConfig(BaseGoogleCredentialsConfig):
  """Model Armor Credentials Configuration for the guardrail plugin.

  Please do not use this in production, as it may be deprecated later.
  """

  def __post_init__(self) -> 'ModelArmorCredentialsConfig':
    """Populate default scope if scopes is None."""
    super().__post_init__()

    if not self.scopes:
      self.scopes = MODEL_ARMOR_SCOPES
    # Set the token cache key.
    self._token_cache_key = MODEL_ARMOR_TOKEN_CACHE_KEY

    return self
