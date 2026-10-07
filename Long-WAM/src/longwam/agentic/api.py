# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
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
# End Long-WAM attribution.

"""Small benchmark boundary: prompts/tools in, model decisions out.

No simulator, robot action schema, task registry or checkpoint loader belongs here.
Only InputError means that a rejected tool call performed no irreversible action.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ._vendor.io import InputError


@dataclass(frozen=True)
class AgentSpec:
    workspace: Path
    instructions: str
    tools: list[dict[str, Any]]
    initial_message: str
    image_tools: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


class AgenticAdapter(Protocol):
    def prepare(self, audit: Path) -> AgentSpec:
        """Create benchmark-owned prompt/workspace assets below a fresh audit root."""
        ...


class AgenticEpisode(Protocol):
    @property
    def finished(self) -> bool: ...

    @property
    def step_id(self) -> int: ...

    def next_call(self) -> dict[str, Any] | None: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate order/limits and execute once; never silently replay actions."""
        ...

    def rejected_input(self, error: InputError, count: int) -> dict[str, Any]:
        """Return benchmark-specific correction context after a no-action rejection."""
        ...
