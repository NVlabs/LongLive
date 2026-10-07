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

"""Model-independent settings for the Agentic API's Codex backend."""

from dataclasses import dataclass
import math

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_EFFORT = "xhigh"


@dataclass(frozen=True)
class AgenticConfig:
    """Exact model IDs/efforts are forwarded; availability is checked by Codex."""

    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    codex: str = "codex"
    allow_model_upload: bool = False
    max_total_tokens: int | None = None
    timeout_seconds: float = 900
    backend: str = "codex"

    def validate(self):
        if self.backend != "codex":
            raise ValueError("Only agentic.backend=codex is currently implemented")
        for name in ("model", "effort", "codex"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"agentic.{name} must be a non-empty string")
            if name != "codex" and any(c.isspace() for c in value):
                raise ValueError(f"agentic.{name} must be an exact ID without whitespace")
        if self.allow_model_upload is not True:
            raise ValueError("Set agentic.allow_model_upload=true to send episode data to OpenAI")
        budget = self.max_total_tokens
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("Set a positive integer agentic.max_total_tokens budget per episode")
        timeout = self.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("agentic.timeout_seconds must be finite and positive")
        return self
