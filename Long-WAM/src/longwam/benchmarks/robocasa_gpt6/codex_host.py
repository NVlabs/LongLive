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

"""Compatibility import path; new integrations use longwam.agentic directly."""

import os

from longwam.agentic import AgenticConfig, CodexAgent
from longwam.benchmarks.robocasa.agentic.adapter import (
    RoboCasaAdapter,
    RoboCasaEpisode,
    prepare_workspace,
    skill_file,
)


class CodexPolicy(CodexAgent):
    def __init__(
        self,
        workspace,
        codex,
        *,
        method,
        timeout=900.0,
        model=None,
        effort=None,
        allow_model_upload=False,
        max_total_tokens=None,
    ):
        super().__init__(
            workspace,
            adapter=RoboCasaAdapter(method),
            config=AgenticConfig(
                codex=codex,
                timeout_seconds=timeout,
                model=model or os.environ.get("ROLLOUT_MODEL", "gpt-6-astra"),
                effort=effort or os.environ.get("ROLLOUT_EFFORT", "xhigh"),
                max_total_tokens=(
                    max_total_tokens
                    if max_total_tokens is not None
                    else int(os.environ.get("CODEX_MAX_TOTAL_TOKENS", "0"))
                ),
                allow_model_upload=allow_model_upload,
            ),
        )

    def run(self, rollout):
        return super().run(RoboCasaEpisode(rollout))
