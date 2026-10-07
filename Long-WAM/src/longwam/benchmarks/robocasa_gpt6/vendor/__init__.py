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

"""Compatibility paths for helpers moved to Agentic core and the RoboCasa adapter."""

from pathlib import Path
import importlib
import sys

_root = Path(__file__).resolve().parents[3]
__path__.extend(
    [
        str(_root / "agentic" / "_vendor"),
        str(_root / "benchmarks" / "robocasa" / "agentic" / "vendor"),
    ]
)

# Share exception/transport classes rather than loading a second copy under aliases.
for _name in ("io", "transport", "image_preview", "network_recovery"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"longwam.agentic._vendor.{_name}")
