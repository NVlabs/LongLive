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

"""Paths for a relocatable, editable research checkout."""

import os
from pathlib import Path


def repository_root() -> Path:
    root = Path(os.environ.get("LONGWAM_ROOT", Path(__file__).resolve().parents[2]))
    if not (root / "configs").is_dir():
        raise FileNotFoundError(
            "Long-WAM configs not found. Use an editable checkout (pip install -e .), "
            "or set LONGWAM_ROOT to the checkout containing configs/."
        )
    return root.resolve()


def setup_model_paths() -> None:
    """Retain the underlying Wan loader's on-disk org/repository layout."""
    os.environ.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH", os.environ.get("LONGWAM_MODEL_ROOT", "./models")
    )
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
