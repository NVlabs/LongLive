#!/usr/bin/env bash
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

# Source from launchers. No machine-specific environment or cluster submission.
set -euo pipefail
LONGWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LONGWAM_ROOT
export PYTHONPATH="$LONGWAM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${LONGWAM_MODEL_ROOT:-$LONGWAM_ROOT/models}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export LONGWAM_DATA_ROOT="${LONGWAM_DATA_ROOT:-$LONGWAM_ROOT/data}"
export LONGWAM_CHECKPOINT_ROOT="${LONGWAM_CHECKPOINT_ROOT:-$LONGWAM_ROOT/checkpoints}"
PYTHON="${PYTHON:-python}"
cd "$LONGWAM_ROOT"
