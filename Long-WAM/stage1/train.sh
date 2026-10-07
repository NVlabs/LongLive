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

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
for arg in "$@"; do
  if [[ "$arg" == --plan ]]; then exec "$PYTHON" "$ROOT/run.py" train "$@"; fi
done
: "${MASTER_ADDR:?Set torchrun MASTER_ADDR}"
: "${NODE_RANK:?Set torchrun NODE_RANK}"
exec "$PYTHON" -m torch.distributed.run --nnodes="${NNODES:-16}" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
  "$ROOT/run.py" train "$@"
