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
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train.sh BENCHMARK [--dry-run] [Hydra key=value ...]" >&2
  exit 2
fi
benchmark="$1"
shift
case "$benchmark" in
  libero|robotwin2|domino|robocasa_gr1|robocasa365) ;;
  *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
esac
args=("task=$benchmark")
case "$benchmark" in
  robocasa_gr1|robocasa365) export NPROC_PER_NODE="${NPROC_PER_NODE:-16}" ;;
  *) export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" ;;
esac
dry_run=0
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then dry_run=1; else args+=("$arg"); fi
done
if (( dry_run )); then
  # Hydra composition only: no dataset, model, checkpoint or GPU is opened.
  exec "$PYTHON" "$LONGWAM_ROOT/scripts/check_config.py" \
    --world-size "$(( ${NNODES:-1} * ${NPROC_PER_NODE:-1} ))" "${args[@]}"
fi
if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
  if [[ "${NNODES:-1}" != 1 ]]; then
    echo "The ACCELERATE_CONFIG wrapper is single-node; launch multi-node Accelerate with an explicit cluster config." >&2
    exit 2
  fi
  exec "$PYTHON" -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG" \
    --num_machines 1 --machine_rank 0 --num_processes "${NPROC_PER_NODE:-1}" \
    "$LONGWAM_ROOT/scripts/train.py" "${args[@]}"
fi
if [[ "${NNODES:-1}" == 1 ]]; then
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
    "$LONGWAM_ROOT/scripts/train.py" "${args[@]}"
fi
: "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
: "${NODE_RANK:?Set NODE_RANK for multi-node training}"
exec "$PYTHON" -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
  --nproc_per_node="${NPROC_PER_NODE:-1}" "$LONGWAM_ROOT/scripts/train.py" "${args[@]}"
