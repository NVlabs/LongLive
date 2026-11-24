#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SFT warm start (rejection-sampling fine-tuning) of Qwen3-8B + LoRA on data/rl/sft/bestof.jsonl -> outputs/rl/sft.
# No servers needed. 89 steps take about 75 minutes on one A100 80GB; several GPUs split each batch:
#   TRAIN_GPUS=0,1 bash scripts/rl/sft.sh [key=value overrides]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/rl/env.sh
NPROC=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
PORT=${MASTER_PORT:-29700}

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m torch.distributed.run --nproc_per_node "$NPROC" --master_port "$PORT" \
  -m agentic_longlive.rl.train --config configs/rl/sft.json "$@"
