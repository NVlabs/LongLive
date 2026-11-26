#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GRPO of the SFT planner with reward version 4 (needs: bash scripts/rl/serve.sh and outputs/rl/sft/adapter_final).
#   bash scripts/rl/grpo.sh                 # seed 0 -> outputs/rl/grpo_s0 (80 steps; resumes if interrupted)
#   SEED=1 TRAIN_GPUS=6 bash scripts/rl/grpo.sh
#   CONFIG=configs/rl/ablations/grpo_text_reward.json bash scripts/rl/grpo.sh   # paper ablations
# Several seeds can share one set of servers (each trainer loads its adapters under its own names).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/rl/env.sh
SEED=${SEED:-0}
CONFIG=${CONFIG:-configs/rl/grpo.json}
NAME=${NAME:-$(basename "$CONFIG" .json)_s$SEED}
PORT=${MASTER_PORT:-$((29710 + SEED))}

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m torch.distributed.run --nproc_per_node 1 --master_port "$PORT" \
  -m agentic_longlive.rl.train --config "$CONFIG" seed="$SEED" run="$NAME" out="outputs/rl/$NAME" \
  rollout_urls="$ROLLOUT_URLS" judge_urls="$JUDGE_URLS" vlm_urls="$JUDGE_URLS" image_urls="$IMAGE_URLS" "$@"
