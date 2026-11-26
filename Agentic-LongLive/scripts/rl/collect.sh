#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Teacher data (needs: bash scripts/rl/serve.sh judge).
#   1. The teacher (Qwen3.6-35B-A3B, temperature 0.2) plays every training story online with the deployed planner
#      prompt -> data/rl/rollouts/states_teacher.jsonl, the GRPO state pool (one state per shot).
#   2. Best of 4 per state (the teacher's output + 3 samples at temperature 0.8), scored with the text reward
#      -> data/rl/sft/bestof.jsonl, the SFT (rejection-sampling fine-tuning) data.
# Both steps resume where they stopped. Skip this script when using the released data.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/rl/env.sh
DATA=${DATA:-data/rl}

python -m agentic_longlive.rl.collect rollouts --stories "$DATA/train/terse" "$DATA/train/full" \
  --planner-urls "$JUDGE_URLS" --planner-model "$JUDGE_MODEL" --tag teacher --out "$DATA/rollouts" --workers 32
python -m agentic_longlive.rl.collect bestofk --states "$DATA/rollouts/states_teacher.jsonl" \
  --planner-urls "$JUDGE_URLS" --planner-model "$JUDGE_MODEL" --judge-urls "$JUDGE_URLS" --judge-model "$JUDGE_MODEL" \
  --k 3 --out "$DATA/sft/bestof.jsonl" --workers 48
