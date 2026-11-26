#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# RL training data (needs: bash scripts/rl/serve.sh judge). The data is not released; this script makes it.
#   0. Training stories, only while data/rl/train/terse holds fewer than 56: Qwen3.6-35B-A3B writes the 64 stories
#      (storygen, fixed ids and seeds) -> data/rl/stories/{full,terse}. tr050-tr057 (terse) are copied to
#      data/rl/dev/terse, the other 56 to data/rl/train/{terse,full}.
#   1. The teacher (Qwen3.6-35B-A3B, temperature 0.2) plays every training story online with the deployed planner
#      prompt -> data/rl/rollouts/states_teacher.jsonl, the GRPO state pool (one state per shot). About 40 minutes
#      with the judge on two A100 GPUs.
#   2. Best of 4 per state (the teacher's output + 3 samples at temperature 0.8), scored with the text reward
#      -> data/rl/sft/bestof.jsonl, the SFT (rejection-sampling fine-tuning) data.
# Every step resumes where it stopped: run the script again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/rl/env.sh
DATA=${DATA:-data/rl}
shopt -s nullglob

train=("$DATA"/train/terse/tr*.json)
if [ "${#train[@]}" -lt 56 ]; then
  python -m agentic_longlive.rl.storygen --urls "$JUDGE_URLS" --model "$JUDGE_MODEL" --out "$DATA/stories"
  written=("$DATA"/stories/terse/tr*.json)
  if [ "${#written[@]}" -lt 64 ]; then
    echo "storygen wrote ${#written[@]} of 64 stories (see $DATA/stories/gen.log); run this script again" >&2
    exit 1
  fi
  mkdir -p "$DATA/train/terse" "$DATA/train/full" "$DATA/dev/terse"
  for f in "${written[@]}"; do
    name=$(basename "$f")
    k=$((10#${name:2:3}))
    if [ "$k" -ge 50 ] && [ "$k" -le 57 ]; then
      cp "$f" "$DATA/dev/terse/"
    else
      cp "$f" "$DATA/train/terse/"
      cp "$DATA/stories/full/$name" "$DATA/train/full/"
    fi
  done
fi

python -m agentic_longlive.rl.collect rollouts --stories "$DATA/train/terse" "$DATA/train/full" \
  --planner-urls "$JUDGE_URLS" --planner-model "$JUDGE_MODEL" --tag teacher --out "$DATA/rollouts" --workers 32
python -m agentic_longlive.rl.collect bestofk --states "$DATA/rollouts/states_teacher.jsonl" \
  --planner-urls "$JUDGE_URLS" --planner-model "$JUDGE_MODEL" --judge-urls "$JUDGE_URLS" --judge-model "$JUDGE_MODEL" \
  --k 3 --out "$DATA/sft/bestof.jsonl" --workers 48
