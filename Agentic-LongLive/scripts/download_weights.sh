#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the checkpoints of the full system on LongLive-2.0 into checkpoints/ (about 60 GB).
#   bash scripts/download_weights.sh             # generator + agent models
#   bash scripts/download_weights.sh generator   # LongLive-2.0 only (enough for configs/ablations/no_memory.yaml)
# The agent's open models (Qwen3.6-35B-A3B, FLUX.2-klein-4B, the EAGLE-3 draft) are fetched by SGLang on first use;
# they are listed here so that they can be pre-downloaded on machines without internet at serving time.
set -euo pipefail
WHAT=${1:-all}
C=${CKPT_DIR:-checkpoints}
mkdir -p "$C"
hf download Efficient-Large-Model/LongLive-2.0-5B model_bf16.pt --local-dir "$C/LongLive-2.0-5B"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir "$C/wan_models/Wan2.2-TI2V-5B" --exclude "assets/*" --exclude "examples/*"
[ "$WHAT" = generator ] && exit 0
# RL memory planner (Qwen3-8B + GRPO LoRA, merged; coming soon)
hf download Efficient-Large-Model/Agentic-LongLive-Planner-Qwen3-8B --local-dir "$C/Agentic-LongLive-Planner" \
  || echo "The RL planner is not released yet: run without it via --ablation configs/ablations/no_rl_planner.yaml"
hf download Tengyunw/qwen3_8b_eagle3
hf download Qwen/Qwen3.6-35B-A3B-FP8
hf download black-forest-labs/FLUX.2-klein-4B
