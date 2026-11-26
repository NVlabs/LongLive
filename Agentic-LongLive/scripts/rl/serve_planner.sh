#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Merge a trained LoRA into Qwen3-8B and serve it as the agent's planner (EAGLE-3 speculative decoding) together with
# the agent's checker and image servers (scripts/serve_agent.sh, configs/agents/oss.yaml).
#   bash scripts/rl/serve_planner.sh [ADAPTER_DIR] [MERGED_DIR]
#   defaults: outputs/rl/grpo_s0/adapter_step0080 -> outputs/rl/planner_merged
# Stop the RL servers first (bash scripts/rl/serve.sh stop) if they share the GPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
ADAPTER=${1:-outputs/rl/grpo_s0/adapter_step0080}
MERGED=${2:-outputs/rl/planner_merged}
BASE=${BASE:-Qwen/Qwen3-8B}

if [ ! -f "$MERGED/config.json" ]; then
  python -m agentic_longlive.rl.merge --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"
fi
PLANNER=$MERGED bash scripts/serve_agent.sh
