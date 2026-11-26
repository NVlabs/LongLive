# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU ids and ports of the RL services (sourced by the other scripts in scripts/rl/). Every value can be overridden
# from the environment. The defaults are a single 8-GPU node for one training seed:
#   GPUs 0,1  Qwen3.6-35B-A3B-FP8: teacher planner, text judge and image auditor   :31100 :31101
#   GPU  2    Qwen3-8B with --enable-lora: GRPO rollouts (adapters hot-loaded)    :30100
#   GPUs 3-6  FLUX.2-klein-4B: G / M renders of the image reward                  :32100-:32103
#   GPU  7    trainer (SFT or one GRPO seed)
# The ports match configs/rl/*.json. If you change them, pass the URLs to train.py as overrides (grpo.sh does).
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}
ROLLOUT_MODEL=${ROLLOUT_MODEL:-Qwen/Qwen3-8B}
IMAGE_MODEL=${IMAGE_MODEL:-black-forest-labs/FLUX.2-klein-4B}

JUDGE_GPUS=${JUDGE_GPUS:-"0 1"};        JUDGE_PORTS=${JUDGE_PORTS:-"31100 31101"}
ROLLOUT_GPU=${ROLLOUT_GPU:-2};          ROLLOUT_PORT=${ROLLOUT_PORT:-30100}
IMAGE_GPUS=${IMAGE_GPUS:-"3 4 5 6"};    IMAGE_PORT0=${IMAGE_PORT0:-32100}
TRAIN_GPUS=${TRAIN_GPUS:-7}

LOGS=${LOGS:-outputs/rl/logs}

urls() {  # urls PORT... -> comma-separated OpenAI-compatible base URLs
  local out="" p
  for p in "$@"; do out="$out${out:+,}http://127.0.0.1:$p/v1"; done
  echo "$out"
}
image_ports() {
  local k=0 g
  for g in $IMAGE_GPUS; do echo $((IMAGE_PORT0 + k)); k=$((k + 1)); done
}
# shellcheck disable=SC2086
JUDGE_URLS=$(urls $JUDGE_PORTS)
ROLLOUT_URLS=$(urls "$ROLLOUT_PORT")
# shellcheck disable=SC2046
IMAGE_URLS=$(urls $(image_ports))
