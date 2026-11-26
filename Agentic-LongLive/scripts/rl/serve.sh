#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the RL services with SGLang (GPU ids and ports: scripts/rl/env.sh).
#   bash scripts/rl/serve.sh            # judges + rollout server + image servers, waits until all answer
#   bash scripts/rl/serve.sh judge      # only the Qwen3.6 judges (enough for collect.sh)
#   bash scripts/rl/serve.sh stop
# Roles: judge = teacher planner, text judge and image auditor (Qwen3.6-35B-A3B-FP8); rollout = Qwen3-8B with LoRA
# hot-loading for GRPO sampling; image = FLUX.2-klein-4B renderers (OpenAI image API) for the image reward.
# On A100 (no FP8 MoE kernels) serve Qwen/Qwen3.6-35B-A3B with --tp 2 instead and keep the served name via
# --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 so the configs stay unchanged.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/rl/env.sh
mkdir -p "$LOGS"

serve() {  # name gpu port args...
  local name=$1 gpu=$2 port=$3; shift 3
  if curl -s -m 5 -o /dev/null "http://127.0.0.1:$port/"; then echo "$name: port $port already serving"; return; fi
  echo "starting $name on GPU $gpu, port $port (log $LOGS/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu nohup sglang serve --host 127.0.0.1 --port "$port" "$@" > "$LOGS/$name.log" 2>&1 &
  echo $! > "$LOGS/$name.pid"
}
wait_ready() {  # port [path]: an OpenAI-compatible server lists its models; an image server just answers HTTP
  local port=$1 path=${2:-/v1/models}
  if [ "$path" = /v1/models ]; then
    until curl -sf -m 5 "http://127.0.0.1:$port$path" > /dev/null; do sleep 10; done
  else
    until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port$path")" != 000 ]; do sleep 10; done
  fi
  echo "port $port ready"
}

start_judges() {
  local gpus=($JUDGE_GPUS) ports=($JUDGE_PORTS) k
  for k in "${!gpus[@]}"; do
    serve "judge$k" "${gpus[$k]}" "${ports[$k]}" --model-path "$JUDGE_MODEL" --reasoning-parser qwen3 \
      --context-length 32768 --max-mamba-cache-size 200 --max-running-requests 40
  done
}
start_rollout() {
  serve rollout "$ROLLOUT_GPU" "$ROLLOUT_PORT" --model-path "$ROLLOUT_MODEL" --enable-lora --max-lora-rank 32 \
    --lora-target-modules all --max-loras-per-batch 8 --max-loaded-loras 24 --context-length 16384
}
start_images() {
  local k=0 g
  for g in $IMAGE_GPUS; do
    serve "image$k" "$g" $((IMAGE_PORT0 + k)) --model-path "$IMAGE_MODEL" --num-gpus 1 \
      --master-port $((30621 + 100 * k))
    k=$((k + 1))
  done
}

case ${1:-all} in
  judge)   start_judges; for p in $JUDGE_PORTS; do wait_ready "$p"; done ;;
  rollout) start_rollout; wait_ready "$ROLLOUT_PORT" ;;
  image)   start_images; for p in $(image_ports); do wait_ready "$p" /; done ;;
  all)
    start_judges; start_rollout; start_images
    for p in $JUDGE_PORTS $ROLLOUT_PORT; do wait_ready "$p"; done
    for p in $(image_ports); do wait_ready "$p" /; done ;;
  stop)
    for f in "$LOGS"/*.pid; do [ -f "$f" ] && { kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; }; done ;;
  *) echo "usage: $0 [all|judge|rollout|image|stop]"; exit 1 ;;
esac
