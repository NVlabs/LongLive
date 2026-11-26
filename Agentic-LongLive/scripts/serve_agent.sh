#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the agent's model servers with SGLang (one GPU each), matching configs/agents/oss.yaml:
#   planner  RL memory planner (Qwen3-8B + GRPO LoRA, merged) with EAGLE-3 speculative decoding   :30000
#   checker  Qwen3.6-35B-A3B-FP8 with MTP: image checks and scene routing                         :31001
#   image    FLUX.2-klein-4B: entity images and G / M memory pictures                             :32001
#
#   bash scripts/serve_agent.sh                      # GPUs 3 4 5, logs in outputs/servers/
#   PLANNER_GPU=5 CHECKER_GPU=6 IMAGE_GPU=7 bash scripts/serve_agent.sh
#   PLANNER=Qwen/Qwen3.6-35B-A3B-FP8 ...             # without the RL planner: see configs/ablations/no_rl_planner.yaml
#   bash scripts/serve_agent.sh stop                 # stop them (same LOGS)
#
# On A100 (no FP8 MoE kernels) serve the BF16 checker on two GPUs and tell the client its name:
#   CHECKER=Qwen/Qwen3.6-35B-A3B CHECKER_TP=2 CHECKER_GPU=3,4 IMAGE_GPU=5 PLANNER_GPU=6 bash scripts/serve_agent.sh
#   export AGL_CHECK_MODEL=Qwen/Qwen3.6-35B-A3B AGL_ROUTE_MODEL=Qwen/Qwen3.6-35B-A3B   # before scripts/run_story.py
set -euo pipefail
PLANNER=${PLANNER:-checkpoints/Agentic-LongLive-Planner}          # merged RL planner (scripts/download_weights.sh)
PLANNER_DRAFT=${PLANNER_DRAFT:-Tengyunw/qwen3_8b_eagle3}           # EAGLE-3 draft head for Qwen3-8B
CHECKER=${CHECKER:-Qwen/Qwen3.6-35B-A3B-FP8}
CHECKER_TP=${CHECKER_TP:-1}
IMAGE=${IMAGE:-black-forest-labs/FLUX.2-klein-4B}
PLANNER_GPU=${PLANNER_GPU:-3}; CHECKER_GPU=${CHECKER_GPU:-4}; IMAGE_GPU=${IMAGE_GPU:-5}
PLANNER_PORT=${PLANNER_PORT:-30000}; CHECKER_PORT=${CHECKER_PORT:-31001}; IMAGE_PORT=${IMAGE_PORT:-32001}
LOGS=${LOGS:-outputs/servers}
PIDS=$LOGS/servers.pid  # "<pid> <name>" per server; each server leads its own process group
mkdir -p "$LOGS"

alive() { kill -0 -- "-$1" 2> /dev/null; }
stop() {  # the whole process group of each server: SGLang's scheduler and detokenizer processes too
  [ -s "$PIDS" ] || { echo "no agent servers recorded in $PIDS"; return 0; }
  local pid name left
  while read -r pid name; do kill -TERM -- "-$pid" 2> /dev/null || true; done < "$PIDS"
  for _ in $(seq 30); do
    left=0; while read -r pid name; do if alive "$pid"; then left=1; fi; done < "$PIDS"
    [ "$left" = 0 ] && break; sleep 1
  done
  while read -r pid name; do kill -KILL -- "-$pid" 2> /dev/null || true; done < "$PIDS"
  rm -f "$PIDS"
  echo "agent servers stopped"
}
case "${1:-start}" in
  start) ;;
  stop) stop; exit 0 ;;
  *) echo "usage: $0 [start|stop]" >&2; exit 2 ;;
esac
if [ -s "$PIDS" ]; then
  while read -r pid name; do
    if alive "$pid"; then echo "agent servers already running ($name, pid $pid); run: bash $0 stop" >&2; exit 1; fi
  done < "$PIDS"
fi
: > "$PIDS"

# SGLang compiles some kernels on first use and needs a CUDA 13 nvcc; an older toolkit fails there (e.g. "namespace cuda
# has no member maximum"). Unless CUDA_HOME holds CUDA 13, use the CUDA 13 compiler wheel that SGLang's install brings.
nvcc13() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --version | grep -qE 'release (1[3-9]|[2-9][0-9])\.'; }
if ! nvcc13 "${CUDA_HOME:-/usr/local/cuda}"; then
  cu=$("$(dirname "$(command -v sglang)")/python" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))' 2> /dev/null || true)
  if [ -n "$cu" ] && nvcc13 "$cu"; then
    # the wheel ships lib/ with versioned libraries only, while the JIT links with -L$CUDA_HOME/lib64 -lcudart
    [ -e "$cu/lib64" ] || ln -s lib "$cu/lib64"
    [ -e "$cu/lib/libcudart.so" ] || ln -s libcudart.so.13 "$cu/lib/libcudart.so"
    export CUDA_HOME=$cu PATH=$cu/bin:$PATH
  else
    echo "warning: no CUDA 13 nvcc; SGLang JIT kernels may fail to build" >&2
  fi
fi
echo "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}"

serve() {  # name gpus port args...
  local name=$1 gpus=$2 port=$3; shift 3
  echo "starting $name on GPU $gpus, port $port (log $LOGS/$name.log)"
  # setsid: detached from this shell's session, so the servers (and the compile workers they start later) survive the
  # end of the terminal / SSH / tmux session that started them
  CUDA_VISIBLE_DEVICES=$gpus setsid nohup sglang serve --host 127.0.0.1 --port "$port" "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
  echo "$! $name" >> "$PIDS"
}

serve checker "$CHECKER_GPU" "$CHECKER_PORT" --model-path "$CHECKER" --tp "$CHECKER_TP" --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --context-length 32768
serve image "$IMAGE_GPU" "$IMAGE_PORT" --model-path "$IMAGE" --num-gpus 1
if [ "$PLANNER" != "$CHECKER" ]; then
  serve planner "$PLANNER_GPU" "$PLANNER_PORT" --model-path "$PLANNER" --served-model-name agentic-longlive-planner \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path "$PLANNER_DRAFT" --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 --context-length 16384
fi

wait_ready() {  # name port
  local pid
  pid=$(awk -v n="$1" '$2 == n {print $1}' "$PIDS")
  until curl -sf -m 5 "http://127.0.0.1:$2/v1/models" > /dev/null; do
    alive "$pid" || { echo "$1 server exited; see $LOGS/$1.log" >&2; exit 1; }
    sleep 5
  done
  echo "$1 ready on port $2"
}
wait_ready checker "$CHECKER_PORT"
wait_ready image "$IMAGE_PORT"
if [ "$PLANNER" != "$CHECKER" ]; then wait_ready planner "$PLANNER_PORT"; fi
