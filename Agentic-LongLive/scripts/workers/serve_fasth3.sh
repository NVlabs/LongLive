#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start / stop FastH3-Omni: the UniServe server (patched for the sink layout, 8 GPUs) and the FastH3 worker that serves
# the agent's job queue through it.
#
#   FH_ROOT=/data/fasth3 bash scripts/workers/serve_fasth3.sh [start|stop|status]
#
# Variables:
#   FH_ROOT      directory written by setup_fasth3.sh; logs and pids go to $FH_ROOT/logs
#   GPUS         GPUs of the server (default 0,1,2,3,4,5,6,7)       PORT   server port on 127.0.0.1 (default 18083)
#   QUEUE        job queue (default $FH_ROOT/queue); the agent's H3 backend reads it from AGL_H3_QUEUE
#   MEDIA        the server's --media-directory: worker pictures and the sink plan file (default $FH_ROOT/media)
#   PRESET       server settings, each overridable by the variables below:
#                  default        UniServe's validated FastH3-Omni settings (8 x H200-class GPUs), BF16 "quality"
#                  80gb           8 x 80 GB GPUs (H100): --max-model-len 15360 --max-condition-rows 16384
#                                 --graph-policy off
#                  b200-realtime  the real-time setting of the paper: 8 x B200, FP8 "performance", DiT Ulysses-8 with the
#                                 text encoder on 4 GPUs (fasth3_placement_8xb200.json)
#   WORKERS QUANT MEM MAXLEN CAPACITIES MAXSEC CONDROWS EXTRA   UniServe options (see the preset defaults below)
#   REF_SCALE (0.5)  SINK_PROMPT (earlier)  WARMUP (job JSON rendered once before ready)  WORKER_PYTHON
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AGL=$(cd "$HERE/../.." && pwd)
FH_ROOT=${FH_ROOT:-$HOME/agentic_longlive/fasth3}
US=$FH_ROOT/uniserve
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-18083}
QUEUE=${QUEUE:-$FH_ROOT/queue}
MEDIA=${MEDIA:-$FH_ROOT/media}
LOGS=$FH_ROOT/logs
PIDS=$LOGS/pids
PRESET=${PRESET:-default}
case "$PRESET" in
  default) ;;
  80gb) MAXLEN=${MAXLEN:-15360} CONDROWS=${CONDROWS:-16384} EXTRA=${EXTRA:---graph-policy off} ;;
  b200-realtime) WORKERS=${WORKERS:-$HERE/fasth3_placement_8xb200.json} QUANT=${QUANT:-'{"mode":"performance"}'} ;;
  *) echo "unknown PRESET $PRESET (default | 80gb | b200-realtime)" >&2; exit 2 ;;
esac
WORKERS=${WORKERS:-configs/minimax_h3/ulysses8-reference.json}
QUANT=${QUANT:-'{"mode":"quality"}'}
MEM=${MEM:-0.92}
MAXLEN=${MAXLEN:-24576}
CAPACITIES=${CAPACITIES:-4096,7168,8192,9216,10240,12288,15360}
MAXSEC=${MAXSEC:-8}
CONDROWS=${CONDROWS:-56192}
EXTRA=${EXTRA:-}
REF_SCALE=${REF_SCALE:-0.5}
SINK_PROMPT=${SINK_PROMPT:-earlier}
WORKER_PYTHON=${WORKER_PYTHON:-$US/.venv/bin/python}
if [ -x "$FH_ROOT/ffmpeg/bin/ffmpeg" ]; then FFBIN=$FH_ROOT/ffmpeg/bin; else FFBIN=$(dirname "$(command -v ffmpeg)"); fi
CUDA_HOME=${CUDA_HOME:-$([ -x "$FH_ROOT/cuda-13.0/bin/nvcc" ] && echo "$FH_ROOT/cuda-13.0" || echo /usr/local/cuda)}

pid_of() { [ -s "$PIDS" ] && awk -v w="$1" '$2 == w {print $1}' "$PIDS" | tail -1; }
running() { local p; p=$(pid_of "$1"); [ -n "$p" ] && kill -0 "$p" 2> /dev/null; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; }

start_server() {
  mkdir -p "$LOGS" "$MEDIA" "$FH_ROOT/cache"
  # UniServe creates Unix sockets under TMPDIR (e.g. uniserve-descriptor-grants/uniserve-publications-<32 hex>.sock, 87
  # characters below it); Linux caps a socket path at 107, so TMPDIR itself must stay within 20 characters
  local tmp=/dev/shm/fh$PORT
  [ -d /dev/shm ] || tmp=/tmp/fh$PORT
  mkdir -p "$tmp"
  echo "server: GPUS=$GPUS WORKERS=$WORKERS QUANT=$QUANT MEM=$MEM MAXLEN=$MAXLEN CONDROWS=$CONDROWS EXTRA=$EXTRA"
  # setsid: the server and the model processes it spawns form one process group, stopped together
  (cd "$US" && unset VIRTUAL_ENV && exec env \
    TMPDIR="$tmp" HF_HOME="$FH_ROOT/hf_home" HF_HUB_OFFLINE=1 CUDA_HOME="$CUDA_HOME" \
    PATH="$US/.venv/bin:$FFBIN:$CUDA_HOME/bin:$PATH" CUDA_VISIBLE_DEVICES="$GPUS" \
    NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" TRITON_CACHE_DIR="$FH_ROOT/cache/triton" XDG_CACHE_HOME="$FH_ROOT/cache" \
    UNISERVE_SINK_PLAN="$MEDIA/sink_plan.json" \
    setsid .venv/bin/uniserve serve "$FH_ROOT/weights/FastH3-Omni" \
      --served-model-name FastH3-Omni --host 127.0.0.1 --port "$PORT" --worker-python .venv/bin/python \
      --workers "$WORKERS" --max-running-requests 2 --max-model-len "$MAXLEN" --video-text-capacities "$CAPACITIES" \
      --max-video-seconds "$MAXSEC" --max-condition-rows "$CONDROWS" --video-resolutions 768p --video-aspect-ratios 16:9 \
      --media-directory "$MEDIA" --ffprobe "$FFBIN/ffprobe" --ffmpeg "$FFBIN/ffmpeg" \
      --mem-fraction-static "$MEM" --dtype bfloat16 --quantization-config "$QUANT" $EXTRA \
    > "$LOGS/server.log" 2>&1 < /dev/null) &
  echo "$! server" >> "$PIDS"
  local t0
  t0=$(date +%s)
  until healthy; do  # the first start on a machine also compiles kernels
    running server || { echo "UniServe exited; see $LOGS/server.log" >&2; exit 1; }
    if [ $(($(date +%s) - t0)) -gt 3600 ]; then echo "UniServe not ready after 1 h; see $LOGS/server.log" >&2; exit 1; fi
    sleep 5
  done
  echo "UniServe ready on port $PORT ($(($(date +%s) - t0)) s)"
}

start_worker() {
  mkdir -p "$QUEUE" "$LOGS"
  rm -f "$QUEUE/ready"
  (cd "$AGL" && exec env PYTHONPATH="$AGL${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 \
    "$WORKER_PYTHON" -m agentic_longlive.workers.fasth3 --queue "$QUEUE" --url "http://127.0.0.1:$PORT" --media "$MEDIA" \
      --sink-plan "$MEDIA/sink_plan.json" --ref-scale "$REF_SCALE" --sink-prompt "$SINK_PROMPT" ${WARMUP:+--warmup "$WARMUP"} \
    > "$LOGS/worker.log" 2>&1 < /dev/null) &
  echo "$! worker" >> "$PIDS"
  for _ in $(seq 600); do
    if [ -f "$QUEUE/ready" ]; then echo "FastH3 worker ready on $QUEUE"; return 0; fi
    running worker || { echo "FastH3 worker exited; see $LOGS/worker.log" >&2; exit 1; }
    sleep 1
  done
  echo "FastH3 worker not ready after 10 min; see $LOGS/worker.log" >&2
  exit 1
}

stop() {
  local p
  p=$(pid_of worker) && [ -n "$p" ] && kill -TERM "$p" 2> /dev/null || true
  p=$(pid_of server)
  if [ -n "$p" ]; then
    kill -TERM -- "-$p" 2> /dev/null || kill -TERM "$p" 2> /dev/null || true
    for _ in $(seq 60); do kill -0 "$p" 2> /dev/null || break; sleep 1; done
    kill -KILL -- "-$p" 2> /dev/null || true
  fi
  rm -f "$MEDIA/sink_plan.json"
  for f in "$QUEUE"/running/*.json; do if [ -e "$f" ]; then mv "$f" "$QUEUE/jobs/"; fi; done  # back to the queue
  : > "$PIDS" 2> /dev/null || true
  echo "FastH3 stopped"
}

case "${1:-start}" in
  start)
    [ -x "$US/.venv/bin/uniserve" ] || { echo "no UniServe build under $US (run setup_fasth3.sh)" >&2; exit 1; }
    mkdir -p "$LOGS"
    if healthy; then echo "a server already answers on port $PORT"; else : > "$PIDS"; start_server; fi
    if running worker; then echo "worker already running"; else start_worker; fi
    ;;
  stop) stop ;;
  status)
    for w in server worker; do echo "$w: $(running $w && echo "running (pid $(pid_of $w))" || echo stopped)"; done
    echo "health: $(healthy && echo ok || echo down)"
    echo "$QUEUE: $(ls "$QUEUE"/jobs/*.json 2> /dev/null | wc -l) queued, $(ls "$QUEUE"/running/*.json 2> /dev/null | wc -l) running"
    ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
