#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start / stop the LongLive-H3 service on one machine: the shared text-encoder service on one GPU and one worker per
# GPU (each worker renders one shot at a time on its GPU).
#
#   H3_ROOT=/data/longlive_h3 TE_GPU=0 GPUS="1 2 3" bash scripts/workers/serve_longlive_h3.sh [start|stop|status]
#
# Variables:
#   H3_ROOT     directory written by setup_longlive_h3.sh (venv/, weights/); logs and pids go to $H3_ROOT/logs
#   TE_GPU      GPU of the text encoder + picture encoder service (default 0)
#   GPUS        GPUs with one worker each (default "1")
#   QUEUE       job queue (default $H3_ROOT/queue); the agent's H3 backend reads it from AGL_H3_QUEUE
#   QUEUE_MODE  shared (default): every worker takes jobs from $QUEUE | per-gpu: the worker on GPU g serves $QUEUE/gpu<g>
#   FORCE=1     start even if a target GPU already has more than 2 GB in use
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AGL=$(cd "$HERE/../.." && pwd)
H3_ROOT=${H3_ROOT:-$HOME/agentic_longlive/longlive_h3}
PY=${PY:-$H3_ROOT/venv/bin/python}
WEIGHTS=${WEIGHTS:-$H3_ROOT/weights}
TE_DIR=${TE_DIR:-$H3_ROOT/te}
QUEUE=${QUEUE:-$H3_ROOT/queue}
QUEUE_MODE=${QUEUE_MODE:-shared}
TE_GPU=${TE_GPU:-0}
GPUS=${GPUS:-1}
LOGS=$H3_ROOT/logs
PIDS=$LOGS/pids

export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$AGL${PYTHONPATH:+:$PYTHONPATH}"

queue_of() { if [ "$QUEUE_MODE" = per-gpu ]; then echo "$QUEUE/gpu$1"; else echo "$QUEUE"; fi; }
alive() {
  [ -s "$PIDS" ] || return 1
  while read -r pid _; do if kill -0 "$pid" 2> /dev/null; then return 0; fi; done < "$PIDS"
  return 1
}

start() {
  [ -x "$PY" ] || { echo "no environment at $PY (run setup_longlive_h3.sh)" >&2; exit 1; }
  if alive; then echo "already running (pids in $PIDS); stop it first" >&2; exit 1; fi
  if [ "${FORCE:-0}" != 1 ] && command -v nvidia-smi > /dev/null; then
    for g in $TE_GPU $GPUS; do
      used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      if [ "$used" -gt 2000 ]; then echo "GPU $g is busy (${used} MiB in use); FORCE=1 to start anyway" >&2; exit 1; fi
    done
  fi
  mkdir -p "$LOGS"
  : > "$PIDS"
  CUDA_VISIBLE_DEVICES=$TE_GPU nohup "$PY" -m agentic_longlive.workers.longlive_h3.text_encoder \
    --weights "$WEIGHTS" --queue "$TE_DIR" --device cuda:0 > "$LOGS/text_encoder.log" 2>&1 < /dev/null &
  echo "$! text_encoder $TE_GPU" >> "$PIDS"
  for g in $GPUS; do
    q=$(queue_of "$g")
    mkdir -p "$q"
    CUDA_VISIBLE_DEVICES=$g nohup "$PY" -m agentic_longlive.workers.longlive_h3.worker \
      --weights "$WEIGHTS" --te "$TE_DIR" --queue "$q" --device cuda:0 > "$LOGS/worker_gpu$g.log" 2>&1 < /dev/null &
    echo "$! worker $g $q" >> "$PIDS"
    sleep 10  # stagger the weight loading
  done
  n=$(echo $GPUS | wc -w | tr -d ' ')
  t0=$(date +%s)
  while :; do
    ready=0
    for g in $GPUS; do
      if grep -q '^ready ' "$LOGS/worker_gpu$g.log" 2> /dev/null; then ready=$((ready + 1)); fi
    done
    te=$([ -f "$TE_DIR/ready" ] && echo 1 || echo 0)
    if [ "$te" = 1 ] && [ "$ready" -eq "$n" ]; then
      echo "LongLive-H3 ready: text encoder on GPU $TE_GPU, $n worker(s) on GPUs $GPUS, queue $QUEUE ($QUEUE_MODE)"
      exit 0
    fi
    while read -r pid what gpu _; do
      kill -0 "$pid" 2> /dev/null || { echo "$what on GPU $gpu exited; see $LOGS" >&2; stop; exit 1; }
    done < "$PIDS"
    if [ $(($(date +%s) - t0)) -gt 1800 ]; then echo "timeout (text encoder $te, workers $ready/$n); see $LOGS" >&2; exit 1; fi
    sleep 10
  done
}

stop() {
  [ -s "$PIDS" ] || { echo "nothing to stop"; return 0; }
  while read -r pid _; do kill -TERM "$pid" 2> /dev/null || true; done < "$PIDS"
  for _ in $(seq 30); do alive || break; sleep 1; done
  while read -r pid _; do kill -KILL "$pid" 2> /dev/null || true; done < "$PIDS"
  n=0  # jobs that were being rendered go back to the queue
  for q in $(awk '$2 == "worker" {print $4}' "$PIDS" | sort -u); do
    for f in "$q"/running/*.json; do [ -e "$f" ] || continue; mv "$f" "$q/jobs/" && n=$((n + 1)); done
  done
  : > "$PIDS"
  echo "LongLive-H3 stopped; requeued $n running job(s)"
}

status() {
  [ -s "$PIDS" ] || { echo "not running"; return 0; }
  while read -r pid what gpu q; do
    state=$(kill -0 "$pid" 2> /dev/null && echo running || echo exited)
    echo "$what gpu=$gpu pid=$pid $state ${q:+queue=$q}"
  done < "$PIDS"
  for q in $(awk '$2 == "worker" {print $4}' "$PIDS" | sort -u); do
    echo "$q: $(ls "$q"/jobs/*.json 2> /dev/null | wc -l) queued, $(ls "$q"/running/*.json 2> /dev/null | wc -l) running, \
$(ls "$q"/done/*.json 2> /dev/null | wc -l) done"
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
