#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up the LongLive-H3 worker: a Python 3.12 venv with the pinned packages (requirements-h3.txt) and the weights.
# Needs uv (https://docs.astral.sh/uv/), an NVIDIA driver for CUDA 13.0 and Hugging Face access: set HF_TOKEN or run
# `hf auth login`, and accept the MiniMax-H3 license on its model page.
#
#   H3_ROOT=/data/longlive_h3 bash scripts/workers/setup_longlive_h3.sh
#
# Writes under $H3_ROOT:
#   venv/                  worker environment
#   weights/LongLive-H3/   LongLive-H3 transformer (transformer/), from $LONGLIVE_H3_REPO
#   weights/MiniMax-H3/    MiniMax-H3 text encoder, tokenizer, processor and VAEs (its transformers are not needed)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AGL=$(cd "$HERE/../.." && pwd)
H3_ROOT=${H3_ROOT:-$HOME/agentic_longlive/longlive_h3}
VENV=${VENV:-$H3_ROOT/venv}
WEIGHTS=${WEIGHTS:-$H3_ROOT/weights}
LONGLIVE_H3_REPO=${LONGLIVE_H3_REPO:-Andyson/LongLive-H3}
MINIMAX_H3_REPO=MiniMaxAI/MiniMax-H3
MINIMAX_H3_REVISION=fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af

command -v uv > /dev/null || { echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
mkdir -p "$H3_ROOT" "$WEIGHTS"

echo "[1/3] environment $VENV"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "torch==2.11.0" "torchvision==0.26.0" \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$VENV/bin/python" -r "$AGL/requirements-h3.txt"
"$VENV/bin/python" -c "import torch, diffusers, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"

echo "[2/3] LongLive-H3 transformer ($LONGLIVE_H3_REPO)"
"$VENV/bin/hf" download "$LONGLIVE_H3_REPO" --local-dir "$WEIGHTS/LongLive-H3"
test -f "$WEIGHTS/LongLive-H3/transformer/config.json" || { echo "no transformer/ in $WEIGHTS/LongLive-H3" >&2; exit 1; }

echo "[3/3] $MINIMAX_H3_REPO @ $MINIMAX_H3_REVISION (text encoder, tokenizer, processor, VAEs)"
"$VENV/bin/hf" download "$MINIMAX_H3_REPO" --revision "$MINIMAX_H3_REVISION" --local-dir "$WEIGHTS/MiniMax-H3" \
  --exclude "transformer/*" --exclude "transformer_ref/*" --exclude "FL2VA/*" --exclude "Ref2VA/*" \
  --exclude "assets/*" --exclude "docs/*"

du -sh "$WEIGHTS/LongLive-H3" "$WEIGHTS/MiniMax-H3" || true
echo "done. Start the service with: H3_ROOT=$H3_ROOT bash scripts/workers/serve_longlive_h3.sh"
