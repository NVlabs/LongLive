#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up FastH3-Omni on UniServe: clone UniServe at the pinned commit, apply the sink-layout patch, build it as its
# README says (Linux, NVIDIA Hopper or Blackwell GPUs, CUDA 13, Python 3.12, a Rust toolchain) and download the weights.
# Install UniServe's system packages first (as root):
#   apt-get install -y build-essential python3.12-dev pkg-config cmake git curl ca-certificates nasm zlib1g-dev xz-utils
#
#   FH_ROOT=/data/fasth3 CUDA_HOME=/usr/local/cuda-13.0 bash scripts/workers/setup_fasth3.sh
#
# Variables:
#   FH_ROOT               workspace (default $HOME/agentic_longlive/fasth3)
#   CUDA_HOME             CUDA 13 toolkit with nvcc; INSTALL_CUDA=1 installs the CUDA 13.0.2 toolkit under $FH_ROOT
#   TORCH_CUDA_ARCH_LIST  kernel architectures (default: those of the visible GPUs, e.g. 9.0 for H100, 10.0 for B200)
#   MAX_JOBS              parallel compile jobs (default 16)
# Hugging Face access: HF_TOKEN or `hf auth login`; accept the MiniMax-H3 license on its model page.
#
# Writes under $FH_ROOT: uniserve/ (patched, built, .venv), ffmpeg/ (FFmpeg 8.1.2), weights/FastH3-Omni/ and hf_home/
# (the MiniMax-H3 text encoder, tokenizer and VAEs at the revision the checkpoint pins, in the Hugging Face cache).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AGL=$(cd "$HERE/../.." && pwd)
FH_ROOT=${FH_ROOT:-$HOME/agentic_longlive/fasth3}
UNISERVE_REPO=https://github.com/hao-ai-lab/uniserve.git
UNISERVE_COMMIT=d58bee14b46475c690ead7540f9f2bbab7790195
FASTH3_REPO=FastVideo/FastVideo-FastH3-Omni-8-Step-V1
FASTH3_REVISION=53e0d26a1691aa473b6262d8196a8feb2c297df0
mkdir -p "$FH_ROOT"
cd "$FH_ROOT"
stamp() { echo "[$(date +%T)] $*"; }

# ---- toolchain: uv, Rust, CUDA 13
# A pinned uv of our own: UniServe's lock builds uniserve-kernels without build isolation, which an older uv on PATH
# (e.g. 0.8) cannot do ("No module named 'setuptools'").
UV_VERSION=0.13.10
if [ "$("$FH_ROOT/bin/uv" --version 2> /dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
  stamp "installing uv $UV_VERSION into $FH_ROOT/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$FH_ROOT/bin" UV_NO_MODIFY_PATH=1 sh > /dev/null
fi
export PATH="$FH_ROOT/bin:$PATH"
if ! command -v cargo > /dev/null; then
  export RUSTUP_HOME=${RUSTUP_HOME:-$FH_ROOT/rustup} CARGO_HOME=${CARGO_HOME:-$FH_ROOT/cargo}
  if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    stamp "installing Rust into $CARGO_HOME"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
  fi
  export PATH="$CARGO_HOME/bin:$PATH"
fi
# UniServe builds with CUDA 13: an older toolkit under CUDA_HOME (e.g. CUDA 12.x in /usr/local/cuda) does not count
nvcc13() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --version | grep -qE 'release (1[3-9]|[2-9][0-9])\.'; }
if ! nvcc13 "${CUDA_HOME:-/usr/local/cuda}"; then
  if [ "${INSTALL_CUDA:-0}" = 1 ]; then
    CUDA_HOME=$FH_ROOT/cuda-13.0
    if ! nvcc13 "$CUDA_HOME"; then
      stamp "installing the CUDA 13.0.2 toolkit into $CUDA_HOME"
      mkdir -p "$FH_ROOT/tmp"
      curl -sSfL --retry 3 -o "$FH_ROOT/tmp/cuda13.run" \
        https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
      sh "$FH_ROOT/tmp/cuda13.run" --silent --toolkit --toolkitpath="$CUDA_HOME" --no-opengl-libs --no-man-page --no-drm \
        --override --tmpdir="$FH_ROOT/tmp"
      rm -f "$FH_ROOT/tmp/cuda13.run"
    fi
  else
    echo "no CUDA 13 nvcc under CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; point CUDA_HOME at a CUDA 13 toolkit or set INSTALL_CUDA=1" >&2
    exit 1
  fi
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -1

# ---- UniServe at the pinned commit, with the sink layout patch
if [ ! -d uniserve/.git ]; then git clone -q "$UNISERVE_REPO" uniserve; fi
git -C uniserve fetch -q origin
git -C uniserve checkout -q "$UNISERVE_COMMIT"
git -C uniserve --no-pager log -1 --format='uniserve %H %ad %s'
python3 "$AGL/agentic_longlive/workers/patch_uniserve_sink.py" "$FH_ROOT/uniserve"
if [ ! -x ffmpeg/bin/ffprobe ]; then
  stamp "building FFmpeg 8.1.2 into $FH_ROOT/ffmpeg"
  sh uniserve/scripts/install-ffmpeg.sh "$FH_ROOT/ffmpeg"
fi
ffmpeg/bin/ffmpeg -version | head -1

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && command -v nvidia-smi > /dev/null; then
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)
fi
[ -n "${TORCH_CUDA_ARCH_LIST:-}" ] && export TORCH_CUDA_ARCH_LIST
stamp "uv sync (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-visible GPUs})"
(cd uniserve && unset VIRTUAL_ENV && MAX_JOBS="${MAX_JOBS:-16}" uv sync --locked --python 3.12 --extra gpu)
uniserve/.venv/bin/python -c "import PIL, torch; print('torch', torch.__version__, torch.version.cuda)"

# ---- weights: the FastH3-Omni denoiser, and the MiniMax-H3 parts UniServe reads from the Hugging Face cache
export HF_HOME=$FH_ROOT/hf_home
HF=(uv tool run --from "huggingface_hub>=1.0" hf)
stamp "downloading $FASTH3_REPO @ $FASTH3_REVISION"
"${HF[@]}" download "$FASTH3_REPO" --revision "$FASTH3_REVISION" --local-dir "$FH_ROOT/weights/FastH3-Omni"
BASE_REVISION=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['base_model_revision'].split('@')[1])" \
  "$FH_ROOT/weights/FastH3-Omni/fastvideo_inference.json")
stamp "downloading MiniMaxAI/MiniMax-H3 @ $BASE_REVISION into $HF_HOME"
"${HF[@]}" download MiniMaxAI/MiniMax-H3 --revision "$BASE_REVISION" \
  --include "*.json" --include "*.md" --include "LICENSE" --include "text_encoder/*" --include "tokenizer/*" \
  --include "processor/*" --include "vae/*" --include "audio_vae/*" --include "scheduler/*" --include "audio_scheduler/*" \
  --exclude "transformer/*" --exclude "transformer_ref/*" --exclude "FL2VA/*" > /dev/null
du -sh "$FH_ROOT/weights/FastH3-Omni" "$HF_HOME" || true
stamp "done. Start the server and the worker with: FH_ROOT=$FH_ROOT bash scripts/workers/serve_fasth3.sh"
