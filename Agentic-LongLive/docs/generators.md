# Video generators

Agentic LongLive drives three video generators. The memory agent is the same for all of them; only the way memory
pictures enter the generator differs (an attention sink of memory frames before a cut to the new shot).

| Generator | What it is | Where it runs | GPUs | Config |
| --- | --- | --- | --- | --- |
| LongLive-2.0 | real-time autoregressive video model (video only) | in process, `backend: longlive` | 3 (generator, memory prefill, VAE decode) | `configs/longlive.yaml` |
| LongLive-H3 | our few-step chunked-AR distillation of MiniMax-H3 (video + audio) | worker process | 1 per shot rendered at a time, plus 1 for the text encoder | `configs/longlive_h3.yaml` |
| FastH3-Omni | public 8-step distillation of MiniMax-H3's reference-to-video model, served by UniServe | worker + UniServe server | 8 (Hopper or Blackwell) | `configs/fasth3.yaml` |

LongLive-2.0 runs inside the agent's own environment (`requirements.txt` plus `../LongLive2.0/requirements.txt`) and
needs no worker. The two MiniMax-H3 generators have their own environments and run as separate processes.

## Workers and jobs

The agent side (`agentic_longlive/backends/h3.py`, `configs/backends/h3.yaml`) talks to a worker through a file queue:
a directory with `jobs/`, `running/` and `done/` that both sides can reach (same machine or a shared file system). Point
the agent at the worker's queue with `AGL_H3_QUEUE=<queue>` (or `backend.params.queue_dir`). The protocol and the full
job schema are in [`agentic_longlive/workers/protocol.py`](../agentic_longlive/workers/protocol.py).

| Job field | LongLive-H3 | FastH3-Omni |
| --- | --- | --- |
| `prompt`, `seed`, `out` | yes | yes |
| `refs` + `ref_seconds` + `video_start_seconds` (sink: pictures at `ref_seconds`, the shot after a cut) | yes | yes (needs the patched server) |
| `refs` without `ref_seconds` (Ref2VA reference layout) | yes | yes |
| no `refs` (text to video) | yes | no: UniServe refuses `ref2va` without a picture |
| `ref_masks` (foreground tokens of a picture, e.g. G without its studio backdrop) | yes | ignored |
| `keyframe` (the video continues from this frame, e.g. robot video) | no | yes |
| `ref_scale` (picture size relative to the 1344x768 canvas) | 1 only | yes, default 0.5 |
| `duration_seconds` | lengths of 17k + 5 frames at 24 fps (default 8 s) | 4 to the server's `--max-video-seconds` (default 8 s) |

Every shot is 1344x768 at 24 fps with stereo audio. A worker exits on a job with `"stop": true`.

## LongLive-H3

Weights: the LongLive-H3 transformer from [`Andyson/LongLive-H3`](https://huggingface.co/Andyson/LongLive-H3) and the text encoder,
tokenizer and VAEs of `MiniMaxAI/MiniMax-H3` at revision `fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af` (accept its license
on Hugging Face; log in with `hf auth login` or set `HF_TOKEN`). Environment: Python 3.12, PyTorch 2.11 (CUDA 13.0),
diffusers at `b88b343`, pinned in [`requirements-h3.txt`](../requirements-h3.txt).

```bash
H3_ROOT=/data/longlive_h3 bash scripts/workers/setup_longlive_h3.sh                  # venv + weights
H3_ROOT=/data/longlive_h3 TE_GPU=0 GPUS="1" bash scripts/workers/serve_longlive_h3.sh # text encoder + 1 worker
AGL_H3_QUEUE=/data/longlive_h3/queue python scripts/run_story.py --config configs/longlive_h3.yaml
```

`serve_longlive_h3.sh stop` stops the service and puts unfinished jobs back in the queue; `status` shows the queue.
With `GPUS="1 2 3"` three workers share one queue and render three shots at a time (offline batches); `QUEUE_MODE=per-gpu`
gives each worker its own queue (one live session each). By hand:

```bash
python -m agentic_longlive.workers.longlive_h3.text_encoder --weights $H3_ROOT/weights --queue $H3_ROOT/te --device cuda:1
python -m agentic_longlive.workers.longlive_h3.worker --weights $H3_ROOT/weights --te $H3_ROOT/te --queue $H3_ROOT/queue --device cuda:0
```

The text-encoder service keeps the MiniMax-H3 text encoder resident on one GPU and serves every worker on the machine
(results are cached by request, so a repeated memory picture is encoded once). Each worker holds the LongLive-H3
transformer, its KV cache and the VAEs on one 80 GB GPU.

G shows its members on a plain studio backdrop, and only its foreground tokens enter LongLive-H3. The agent side
computes the mask when G is committed (`backend.params.g_masks`): `matte` (default, as in the paper) is BiRefNet
matting OR a backdrop color key (`ZhengPeng7/BiRefNet`, pinned revision, about 6 s per picture on CPU, or set
`mask_device: cuda:N`); `key` uses only the color key and needs no model, but can drop white fur or light clothing;
`false` keeps the whole picture. FastH3-Omni ignores masks.

The released worker is the reference implementation: an exact KV-cached form of LongLive-H3's chunked sampler (memory
pictures as a pinned chunk after the text), one shot per GPU, not real time. The real-time LongLive-H3 numbers in the
paper use 5 B200 GPUs with an optimized runtime that is not part of this release.

## FastH3-Omni

Model: [`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
at revision `53e0d26a1691aa473b6262d8196a8feb2c297df0`, served by [UniServe](https://github.com/hao-ai-lab/uniserve) at
commit `d58bee14b46475c690ead7540f9f2bbab7790195` (task `ref2va`; Linux, NVIDIA Hopper or Blackwell GPUs, CUDA 13).

UniServe's packing places reference pictures right after the prompt and the generated video right after them.
[`patch_uniserve_sink.py`](../agentic_longlive/workers/patch_uniserve_sink.py) adds an opt-in sink layout: while the file
named by `UNISERVE_SINK_PLAN` exists, picture k sits at `40 * ref_seconds[k]` rotary ticks after the prompt and the video
starts at `40 * video_start_seconds` (H3 counts 40 ticks per second). The worker writes that file before each sink request
and removes it afterwards, so run one worker per server. The worker also rewrites the agent's sink prompt into one
sentence saying that the pictures are frames from earlier shots, followed by the shot (`--sink-prompt earlier`, the
default; `r5` sends the prompt unchanged).

```bash
FH_ROOT=/data/fasth3 CUDA_HOME=/usr/local/cuda-13.0 bash scripts/workers/setup_fasth3.sh  # UniServe + patch + weights
# no CUDA 13 toolkit on the machine: add INSTALL_CUDA=1 (installs CUDA 13.0.2 under FH_ROOT)
FH_ROOT=/data/fasth3 PRESET=80gb bash scripts/workers/serve_fasth3.sh                      # server on 8 GPUs + worker
AGL_H3_QUEUE=/data/fasth3/queue python scripts/run_story.py --config configs/fasth3.yaml
```

Presets of `serve_fasth3.sh` (each option can be overridden, see the script header):

| `PRESET` | GPUs | Settings |
| --- | --- | --- |
| `default` | 8 x H200 / B200 class | UniServe's validated FastH3-Omni options, `configs/minimax_h3/ulysses8-reference.json`, BF16 `quality` |
| `80gb` | 8 x 80 GB (H100) | as `default` with `--max-model-len 15360 --max-condition-rows 16384 --graph-policy off` |
| `b200-realtime` | 8 x B200 | the paper's real-time setting: FP8 `performance`, DiT Ulysses-8 with the text encoder on 4 GPUs ([`fasth3_placement_8xb200.json`](../scripts/workers/fasth3_placement_8xb200.json)) |

With `b200-realtime`, an 8 s shot with up to three memory pictures takes about 6.6 s in our measurements. Precision modes
other than `quality` change the generated media. The worker alone:

```bash
python -m agentic_longlive.workers.fasth3 --queue $FH_ROOT/queue --url http://127.0.0.1:18083 --media $FH_ROOT/media
```
