<!--
Long-WAM file attribution; existing upstream notices are retained below.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Original Long-WAM robot-video integration, adapted from the author branch.
Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/README.md
License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
End Long-WAM attribution.
-->

# Stage 1: LongLive 2.0 Robot

Based on [NVlabs/LongLive (LongLive 2.0)](https://github.com/NVlabs/LongLive/tree/main/LongLive2.0).
The required upstream implementation and original license are under
`third_party/LongLive/`; the directory name is upstream's, not a v1 designation.
Imported from the author branch at `87b87ceb65582c46744792a77f2f61754f8274f0`.
S/M/L are the three stages of the long-video curriculum, not disposable test runs.
Full training/validation datasets, manifests and their integrity receipt remain
external assets (**download links TBD**). Standalone inference below includes
seven first-frame examples and does not require those datasets or a receipt.

From any directory use `bash /path/to/Long-WAM/stage1/train.sh ...` or
`bash /path/to/Long-WAM/stage1/eval.sh ...` with the arguments below. Append
`--plan` for CPU-only inspection without launching torchrun.

## Setup

Use Python 3.10, PyTorch 2.8.0+cu128, FlashAttention 2.8.3 and FFmpeg (including ffprobe). Dependencies: `third_party/LongLive/requirements.txt` and `requirements/longlive2-bf16-cu128.lock.txt`. Work from this directory.

Supply the Wan2.2-TI2V-5B assets under `$ASSETS_ROOT/wan_models/Wan2.2-TI2V-5B/`, including the tokenizer, T5 encoder and VAE. Dataset manifests and their matching integrity receipt are external; paths in the receipt and manifests must match the mounted data.

```bash
export LONGWAM_VIDEO_DATA_ROOT=/path/to/robot-video-data
export LONGWAM_VIDEO_INDEX_CACHE=/path/to/robot-video-index
export ASSETS_ROOT=/path/to/assets
python scripts/longlive/prebuild_video_indexes.py --help
```

## Image-to-video inference

The standalone [`infer.py`](infer.py) reuses the bundled LongLive 2.0 causal
pipeline. It loads the online generator only: no trainer, dataset loader,
optimizer, W&B, distributed process group or API credentials.

Install the Stage 1 environment above, including Pillow, `imageio` and
`imageio-ffmpeg`. Supply the local Wan assets:

```text
/path/to/video-assets/wan_models/Wan2.2-TI2V-5B/
├── config.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.2_VAE.pth
└── google/umt5-xxl/
```

From the Long-WAM project root (`LongLive/Long-WAM/`), download Robot-S or use your own compatible trained
generator. Base diffusion weights are not loaded.

```bash
python scripts/download_checkpoint.py stage1_robot_s --output checkpoints/robot-s
bash stage1/infer.sh --list
bash stage1/infer.sh --example all --dry-run

bash stage1/infer.sh --example short_00 \
  --checkpoint checkpoints/robot-s/model.pt \
  --assets-root /path/to/video-assets --output-dir outputs/video-demo
```

Result: `outputs/video-demo/short_00.mp4`. `--example all` runs the seven supplied
examples sequentially with one model load. Existing outputs are never overwritten.
`--checkpoint` (alias `--generator`) accepts a bare state dictionary or a
`model.pt` containing online `generator` weights; loading is strict.
EMA-only, LoRA and quantized exports are not supported by this reference demo.

| Examples | Latent / output frames | Duration at 24 FPS |
| --- | --- | --- |
| `short_00`, `short_03`, `short_04`, `short_05` | 32 / 125 | 5.21 s |
| `long_00`, `long_03`, `long_04` | 64 / 253 | 10.54 s |

The [example manifest](examples/robot_s/examples.json) preserves each supplied
image hash, prompt, sample ID and original seed. Images are lossless 1280×704 RGB
frame-zero extracts from [RoVid-X](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X),
under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Only complete examples are included; the historical full validation panel is separate.

For your own input:

```bash
bash stage1/infer.sh \
  --checkpoint /path/to/trained-generator.pt --assets-root /path/to/video-assets \
  --image /path/to/1280x704-rgb.png --prompt "Place the object into the container" \
  --latent-frames 64 --seed 42 --output-dir outputs/my-video
```

This writes `custom.mp4`. The reference requires a 1280×704 RGB image;
`--latent-frames` must be divisible by 8, with `4 × latent_frames − 3` decoded
frames. Choose a length supported by your checkpoint's training.

[`configs/inference_robot.yaml`](configs/inference_robot.yaml) retains **50 UniPC
steps, CFG 3, shift 5, 8 latent frames/block, local attention 64 and sink 0**.
Use `--config` for a compatible config. A four-step distilled preset is not
equivalent for these weights. This is single-GPU BF16 inference and can require
substantial VRAM; it is not a measured consumer-GPU latency/memory claim.
CPU checks cover inputs/loading contracts; end-to-end GPU generation remains
to be validated in the target environment.

## Train S, M, L

The configurations use 128 workers, microbatch 1 and gradient accumulation 1. Sequence-parallel workers share a sample; global batch counts independent samples.

| Stage | Sequence parallelism | Global batch size | Stop at optimizer step |
| --- | ---: | ---: | ---: |
| S | 2 | 64 | 28602 |
| M | 4 | 32 | 11155 |
| L | 8 | 16 | 3972 |

Batch size is fixed within each stage. Run on each of 16 nodes with eight GPUs per node, setting the torchrun rendezvous variables for your cluster:

```bash
torchrun --nnodes=16 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  run.py train --stage S --output /path/to/output/S \
  --assets-root "$ASSETS_ROOT" --generator /path/to/initial_generator.pt
```

For M and L, change `--stage`, use a new output directory and initialize `--generator` from the previous stage's final `model.pt`. Resume the same stage with `--resume-step N` and its existing output directory. `--plan` resolves configuration without starting training; `--stop-step` selects an earlier stop for your own schedule.

## Validation / long-context demos

This is the inference path used for the robot W&B videos: first-frame-conditioned, online-generator weights, seed **20260711**, **50** sampling steps, **CFG 3**, 1280×704 at 24 FPS.

| Stage | Latent frame lengths | Fixed videos |
| --- | --- | ---: |
| S | 32, 64 | 12 |
| M | 32, 64, 96, 120 | 18 |
| L | 32, 64, 96, 120, 128, 192 | 24 |

```bash
torchrun --standalone --nproc_per_node=8 run.py validation \
  --stage S --training-step 28602 \
  --generator /path/to/S/checkpoint_model_028602/model.pt \
  --assets-root "$ASSETS_ROOT" --output /path/to/new/validation-S

python validation_artifacts.py --output /path/to/new/validation-S
```

The verifier fully decodes each video and checks frames, dimensions, prompts and seeds. To upload the panel to a new W&B run:

```bash
python validation_artifacts.py --output /path/to/new/validation-S \
  --upload --entity YOUR_ENTITY --project YOUR_PROJECT --run-id NEW_RUN_ID
```

Use `--stage M` or `L` with the corresponding checkpoint and real training step for longer contexts. Evaluation runs no optimizer updates.
