<!--
Long-WAM file attribution; existing upstream notices are retained below.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Original Long-WAM implementation.
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

# Training recipes

Use `scripts/<benchmark>/train.sh`; append `--dry-run` for CPU-only composition.
The five benchmark tasks share one model definition. Each parameter below is
covered by a configuration regression check. Paths are environment variables or
explicit YAML overrides; the original cluster launch controllers are not needed.

| Benchmark | Default ranks × batch × accumulation | Global batch | LR | Training | History / action chunk |
| --- | --- | --- | --- | --- | --- |
| LIBERO | 8 × 8 × 2 | 128 | 1e-4 | 10 epochs, cosine | P48 / 32 |
| RoboTwin 2.0 | 8 × 16 × 8 | 1024 | 1e-4 | 5 epochs, cosine | P48 / 32 |
| Domino | 8 × 16 × 8 | 1024 | 3e-6 | 300 updates, cosine | P48 / 32 |
| RoboCasa GR1 | 16 × 30 × 1; P768 below | 480 | 3e-5 | 30K updates, cosine_hf | selected P / 16 |
| RoboCasa365 | 16 × 16 × 1 | 256 | 1e-4 | fresh 100K updates, cosine | P48 / 32 |

All use BF16, seed 42, gradient clipping 1.0, and K2 future latents. AdamW is
beta=(0.9,0.95), epsilon=1e-8, weight decay=0.01 except GR1, which uses
beta=(0.95,0.999), weight decay=1e-5. GR1/365 use 5% warmup. RoboCasa365
warms up over 5,000 updates, then cosine-decays to 1% of peak LR at 100K.
Its schedule is fixed at launch, not extended partway through training.

RoboCasa365 uses Human300's seed-0 random 100 demonstrations/task, the left-main
384×320 three-camera mosaic, 12-D actions and 16-D state. Hierarchical sampling
uses atomic/composite weights **0.36/0.64**, group-uniform sampling within each,
and 512,000 samples/epoch. This is the author's selected public fresh-100K
recipe; released checkpoint metadata remains authoritative for an existing model.

Domino exposes the selected 50:50 RoboTwin/Domino replay fine-tune: 16 RoboTwin
tasks / 880 demos plus 11 Domino tasks / 550 demos, 307,200 samples total. Set
`LONGWAM_ROBOTWIN_CHECKPOINT` to the full RoboTwin initializer. OOD evaluation
instead directly loads that pretrained policy without Domino fine-tuning.

## GR1 scaling

All lengths use the pinned **good Teleop** release, 24 tasks / 24,000 demos,
29-D actions, 58-D state, 20 Hz, the same 30K schedule and global batch 480.

| `context=` | History (seconds) | Raw frames | Sampled frames | Microbatch × accumulation at 16 ranks |
| --- | --- | --- | --- | --- |
| `p0` | 0 | 33 | 9 | 30 × 1 |
| `p48` | 2.4 | 81 | 21 | 30 × 1 |
| `p96` | 4.8 | 129 | 33 | 30 × 1 |
| `p192` | 9.6 | 225 | 57 | 30 × 1 |
| `p384` | 19.2 | 417 | 105 | 30 × 1 |
| `p768` | 38.4 | 801 | 201 | 6 × 5 |

`configs/context/` drives **both model and data**; do not edit only the model's
history. P768 uses the memory-safe recorded microbatching; 10×3 is an alternative
with the same global batch, but is not a promise of bitwise-identical training.
Evaluation automatically reads the matching checkpoint's context, retains H16
and executes 16 actions before replanning. Use `expected_history=192`, for example,
to reject an accidentally mismatched release bundle.

```bash
bash scripts/robocasa_gr1/train.sh context=p192 --dry-run
# All context lengths use the shared public Robot-S video initializer.
bash scripts/robocasa_gr1/train.sh context=p192 \
  output_dir=outputs/gr1-p192-robot
```

All six GR1 contexts default to the public LongLive2.0-Robot-S video generator.
Context selection does not change initialization, data or optimizer settings.
The mapped ActionDiT backbone and GR1-specific I/O layers remain separate from
the video initializer; no RoboTwin action head is copied into GR1.

## IDM and CodeDenoise

LIBERO/RoboTwin support `denoising=idm` (default) and `denoising=codenoise`.
These select the actual training attention regime, not a visualization flag.
IDM uses four partial video steps down to sigma=0.9, followed by action denoising.
CodeDenoise uses joint future-video/action attention and a full schedule to zero.
Evaluation uses `inference_mode=idm` or `inference_mode=codenoise` with a matching
checkpoint/config; it refuses to reinterpret IDM weights as CodeDenoise weights.

## Initialization, fine-tuning and resume

**Video expert:** all fresh LIBERO, RoboTwin 2.0, RoboCasa GR1 and RoboCasa365
runs use the public
[LongLive2.0-Robot-S](https://huggingface.co/Efficient-Large-Model/LongLive2.0-Robot-S)
generator, not Robot-M or the general LongLive 2.0 AR checkpoint. Download key
`stage1_robot_s` pins revision `9f23acf1a619afe69d866f105c17209880851894` in
[`configs/checkpoints.yaml`](../configs/checkpoints.yaml). Its `model.pt` SHA-256 is
`f7e9693ef4e6d19c2e13d1cfaf7abf306ec70b096da5520dd9f0f0ef96f3962d`.

```bash
python scripts/download_checkpoint.py stage1_robot_s \
  --output "${LONGWAM_CHECKPOINT_ROOT:-./checkpoints}/LongLive2.0-Robot-S"
(cd "${LONGWAM_CHECKPOINT_ROOT:-./checkpoints}/LongLive2.0-Robot-S" && sha256sum -c SHA256SUMS)
# Optional override when the same Robot-S model.pt is stored elsewhere:
export LONGWAM_ROBOT_VIDEO_CHECKPOINT=/absolute/path/to/LongLive2.0-Robot-S/model.pt
```

The default loads this file directly, strips the generator's `model.` prefix,
and strictly loads the video expert. No video conversion step or private
checkpoint is required. `model_id: Wan-AI/Wan2.2-TI2V-5B` still supplies the
architecture/VAE assets; its video DiT weights are skipped.

**ActionDiT is independent of video initialization.** RoboTwin uses a randomly
initialized **entire ActionDiT**, including its 14-D action interface:
`action_dit_pretrained_path: null` is intentional, not a missing mapping path.
LIBERO and RoboCasa retain their separate mapped action-backbone configuration
under `$LONGWAM_CHECKPOINT_ROOT/longwam/ActionDiT_from_longlive_ar_1024hdim.pt`.
Generate it directly from the public Robot-S video checkpoint (no separate
ActionDiT download or pretraining run):

```bash
python tools/weights/preprocess_action_dit_backbone.py
```

This CPU-only command reuses the local Robot-S `model.pt`, or downloads the pinned
release when absent. It copies matching tensors, interpolates resized tensors with
alpha scaling, and excludes the action input/output layers, which are initialized
randomly when training. The output goes to the path already used by the training
configs; the filename is retained for compatibility. Its source SHA-256 and public
revision are recorded in the generated file. Use a CPU machine with 32 GB RAM;
allow about 4 GB for the output, plus space for Robot-S if not already downloaded.

For an existing local source or a custom output:

```bash
python tools/weights/preprocess_action_dit_backbone.py \
  --video-weights /path/to/LongLive2.0-Robot-S/model.pt \
  --output /path/to/action_initializer.pt
# Pass a custom output to a fresh training run:
bash scripts/robocasa_gr1/train.sh model.action_dit_pretrained_path=/path/to/action_initializer.pt
```

`LONGWAM_ROBOT_VIDEO_CHECKPOINT` also selects a local source. Existing outputs are
not overwritten unless `--overwrite` is explicit. Conversion is a preparation
step for **fresh training**, not for resume or inference; changing the video path
alone does not regenerate ActionDiT. A different source checkpoint produces a
different initializer, not a bitwise reproduction of an earlier conversion.

**Domino** loads the complete robot-pretrained RoboTwin policy through
`LONGWAM_ROBOTWIN_CHECKPOINT`, retaining its trained video/action experts and
normalization. Its separate video/action initialization fields are therefore
null. Hugging Face policy-bundle `config.yaml` files are **inference settings**:
null initializers there avoid loading pretraining weights before the full policy.

Precompute text embeddings before training as described in [DATA.md](DATA.md).

### Fine-tuning and resume

`resume=/path/to/weights.pt` starts a new optimization run from full policy
weights. `resume=/path/to/checkpoints/state/step_NNNNNN` restores a compatible
optimizer/scheduler/dataloader state. Neither should silently change robot
dimensions, history, normalization or denoising regime. Evaluation never loads
the training initialization assets again.

## Topology and storage

```bash
# Four GPUs, same GR1 global batch:
NPROC_PER_NODE=4 bash scripts/robocasa_gr1/train.sh context=p96 --dry-run \
  batch_size=15 gradient_accumulation_steps=8
# Optional single-node ZeRO-1, after pip install -e '.[distributed]':
ACCELERATE_CONFIG=configs/distributed/zero1.yaml bash scripts/robocasa_gr1/train.sh
```

Multi-node torchrun uses `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`
and `MASTER_PORT`. Adjust batch/accumulation to preserve global batch. Historical
LIBERO and RoboTwin training used other topologies; equal global batch is not a
claim of bitwise-identical sample ordering. Dry-run validation cannot prove GPU
memory fit. The convenience Accelerate wrapper supports one node only.

Final weights and optimizer state are retained. RoboCasa365 retains the newest
three committed weight/state pairs; pin selected evaluations via
`protected_checkpoint_steps=[20000,60000]` or a `.keep` file inside that state
directory. Incomplete saves, symlinks and weights without a committed state pair
are not auto-pruned. This mechanism only acts inside the current training output,
not the original research directories. Benchmark evaluation is a separate script.
