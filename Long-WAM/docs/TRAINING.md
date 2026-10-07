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
# Robot-video initialization variant: replace only the video expert.
bash scripts/robocasa_gr1/train.sh context=p192 \
  model.longlive_video_weights=/path/to/LongLive-2.0-Robot.pt \
  output_dir=outputs/gr1-p192-robot
```

The scaling default retains the original video/action initializer pair.
The Robot-initialized P96/P192 runs use the same length/data/optimizer parameters
with only `model.longlive_video_weights` replaced; a Robot-S joint payload is
supported by extracting its video expert. Its robot-specific action head is
not copied into the GR1 embodiment.

## IDM and CodeDenoise

LIBERO/RoboTwin support `denoising=idm` (default) and `denoising=codenoise`.
These select the actual training attention regime, not a visualization flag.
IDM uses four partial video steps down to sigma=0.9, followed by action denoising.
CodeDenoise uses joint future-video/action attention and a full schedule to zero.
Evaluation uses `inference_mode=idm` or `inference_mode=codenoise` with a matching
checkpoint/config; it refuses to reinterpret IDM weights as CodeDenoise weights.

## Initialization, fine-tuning and resume

The shared model points to the LongLive 2.0 video expert and mapped ActionDiT
under `$LONGWAM_CHECKPOINT_ROOT/longwam/`. These assets are separate from a full
fine-tuned benchmark policy. Their download URLs are TBD; converters are under
`tools/weights/`. RoboTwin reinitializes its 14-D action interface; GR1/365 retain
their own embodiment-specific I/O layers. Precompute text embeddings before
training as described in [DATA.md](DATA.md).

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
