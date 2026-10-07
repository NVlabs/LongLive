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

# Data preparation

Set `LONGWAM_DATA_ROOT` to an absolute path, or override `data.train.dataset_dirs`.
Data is never downloaded as a side effect of the training launcher. The dataset
download helper downloads only; inspect archives and extract them explicitly.
Retain upstream license/access requirements and pin dataset revisions in release
metadata. Simulator assets are separate from demonstration datasets.

## LIBERO

The merged recipe uses the four LeRobot archives from
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam),
not the original HDF5 loader:

```bash
python scripts/download_data.py libero --output data/downloads/libero
```

Extract the four archives so the configured directories exist under
`$LONGWAM_DATA_ROOT/libero_mujoco3.3.2/`:
`libero_10_no_noops_lerobot`, `libero_goal_no_noops_lerobot`,
`libero_object_no_noops_lerobot`, `libero_spatial_no_noops_lerobot`.
Each contains `meta/`, `data/` and `videos/`. The dataset name reflects the
recorded MuJoCo data version; do not silently replace it with a different release.

## RoboTwin 2.0

```bash
python scripts/download_data.py robotwin2 --output data/downloads/robotwin2
```

The source is [yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam).
Its split archive must be reassembled in numeric part order before extraction.
The configured dataset is `$LONGWAM_DATA_ROOT/robotwin2.0/robotwin2.0/`, with
normalization at `$LONGWAM_DATA_ROOT/robotwin2.0/dataset_stats.json`.
Check the resolved YAML with `scripts/robotwin2/train.sh --dry-run`.

## Domino

The project evaluates **35 level-1 clean dynamic tasks**. The released fine-tune
uses the selected 11-task, 50-demonstration/task subset together with RoboTwin
replay described below. These are subsets of the larger official release.
Obtain the corresponding archives from
[H-EmbodVis/DOMINO](https://huggingface.co/datasets/h-embodvis/DOMINO).
Use `scripts/download_data.py domino --include '<desired-pattern>' --output ...`
to avoid downloading unrelated levels.

Convert upstream HDF5 demonstrations to LeRobot using the converter supplied
under the pinned DOMINO `policy/PUMA` tree. The included
`domino_lerobot_compat.patch` adapts its LeRobot schema compatibility; this does
not require using the PUMA model. The old cluster-specific bulk download and
conversion launcher is not a public entry point yet: its portable orchestration
is listed in [BENCHMARKS.md](BENCHMARKS.md). Prepared LeRobot data can be used now
by matching the paths in `configs/data/domino.yaml`.

Keep the 10-Hz temporal recipe: source-frame sampling and dynamic-motion rate
are part of the experiment. Do not change rates independently of history/action
sampling. The released training recipe uses 16 selected RoboTwin tasks (880
demonstrations) and 11 selected Domino tasks (550 demonstrations), with 50:50
sampling weights. The remaining Domino tasks are still evaluated.
Selection/materialization helpers are under `tools/data/`; the exact selections
and domain weights are in `configs/data/domino.yaml` and
`configs/task/domino.yaml`. Avoid replacing that mixture with a concatenation
whose sampling ratio is determined by dataset size.

## RoboCasa GR1: use the good Teleop release

Use [PhysicalAI-Robotics-GR00T-Teleop-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim)
at revision `09c6de8af50168090e7e9cc01e1ec3bce788de24`: 24 tasks, 24,000 episodes,
5,820,277 frames, 20 Hz. Do not substitute the older X-Embodiment release.

```bash
python scripts/download_data.py robocasa_gr1 \
  --include 'LeRobot/gr1_unified.*/*' \
  --output data/robocasa_gr1_tabletop/PhysicalAI-Robotics-GR00T-Teleop-Sim
export ROBOCASA_GR1_TABLETOP_DATA_ROOT="$LONGWAM_DATA_ROOT/robocasa_gr1_tabletop/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot"
export ROBOCASA_GR1_TABLETOP_ACTION_STATS="$LONGWAM_DATA_ROOT/robocasa_gr1_tabletop/teleop_derived/global_action_minmax.json"
python tools/data/build_robocasa_gr1_tabletop_stats.py \
  --data-root "$ROBOCASA_GR1_TABLETOP_DATA_ROOT" \
  --inventory configs/data/gr1_teleop_inventory.json \
  --output "$ROBOCASA_GR1_TABLETOP_ACTION_STATS"
```

The statistics builder reads CPU metadata, not videos. It validates inventory
and selects the official 29 action fields from the 44-D raw representation;
proprioception becomes 58-D. Episode remarks provide the Teleop prompts.

## RoboCasa365

Use the pinned RoboCasa download utility:

```bash
python /path/to/RoboCasa/robocasa/scripts/download_datasets.py --source human --split pretrain
```

Configure the upstream dataset download directory, then set `ROBOCASA365_DATA_ROOT`
to the root containing `v1.0/pretrain/`. The complete inventory is in
`configs/data/robocasa365_paths.yaml`: 65 atomic and 235 composite tasks.
Training selects the official seed-0 random 100 demonstrations per task, **not**
the first 100. Sampling weights are atomic 0.36 / composite 0.64, task-balanced
inside each family.

```bash
export ROBOCASA365_NORM_STATS="$LONGWAM_DATA_ROOT/robocasa365/cache/human300_norm_stats.json"
python tools/data/build_robocasa_norm_stats.py \
  --data-root "$ROBOCASA365_DATA_ROOT" --output "$ROBOCASA365_NORM_STATS"
```

Evaluation with a released policy must use its packaged statistics rather than
rebuilding them from a different release or subset. The selected policy uses the
left-main camera mosaic, 12-D actions, 16-D proprioception and 20-Hz observations.

## Text embeddings

Training reads cached UMT5 embeddings. First inspect the CPU-only prompt plan:

```bash
python scripts/precompute_text_embeds.py task=robocasa_gr1 +text_cache_mode=plan
# Encode missing entries using a prepared model environment:
python scripts/precompute_text_embeds.py task=robocasa_gr1
# Verify cache completeness before a training allocation:
python scripts/precompute_text_embeds.py task=robocasa_gr1 +text_cache_mode=verify
```

Replace the task name for other benchmarks. Cache locations are in each data
YAML (`text_embedding_cache_dir`). Encoding needs the T5 weights and sufficient
compute; planning, statistics and inventory validation belong on CPU. Prompt
hashes, padding and masks are shared between training and inference.
