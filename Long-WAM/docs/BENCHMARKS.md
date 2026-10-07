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

# Benchmark support and simulator setup

The five public benchmarks have separate train/eval scripts. Model inference
uses strict checkpoint loading, matched camera/action normalization and the
checkpoint's history. Config checks and mocked rollouts run on CPU; they do not
establish simulator success rates. No GPU experiment was rerun during this merge.

| Benchmark | Public support |
| --- | --- |
| LIBERO | Training; IDM/CodeDenoise inference; four-suite evaluation |
| RoboTwin 2.0 | Training; IDM/CodeDenoise native policy; 50-task clean/randomized evaluation |
| Domino | Selected balanced-replay fine-tuning; OOD and fine-tuned evaluation with SR/MS |
| RoboCasa GR1 | Good Teleop data; all six scaling contexts; history-matched 24-task evaluation |
| RoboCasa365 | Fresh 100K Human300 training; Target50 inference/evaluation; optional [Agentic API](AGENTIC.md) |

Historical runs and model cards exist for these benchmarks. The newly imported
branch adds stage-1 video training/validation and the selected GPT experiment;
see [MERGE.md](MERGE.md). Release weights, resolved configs/statistics, dataset
assets and final checkpoint-specific protocols remain external inputs. Complete
raw-result packages are not bundled, so this repository does not assert that
the migrated entry points have already reproduced the paper's numbers.

## Pinned simulator sources

LIBERO, RoboTwin and Domino now use the shared control runtime and need the
`.[infra]` deployment extra in their inference environment. Keep this separate
from `.[train]` because their dataset-library pins differ. See
[infra/README.md](../infra/README.md) for async execution, device profiles and
the exact observation/action interface. RoboCasa GR1/365 remain separate socket
adapters; installing infra does not enable edge acceleration for them.

Install upstream dependencies and assets under their own licenses:

| Simulator | Source | Audited commit |
| --- | --- | --- |
| LIBERO | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| RoboTwin 2.0 | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| Domino | [H-EmbodVis/DOMINO](https://github.com/H-EmbodVis/DOMINO) | `93d485a34a6e710bb4c895d83e2bb0520a9f169e` |
| RoboCasa GR1 | [robocasa/robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) | `4840e671596f93ca03651524b9f72ffb1aadfeff` |
| RoboCasa365 | [robocasa/robocasa](https://github.com/robocasa/robocasa) | `b4684e6ee37d377cc392e98302a6b916d588b415` |

GR1 used Isaac-GR00T `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` and robosuite
`51cc01785bab80ffeed20da15e67d7dd4140e76a`. RoboCasa365 used robosuite
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. Install their conflicting `robocasa`
packages in separate simulator environments. `simulator_python` chooses the
worker environment; it needs NumPy, Hydra/OmegaConf and this checkout's `src/`
in addition to the simulator (SciPy/Pillow/ImageIO for the GPT experiment).

For RoboTwin, Domino and RoboCasa365 apply the preserved protocol patches to
your own writable checkout:

```bash
python scripts/prepare_simulator.py robotwin2 /path/to/RoboTwin --check-only
python scripts/prepare_simulator.py robotwin2 /path/to/RoboTwin
python scripts/prepare_simulator.py domino /path/to/DOMINO
python scripts/prepare_simulator.py robocasa365 /path/to/RoboCasa
```

The helper checks the commit and patch applicability; it never resets a checkout
or downloads assets. Domino's optional `--arm` patch is for the audited ARM
runtime only. Renderer/OIDN/physics compatibility still requires validation on
the target machine. Native RoboTwin/Domino evaluation creates a checked
`policy/longwam_policy` symlink; conflicting paths are rejected. These native
adapters need policy and simulator dependencies in the same worker environment.

## Remaining external inputs

- Author-selected Hugging Face checkpoint bundles, including both denoising
  modes and all GR1 contexts, exact resolved configs and statistics/prompt cache.
- Stage-1 media, manifests, matching integrity receipt, initializer and fixed
  demo inputs. The code is now present; these assets are not.
- Domino raw HDF5-to-LeRobot conversion uses the converter in the pinned upstream
  PUMA tree plus the included compatibility patch. Prepared LeRobot data is
  supported; the former cluster-specific bulk conversion orchestration is not
  a portable public launcher. See [DATA.md](DATA.md).

LIBERO-Plus, RoboDojo, prompt sweeps and historical imagination ablations are
intentionally excluded. They are not missing prerequisites for the five supported
benchmarks.
