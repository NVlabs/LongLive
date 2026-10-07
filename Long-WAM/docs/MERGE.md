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

# Selective branch integration

## Official multi-project layout

Long-WAM is an independent project at `NVlabs/LongLive/Long-WAM/`, beside
`LongLive1.0/`, `LongLive2.0/` and `LongLive-Plug/`. The initial snapshot preserves
all 1,458 tracked files from the author release at
`a1356d20ffcc629f8f9d3a1b526238f4ee7d8f13`; local outputs, credentials and Git
metadata are not imported. Launchers resolve the Long-WAM project root, not the
enclosing Git root. Training recipes, inference code, model/state-dict names,
public checkpoint revisions and bundled Stage 1 source pins are unchanged.
The existing three projects are untouched; shared changes are limited to the
repository landing page and a comment/header in the root ignore file.
The documented partial clone (`--filter=blob:none --sparse`) checks out only
`Long-WAM/` and shared root files, without fetching the neighboring projects'
source or assets. The bundled Stage 1 core and infra sources stay self-contained.

## Standalone robot-video inference

The author-provided `longlive2.0-robot` demo is integrated as `stage1/infer.py`
and `stage1/infer.sh`, reusing the bundled LongLive 2.0 pipeline. Seven complete
first-frame/prompt examples are included; five image files referenced by the
source's 12-item manifest were absent and are not advertised as runnable.
Original sampling settings, image hashes, sample slots and seeds are preserved.
Images retain RoVid-X attribution and CC-BY-4.0 licensing. The S/M/L training
and validation recipes are unchanged apart from public `LONGWAM_VIDEO_*` path
names; the neutral dataset/index helper names replace internal experiment names.

## Inference infrastructure integration

G1 follow-up: deployment contracts were inspected at
`kaiknower/Long-WAM-G1-Dynamic-Task-Deploy@a0eda8d2f269635dfef87f7ce3f34cf94a85b777`.
Only the 16D dual-arm / 8D right-arm observation/action contracts and the optional
external bridge binding are integrated into the existing runtime. No training,
collection, task-specific weights/prompts, source runtime, server, smoothing
scheduler or source hardware-acceleration implementation is imported. YAM,
Franka and G1 share a finite deployment runner with read-only defaults. The
three YAML templates leave checkpoint/config/statistics unset; operators provide
their own trained policy and calibrated driver. The G1 bridge/SDK remains an
optional operator-installed dependency, not part of the model/runtime package.

The `infra` branch at `fbe5e556b7897c16aac1ce8cb4706764e00c74e1`
has a separate Git root from `final_version`. Its runtime, model optimizations,
robot interfaces, native-source bundles and tests were integrated selectively;
the final merge records both histories.

- `longwam.runtime` is now a package. Existing Hydra factory names remain
  importable; the single control-policy entry point is `create_policy`.
- Shared reference/async execution, streaming VAE, LeRobot feature contracts,
  YAM/Franka converters and the RTX 5090/Spark/Thor profiles are included.
- Existing training hyperparameters, public checkpoints, GR1 contexts,
  IDM/COD variants, Agentic API and release documentation are retained.
  Deployment defaults remain synchronous/reference, with the release's
  `LONGWAM_OUTPUT_ROOT` convention.
- LeRobot is isolated in `.[infra]`, not the base or training dependencies:
  its datasets-4.x requirement conflicts with the training datasets-3.6 pin.
- Contributor files, device patches and vendored source retain per-file
  provenance and NVIDIA/upstream license notices. Build copies and verification
  artifacts stay outside the checkout.

See [infra/README.md](../infra/README.md) for deployment commands and
[VERIFICATION.md](VERIFICATION.md) for measured validation scope; source
integration does not constitute hardware or robot success-rate validation.

## Earlier data, video and Agentic import

Imported author branch:
`Robocasa365+GPT6+Video_训练`, commit
`87b87ceb65582c46744792a77f2f61754f8274f0`.
This branch is an independent export (no merge base with the original working
tree); its files were selectively integrated, not merged over local experiments.

- RoboCasa365: Human300 data contract, official per-task episode selection,
  left-main camera layout and fresh-100K training recipe. Sample identity and
  proprioception padding alignment fixes are included in the shared data path.
- Model: streaming IDM and joint CodeDenoise implementations, keeping tensor
  names unchanged and preserving the portable Robot-S video-extraction loader.
- Video: LongLive 2.0 S/M/L trainer/configs, sequence-parallel validation,
  manifest-index builder and video-artifact verifier, in a separate `stage1/`.
- GPT: selected hybrid_decompose v1 host, prompts and bounded-action tools,
  connected to the shared RoboCasa policy server. Prompt-sweep variants excluded.

Release scope follows the author's clarification: fresh 100K only for RoboCasa365;
all six GR1 context lengths; IDM/CodeDenoise for LIBERO/RoboTwin; selected Domino
balanced replay. No LIBERO-Plus, historical imagination-ablation recipes,
continuation job chains, allocation controllers or test-run artifacts.

Still external: stage-1 dataset media/manifests/receipt and full validation-panel
assets beyond the seven standalone examples above. Published checkpoint links
are maintained in `configs/checkpoints.yaml`. Imported code availability does not imply all data assets are included
or that the reorganized checkout has rerun the paper's GPU experiments.
