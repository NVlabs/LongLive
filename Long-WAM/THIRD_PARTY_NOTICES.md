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

# License and third-party notices

## Long-WAM license

Original Long-WAM code and NVIDIA-authored modifications use **Apache-2.0**;
see [LICENSE](LICENSE) and [NOTICE](NOTICE). NVIDIA copyright statements cover
our contributions, not underlying third-party work. Existing third-party
copyright and license notices are preserved.

Long-WAM is independently organized and does not require the FastWAM Python
package, checkout or launcher. Parts of the implementation derive from FastWAM;
independence does not mean unrelated code ancestry or third-party endorsement.

Each source/configuration/script/documentation file has an attribution header.
JSON, patches, checksum manifests, plain-text inputs and literal GPT-6 agent
prompts (including skill YAML frontmatter) use an adjacent
`.license` file, so their parser-sensitive contents remain unchanged. License
texts themselves are kept verbatim and do not receive replacement headers.

Headers identify original work or copied/modified sources, upstream paths and
available source revisions. `Apache-2.0 AND MIT` or `Apache-2.0 AND BSD-3-Clause`
means that the file contains portions under those respective licenses, with
their associated obligations; it is **not** a choice between licenses and does
not relicense third-party portions.

## Included components

### G1 deployment contract reference

The inference-only G1 adapter and optional driver binding are original Long-WAM
implementations referencing the interfaces in
[Long-WAM-G1-Dynamic-Task-Deploy](https://github.com/kaiknower/Long-WAM-G1-Dynamic-Task-Deploy)
at `a0eda8d2f269635dfef87f7ce3f34cf94a85b777`: `WORKSTATION_START.md`,
`code/LongWAM/deploy/unitree_g1/contract.py`, and the `ial_g1d` model/robot interfaces.
The external `ial_g1d` package declares Apache-2.0 and carries its own Unitree
SDK/driver notices. It is not vendored, downloaded automatically, or relicensed
here. No source checkpoints, private host addresses, training pipelines, old
model server, native kernel fork or hardware SDK code are copied into this release.

### Bundled components

| Component / source | Location or incorporated material | Retained license |
| --- | --- | --- |
| [FastWAM](https://github.com/yuantianyuan01/FastWAM) | Model, processor, dataset, training and benchmark helpers identified in their headers | [MIT](licenses/FastWAM-MIT.txt), except individually licensed upstream portions |
| [LeRobot](https://github.com/huggingface/lerobot) | LeRobot dataset helpers and constants | [Apache-2.0](licenses/Apache-2.0.txt); Hugging Face notices retained |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Wan DiT/VAE/text encoder, loading, conversion and gradient helpers incorporated through FastWAM | [Apache-2.0](licenses/Apache-2.0.txt) |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Rotation conversion helpers under `src/longwam/datasets/lerobot/` | [BSD-3-Clause](licenses/PyTorch3D-BSD-3-Clause.txt) |
| [OpenVLA](https://github.com/openvla/openvla) | LIBERO evaluation utilities, via FastWAM | [MIT](licenses/OpenVLA-MIT.txt) |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | Quaternion conversion in LIBERO helpers | [MIT](licenses/robosuite-MIT.txt) |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | RoboTwin integration patches and task identifiers | [MIT](licenses/RoboTwin-MIT.txt) |
| [DOMINO](https://github.com/H-EmbodVis/DOMINO) | DOMINO integration patch and task identifiers | [Apache-2.0](licenses/Apache-2.0.txt) |
| [RoboCasa](https://github.com/robocasa/robocasa) | RoboCasa integration patch | [MIT](licenses/RoboCasa-MIT.txt), with upstream exceptions |
| [LongLive 2.0](https://github.com/NVlabs/LongLive) | `stage1/third_party/LongLive/`; robot additions and wrappers come from our author branch | [Apache-2.0](stage1/third_party/LongLive/LICENSE), except nested components below |
| [RoVid-X](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X) | Seven first-frame PNGs and original instructions in `stage1/examples/robot_s/`; source IDs and image hashes retained | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); RoVid-X contributors, not NVIDIA-created artwork |
| [Wan](https://github.com/Wan-Video/Wan2.2) | Wan model, VAE, text encoder and utilities included through LongLive 2.0 | [Apache-2.0](licenses/Apache-2.0.txt); Alibaba notices retained |
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | LongLive training, model and pipeline components | [Apache-2.0](licenses/Apache-2.0.txt) |
| [Stable Video Infinity](https://github.com/vita-epfl/Stable-Video-Infinity) | LongLive error buffer | [Apache-2.0](licenses/Apache-2.0.txt) |
| [FramePack](https://github.com/lllyasviel/FramePack) | LongLive memory helpers | [Apache-2.0](licenses/Apache-2.0.txt) |
| [Diffusers](https://github.com/huggingface/diffusers) | Wan solver implementations | [Apache-2.0](licenses/Apache-2.0.txt) |
| [Transformers](https://github.com/huggingface/transformers) | Wan T5 implementation | [Apache-2.0](licenses/Apache-2.0.txt) |
| [qwen-vl-utils](https://github.com/kq-chen/qwen-vl-utils) | Wan vision-language utilities | [Apache-2.0](licenses/Apache-2.0.txt) |
| [FourOverSix](https://github.com/mit-han-lab/fouroversix) | Optional stage-1 FP4 implementation | [MIT](stage1/third_party/LongLive/fouroversix/LICENSE.md); Jack Cook and original per-file notices retained |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Adapted FourOverSix CUDA headers and build helpers | [BSD-3-Clause](licenses/FlashAttention-BSD-3-Clause.txt); Tri Dao |
| [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) | FourOverSix Hadamard transform | [BSD-3-Clause](licenses/fast-hadamard-transform-BSD-3-Clause.txt); Tri Dao |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | FourOverSix FP4 GEMM examples and CUDA copy helper | [BSD-3-Clause](licenses/CUTLASS-BSD-3-Clause.txt) |
| [DALI](https://github.com/NVIDIA/DALI) | FourOverSix static-switch helper | [Apache-2.0](licenses/Apache-2.0.txt) |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | FourOverSix quantization utility | [Apache-2.0](licenses/Apache-2.0.txt) |
| [wheel](https://github.com/pypa/wheel) | Archive handling snippet in FourOverSix build setup | [MIT](licenses/wheel-MIT.txt) |
| eval-of-gpt-6-astra-as-policy | Agentic host/helpers and RoboCasa365 action adapter, imported through the author branch | [MIT](licenses/Agentic-MIT.txt); Yu-Mool Shu and Lipxin Zheng |

The legacy PyTorch3D comment referring to a BSD license in its root
`LICENSE` refers to the **upstream** project. In this repository that license
is [licenses/PyTorch3D-BSD-3-Clause.txt](licenses/PyTorch3D-BSD-3-Clause.txt).

## Source snapshots and changes

- The standalone robot-video inference wrapper and reference settings were
  adapted from the author-provided `longlive2.0-robot` demo. Its original
  `inference.py` SHA256 is `21f4a911a3e61d2f72680681a083901569a0c2b6ddc7fb5d6debd24896e81a29`.
  It reuses the bundled LongLive 2.0 pipeline; no second model implementation
  is imported. The seven supplied PNGs are unchanged lossless RGB first frames
  extracted from RoVid-X. Their prompts, source IDs, seeds and hashes remain in
  `stage1/examples/robot_s/examples.json`. The upstream dataset card declares
  CC-BY-4.0; this does not relicense those images under the code's Apache license.

- Accelerated runtime/infra contribution: branch `infra`, commit
  `fbe5e556b7897c16aac1ce8cb4706764e00c74e1`, by Bohan Zhang, in
  [Long-WAM](https://github.com/Aaronhuang-778/Long-WAM/tree/infra).
  Its model/runtime integration is attributed per file; existing FastWAM-derived
  portions retain their MIT notices alongside NVIDIA's Apache-2.0 changes.
- The runtime bundles [FourOverSix](third_party/fouroversix/LICENSE.md)
  (`4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb`, Jack Cook, MIT) and
  [CUTLASS](third_party/fouroversix/third_party/cutlass/LICENSE.txt)
  (`ec8daf642d69fc31352ac6fa6e14a0de9019604b`, NVIDIA, BSD-3-Clause).
  Nested BSD/Apache/MIT snippets keep their component-specific attribution.
  These deployment sources are separate from the Stage-1 snapshot. Added
  headers do not relicense upstream code. Runtime device patches have sidecars;
  hunk locations and end-of-file markers are rebased for the added attribution,
  without changing their kernel or Python additions/removals.
- LeRobot is an external Apache-2.0 dependency of the optional `infra` extra;
  it is not vendored and no robot SDK or hardware credentials are bundled.

- FastWAM comparison snapshot: `7faa71108368fbb3b6885649f112af607427a2d4`.
  Headers record the corresponding upstream file paths. This is a comparison
  baseline for attribution, not a claim that every research modification was
  originally made against that exact revision.
- LongLive 2.0 import baseline: `0308b126accba9440b8caa45bcf7bec0877933e1`.
- DiffSynth-Studio attribution comparison: `974cfa37f27ac55eba3b6d10efa21f876900572d`.
  This identifies a verified nested implementation, not an exact import revision.
- Author integration/export snapshot in
  [Long-WAM](https://github.com/Aaronhuang-778/Long-WAM):
  `87b87ceb65582c46744792a77f2f61754f8274f0`. This export includes working-tree
  changes; a copied file from this snapshot need not be byte-identical to its
  original upstream repository.
- The exported GPT-6 policy source records upstream revision
  `79f8be5905102d6b16000c0f02a9c2195b51bb61`. A public upstream repository
  URL was not established from the import records; the author snapshot,
  source paths and retained MIT notice identify the material we received.

“Copied” means that the file body matched the identified snapshot before this
attribution pass; adding the attribution block itself is not an algorithmic
change. “Modified” covers Long-WAM import relocation, portable configuration,
robot data contracts, checkpoint adaptation, history/context support and/or
training/evaluation integration. Original upstream comments remain below the
new header. Consult version history for the exact changes.

## External resources and release checks

Benchmark simulator trees, third-party model weights, datasets and assets are
obtained separately. Their own terms still apply; this repository's license
does not grant rights to those resources or imply access to hosted services.

Run the lightweight attribution check before adding files to a release:

```bash
python scripts/check_license_headers.py
```

It verifies coverage and header structure, not legal ownership. New vendored
code must also preserve upstream notices and receive a source/license review.
Public release remains subject to the copyright holder's normal approval.
