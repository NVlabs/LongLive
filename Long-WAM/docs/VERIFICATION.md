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

# Verification scope

## Official repository layout — 2026-10-07

After moving into `NVlabs/LongLive/Long-WAM/`, **87 CPU regression tests** and
**41 CPU-only CLI checks** passed from the enclosing repository root. These
cover all five benchmark train/eval launchers, IDM/COD selection, six GR1
contexts, Agentic planning, three robot deployment plans, three native patch
checks, S/M/L video launch plans and standalone video inputs. Attribution and
credential guards also pass within a multi-project Git checkout; tests ensure
they do not inspect neighboring projects.

The 1,458-file source snapshot is complete. Only documentation and two regression
test files differ from that snapshot; model, training and inference code/configs
are byte-identical. The existing three projects have no changes. Python/shell
syntax, static checks, 116 local documentation links and 35 README shell blocks
passed. All 13 listed ELM model repositories were confirmed public and ungated
through unauthenticated metadata requests; no weights were downloaded.

No GPU jobs, simulator rollouts, physical robot commands or paid API calls were
run. Checks used a disposable Python 3.12 environment with the existing PyTorch
2.5.1 CPU runtime, not a fresh installation of the declared training/deployment
pins. Temporary check environments and outputs are removed after verification.

## Video inference and public-release safeguards — 2026-10-07

**146 focused CPU tests passed, 2 CUDA tests skipped; 59 bundled video tests
passed** (plus 5 subtests). Coverage includes all seven image hashes, original
seeds/sampling settings, custom-image validation, strict online-weight loading,
real CPU tensor preprocessing, cache cleanup, MP4 encoding/no-overwrite,
manifest-index creation/reuse, S/M/L launch resolution, Agentic credential
redaction without changing RPC payloads, and owner-only policy sockets.
The video tests use synthetic models/tensors, not a trained generator.

Six relocated train/validation shell plans, the standalone inference dry-run,
index-builder help, shell syntax, Python syntax, static checks, attribution and
74 local documentation links passed. Comparison against the preceding commit
confirms that the six training/validation recipes differ only in environment
variable names. The M/L evaluation source pins were refreshed after verifying
that their only difference from the original pinned files was attribution headers.

The release credential guard found no matching keys, private-key blocks,
credential files or private cluster paths in the checkout or the 2,765 unique
blobs reachable from the preceding HEAD. It never prints matched values or reads
the operator's credential stores. This is a pattern-based release audit, not a
complete security certification or dependency vulnerability audit.

```bash
python scripts/check_public_release.py --history
python scripts/check_license_headers.py
```

Agentic authentication follows the user's local Codex login. Model shell tools
receive a minimal environment; audit copies redact recognized credential fields
and token formats. No real API call, GPU allocation, trained-checkpoint inference,
simulator episode or physical robot command was executed. Live Codex/sandbox
compatibility and end-to-end video generation remain target-environment checks.
Tests used a disposable Python 3.12 / PyTorch 2.5.1 / NumPy 1.26.4 environment,
not a fresh installation of the declared Stage 1 or Stage 2 pins. Temporary test
environments and outputs are removed rather than added to the release.

## Earlier verification

The CPU suite checks configuration composition, global
batch, GR1's six temporal contracts, both denoising modes, action/camera/padding
contracts, checkpoint selection/retention, mocked policy rollouts and stage-1
launch planning. No training, simulator rollouts or paid GPT calls are started
by these checks.

Audit on 2026-09-29: all **129 main-suite tests** passed, including the added
variant/import/schedule regressions. All **59 bundled video-module tests** also
passed. A relocated temporary checkout
passed 27 train/eval planning invocations and every shell syntax check; planning
did not create output directories. Static undefined-name checks passed.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src pytest -q
ruff check src scripts tools tests
# Stage-1 tests use the upstream module root, in a separate process:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=stage1/third_party/LongLive \
  pytest -q stage1/third_party/LongLive/tests
bash scripts/robocasa365/train.sh --dry-run
bash scripts/robocasa_gr1/train.sh context=p768 --dry-run
bash scripts/robotwin2/eval.sh --dry-run inference_mode=codenoise
bash scripts/robocasa365/eval.sh --agentic --dry-run
```

The audit environment is Python 3.12.13, PyTorch 2.10.0+cu128,
Accelerate 1.13 and NumPy 2.4.5. It is **not** a fresh installation of the package's
proposed stage-2 pins. Stage 1 separately records its Python 3.10 / PyTorch 2.8
environment. See the dated GPU checks below for the narrower set of runtime
paths actually exercised. A clean dependency installation, full benchmark
reproduction and GPT host/account compatibility remain unverified. Historical
scores are not new measurements of this checkout.

## G1 inference-only deployment examples — 2026-10-07

Added a Unitree G1/Dex1 adapter (16D dual-arm or native 8D right-arm), an optional
binding to the operator-installed `ial_g1d` bridge, and one shared deployment
runner for YAM, Franka and G1. All deployment checkpoint/config/statistics paths
are blank. The source's training code, checkpoints, old model server and separate
acceleration/scheduling implementation are not imported. Existing model kernels,
benchmark training recipes and acceleration profiles are unchanged.

Validation: **260 main-suite tests passed, 10 skipped**. After the final processor
camera-order guard and partial-startup cleanup regressions, **27 focused tests
passed**. These runs overlap; they are not 287 distinct tests. Coverage includes
the real LeRobot policy class with fake model execution, native 8D/16D ordering,
standard/legacy observation conversion, real processor normalization/history,
right-wrist mosaic placement and black missing-wrist padding, read-only default,
stale-action rejection, calibrated target checks and stop/disconnect on failure.
Eight CLI dry-runs, documentation links/shell examples, AST/static checks and
attribution checks also passed (1,375 covered files; no missing headers).

Tests used a disposable system-site-packages environment with LeRobot 0.4.1 and
the inherited Python 3.12 / PyTorch 2.10.0+cu128 / NumPy 2.4.5 stack. This is not
a clean installation of the declared deployment pins. No physical robot,
network bridge, GPU allocation, actual model checkpoint or simulator was used;
all robot lifecycle/command tests use mocks. Hardware safety, SDK/DDS/bridge
compatibility, target-device compilation/performance and new task success rates
still require on-site validation. An operator-provided driver is required for
YAM/Franka; the optional G1 bridge package is external and not auto-installed.

See [robot deployment demos](../infra/README.md#real-robot-deployment-demos).
Temporary source inspection and test environments are removed after verification;
no training artifacts, weights or diagnostic reports are added to the release.

## Infra integration — 2026-10-07

Integrated the contributor's `infra` snapshot
`fbe5e556b7897c16aac1ce8cb4706764e00c74e1` into `final_version`, preserving
the release's training recipes, GR1 context variants, public checkpoint registry
and benchmark-neutral Agentic API. See [infra/README.md](../infra/README.md).

- **244 tests passed, 10 skipped** in the complete main CPU suite. The earlier
  26-test runtime/LeRobot/native-build subset also passed; these runs overlap.
  Coverage includes real LeRobot base classes with mocked policies/drivers,
  spawned async scheduling, causal streaming-VAE equivalence, model contracts
  and existing Agentic regressions. CUDA-only tests remain skipped. Without
  LeRobot installed, collection also succeeds (232 tests); optional LeRobot
  fixtures import that dependency only when used.
- **31 CLI checks passed:** all three device patch checks, benchmark train/eval
  dry-runs, IDM/COD and async settings, all six GR1 contexts, and new/legacy
  Agentic entry points. GR1 training uses `context=pN`; evaluation checks the
  checkpoint with `expected_history=N`, not a training-context override.
- Shell syntax, Python AST parsing, Ruff undefined-name checks and **91 relative
  documentation links** passed. Attribution covers **1,368 files**, with no
  missing headers/sidecars. This checks coverage, not legal approval.
- All three device patches apply to isolated copies of the attributed source.
  Their functional additions/removals match the imported patches; only hunk
  locations and EOF annotations changed for the headers. Source/build cache
  isolation and local-edit preservation passed without compiling an extension.

LIBERO rendering imports are now deferred until an environment is created, so
adapter diagnostics do not require EGL. The earlier native-config propagation
and NumPy initial-state loading issues are covered by CPU regressions; a fresh
end-to-end LIBERO simulator run is still required to close the historical audit.

The test environment used Python 3.12, inherited PyTorch **2.10.0+cu128** and
NumPy **2.4.5**, plus LeRobot **0.4.1** and small import dependencies in a disposable
virtual environment. It was **not** a clean installation of the declared pins
or a fully dependency-resolved LeRobot deployment. The deployment extra is
separate from training because their `datasets` requirements differ.

No GPU allocation, target-device CUDA compilation, real checkpoint inference,
simulator episode, paid model call or physical robot command was executed in
this integration. RTX 5090/Spark/Thor ABI compatibility, numerical equivalence,
latency and real-robot behavior remain target-hardware validation tasks. The
infra guide's latency table is attributed to the contributor snapshot, not this
audit. Verification artifacts stay outside the repository and are removed after
checks; no model weights, build binaries or temporary reports are committed.

## Agentic API refactor — 2026-10-06

The model/tool runtime now lives in `longwam.agentic`; RoboCasa365 prompts,
robot contracts and rollout tools live in `longwam.benchmarks.robocasa.agentic`.
Models/efforts are explicit per-session parameters, defaulting to
`gpt-6-astra` / `xhigh`. Legacy CLI/config/import paths remain compatibility
entry points. The missing dynamic-tool event handling in the earlier cleaned
host was restored against the imported source snapshot and covered by offline tests.

Validation: **172 main-suite tests passed**, followed by **42 focused tests**
after the final relative-path and model-neutral prompt-wording adjustments.
These are overlapping test runs, not 214 distinct tests. Checks cover a synthetic
non-RoboCasa adapter, the RoboCasa tool-binding cycle, exact model forwarding,
consent/budget guards, no model fallback, timeout/error handling, same-thread
recovery and legacy entry points. New/legacy shell dry-runs, both module help
entries, README/guide links, attribution headers and static checks also passed.
Prompt comparison confirms that only model-identity wording changed; action/gate
and native-termination instructions are unchanged.

No real Codex model calls, GPU jobs or simulator rollouts were launched in this
pass. Live account/model access and end-to-end simulator results remain unverified.
The test output used a disposable temporary directory outside the repository;
no temporary reports or diagnostic scripts were added.

## Attribution audit — 2026-10-06

The release now uses Apache-2.0 for original Long-WAM code and NVIDIA-authored
changes, with the original MIT/Apache/BSD notices retained for incorporated
third-party code. Per-file headers or parser-safe sidecars cover 375 files;
license texts are preserved separately. See [third-party notices](../THIRD_PARTY_NOTICES.md).

Validation: **136 main-suite tests and 59 stage-1 tests passed** on CPU;
all 16 shell scripts passed syntax checks. The 258 pre-existing Python files
have identical ASTs before/after attribution. Source/config bodies, patch and
checksum contents and all four GPT-6 prompt payloads are unchanged; intended
non-header changes are limited to
license documents, release documentation and package license metadata.
The existing runtime-audit content below was preserved. This pass did not
rerun training, simulator evaluation or paper results, and does not resolve
the runtime issues recorded below. The header checker verifies structure and
coverage, not copyright ownership or legal approval.

```bash
python scripts/check_license_headers.py
```

## Runtime audit — 2026-10-01

Tested release: `31fb322658e827aec3ca1b6fdab80d37d0371813`. The main
129-test suite and the 59 bundled stage-1 tests were rerun successfully. These
tests did not catch every real simulator integration issue; see the failures
below. No training/inference implementation was changed during this audit.

Hardware: ARM64 GB200, four GPUs per allocation (the cluster's minimum).
Training used four DDP ranks; independent simulator checks used one GPU each.
Runs had finite timeouts, disabled W&B and disabled weight/optimizer-state
saves. Existing checkpoints were read-only inputs. This is **smoke testing,
not a rerun of paper results or all published training recipes**.

| Path | Actual execution | Result |
| --- | --- | --- |
| GR1 P96 training | Robot-S video initialization; good Teleop dataset; global batch 4; two optimizer steps on four GPUs | Passed; total losses 2.0447 / 1.9051, action losses 1.6680 / 1.5124; no NaN/OOM observed |
| GR1 P0/P48/P96/P192 evaluation | Matching local 30K checkpoints; one CupToDrawer episode per context, capped at 32 environment steps; two policy calls each | Passed the short closed-loop path; all selected inventories completed, with no task successes within this short cap |
| RoboCasa365 evaluation | Historical effective-100K checkpoint (payload/local filename step 60000); one OpenDrawer episode | Passed; six policy calls, 161 environment steps, task success |
| Domino fine-tuned evaluation | Robot-S-initialized local step-300 checkpoint; one adjust_bottle dynamic episode, seed 4200000 | Completed 400 steps and wrote validated SR/MS files; 0/1 successes, MS 20.204; not an aggregate benchmark result |
| LIBERO evaluation | Local step-86790 checkpoint loaded; official pinned simulator checkout | Blocked by configuration propagation bug and reset-state compatibility issue described below |
| RoboTwin 2.0 evaluation | Public script and patched pinned simulator checkout; local Robot-S bundle | Blocked before model loading: simulator environment lacks Open3D |

The smoke-training overrides deliberately reduce the training horizon and batch;
they are not replacements for the public recipe. Both `max_steps` and
`scheduler_max_steps` were set to 2. No new checkpoint was saved, so the simulator
checks use the pre-existing trained weights, not the two-step smoke model.

### Confirmed issues and environment requirements

1. **Native evaluation config propagation:** `native_config()` assigns an
   OmegaConf node to `run.EVALUATION` before adding benchmark-specific fields to
   the original node. OmegaConf copies the node, so the later fields are absent
   from the returned config. LIBERO fails with missing `EVALUATION.num_trials`.
   The missing RoboTwin/Domino-specific fields are also reproducible on CPU;
   some are passed separately to the native launcher. This needs a code fix
   and regression assertions covering the returned benchmark-specific fields.
2. **LIBERO reset states / PyTorch:** the pinned upstream simulator loads NumPy
   reset-state files with `torch.load()` without an explicit loading policy.
   This fails under the audit's weights-only default. The old research checkout
   had a local compatibility change not present upstream. A temporary, scoped
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` diagnostic on trusted local assets got
   past this failure and exposed issue 1; it is not a default-path pass or a
   recommended permanent global setting.
3. **RoboTwin native dependencies:** its pinned simulator imports `open3d`
   unconditionally. An isolated ARM64 Open3D installation was inspected, but
   also needs its plotting/UI dependencies; it did not establish a working
   simulator environment. No existing environment was modified.
4. **Training environment:** Accelerate imports the installed DeepSpeed while
   unwrapping DDP. The inherited environment therefore needed `CUDA_HOME` set
   to its CUDA toolkit. After that and the smoke scheduler correction, training
   completed normally. Neither initial failed attempt is counted as a pass.
5. **GR1 simulator worker:** the existing separate Python 3.10 simulator lacked
   Hydra/OmegaConf. Isolated worker dependencies supplied Hydra 1.3.2,
   OmegaConf 2.3.0 and antlr4-python3-runtime 4.9.3. Gym observation-space and
   unused-controller-component warnings remained; full protocol validation
   should review those warnings before claiming numerical reproduction.
6. **Domino renderer:** the isolated run used the existing ARM compatibility
   patches and OIDN 2.4.1 bundle. Vulkan device-probing and Warp driver-entry-point
   warnings occurred, but the 400-step rollout and result aggregation completed.
   This is not a newly certified rendering/physics environment or a fair
   comparison with historical aggregate scores.

### Evidence and remaining scope

Local logs, exact resolved configs, result JSON and per-GPU telemetry are under
`outputs/validation/20261001/` (ignored; not shipped as release scripts).
Training and GR1 P96/P192/RoboCasa365 checks ran in job `7580995`; GR1 P0/P48,
LIBERO and RoboTwin checks ran in job `7581079`. Domino also ran in `7580995`.
That job completed with exit code 0; `7581079` exited 1 because LIBERO/RoboTwin
failed, despite both GR1 checks completing. Earlier failed training
preflights were `7580267` (smoke scheduler mismatch) and `7580451` (CUDA_HOME).
All four owned GPU allocations have exited. Temporary cloned simulators and
unused Open3D diagnostic files were removed; logs/configs/results were retained.

Not established by this audit: all five benchmarks' training updates, full
task/seed inventories, GR1 P384/P768 GPU inference, CodeDenoise checkpoint
inference, stage-1 GPU training/demo evaluation, or paid GPT-6 calls. The located
policy bundles used here are IDM bundles; changing the mode of an IDM checkpoint
would not constitute a valid CodeDenoise test. Public dependency pins were not
fresh-installed, and a 32-step GR1 cap does not test a fully populated long
history or provide a meaningful success-rate estimate.
