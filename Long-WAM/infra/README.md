<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/README.md
License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
-->

# Infra: accelerated inference and robot integration

Long-WAM inference with **pure asynchronous execution**, **streaming VAE**, and
**device acceleration** for RTX 5090, DGX Spark and Jetson AGX Thor. Control
applications use a LeRobot policy interface; LIBERO, RoboTwin 2, YAM, Franka and Unitree G1
converters are included. Video compute supports NVFP4; action compute and KV
storage remain BF16.

[Quick Start](#quick-start) · [Build](#build) · [Usage](#usage) ·
[Performance](#performance) · [Developer API](#developer-api) ·
[Robot Demos](#real-robot-deployment-demos) · [Code Structure](#code-structure)

## Quick Start

Use a **separate inference environment** and run commands from the Long-WAM
project root (`LongLive/Long-WAM/` in the official repository).
Do not combine `.[infra]` and `.[train]`: LeRobot 0.4.1 requires `datasets` 4.x,
while the reproducible training extra pins `datasets==3.6.0`. The deployment extra
is optional so existing training and RoboCasa environments do not acquire that
conflict. LIBERO, RoboTwin and Domino's shared control path need `.[infra]`;
RoboCasa GR1/365 keep their benchmark-specific socket protocols.

Install the deployment environment:

```bash
pip install -e '.[infra]'
```

Install your simulator using the [benchmark setup guide](../docs/BENCHMARKS.md).
Prepare a matching checkpoint, resolved training configuration and normalization
statistics, plus the VAE/tokenizer/text assets listed under
[Checkpoints](../README.md#checkpoints). Checkpoint history, embodiment and
denoising mode determine which inference settings are valid.

Start with one LIBERO task and the reference backend:

```bash
longwam eval libero \
  checkpoint=/path/model.pt model_config=/path/resolved-config.yaml \
  stats=/path/dataset_stats.json suite=libero_spatial task_id=0 episodes=1 \
  action_horizon=32 execute_steps=24 \
  output_dir=/tmp/longwam/libero-sync
```

**H** (`action_horizon`) is the number of actions predicted in a chunk.
**R** (`execute_steps`) is the handoff/replanning boundary measured from that
chunk's observation anchor. **S** (`trigger_stride`) is the next inference trigger
measured from the same anchor. All three count control steps. Sync requires
R ≤ H; async requires R/2 ≤ S < R ≤ H.

Enable pure async independently of device acceleration:

```bash
longwam eval libero \
  checkpoint=/path/model.pt model_config=/path/resolved-config.yaml \
  stats=/path/dataset_stats.json suite=libero_spatial task_id=0 episodes=1 \
  execution=async action_horizon=32 execute_steps=24 trigger_stride=16 \
  streaming_vae=true hardware=reference \
  output_dir=/tmp/longwam/libero-async
```

With H=32, R=24, S=16, prediction A starts at step 0, B is requested at step 16,
and control switches to B[8] at step 24. The elapsed prefix of B is dropped. A
late prediction blocks at handoff; actions are not blended. Streaming VAE encodes
observed history incrementally in the same inference worker. Set
`streaming_vae=false` to use whole-window encoding.

For RTX 5090 acceleration, build once on the target device, then add
`hardware=rtx5090` to the evaluation command:

```bash
pip install -e '.[infra,native]'
longwam build rtx5090
```

## Build

Use Linux, compatible CUDA-enabled PyTorch, an NVIDIA driver, CUDA toolkit
(`nvcc`), Git and C++ build tools. Package dependencies are declared in
[`pyproject.toml`](../pyproject.toml); the supported PyTorch series is 2.7.x.
On Spark/Thor, use the platform-compatible PyTorch installation, install the
remaining dependencies (including LeRobot 0.4.1, Ninja, packaging, psutil and setuptools ≥77.0.3)
and run `pip install -e . --no-deps`.

| Device | Build command | GPU compute capability |
| --- | --- | --- |
| RTX 5090 | `longwam build rtx5090` | 12.0 |
| DGX Spark | `longwam build spark` | 12.1 |
| Jetson AGX Thor | `longwam build thor` | 11.0 |

FourOverSix and CUTLASS headers are bundled in
[`third_party/fouroversix`](../third_party/fouroversix). The builder copies
source to an external cache, applies the selected device patch and installs the
extension into the active Python environment. No external clone is needed.

```bash
# Patch compatibility check; no compilation or installation.
longwam build rtx5090 --check-only

# Choose an external build-cache root or an explicit build directory.
LONGWAM_BUILD_ROOT=/path/cache longwam build rtx5090
longwam build rtx5090 --build-dir /path/builds/rtx5090

python -c 'import fouroversix._C'
```

The default cache is `~/.cache/longwam/<hardware>/fouroversix-<key>`. Source,
patch and toolchain changes select a new cache. Locally modified copies are
preserved; choose a fresh explicit build directory when its identity differs.
The bundled manifest records the attributed source checksum as well as the
original import checksum. A source checksum difference produces a warning and a
patch compatibility check.
The CUDA/PyTorch toolchain must be ABI-compatible. Compilation and warmup may occur on the
first inference call. The reference backend needs no FourOverSix build.

## Usage

### Benchmark evaluation

LIBERO suite/task selection uses `suite` and `task_id`; RoboTwin uses `task` and
`setting`. Leave their defaults to evaluate the full configured inventory.

```bash
longwam eval robotwin2 \
  checkpoint=/path/model.pt model_config=/path/resolved-config.yaml \
  stats=/path/dataset_stats.json simulator_root=/path/RoboTwin \
  task=adjust_bottle setting=demo_clean episodes=1 \
  execution=async action_horizon=32 execute_steps=24 trigger_stride=16 \
  streaming_vae=true hardware=reference \
  output_dir=/tmp/longwam/robotwin-async

# Validate settings without opening weights or simulator assets.
longwam eval libero --dry-run execution=async \
  action_horizon=32 execute_steps=24 trigger_stride=16

# A complete custom evaluation YAML can replace the benchmark defaults.
longwam eval libero --config /path/eval.yaml hardware=rtx5090 \
  optimization.action_segmented_attention=false
```

Evaluation owns task selection, seeds, scene admission, resets and scoring.
LIBERO shares one loaded policy across its task inventory; RoboTwin uses the
upstream task evaluator and loads one policy per task/setting subprocess.
Results and resolved configurations are written to `output_dir`; use a fresh
path for each run.

### Important arguments

Settings use YAML plus `key=value` overrides. Defaults are in
[`libero.yaml`](../configs/eval/libero.yaml) and
[`robotwin2.yaml`](../configs/eval/robotwin2.yaml).

| Argument | Default | Meaning |
| --- | --- | --- |
| `checkpoint`, `model_config`, `stats` | Required | Matching weights, training configuration and normalization statistics |
| `execution` | `sync` | Synchronous or pure asynchronous control |
| `action_horizon` | LIBERO: 80; RoboTwin: checkpoint | H; device profiles require 32 |
| `execute_steps` | `replan_steps`: LIBERO 10, RoboTwin 24 | R; must be ≤ H |
| `trigger_stride` | Async: ceil(R/2) | S; R/2 ≤ S < R; omit for sync |
| `streaming_vae` | `false` | Incremental history encoding; requires video-action history and non-tiled VAE |
| `hardware` | `reference` | `reference`, `rtx5090`, `spark`, `thor` |
| `optimization.*` | Device profile | Override individual device optimizations; reference rejects overrides |
| `num_inference_steps` | LIBERO: 20; RoboTwin: 10 | Action denoising steps; video schedule comes from the checkpoint |
| `inference_mode` | `idm` | Benchmark checkpoint regime: `idm` or `codenoise` |
| `episodes`, `seed` | Benchmark configuration | Evaluation protocol |
| `gpu_id` | 0 | CUDA visibility for CLI evaluation |
| `output_dir` | Under `LONGWAM_OUTPUT_ROOT` (default `./outputs`) | Results directory |
| `--config`, `--dry-run` | Unset / off | Custom YAML or validation only |
| `--check-only`, `--build-dir` | Build only | Patch check or explicit external build path |

Co-denoising uses its joint solver and disables sequential IDM cache
transformations. Thor's FlexAttention profile requires the LIBERO grid [6,7,14];
other grids need a validated profile. Runtime hardware profiles do not change
checkpoint history or embodiment.

### Offline generation and socket serving

Offline image-to-video generation uses `longwam.runtime.run_inference(cfg)` with
a model configuration and an `inference` block specifying checkpoint/image/output
paths, dimensions, prompt, frame count, sampling steps and device. It is
synchronous and does not maintain a control history. A full checkpoint can use
`checkpoint_strict=true` to skip loading pretrained DiT weights before restoration;
`inference.action_horizon` defaults to the training action chunk. Optional
`inference.proprio` must already match model-space state dimensions and units.

`longwam infer` starts the RoboCasa GR1/365 socket service:

```bash
longwam infer robocasa_gr1 --config /path/gr1-eval.yaml \
  socket=/tmp/longwam-policy.sock
```

Training starts with `longwam train <benchmark>`; see the
[training guide](../docs/TRAINING.md).

## Performance

The following measurements were reported in the imported `infra` branch
(`fbe5e556b7897c16aac1ce8cb4706764e00c74e1`); **they were not remeasured during
this integration**. CPU contract tests and patch checks do not establish hardware
latency or robot task success.

Reported Long-WAM IDM V4/A4 latency covers the complete observation VAE and
video/action inference. Video compute uses NVFP4; action/KV remain BF16. Each
entry is the mean of two independent-process medians, with five excluded warmups
and 30 steady-state samples per process. Setup, preprocessing, text encoding,
controller work and IPC are outside the measured interval.

| Hardware | BF16 eager | Shared optimizations | + Device tuning | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 356.0 ms | 126.9 ms | **107.4 ms** | **3.3×** |
| DGX Spark | 1342.8 ms | 419.9 ms | **328.2 ms** | **4.1×** |
| Jetson AGX Thor | 1215.2 ms | 468.2 ms | **378.7 ms** | **3.2×** |

The optimization columns are cumulative. Shared optimizations include compiled
execution/CUDA Graphs, KV/RoPE reuse, shared quantization and segmented attention.
Streaming VAE moves encoding ahead of the online trigger; this table includes
full VAE work and does not measure that overlap.

## Developer API

`longwam.runtime.create_policy(settings, camera_keys=...)` returns a LeRobot
`PreTrainedPolicy`. `reset()` starts an episode; `select_action(batch)` returns
one unnormalized action tensor of shape [1,D] per control step. The checkpoint
processor owns normalization, image preprocessing and history inside the worker.

| Input feature | Contract |
| --- | --- |
| `observation.state` | [1,D], unnormalized state in checkpoint order and units |
| `observation.images.<camera>` | [1,3,H,W], RGB uint8 or float in [0,1] |
| `task` | One instruction string or a one-element list |

Assemble CPU batches; GPU preprocessing and model execution run in the worker.
Provide the current observation at **every** control step. Call `reset()` before
changing tasks. Both sync and async use a spawned inference process: guard
application startup with `if __name__ == "__main__":` and close the policy with a
context manager. One policy controls one environment (`batch_size=1`). A policy
loads its model once and caches text conditioning per task.

Use the policy directly rather than an outer action queue, normalizer or RTC
scheduler. `predict_action_chunk` is unsupported because runtime owns scheduling.
The policy uses LeRobot base classes. Hub weight export and automatic
`lerobot-rollout` plugin discovery are not provided. Keep the checkpoint/config/
stats bundle. `policy.metrics` exposes request counts, late handoffs and controller
wait time, which are separate from model inference latency.

### Connect a new benchmark or robot

A new platform that supplies the standard batch above does **not** need a runtime
adapter. Omit `benchmark` and `robot`; runtime uses the checkpoint processor.
Map each checkpoint camera to its incoming policy feature name:

```python
import numpy as np
import torch
from longwam.runtime import create_policy


def control_step(policy, observation, task, state_names, camera_names):
    # LeRobot Robot.get_observation(): named scalars and RGB HWC arrays.
    batch = {
        "observation.state": torch.tensor(
            [[observation[name] for name in state_names]], dtype=torch.float32),
        "task": [task],
    }
    for camera in camera_names:
        image = np.ascontiguousarray(observation[camera])
        batch[f"observation.images.{camera}"] = (
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))
    return policy.select_action(batch)[0].tolist()


```

This example illustrates batch packing only; it does not send hardware commands.
Use `create_policy(settings, camera_keys=...)` from a guarded application entry
point with your own checkpoint settings. For connection, timing and hardware
commands, use the shared [deployment runner](#real-robot-deployment-demos) and
an explicitly calibrated driver instead of adding a second action scheduler.

```python
settings = {
    "checkpoint": "/path/model.pt", "model_config": "/path/config.yaml",
    "stats": "/path/dataset_stats.json", "execution": "async",
    "action_horizon": 32, "execute_steps": 24, "trigger_stride": 16,
    "streaming_vae": True,
}
camera_keys = {"image": "front", "wrist_image": "wrist"}
```

`camera_keys` maps **checkpoint camera → policy feature/driver camera**. All
checkpoint cameras must be covered once. The checkpoint must expose one merged
state field and one merged action field and a supported camera layout
(horizontal, vertical or RoboTwin mosaic). Its processor must support the
platform's transforms. If a new modality needs a new processor, implement it in
`datasets/`, where training and inference can share it.

For a benchmark, build the same batch from `env` observations and pass the
returned action to `env.step`; the evaluator owns episodes, seeds, termination
and scores. The included [LIBERO](../src/longwam/benchmarks/libero/adapter.py) and
[RoboTwin](../src/longwam/benchmarks/robotwin/adapter.py) converters illustrate this boundary.
For a robot, the driver owns camera capture, connection, calibration, units,
control rate and stopping. Driver availability and action semantics must be
checked for the actual hardware; installing LeRobot alone does not configure a
robot controller.

### YAM, Franka and G1 convenience converters

These converters do not include hardware drivers. Supply a LeRobot
`Robot`-compatible driver backed by your robot SDK. Add `robot` and
`adapter_options` to the same settings and call
`policy.select_robot_action(robot.get_observation(), task=...)`, then pass the
returned named scalar commands to `robot.send_action`. A standard batch pipeline
can call `policy.robot_action(policy.select_action(batch))` instead.

**YAM** uses 14D state/actions and three cameras:

```python
settings.update({
    "robot": "yam",
    "adapter_options": {
        "state_names": checkpoint_state_names,   # 14 driver names in trained order
        "action_names": checkpoint_action_names, # 14 driver names in trained order
        "camera_keys": {"top": "top", "left_wrist": "left_wrist",
                        "right_wrist": "right_wrist"},
    },
})
```

Alternatively provide a 14-element raw `state` vector instead of `state_names`.
The trained YAM layout is a 384×320 mosaic with the top camera above both wrists.

**Franka** Cartesian mode uses an 8D pose/finger state and 7D delta/gripper action:

```python
settings.update({
    "robot": "franka",
    "adapter_options": {
        "action_mode": "cartesian_delta",
        "position_scale": calibrated_position_scale,
        "rotation_scale": calibrated_rotation_scale,
        "gripper_open_width": calibrated_open_width,
        "camera_keys": {"image": "front", "wrist_image": "wrist"},
    },
})
```

Supply `eef_position` in meters, `eef_quaternion` in xyzw order and either
`gripper_qpos` (two finger positions) or `gripper_width` in meters. Output names
are `delta_position.{x,y,z}`, `delta_rotation.{x,y,z}` and `gripper.width`.
Override the seven `action_names` to match the driver's semantics. Cartesian
position/rotation scales and gripper opening width require calibration.
`action_mode=joint_position` uses an 8D joint/width state and 8D action, ordered
`action_names` and a matching joint-control checkpoint. Use a checkpoint trained
for the actual robot and controller semantics.

Before motion, check camera orientation, state/action order, units and output
commands with the driver's command sink disabled. Physical movement and task
success require separate validation on the target robot.

## Real-robot deployment demos

These are **inference-only, bring-your-own-checkpoint examples**. Train or
fine-tune Long-WAM on your own robot/task data first. No deployment checkpoint is
provided, selected from our model registry, or downloaded automatically.
The existing benchmark training recipes are separate; this integration does not
import the G1 source repository's training, collection or conversion pipelines.

### Common entry point

Three small templates live in
[configs/deploy](../configs/deploy): [YAM](../configs/deploy/yam.yaml),
[Franka](../configs/deploy/franka.yaml), [G1](../configs/deploy/g1.yaml).
Copy the appropriate YAML to your own configuration location and fill:

- `policy.checkpoint`, `policy.model_config`, `policy.stats`: your matching weights,
  resolved training config and normalization statistics; all default to `null`.
- `policy.text_cache`: your task's optional text embeddings, or configure the
  text encoder assets in your own model configuration.
- `task`, `control_hz`, camera/joint mapping and calibrated driver parameters.
  Preserve the training sampling cadence and history; `action_horizon: null`
  reads the action chunk from your config. Ensure `execute_steps` does not
  exceed it.
- `driver.factory` and `driver.options`: your site-specific driver. YAM/Franka
  do not bundle SDK drivers. G1 has the optional bridge wrapper described below.

```bash
# Configuration inspection only; blank placeholders are intentionally allowed.
longwam deploy yam --dry-run
longwam deploy franka --dry-run
longwam deploy g1 --dry-run

# Inference with your own bundle/driver; connects read-only and sends NO actions.
longwam deploy g1 --config /path/to/your-g1.yaml

# Right-arm policy: requires its native 8D/two-camera checkpoint, not a 16D model.
longwam deploy g1 --config /path/to/your-right-arm-g1.yaml \
  policy.adapter_options.control_side=right

# The SAME hardware and scheduling settings as the benchmark runtime.
longwam deploy g1 --config /path/to/your-g1.yaml --dry-run \
  policy.hardware=rtx5090 policy.execution=async \
  policy.action_horizon=32 policy.execute_steps=24 policy.trigger_stride=16 \
  policy.streaming_vae=true
```

Build a hardware backend separately with `longwam build <target>`; there is no
G1-specific acceleration fork. The example loads the policy once, warms inference
before arming, and executes a finite `max_steps` loop (default 300). The shared
runtime alone owns action chunks, resets, async handoff and video history. No
legacy WebSocket model server, action blending, smoothing or outer action queue
is used. Inference failures, stale actions and interrupts leave the control loop
through its stop/disconnect path.

### Driver interface and motion gate

`driver.factory=your_module:create_driver` must return an object implementing:

| Method | Required behavior |
| --- | --- |
| `create_driver(adapter=..., **options)` | Construct without moving or connecting hardware |
| `connect(read_only=True)` | Connect observations only; must not initialize/move actuators |
| `get_observation()` | Fresh RGB/state values accepted by the selected adapter; reject stale sensors |
| `arm(confirm_safety_area_clear=True)` | Explicitly enter the site's calibrated control lifecycle |
| `send_action(command)` | Apply named absolute/delta targets as specified by the checkpoint; enforce hardware limits |
| `stop()` | Stop policy control and establish a safe hold; must tolerate partial arming |
| `disconnect()` | Release connections; must not launch a home or other movement trajectory |

Use a small wrapper if an existing LeRobot/SDK driver has different lifecycle
method names. A plain LeRobot `Robot` object is not automatically safe to arm via
this demo. Driver construction must have no hardware side effects.

Read-only inference is the default. Before motion, inspect output commands with
the command sink disabled; verify camera orientation, calibration, action units,
task prompt, state/action ordering and controller frequency. Set a site-tested
`max_action_age_s` and driver limits, ensure the hardware watchdog and physical
emergency stop work, clear the area, and have an operator present. Only then:

```bash
longwam deploy g1 --config /path/to/your-g1.yaml \
  --enable-motion --confirm-safety-area-clear
```

The flags are permission gates, **not a safety certification**. A hung process or
network failure still requires the robot-side watchdog/emergency stop. Slow
inference can miss the configured cadence; do not relax safety deadlines to mask
it. For new tasks or checkpoint changes, return to read-only validation.

### Unitree G1 + Dex1 contract

Use `policy.robot: g1` with `policy.adapter_options.control_side` set explicitly:

| Mode | State and action order | Ordered checkpoint cameras | Image layout |
| --- | --- | --- | --- |
| `both` | Left arm 7, right arm 7, left Dex1, right Dex1 (16D) | `color_0`, `color_2`, `color_3` | Head above left/right wrists |
| `right` | Right arm 7, right Dex1 (8D) | `color_0`, `color_3` | Head above a black left tile and right wrist |

Both layouts are 384×320: the head tile is 256×320 and wrist tiles are 128×160.
The right-only layout is named `robotwin_right` in the resolved model config.
Black padding is applied before normalization, so the absent tile is **−1**, not
zero, in the model's [-1,1] input. The processor's training resize/normalization
is reused. The reference source used 480×640 RGB inputs resized to 240×320 before
tiling; your resolved processor must match your own training data.

`color_0` is the **left head eye** in both modes, not the other stereo eye.
Default driver camera names are `cam_high`, `cam_left_wrist`, `cam_right_wrist`;
override `camera_keys` only to map checkpoint keys to equivalent driver views.
RGB arrays must be uint8 H×W×3. Named scalar observations, native `state` /
`observation.state`, and legacy `observation/state` with `observation/color_*`
are accepted. The latter is a data-conversion convenience, not compatibility
with the old WebSocket server protocol.

The seven joints on each arm are shoulder pitch/roll/yaw, elbow, wrist
roll/pitch/yaw, in that order. Arms use **absolute joint targets in radians**;
Dex1 values remain in the calibrated driver's native units. Never reinterpret
them as Cartesian deltas or normalized [-1,1] gripper commands.
Default command names are `kLeft*.pos`, `kRight*.pos`, `left_gripper.pos` and
`right_gripper.pos`; `state_names`/`action_names` may be supplied for a different
driver in the same checkpoint order. No 8D↔16D padding or implicit side switching
is performed. This adapter does not expose walking, legs, waist or base motion.

### Optional G1 bridge wrapper

The [included wrapper](../src/longwam/runtime/robots/g1_driver.py) binds the
operator-installed `ial_g1d.robot.model.G1DModelInterface` to the common driver
API. Contract reference:
[Long-WAM-G1-Dynamic-Task-Deploy](https://github.com/kaiknower/Long-WAM-G1-Dynamic-Task-Deploy),
revision `a0eda8d2f269635dfef87f7ce3f34cf94a85b777`
(`ial_g1d` 0.3.0; robot bridge protocol 3).
That package/SDK is **not bundled or fetched by Long-WAM**; use your reviewed
installation, or implement the driver interface above with your own G1 SDK.
Access to that source repository may require its owner's permission.

The GPU workstation runs Long-WAM and the thin driver. The robot PC keeps its
existing bridge, camera services, SDK/DDS, calibration and watchdog. No old
model server is needed. For a reviewed local source installation:

```bash
# In the deployment environment; do not install the old Long-WAM model package.
pip install --no-deps /path/to/reviewed-g1-deployment/ial_g1d
pip install 'pyzmq>=25,<28'
```

The Long-WAM environment already supplies NumPy, OpenCV and PyYAML. Configure
your bridge/camera YAML locally, outside the source tree, and set
`driver.options.deployment_file` to its path. This file must describe your own
robot/camera hosts and ports; no personal host address is embedded in our
template. The reviewed driver accepts a top-level `deployment` mapping.

Required bridge settings are `verified_end_effector: dex1`,
`teleop_home_mode: true` (reviewed fixed-column, hold-on-exit lifecycle), and
`agv.enabled: false`. Enable the appropriate head/wrist camera services and
verify `cam_high` is the left stereo eye. Before enabling motion, configure
`driver.options.max_arm_delta` (radians) and `max_gripper_delta` (Dex1 native units)
using limits calibrated for your robot. These default to `null`, not guessed
“safe” numbers.

The wrapper connects read-only, retains the upstream bridge's explicit startup
checks and watchdog, checks commands against fresh measured state, and requests
`abort_hold` on stop instead of an automatic home trajectory. Enabling motion
may perform the bridge's configured startup movement: review it at the robot
before using the flags. The inactive arm in right-only mode is held by the
external bridge; policy output remains strictly 8D. The bridge's own Dex1 limits
still apply. This is not a universal G1 hardware/hand driver.

The integrated examples are checked with CPU/mock tests only. No new physical
G1/YAM/Franka run or GPU checkpoint rollout is claimed.

## Code Structure

```text
longwam/
├── benchmarks/
│   ├── libero/adapter.py             # Observation → LeRobot batch
│   ├── libero/eval_libero_single.py  # Episodes, replay, timing and scoring
│   └── robotwin/                    # Native evaluator launcher + policy bridge
└── runtime/
    ├── factory.py                   # Model factories, training and offline generation
    ├── lerobot.py                   # Public policy/features and command helpers
    ├── policy.py                    # Model/processor setup and observation history
    ├── libero.py, robotwin.py       # Checkpoint-specific inference components
    ├── robots/{yam,franka,g1}.py     # Robot observation/command conversion
    ├── robots/g1_driver.py          # Optional external G1 bridge binding
    ├── deploy.py                    # Shared finite, read-only-by-default demo
    ├── execution.py                 # Sync/async action scheduling
    ├── worker.py                    # Ordered process transport
    ├── streaming_vae.py             # Incremental causal VAE sessions
    ├── optim/                      # Shared model/operator optimizations
    └── backends/                   # Hardware profiles, build and device patches
```

Build: CLI → `backends/build.py` → bundled source copy → device patch → extension.
Control: evaluator/driver → LeRobot policy → scheduler → worker → processor →
model → action denormalization → evaluator/driver. Frames are preprocessed once
per observation; predictions reuse the current processed frame. Streaming VAE
supplies clean latents directly, avoiding an additional full-window encoding.
Offline generation uses `factory.run_inference` in-process. Benchmark diagnostics
can use an already loaded model with an internal in-process transport; RoboCasa
GR1 vectorized serving uses its benchmark-specific RPC protocol.
