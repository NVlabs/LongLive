# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/datasets/lerobot/robocasa_contract.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""RoboCasa365 data and camera-layout contracts shared by train and eval."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torchvision.transforms.functional as transforms_F


ROBOCASA_CAMERA_KEYS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
ROBOCASA_CAMERA_SHAPE = (3, 256, 256)
ROBOCASA_MOSAIC_LAYOUT = "robocasa"
ROBOCASA_LEFT_MAIN_LAYOUT = "robocasa_left_main"
ROBOCASA_RGB_LAYOUTS = (
    ROBOCASA_MOSAIC_LAYOUT,
    ROBOCASA_LEFT_MAIN_LAYOUT,
)
ROBOCASA_RGB_FRAME_SHAPE = (3, 384, 320)
ROBOCASA_LATENT_SLOT_LAYOUT = "robocasa_latent_slots"
ROBOCASA_ACTION_DIM = 12
ROBOCASA_STATE_DIM = 16
ROBOCASA_FPS = 20
ROBOCASA_HUMAN300_GROUP_COUNTS = {"atomic": 65, "composite": 235}

# Dataset action order:
# [base(4), control_mode(1), eef_delta_pos(3), eef_delta_rot(3), gripper(1)].
# Padded continuous deltas must become zero; mode and gripper hold their last value.
ROBOCASA_DELTA_ACTION_DIM_MASK = (
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
)


class RoboCasaCameraContractError(ValueError):
    """Raised when a RoboCasa camera input cannot preserve the slot contract."""


def validate_robocasa_rgb_camera_metadata(
    image_metadata: Sequence[Mapping],
) -> None:
    """Require native RGB cameras in the positional order used by mosaics."""
    camera_keys = tuple(metadata.get("key") for metadata in image_metadata)
    if camera_keys != ROBOCASA_CAMERA_KEYS:
        raise RoboCasaCameraContractError(
            "RoboCasa RGB layouts require camera metadata in the exact order "
            f"{ROBOCASA_CAMERA_KEYS}, got {camera_keys}."
        )

    for metadata in image_metadata:
        key = metadata["key"]
        raw_shape = tuple(metadata.get("raw_shape", ()))
        processed_shape = tuple(metadata.get("shape", ()))
        if raw_shape != ROBOCASA_CAMERA_SHAPE:
            raise RoboCasaCameraContractError(
                f"RoboCasa RGB camera {key} raw_shape must be "
                f"{ROBOCASA_CAMERA_SHAPE}, got {raw_shape}."
            )
        if processed_shape != ROBOCASA_CAMERA_SHAPE:
            raise RoboCasaCameraContractError(
                f"RoboCasa RGB camera {key} shape must remain "
                f"{ROBOCASA_CAMERA_SHAPE}, got {processed_shape}."
            )


def validate_robocasa_latent_slot_camera_metadata(
    image_metadata: Sequence[Mapping],
) -> None:
    """Require the canonical three-camera order and unmodified spatial shape."""
    camera_keys = tuple(metadata.get("key") for metadata in image_metadata)
    if camera_keys != ROBOCASA_CAMERA_KEYS:
        raise RoboCasaCameraContractError(
            "RoboCasa latent slots require camera metadata in the exact order "
            f"{ROBOCASA_CAMERA_KEYS}, got {camera_keys}."
        )

    for metadata in image_metadata:
        key = metadata["key"]
        raw_shape = tuple(metadata.get("raw_shape", ()))
        processed_shape = tuple(metadata.get("shape", ()))
        if raw_shape != ROBOCASA_CAMERA_SHAPE:
            raise RoboCasaCameraContractError(
                f"RoboCasa latent-slot camera {key} raw_shape must be "
                f"{ROBOCASA_CAMERA_SHAPE}, got {raw_shape}."
            )
        if processed_shape != ROBOCASA_CAMERA_SHAPE:
            raise RoboCasaCameraContractError(
                f"RoboCasa latent-slot camera {key} shape must remain "
                f"{ROBOCASA_CAMERA_SHAPE}, got {processed_shape}."
            )


def validate_robocasa_latent_slot_cameras(cameras: torch.Tensor) -> None:
    """Validate decoded cameras before any normalization or VAE encoding."""
    if cameras.ndim != 5:
        raise RoboCasaCameraContractError(
            "RoboCasa latent-slot cameras must have shape [V, T, C, H, W], "
            f"got {tuple(cameras.shape)}."
        )
    if (
        cameras.shape[0] != len(ROBOCASA_CAMERA_KEYS)
        or cameras.shape[1] <= 0
        or tuple(cameras.shape[2:]) != ROBOCASA_CAMERA_SHAPE
    ):
        raise RoboCasaCameraContractError(
            "RoboCasa latent slots require exactly three non-empty "
            "left/right/wrist videos with shape [3, T, 3, 256, 256], got "
            f"{tuple(cameras.shape)}."
        )


def discover_human300_dataset_dirs(
    pretrain_root: str | Path,
    *,
    expected_group_counts: Mapping[str, int] = ROBOCASA_HUMAN300_GROUP_COUNTS,
) -> tuple[Path, ...]:
    """Return the deterministic Human300 LeRobot roots and reject partial data."""
    pretrain_root = Path(pretrain_root).expanduser().resolve()
    info_paths = sorted(pretrain_root.glob("*/*/*/lerobot/meta/info.json"))

    datasets: list[tuple[str, str, Path]] = []
    for info_path in info_paths:
        relative = info_path.relative_to(pretrain_root)
        if len(relative.parts) != 6:
            continue
        group, task_name, _, lerobot, meta, filename = relative.parts
        if (lerobot, meta, filename) != ("lerobot", "meta", "info.json"):
            continue
        datasets.append((group, task_name, info_path.parent.parent))

    actual_group_counts = Counter(group for group, _, _ in datasets)
    expected_group_counts = dict(expected_group_counts)
    if dict(actual_group_counts) != expected_group_counts:
        raise ValueError(
            "Incomplete RoboCasa Human300 data: expected group counts "
            f"{expected_group_counts}, got {dict(actual_group_counts)} under "
            f"{pretrain_root}."
        )

    task_names = [task_name for _, task_name, _ in datasets]
    if len(task_names) != len(set(task_names)):
        duplicates = sorted(
            task_name for task_name, count in Counter(task_names).items() if count > 1
        )
        raise ValueError(f"Duplicate RoboCasa Human300 task directories: {duplicates}")

    return tuple(path for _, _, path in datasets)


def validate_human300_dataset_inventory(
    dataset_dirs: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Require an exact, duplicate-free match to the mounted Human300 tree."""
    dataset_dirs = tuple(Path(path).expanduser().resolve() for path in dataset_dirs)
    if not dataset_dirs:
        raise ValueError("RoboCasa Human300 dataset inventory is empty.")
    if len(dataset_dirs) != len(set(dataset_dirs)):
        raise ValueError("RoboCasa Human300 dataset inventory contains duplicates.")

    pretrain_roots = {path.parents[3] for path in dataset_dirs if len(path.parents) >= 4}
    if len(pretrain_roots) != 1:
        raise ValueError(
            "RoboCasa Human300 paths must share one v1.0/pretrain root, got "
            f"{sorted(str(path) for path in pretrain_roots)}."
        )
    discovered = discover_human300_dataset_dirs(next(iter(pretrain_roots)))
    if set(dataset_dirs) != set(discovered):
        missing = sorted(str(path) for path in set(discovered) - set(dataset_dirs))
        extra = sorted(str(path) for path in set(dataset_dirs) - set(discovered))
        raise ValueError(
            "Configured RoboCasa Human300 paths do not match the mounted dataset: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    return dataset_dirs


def validate_robocasa_lerobot_contract(
    dataset_dirs: Iterable[str | Path],
    *,
    expected_fps: int = ROBOCASA_FPS,
) -> None:
    """Validate the schema fields Long-WAM relies on before a long run starts."""
    required_features = {
        *(f"observation.images.{key}" for key in ROBOCASA_CAMERA_KEYS),
        "observation.state",
        "action",
    }
    errors = []
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        info_path = dataset_dir / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = info["features"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{info_path}: invalid metadata ({exc})")
            continue

        missing = sorted(required_features.difference(features))
        if missing:
            errors.append(f"{info_path}: missing features {missing}")
            continue
        if int(info.get("fps", -1)) != expected_fps:
            errors.append(f"{info_path}: fps={info.get('fps')} (expected {expected_fps})")
        for key in ROBOCASA_CAMERA_KEYS:
            shape = list(features[f"observation.images.{key}"].get("shape", ()))
            if shape != [256, 256, 3]:
                errors.append(f"{info_path}: camera {key} shape={shape} (expected [256, 256, 3])")
        state_shape = list(features["observation.state"].get("shape", ()))
        action_shape = list(features["action"].get("shape", ()))
        if state_shape != [ROBOCASA_STATE_DIM]:
            errors.append(
                f"{info_path}: state shape={state_shape} (expected [{ROBOCASA_STATE_DIM}])"
            )
        if action_shape != [ROBOCASA_ACTION_DIM]:
            errors.append(
                f"{info_path}: action shape={action_shape} (expected [{ROBOCASA_ACTION_DIM}])"
            )

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:10])
        suffix = "" if len(errors) <= 10 else f"\n- ... {len(errors) - 10} more"
        raise ValueError(f"RoboCasa LeRobot contract validation failed:\n{preview}{suffix}")


def build_robocasa_camera_layout(
    cameras: torch.Tensor,
    *,
    layout: str,
) -> torch.Tensor:
    """Compose fixed-order left/right/wrist cameras into a named RGB layout.

    ``cameras`` must follow :data:`ROBOCASA_CAMERA_KEYS` and have shape
    ``[3, ..., 3, H, W]``. Both supported layouts return ``[..., 3, 384, 320]``.
    """
    if not isinstance(cameras, torch.Tensor):
        raise TypeError(f"RoboCasa cameras must be a torch.Tensor, got {type(cameras).__name__}.")
    if cameras.ndim < 4:
        raise ValueError(
            f"RoboCasa cameras must have shape [3, ..., C, H, W], got {tuple(cameras.shape)}."
        )
    if cameras.shape[0] != len(ROBOCASA_CAMERA_KEYS):
        raise ValueError(
            f"RoboCasa mosaic requires {len(ROBOCASA_CAMERA_KEYS)} cameras in "
            f"{ROBOCASA_CAMERA_KEYS}, got shape {tuple(cameras.shape)}."
        )
    if cameras.shape[-3] != ROBOCASA_CAMERA_SHAPE[0]:
        raise ValueError(
            f"RoboCasa mosaic cameras must have 3 RGB channels, got shape {tuple(cameras.shape)}."
        )
    if cameras.shape[-2] <= 0 or cameras.shape[-1] <= 0:
        raise ValueError(
            "RoboCasa mosaic cameras must have non-empty spatial dimensions, "
            f"got shape {tuple(cameras.shape)}."
        )
    if layout not in ROBOCASA_RGB_LAYOUTS:
        raise ValueError(
            f"Unsupported RoboCasa RGB layout {layout!r}; expected one of {ROBOCASA_RGB_LAYOUTS}."
        )

    resize_kwargs = {
        "interpolation": transforms_F.InterpolationMode.BILINEAR,
        "antialias": True,
    }
    if layout == ROBOCASA_MOSAIC_LAYOUT:
        external_left = transforms_F.resize(cameras[0], size=[256, 160], **resize_kwargs)
        external_right = transforms_F.resize(cameras[1], size=[256, 160], **resize_kwargs)
        wrist = transforms_F.resize(cameras[2], size=[128, 320], **resize_kwargs)
        top = torch.cat((external_left, external_right), dim=-1)
        return torch.cat((top, wrist), dim=-2)

    left = transforms_F.resize(cameras[0], size=[256, 320], **resize_kwargs)
    right = transforms_F.resize(cameras[1], size=[128, 160], **resize_kwargs)
    wrist = transforms_F.resize(cameras[2], size=[128, 160], **resize_kwargs)
    bottom = torch.cat((right, wrist), dim=-1)
    return torch.cat((left, bottom), dim=-2)


def build_robocasa_camera_mosaic(cameras: torch.Tensor) -> torch.Tensor:
    """Build the legacy equal-area left/right/wrist RoboCasa mosaic."""
    return build_robocasa_camera_layout(
        cameras,
        layout=ROBOCASA_MOSAIC_LAYOUT,
    )
