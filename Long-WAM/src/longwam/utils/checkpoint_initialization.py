# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/utils/checkpoint_initialization.py
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

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping

import torch

from .logging_config import get_logger

logger = get_logger(__name__)


LIBERO_TO_ROBOCASA = "libero_to_robocasa"

_ACTION_ENCODER_WEIGHT = "mixtures.action.action_encoder.weight"
_ACTION_HEAD_WEIGHT = "mixtures.action.head.weight"
_ACTION_HEAD_BIAS = "mixtures.action.head.bias"
_ACTION_BOUNDARY_KEYS = frozenset({_ACTION_ENCODER_WEIGHT, _ACTION_HEAD_WEIGHT, _ACTION_HEAD_BIAS})

_EXPECTED_MOT_KEYS = 1649
_EXPECTED_COMPATIBLE_MOT_KEYS = 1646
_LIBERO_ACTION_DIM = 7
_ROBOCASA_ACTION_DIM = 12
_LIBERO_EEF = slice(0, 6)
_ROBOCASA_EEF = slice(5, 11)
_LIBERO_GRIPPER = 6
_ROBOCASA_GRIPPER = 11


@dataclass(frozen=True)
class InitializationReport:
    checkpoint: str
    adapter: str
    checkpoint_step: int | None
    loaded_keys: int
    mapped_keys: tuple[str, ...]
    reinitialized: tuple[str, ...]
    gripper_mapped: bool


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _validate_gripper_spec(gripper: Mapping[str, Any] | None) -> bool:
    """Validate model-space endpoints for an exact gripper sign flip."""
    if gripper is None:
        return False

    required = (
        "normalized_source_min",
        "normalized_source_max",
        "normalized_target_min",
        "normalized_target_max",
    )
    missing = [key for key in required if gripper.get(key) is None]
    if missing:
        raise ValueError(
            "Gripper transfer requires explicit source/target min and max values "
            f"in normalized model space; missing={missing}."
        )

    source_min, source_max, target_min, target_max = (float(gripper[key]) for key in required)
    values = (source_min, source_max, target_min, target_max)
    if not all(isfinite(value) for value in values):
        raise ValueError(f"Gripper min/max values must be finite, got {values}.")
    if source_min >= source_max or target_min >= target_max:
        raise ValueError(
            "Normalized gripper ranges must have min < max, got "
            f"source=[{source_min}, {source_max}], "
            f"target=[{target_min}, {target_max}]."
        )

    if not (
        isclose(source_min, -target_max, rel_tol=0.0, abs_tol=1e-8)
        and isclose(source_max, -target_min, rel_tol=0.0, abs_tol=1e-8)
    ):
        raise ValueError(
            "LIBERO->RoboCasa initialization only supports an exact gripper sign "
            "flip in normalized model space. Expected "
            "normalized_source_min=-normalized_target_max and "
            "normalized_source_max=-normalized_target_min, got "
            f"source=[{source_min}, {source_max}], "
            f"target=[{target_min}, {target_max}]."
        )
    return True


def _validate_proprio_is_reinitialized(model: Any, payload: Mapping[str, Any]) -> tuple[int, int]:
    source = payload.get("proprio_encoder")
    target = getattr(model, "proprio_encoder", None)
    if not isinstance(source, Mapping) or "weight" not in source:
        raise ValueError(
            "LIBERO checkpoint must contain `proprio_encoder.weight` for source schema validation."
        )
    if target is None:
        raise ValueError("RoboCasa model must have a 16D `proprio_encoder`; got None.")

    source_weight = source["weight"]
    target_weight = target.state_dict().get("weight")
    if not isinstance(source_weight, torch.Tensor) or not isinstance(target_weight, torch.Tensor):
        raise TypeError("Source and target proprio weights must be tensors.")
    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise ValueError(
            "Source and target proprio weights must be matrices, got "
            f"{_shape(source_weight)} and {_shape(target_weight)}."
        )

    source_dim = int(source_weight.shape[1])
    target_dim = int(target_weight.shape[1])
    if source_dim != 8 or target_dim != 16:
        raise ValueError(
            f"LIBERO->RoboCasa proprio schema must be 8D->16D, got {source_dim}D->{target_dim}D."
        )
    return source_dim, target_dim


def _validate_mot_states(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    source_keys = set(source)
    target_keys = set(target)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    if missing or unexpected:
        raise ValueError(
            "LIBERO and RoboCasa MoT key sets differ: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )
    if len(target) != _EXPECTED_MOT_KEYS:
        raise ValueError(
            "Unexpected RoboCasa MoT architecture: expected "
            f"{_EXPECTED_MOT_KEYS} state keys, got {len(target)}."
        )

    non_tensors = sorted(
        key
        for key in target_keys
        if not isinstance(source[key], torch.Tensor) or not isinstance(target[key], torch.Tensor)
    )
    if non_tensors:
        raise TypeError(f"MoT state entries must be tensors: {non_tensors[:8]}.")

    mismatched = {key for key in target_keys if _shape(source[key]) != _shape(target[key])}
    if mismatched != _ACTION_BOUNDARY_KEYS:
        extra = sorted(mismatched - _ACTION_BOUNDARY_KEYS)
        absent = sorted(_ACTION_BOUNDARY_KEYS - mismatched)
        raise ValueError(
            "Shape mismatch allowlist violation for LIBERO->RoboCasa MoT: "
            f"extra={extra}, expected_but_absent={absent}."
        )

    expected_shapes = {
        _ACTION_ENCODER_WEIGHT: (
            (int(target[_ACTION_ENCODER_WEIGHT].shape[0]), _LIBERO_ACTION_DIM),
            (int(target[_ACTION_ENCODER_WEIGHT].shape[0]), _ROBOCASA_ACTION_DIM),
        ),
        _ACTION_HEAD_WEIGHT: (
            (_LIBERO_ACTION_DIM, int(target[_ACTION_HEAD_WEIGHT].shape[1])),
            (_ROBOCASA_ACTION_DIM, int(target[_ACTION_HEAD_WEIGHT].shape[1])),
        ),
        _ACTION_HEAD_BIAS: (
            (_LIBERO_ACTION_DIM,),
            (_ROBOCASA_ACTION_DIM,),
        ),
    }
    bad_boundary_shapes = []
    for key, (expected_source, expected_target) in expected_shapes.items():
        actual_source, actual_target = _shape(source[key]), _shape(target[key])
        if actual_source != expected_source or actual_target != expected_target:
            bad_boundary_shapes.append(
                f"{key}: source={actual_source}/{expected_source}, "
                f"target={actual_target}/{expected_target}"
            )
    if bad_boundary_shapes:
        raise ValueError("Unexpected action-boundary shapes: " + "; ".join(bad_boundary_shapes))

    compatible = tuple(sorted(target_keys - _ACTION_BOUNDARY_KEYS))
    if len(compatible) != _EXPECTED_COMPATIBLE_MOT_KEYS:
        raise ValueError(
            "Expected exactly "
            f"{_EXPECTED_COMPATIBLE_MOT_KEYS} shape-compatible MoT keys, "
            f"got {len(compatible)}."
        )
    return compatible


@torch.no_grad()
def initialize_from_checkpoint(
    model: Any,
    *,
    checkpoint_path: str | Path,
    adapter: str,
    gripper: Mapping[str, Any] | None = None,
) -> InitializationReport:
    """Initialize RoboCasa weights from LIBERO without restoring train state.

    The checkpoint is memory-mapped on CPU and copied tensor-by-tensor so it can
    run before optimizer / ZeRO master parameters are created without making a
    second in-memory copy of the full MoT checkpoint.
    """
    if adapter != LIBERO_TO_ROBOCASA:
        raise ValueError(
            f"Unsupported checkpoint initialization adapter: {adapter!r}. "
            f"Expected {LIBERO_TO_ROBOCASA!r}."
        )

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")

    # This is deliberately not model.load_checkpoint(): strict=False still
    # rejects size mismatches and would also try to load the 8D proprio encoder.
    payload = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or not isinstance(payload.get("mot"), Mapping):
        raise ValueError(f"Initialization checkpoint must contain a `mot` state dict: {path}")

    mot = getattr(model, "mot", None)
    if mot is None:
        raise ValueError("Target model has no `mot` module.")
    source_state = payload["mot"]
    target_state = mot.state_dict()
    compatible = _validate_mot_states(source_state, target_state)
    source_proprio_dim, target_proprio_dim = _validate_proprio_is_reinitialized(model, payload)
    map_gripper = _validate_gripper_spec(gripper)

    for key in compatible:
        target_state[key].copy_(source_state[key], non_blocking=False)

    target_encoder = target_state[_ACTION_ENCODER_WEIGHT]
    source_encoder = source_state[_ACTION_ENCODER_WEIGHT]
    target_encoder[:, _ROBOCASA_EEF].copy_(source_encoder[:, _LIBERO_EEF], non_blocking=False)

    target_head = target_state[_ACTION_HEAD_WEIGHT]
    source_head = source_state[_ACTION_HEAD_WEIGHT]
    target_head[_ROBOCASA_EEF, :].copy_(source_head[_LIBERO_EEF, :], non_blocking=False)

    target_bias = target_state[_ACTION_HEAD_BIAS]
    source_bias = source_state[_ACTION_HEAD_BIAS]
    target_bias[_ROBOCASA_EEF].copy_(source_bias[_LIBERO_EEF], non_blocking=False)

    if map_gripper:
        target_encoder[:, _ROBOCASA_GRIPPER].copy_(
            -source_encoder[:, _LIBERO_GRIPPER], non_blocking=False
        )
        target_head[_ROBOCASA_GRIPPER, :].copy_(
            -source_head[_LIBERO_GRIPPER, :], non_blocking=False
        )
        target_bias[_ROBOCASA_GRIPPER].copy_(-source_bias[_LIBERO_GRIPPER], non_blocking=False)
        reinitialized = (
            "action[0:5] (base+control_mode)",
            f"proprio_encoder ({target_proprio_dim}D)",
        )
    else:
        logger.warning(
            "LIBERO->RoboCasa gripper transfer skipped: explicit source/target "
            "normalized min/max were not provided. Target action[11] remains "
            "newly initialized."
        )
        reinitialized = (
            "action[0:5] (base+control_mode)",
            "action[11] (gripper)",
            f"proprio_encoder ({target_proprio_dim}D)",
        )

    mapped_keys = tuple(sorted(_ACTION_BOUNDARY_KEYS))
    report = InitializationReport(
        checkpoint=str(path),
        adapter=adapter,
        checkpoint_step=(int(payload["step"]) if payload.get("step") is not None else None),
        loaded_keys=len(compatible),
        mapped_keys=mapped_keys,
        reinitialized=reinitialized,
        gripper_mapped=map_gripper,
    )
    logger.info(
        "Checkpoint initialization complete: adapter=%s checkpoint=%s step=%s "
        "loaded=%d mapped=%d reinitialized=%s",
        report.adapter,
        report.checkpoint,
        report.checkpoint_step,
        report.loaded_keys,
        len(report.mapped_keys),
        ", ".join(report.reinitialized),
    )
    logger.info(
        "Action transfer: LIBERO EEF action[0:6] -> RoboCasa action[5:11]; "
        "gripper_sign_flip=%s. Source proprio %dD ignored; target proprio %dD "
        "keeps its new initialization.",
        report.gripper_mapped,
        source_proprio_dim,
        target_proprio_dim,
    )
    return report
