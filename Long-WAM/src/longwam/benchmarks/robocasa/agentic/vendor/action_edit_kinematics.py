# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/action_edit_kinematics.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Robot-only FK and bounded edits; never reads objects or advances physics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


def pose(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return dict(position=t[:3, 3].tolist(), quaternion_wxyz=q[[3, 0, 1, 2]].tolist())


def edited_targets(trajectory, steps, edit):
    """Smooth root-frame offset relative to the selected student prefix.

    Gripper overrides apply explicitly from the first step. Closure after an
    approach must be issued in a later decision using a fresh observation.
    """
    if type(steps) is not int or not 1 <= steps <= 15:
        raise ValueError("Execution prefix must be 1..15 steps")
    if not isinstance(edit, dict) or set(edit) != {
        "delta_position",
        "delta_rotation_vector",
        "gripper",
    }:
        raise ValueError("Edit requires delta_position, delta_rotation_vector, gripper")
    dp, dr = (
        np.asarray(edit["delta_position"], float),
        np.asarray(edit["delta_rotation_vector"], float),
    )
    if dp.shape != (3,) or dr.shape != (3,) or not np.isfinite([dp, dr]).all():
        raise ValueError("Finite 3D edit vectors required")
    if np.linalg.norm(dp) > 0.05 + 1e-10 or np.linalg.norm(dr) > 0.35 + 1e-10:
        raise ValueError("Edit exceeds 5 cm or 0.35 rad short-correction limit")
    if edit["gripper"] not in ("keep", "open", "closed"):
        raise ValueError("Unknown gripper override")
    targets = []
    for i, original in enumerate(trajectory[:steps]):
        t = transform(original["position"], original["quaternion_wxyz"])
        alpha = (i + 1) / steps
        t[:3, 3] += alpha * dp
        t[:3, :3] = Rotation.from_rotvec(alpha * dr).as_matrix() @ t[:3, :3]
        grip = (
            original["gripper_closed"] if edit["gripper"] == "keep" else edit["gripper"] == "closed"
        )
        targets.append(dict(index=i, **pose(t), gripper_closed=grip))
    return targets
