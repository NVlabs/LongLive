# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/robots/franka.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Franka checkpoint observation and action conversion; hardware stays in the driver."""
import numpy as np
from ._common import _vector, _images, _names


class FrankaAdapter:
    """Panda Cartesian-delta or joint-position interface, with explicit units."""

    def __init__(
        self, *, action_mode, action_names=None, state_names=None, camera_keys=None,
        position_scale=None, rotation_scale=None, gripper_open_width=None,
    ):
        if action_mode not in {"cartesian_delta", "joint_position"}:
            raise ValueError("Franka action_mode must be cartesian_delta or joint_position")
        self.action_mode = action_mode
        self.state_dim, self.action_dim = (8, 7) if action_mode == "cartesian_delta" else (8, 8)
        self.camera_keys = dict(camera_keys or {"image": "front", "wrist_image": "wrist"})
        if set(self.camera_keys) != {"image", "wrist_image"}:
            raise ValueError("Franka requires image and wrist_image checkpoint cameras")
        self.action_names = None
        self.state_names = None
        if action_mode == "joint_position":
            self.action_names = _names(action_names, 8, "action_names")
            self.state_names = None if state_names is None else _names(state_names, 8, "state_names")
        else:
            self.action_names = _names(action_names or [
                "delta_position.x", "delta_position.y", "delta_position.z",
                "delta_rotation.x", "delta_rotation.y", "delta_rotation.z", "gripper.width",
            ], 7, "action_names")
            if state_names is not None:
                raise ValueError("Cartesian Franka uses explicit EEF pose fields, not state_names")
            scales = (position_scale, rotation_scale, gripper_open_width)
            if any(v is None or not np.isfinite(v) or v <= 0 for v in scales):
                raise ValueError(
                    "Cartesian control requires positive position_scale, rotation_scale "
                    "and gripper_open_width"
                )
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_open_width = gripper_open_width

    def observation(self, raw):
        if "observation.state" in raw:
            # Already assembled checkpoint state: pose/axis-angle/fingers or joints/width.
            state = _vector(raw["observation.state"], 8, "Franka state")
        elif self.action_mode == "joint_position" and self.state_names is not None:
            state = _vector([raw[key] for key in self.state_names], 8, "Franka state")
        elif self.action_mode == "joint_position":
            state = np.concatenate([
                _vector(raw["joint_positions"], 7, "joint_positions"),
                _vector([raw["gripper_width"]], 1, "gripper_width"),
            ])
        else:
            import torch
            from longwam.datasets.lerobot.utils.rotation import quaternion_to_axis_angle

            position = _vector(raw["eef_position"], 3, "eef_position")
            quaternion = _vector(raw["eef_quaternion"], 4, "eef_quaternion (xyzw)")
            norm = np.linalg.norm(quaternion)
            if norm < 1e-8:
                raise ValueError("eef_quaternion cannot have zero norm")
            quaternion /= norm
            rotation = quaternion_to_axis_angle(torch.from_numpy(quaternion[[3, 0, 1, 2]])).numpy()
            if "gripper_qpos" in raw:
                fingers = _vector(raw["gripper_qpos"], 2, "gripper_qpos")
            else:
                width = _vector([raw["gripper_width"]], 1, "gripper_width")[0]
                if width < 0:
                    raise ValueError("gripper_width must be nonnegative meters")
                fingers = np.full(2, width / 2, dtype=np.float32)
            state = np.concatenate([position, rotation, fingers])
        return {"images": _images(raw, self.camera_keys), "state": state}

    def action(self, values):
        values = _vector(values, self.action_dim, "Franka action")
        if self.action_mode == "joint_position":
            return dict(zip(self.action_names, map(float, values)))
        # Training gripper convention is 0=closed, 1=open, before simulator conversion.
        return {
            "delta_position": values[:3] * self.position_scale,
            "delta_rotation": values[3:6] * self.rotation_scale,
            "gripper_width": float(np.clip(values[6], 0, 1) * self.gripper_open_width),
        }

    def lerobot_action(self, values):
        """Named scalar commands matching a LeRobot driver's action_features."""
        command = self.action(values)
        if self.action_mode == "joint_position":
            return command
        values = np.concatenate([
            command["delta_position"], command["delta_rotation"], [command["gripper_width"]],
        ])
        return dict(zip(self.action_names, map(float, values)))
