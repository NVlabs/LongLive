# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation; G1 deployment contracts referenced from the source below.
# Source: https://github.com/kaiknower/Long-WAM-G1-Dynamic-Task-Deploy @ a0eda8d2f269635dfef87f7ce3f34cf94a85b777
# Changes: Inference-only adapter for the shared Long-WAM runtime; no upstream runtime or training code copied.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

"""Unitree G1 arms + Dex1: native 16D dual-arm or 8D right-arm contracts."""
from ._common import _images, _names, _vector

_ARM_PARTS = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow",
              "WristRoll", "WristPitch", "WristYaw")
LEFT_ARM = tuple(f"kLeft{part}.pos" for part in _ARM_PARTS)
RIGHT_ARM = tuple(f"kRight{part}.pos" for part in _ARM_PARTS)
JOINT_NAMES = {
    "both": (*LEFT_ARM, *RIGHT_ARM, "left_gripper.pos", "right_gripper.pos"),
    "right": (*RIGHT_ARM, "right_gripper.pos"),
}
CAMERAS = {
    "both": {"color_0": "cam_high", "color_2": "cam_left_wrist", "color_3": "cam_right_wrist"},
    "right": {"color_0": "cam_high", "color_3": "cam_right_wrist"},
}


class G1Adapter:
    """Absolute arm radians and native Dex1 positions; never base/leg commands.

    Vector order is left arm 7, right arm 7, left/right Dex1 for dual-arm;
    right arm 7, right Dex1 for right-arm. No padding between the two contracts.
    """

    def __init__(self, *, control_side, state_names=None, action_names=None, camera_keys=None):
        if control_side not in JOINT_NAMES:
            raise ValueError("G1 control_side must be both (16D) or right (8D)")
        self.control_side = control_side
        names = JOINT_NAMES[control_side]
        self.state_dim = self.action_dim = len(names)
        self.state_names = _names(names if state_names is None else state_names,
                                  self.state_dim, "state_names")
        self.action_names = _names(names if action_names is None else action_names,
                                   self.action_dim, "action_names")
        self.camera_keys = dict(CAMERAS[control_side] if camera_keys is None else camera_keys)
        if set(self.camera_keys) != set(CAMERAS[control_side]):
            raise ValueError(f"G1 {control_side} requires checkpoint cameras {tuple(CAMERAS[control_side])}")
        _names(self.camera_keys.values(), len(self.camera_keys), "camera feature names")

    def observation(self, raw):
        state_key = next((key for key in ("observation.state", "observation/state", "state")
                          if key in raw), None)
        state = raw[state_key] if state_key else [raw[name] for name in self.state_names]
        source = raw.get("images", raw)
        images = {}
        for model_key, driver_key in self.camera_keys.items():
            candidates = (driver_key, f"observation.images.{driver_key}",
                          f"observation/{model_key}", model_key)
            key = next((key for key in candidates if key in source), None)
            if key is None:
                raise ValueError(f"Missing G1 RGB camera: {driver_key} ({model_key})")
            images[driver_key] = source[key]
        return {"state": _vector(state, self.state_dim, "G1 state"),
                "images": _images(images, self.camera_keys)}

    def action(self, values):
        values = _vector(values, self.action_dim, "G1 action")
        return dict(zip(self.action_names, map(float, values)))

    def validate_checkpoint(self, data):
        """Reject camera/embodiment swaps before any checkpoint is loaded."""
        expected = tuple(CAMERAS[self.control_side])
        shapes = [data.shape_meta]
        processor = data.get("processor")
        if processor is not None and "shape_meta" in processor:
            shapes.append(processor.shape_meta)
        for shape in shapes:
            for category, dim in (("state", self.state_dim), ("action", self.action_dim)):
                fields = shape[category]
                if len(fields) != 1:
                    raise ValueError(f"G1 requires one merged {category} field")
                raw_dim = fields[0].get("raw_shape", fields[0].shape)
                raw_dim = (raw_dim,) if isinstance(raw_dim, int) else tuple(raw_dim)
                if raw_dim != (dim,):
                    raise ValueError(f"G1 {self.control_side} requires {dim}D {category}")
            if tuple(meta.key for meta in shape.images) != expected:
                raise ValueError(f"G1 checkpoint camera order must be {expected}")
        layout = "robotwin" if self.control_side == "both" else "robotwin_right"
        if data.concat_multi_camera != layout or tuple(data.video_size) != (384, 320):
            raise ValueError(f"G1 {self.control_side} requires {layout} layout and 384x320 video_size")
