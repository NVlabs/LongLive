# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/libero/adapter.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Convert native LIBERO observations to LeRobot policy features."""
import numpy as np
import math
import torch


def policy_batch(observation, task):
    """Bridge the native simulator callback to the sole public policy API."""
    images = get_libero_image(observation)
    state = torch.cat([
        torch.as_tensor(observation["robot0_eef_pos"], dtype=torch.float32),
        torch.as_tensor(quat2axisangle(observation["robot0_eef_quat"].copy()), dtype=torch.float32),
        torch.as_tensor(observation["robot0_gripper_qpos"], dtype=torch.float32),
    ])
    return {
        "observation.state": state.unsqueeze(0), "task": [task],
        "observation.images.image": torch.from_numpy(np.ascontiguousarray(
            images["image"])).permute(2, 0, 1).unsqueeze(0),
        "observation.images.image2": torch.from_numpy(np.ascontiguousarray(
            images["wrist_image"])).permute(2, 0, 1).unsqueeze(0),
    }


def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing

    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing

    return {"image": img, "wrist_image": wrist_img}


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
