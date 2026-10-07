# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/robotwin/adapter.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Translate LeRobot policy features to the existing RoboTwin inference contract."""
import numpy as np
import torch


def policy_batch(observation, task):
    """Bridge native RoboTwin observations to LeRobot policy features."""
    return {
        "observation.state": torch.as_tensor(
            observation["joint_action"]["vector"], dtype=torch.float32
        ).unsqueeze(0),
        "task": [task],
        **{f"observation.images.{key}": torch.from_numpy(np.ascontiguousarray(
            observation["observation"][key]["rgb"]
        )).permute(2, 0, 1).unsqueeze(0)
           for key in ("head_camera", "left_camera", "right_camera")},
    }
