# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/runtime/conftest.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

import numpy as np
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf


class RecordingPolicy:
    def __init__(self, dim):
        self.dim, self.observations, self.instructions = dim, [], []
        self.metrics = {}
        self.closed = False

    def reset(self, instruction=""):
        self.instructions.append(instruction)

    def act(self, observation):
        self.observations.append(observation)
        return np.arange(self.dim, dtype=np.float32)

    def close(self):
        self.closed = True


def config(platform):
    # Only LeRobot-boundary tests need the optional deployment dependency.
    from lerobot.configs.types import FeatureType, PolicyFeature
    from longwam.runtime.lerobot import LongWAMConfig

    robot = platform in {"yam", "franka"}
    state_dim = 14 if platform in {"yam", "robotwin2"} else 8
    action_dim = 14 if state_dim == 14 else 7
    cameras = {
        "libero": ["image", "image2"],
        "robotwin2": ["head_camera", "left_camera", "right_camera"],
        "yam": ["top", "left_wrist", "right_wrist"],
        "franka": ["front", "wrist"],
    }[platform]
    settings = {"robot": platform if robot else None,
                "benchmark": None if robot else platform,
                "replan_steps": 24, "action_horizon": 32}
    if platform == "yam":
        settings["adapter_options"] = {"action_names": [f"j{i}" for i in range(14)]}
    elif platform == "franka":
        settings["adapter_options"] = {"action_mode": "cartesian_delta",
                                      "position_scale": 0.05, "rotation_scale": 0.5,
                                      "gripper_open_width": 0.08}
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))}
    inputs.update({f"observation.images.{key}": PolicyFeature(FeatureType.VISUAL, (3, 4, 5))
                   for key in cameras})
    return LongWAMConfig(runtime_settings=settings, input_features=inputs,
                        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
                        device="cpu", push_to_hub=False)


def batch(cfg):
    result = {key: torch.zeros((1, *feature.shape)) for key, feature in cfg.input_features.items()}
    # Distinct pixels detect orientation/channel/layout changes.
    for key in cfg.image_features:
        result[key] = torch.arange(60, dtype=torch.uint8).reshape(1, 3, 4, 5)
    result["task"] = ["pick up the object"]
    return result


def yam_observation(step=0):
    return {**{name: np.zeros((12, 16, 3), np.uint8) for name in (
        "top", "left_wrist", "right_wrist"
    )}, "state": np.full(14, step, np.float32)}


def franka_observation():
    return {
        "front": np.zeros((12, 16, 3), np.uint8),
        "wrist": np.zeros((12, 16, 3), np.uint8),
        "eef_position": [0.1, 0.2, 0.3], "eef_quaternion": [0, 0, 0, 1],
        "gripper_width": 0.04,
    }


def fake_robot_initializer(settings):
    """Install fake model components inside the actual spawned GPU-handler path."""
    import longwam.runtime.policy as real
    from longwam.runtime.policy import initialize_policy

    def components(settings):
        run = OmegaConf.create({"data": {"train": {"past_obs_size": 8, "action_video_freq_ratio": 1}}})
        model = SimpleNamespace(
            num_imagine_frames=2,
            configure_inference_optimizations=lambda options: None,
            prepare_inference_optimizations=lambda: None,
        )
        def image(obs):
            return torch.as_tensor(next(iter(obs["images"].values()))).permute(2, 0, 1).float()
        def infer(obs, instruction, frames, clean):
            anchor = int(obs["state"][0])
            dim = 14 if settings.robot == "yam" else 7
            return np.repeat((np.arange(anchor, anchor + settings.action_horizon) / 100)[:, None], dim, axis=1)
        return run, model, image, infer

    real.load_robot_components = components
    return initialize_policy(settings)
