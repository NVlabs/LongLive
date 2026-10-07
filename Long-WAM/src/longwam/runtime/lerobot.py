# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/lerobot.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
# Changes: Added Unitree G1 inference/deployment integration while retaining the shared runtime.

"""The LeRobot policy API over the shared ordered inference runtime.

Inputs are unnormalized LeRobot features. Checkpoint processors stay in the
inference worker; do not normalize them a second time in an outer pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pretrained import PreTrainedPolicy
except ModuleNotFoundError as exc:
    if exc.name == "lerobot":
        raise ImportError(
            "Install the deployment extra in a separate inference environment: pip install -e '.[infra]'"
        ) from exc
    raise


@PreTrainedConfig.register_subclass("longwam_runtime")
@dataclass
class LongWAMConfig(PreTrainedConfig):
    runtime_settings: dict = field(default_factory=dict)
    camera_keys: dict = field(default_factory=dict)

    @property
    def observation_delta_indices(self):
        return None  # The worker owns video history, updated every control step.

    @property
    def action_delta_indices(self):
        return list(range(self.runtime_settings.get("action_horizon") or 32))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("Inference wrapper; train with longwam train")

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        if "observation.state" not in self.input_features:
            raise ValueError("Declare observation.state and camera input_features")
        if not self.image_features or "action" not in self.output_features:
            raise ValueError("Declare camera input_features and action output_features")


def _observation(batch):
    """Unbatch one policy observation; preserve uint8 RGB or decode float [0,1]."""
    raw = {}
    for key, value in batch.items():
        if not key.startswith("observation."):
            continue
        value = torch.as_tensor(value).detach().cpu()
        if value.ndim < 1 or value.shape[0] != 1:
            raise ValueError("LongWAM runtime supports batch_size=1; use one policy per environment")
        value = value[0]
        if key.startswith("observation.images."):
            if value.ndim != 3 or value.shape[0] != 3 or min(value.shape[1:]) == 0:
                raise ValueError("Policy images must have shape [1,3,H,W]")
            if value.dtype != torch.uint8:
                if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                    raise ValueError("Policy images must be uint8 or unnormalized floats in [0,1]")
                value = value.mul(255).round().to(torch.uint8)
            value = value.permute(1, 2, 0)
        elif not torch.isfinite(value).all():
            raise ValueError("Observation features must be finite")
        raw[key] = np.ascontiguousarray(value.numpy())
    return raw


class LongWAMPolicy(PreTrainedPolicy):
    """LeRobot reset/select_action contract; pure async remains inside the runtime."""
    config_class = LongWAMConfig
    name = "longwam_runtime"

    def __init__(self, config, *, policy=None):
        super().__init__(config)
        config.validate_features()
        from .settings import normalize_settings
        from .robots import make_adapter
        from .policy import _create_execution

        self.settings = normalize_settings(config.runtime_settings)
        if self.settings.get("robot") and config.camera_keys:
            self.settings.adapter_options.camera_keys = config.camera_keys
        self.adapter = (
            make_adapter(self.settings.robot, self.settings.adapter_options)
            if self.settings.get("robot") else None
        )
        if not self.adapter and self.settings.get("benchmark") not in {None, "libero", "robotwin2", "domino"}:
            raise ValueError("LeRobot wrapper supports LIBERO, RoboTwin, YAM, Franka and G1")
        state_dim, action_dim = (
            (self.adapter.state_dim, self.adapter.action_dim) if self.adapter else
            ((8, 7) if self.settings.get("benchmark") == "libero" else (14, 14))
            if self.settings.get("benchmark") else
            (config.input_features["observation.state"].shape[0], config.output_features["action"].shape[0])
        )
        if tuple(config.input_features["observation.state"].shape) != (state_dim,) or tuple(
            config.output_features["action"].shape
        ) != (action_dim,):
            raise ValueError("Policy features do not match the platform state/action dimensions")
        defaults = {
            "libero": {"image": "image", "wrist_image": "image2"},
            "robotwin2": {"cam_high": "head_camera", "cam_left_wrist": "left_camera",
                          "cam_right_wrist": "right_camera"},
            "domino": {"cam_high": "head_camera", "cam_left_wrist": "left_camera",
                       "cam_right_wrist": "right_camera"},
        }
        self.camera_keys = dict(config.camera_keys or (
            self.adapter.camera_keys if self.adapter else defaults.get(self.settings.get("benchmark"),
                {key.removeprefix("observation.images."): key.removeprefix("observation.images.")
                 for key in config.image_features})
        ))
        if set(self.camera_keys.values()) != {
            key.removeprefix("observation.images.") for key in config.image_features
        }:
            raise ValueError("camera_keys must cover each policy camera exactly once")
        self.runtime = policy if policy is not None else _create_execution(self.settings, normalized=True)
        self._instruction = None

    @classmethod
    def from_runtime(cls, settings, *, camera_keys=None):
        """Read checkpoint feature metadata; retain the original checkpoint/config/stats."""
        from omegaconf import OmegaConf
        from longwam.inference import load_run_config

        settings = OmegaConf.to_container(OmegaConf.create(settings), resolve=True)
        domain = settings.get("domain")
        if domain is None and settings.get("benchmark") in {"robotwin2", "domino"}:
            domain = "domino" if settings["benchmark"] == "domino" else "robotwin"
        data = load_run_config(settings["model_config"], domain=domain).data.train
        if settings.get("robot") == "g1":
            from .robots import make_adapter
            make_adapter("g1", settings.get("adapter_options", {})).validate_checkpoint(data)
        if settings.get("action_horizon") is None:
            settings["action_horizon"] = int(data.action_chunk)
        keys = camera_keys or settings.get("adapter_options", {}).get("camera_keys")
        if not keys:
            if settings.get("benchmark") == "libero":
                keys = {"image": "image", "wrist_image": "image2"}
            elif settings.get("benchmark") in {"robotwin2", "domino"}:
                keys = {"cam_high": "head_camera", "cam_left_wrist": "left_camera",
                        "cam_right_wrist": "right_camera"}
            elif settings.get("robot") == "yam":
                keys = {k: k for k in ("top", "left_wrist", "right_wrist")}
            elif settings.get("robot") == "franka":
                keys = {"image": "front", "wrist_image": "wrist"}
            elif settings.get("robot") == "g1":
                from .robots import make_adapter
                keys = make_adapter("g1", settings.get("adapter_options", {})).camera_keys
            else:
                keys = {meta.key: meta.key for meta in data.shape_meta.images}
        if set(keys) != {meta.key for meta in data.shape_meta.images} or len(set(keys.values())) != len(keys):
            raise ValueError("camera_keys must map every checkpoint camera to a unique policy feature")
        inputs = {
            f"observation.images.{key}": PolicyFeature(
                type=FeatureType.VISUAL, shape=tuple(meta.get("raw_shape", meta.shape))
            ) for meta in data.shape_meta.images for key in [keys[meta.key]]
        }
        def dimension(meta):
            size = meta.get("raw_shape", meta.shape)
            return (int(size),) if isinstance(size, int) else tuple(size)
        inputs["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=dimension(data.shape_meta.state[0])
        )
        outputs = {"action": PolicyFeature(
            type=FeatureType.ACTION, shape=dimension(data.shape_meta.action[0])
        )}
        return cls(LongWAMConfig(runtime_settings=settings, camera_keys=dict(keys),
                                input_features=inputs, output_features=outputs,
                                device=settings.get("device", "cuda"), push_to_hub=False))

    def reset(self):
        # The next observation supplies the task and sends one ordered reset.
        self._instruction = None

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        if kwargs:
            raise ValueError("External noise/RTC overrides are unsupported; configure the runtime")
        task = batch.get("task", "")
        if isinstance(task, (list, tuple)):
            if len(task) != 1:
                raise ValueError("One task is required for batch_size=1")
            task = task[0]
        if not isinstance(task, str):
            raise TypeError("task must be a string or a one-element list")
        raw = _observation(batch)
        for key, feature in self.config.input_features.items():
            if key not in raw:
                raise ValueError(f"Missing policy feature: {key}")
            if feature.type != FeatureType.VISUAL and raw[key].shape != feature.shape:
                raise ValueError(f"Incorrect feature shape: {key}")
        obs = {"state": raw["observation.state"], "images": {
            model_key: raw[f"observation.images.{feature}"]
            for model_key, feature in self.camera_keys.items()
        }}
        return torch.from_numpy(self._select(obs, task)).unsqueeze(0)

    def _select(self, observation, task):
        if self._instruction is None:
            self.runtime.reset(task)
            self._instruction = task
        elif task != self._instruction:
            raise ValueError("Call reset() before changing the task")
        action = np.asarray(self.runtime.act(observation), dtype=np.float32)
        if action.shape != self.config.output_features["action"].shape or not np.isfinite(action).all():
            raise ValueError("Runtime returned an invalid action feature")
        return action

    def robot_action(self, action):
        """Convert the returned [1,D] tensor into calibrated driver commands."""
        if self.adapter is None:
            raise ValueError("robot_action is only defined for real robots")
        value = torch.as_tensor(action).detach().cpu()
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError("Expected one action with shape [1,D]")
        convert = getattr(self.adapter, "lerobot_action", self.adapter.action)
        return convert(value[0].numpy())

    def select_robot_action(self, observation, *, task=""):
        """Convenience bridge from Robot.get_observation to Robot.send_action."""
        if self.adapter is None:
            raise ValueError("select_robot_action requires a robot adapter")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        packed = self.adapter.observation(observation)
        action = self._select(packed, task)
        convert = getattr(self.adapter, "lerobot_action", self.adapter.action)
        return convert(action)

    def predict_action_chunk(self, batch, **kwargs):
        raise NotImplementedError("Use select_action each control step; runtime owns chunk scheduling")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("Use from_runtime with the original checkpoint/config/stats")

    def forward(self, batch, **kwargs):
        raise NotImplementedError("Inference wrapper; train with longwam train")

    def get_optim_params(self):
        raise NotImplementedError("Inference wrapper; train with longwam train")

    def _save_pretrained(self, save_directory: Path):
        raise NotImplementedError("Keep the original checkpoint/config/stats; this wrapper has no weights")

    @property
    def metrics(self):
        return self.runtime.metrics

    def close(self):
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
