# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/runtime/test_real_robot.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
# Changes: Added Unitree G1 inference/deployment integration while retaining the shared runtime.

"""Real-robot data contracts and shared scheduling, without driving hardware."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from longwam.runtime.robots import YAMAdapter, FrankaAdapter, G1Adapter, make_adapter
from longwam.runtime.execution import AsyncPolicy, SyncPolicy, Schedule, SyncSchedule
from longwam.runtime.worker import SpawnedInferenceProcess
from longwam.runtime.settings import normalize_settings
from conftest import yam_observation, franka_observation


def test_yam_mapping_preserves_checkpoint_order_and_supports_named_features():
    names = [f"command_{i}" for i in range(14)]
    adapter = YAMAdapter(action_names=names)
    obs = adapter.observation(yam_observation())
    assert list(obs["images"]) == ["top", "left_wrist", "right_wrist"]
    assert adapter.action(np.arange(14)) == dict(zip(names, map(float, range(14))))
    named = YAMAdapter(action_names=names, state_names=names)
    raw = {**yam_observation(), **dict(zip(names, range(14)))}
    np.testing.assert_array_equal(named.observation(raw)["state"], np.arange(14))
    flat = {f"observation.images.{k}": v for k, v in obs["images"].items()}
    flat["observation.state"] = obs["state"]
    np.testing.assert_array_equal(adapter.observation(flat)["state"], obs["state"])


def test_franka_pose_units_quaternion_order_and_gripper_conversion():
    adapter = FrankaAdapter(
        action_mode="cartesian_delta", position_scale=0.05,
        rotation_scale=0.5, gripper_open_width=0.08,
    )
    obs = franka_observation()
    obs["eef_quaternion"] = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    state = adapter.observation(obs)["state"]
    np.testing.assert_allclose(state, [0.1, 0.2, 0.3, 0, 0, np.pi / 2, 0.02, 0.02], atol=1e-6)
    command = adapter.action([1, 0, -1, 0, 1, -1, 0.25])
    np.testing.assert_allclose(command["delta_position"], [0.05, 0, -0.05])
    np.testing.assert_allclose(command["delta_rotation"], [0, 0.5, -0.5])
    assert command["gripper_width"] == pytest.approx(0.02)
    assert adapter.action([0]*6 + [2])["gripper_width"] == pytest.approx(0.08)


def test_franka_joint_mode_does_not_reinterpret_as_cartesian():
    adapter = FrankaAdapter(action_mode="joint_position", action_names=[f"joint_{i}" for i in range(8)])
    raw = franka_observation()
    raw["joint_positions"] = np.arange(7) / 10
    np.testing.assert_allclose(adapter.observation(raw)["state"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04])
    assert adapter.action(np.arange(8))["joint_6"] == 6


@pytest.mark.parametrize("bad", [np.zeros(13), np.full(14, np.nan)])
def test_yam_rejects_invalid_action_vectors(bad):
    with pytest.raises(ValueError, match="finite vector"):
        YAMAdapter(action_names=[str(i) for i in range(14)]).action(bad)


def test_missing_units_cameras_and_invalid_quaternion_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        FrankaAdapter(action_mode="cartesian_delta")
    with pytest.raises(ValueError, match="distinct"):
        YAMAdapter(action_names=["joint"] * 14)
    adapter = FrankaAdapter(action_mode="cartesian_delta", position_scale=1, rotation_scale=1, gripper_open_width=0.08)
    raw = franka_observation()
    raw["eef_quaternion"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="zero norm"):
        adapter.observation(raw)
    raw = yam_observation()
    raw["top"] = raw["top"].astype(np.float32)
    with pytest.raises(ValueError, match="RGB uint8"):
        YAMAdapter(action_names=[str(i) for i in range(14)]).observation(raw)


def test_real_robot_settings_default_async_stride():
    cfg = normalize_settings({"robot": "yam", "benchmark": None, "replan_steps": 24,
                              "action_horizon": 32, "execution": "async"})
    assert cfg.trigger_stride == 12


@pytest.mark.parametrize("robot", ["yam", "franka", "g1_both", "g1_right"])
def test_real_processor_history_and_action_unnormalization(robot, tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import longwam.inference as inference
    from longwam.runtime.policy import load_robot_components

    data = OmegaConf.load(Path(__file__).parents[2] / "configs/data/libero.yaml").train
    is_g1 = robot.startswith("g1_")
    side = robot.removeprefix("g1_") if is_g1 else None
    dim = (16 if side == "both" else 8) if is_g1 else (14 if robot == "yam" else 7)
    state_dim = 8 if robot == "franka" else dim
    if robot == "yam" or is_g1:
        keys = tuple(G1Adapter(control_side=side).camera_keys) if is_g1 else (
            "top", "left_wrist", "right_wrist")
        data.shape_meta.images = [{"key": key, "raw_shape": [3, 224, 224], "shape": [3, 224, 224]}
                                  for key in keys]
        for category in ("action", "state"):
            data.shape_meta[category][0].shape = dim
            data.shape_meta[category][0].raw_shape = dim
        data.video_size = [384, 320]
        data.concat_multi_camera = "robotwin_right" if side == "right" else "robotwin"
        data.processor.action_output_dim = data.processor.proprio_output_dim = dim
        data.processor.num_output_cameras = len(keys)
        data.processor.delta_action_dim_mask = None
        data.processor.norm_default_mode = "z-score"
    # Resolve processor interpolations against a real run configuration.
    run = OmegaConf.create({"data": {"train": data}, "model": {"longwam_past_obs_size": 48}})
    stats = {category: {"default": {
        "global_mean": [0.25] * size, "global_std": [0.5] * size,
        "global_min": [0] * size, "global_max": [1] * size,
    }} for category, size in (("action", dim), ("state", state_dim))}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        return {"action": torch.zeros(32, dim)}

    model = SimpleNamespace(
        vae=SimpleNamespace(temporal_downsample_factor=4), num_clean_frames=4, num_imagine_frames=2,
        device="cpu", torch_dtype=torch.float32, infer_joint_ar=predict,
        encode_prompt=lambda prompt: (torch.zeros(1, 2, 4096), torch.ones(1, 2)),
    )
    monkeypatch.setattr(inference, "load_run_config", lambda *args, **kwargs: run)
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: model)
    adapter_options = ({"action_names": [str(i) for i in range(14)]} if robot == "yam" else
                       {"action_mode": "cartesian_delta", "position_scale": 1, "rotation_scale": 1, "gripper_open_width": 0.08})
    if is_g1:
        adapter_options = {"control_side": side}
    settings = OmegaConf.create({"robot": "g1" if is_g1 else robot, "adapter_options": adapter_options,
        "model_config": "unused", "checkpoint": "unused", "stats": str(stats_path),
        "device": "cpu", "seed": 0, "text_cache": None, "action_horizon": 32,
        "num_inference_steps": 4, "policy_options": {}})
    _, _, image, infer = load_robot_components(settings)
    adapter = make_adapter(settings.robot, adapter_options)
    raw = yam_observation() if robot == "yam" else franka_observation()
    if is_g1:
        raw = {"state": np.zeros(dim), **{
            key: np.full((12, 16, 3), 255 if key == "cam_high" else 128, np.uint8)
            for key in adapter.camera_keys.values()}}
    if robot == "yam":
        raw["top"].fill(255)
        raw["right_wrist"].fill(128)
    observation = adapter.observation(raw)
    frame = image(observation)
    assert tuple(frame.shape) == (3, *data.video_size)
    if robot == "yam":
        torch.testing.assert_close(frame[:, :256], torch.ones_like(frame[:, :256]))
        torch.testing.assert_close(frame[:, 256:, :160], -torch.ones_like(frame[:, 256:, :160]))
    if is_g1:
        torch.testing.assert_close(frame[:, :256], torch.ones_like(frame[:, :256]))
        torch.testing.assert_close(frame[:, 256:, 160:], torch.full_like(frame[:, 256:, 160:], 128 / 255 * 2 - 1))
        if side == "right":
            torch.testing.assert_close(frame[:, 256:, :160], -torch.ones_like(frame[:, 256:, :160]))
    actions = infer(observation, "pick up", [frame], None)
    assert actions.shape == (32, dim)
    np.testing.assert_allclose(actions, 0.5 if robot == "franka" else 0.25)
    assert calls[-1]["obs_window"].shape == (1, 3, 13, *data.video_size)
    assert calls[-1]["proprio"].shape == (1, state_dim)
    clean = torch.zeros(1, 2, 4, 3, 4)
    infer(observation, "pick up", [frame], clean)
    assert calls[-1]["clean_latents"] is clean
    assert "obs_window" not in calls[-1]
