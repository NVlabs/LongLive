# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/runtime/test_lerobot_policy.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""LeRobot boundary tests using the real upstream base classes when installed."""
from pathlib import Path
from types import SimpleNamespace
import json
from omegaconf import OmegaConf
import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from longwam.runtime.lerobot import LongWAMConfig, LongWAMPolicy
from conftest import config, batch, RecordingPolicy, yam_observation, franka_observation, fake_robot_initializer


@pytest.mark.parametrize("platform", ["libero", "robotwin2", "yam", "franka"])
def test_standard_policy_maps_features_without_second_action_queue(platform):
    cfg = config(platform)
    inner = RecordingPolicy(cfg.output_features["action"].shape[0])
    with LongWAMPolicy(cfg, policy=inner) as policy:
        assert isinstance(policy, PreTrainedPolicy)
        data = batch(cfg)
        for _ in range(3):
            action = policy.select_action(data)
            assert action.shape == (1, inner.dim)
        assert len(inner.observations) == 3  # every control step reaches video history
        assert inner.instructions == ["pick up the object"]
        obs = inner.observations[0]
        expected = data[next(iter(cfg.image_features))][0].permute(1, 2, 0).numpy()
        if platform == "libero":
            np.testing.assert_array_equal(obs["images"]["image"], expected)
            np.testing.assert_array_equal(obs["state"], np.zeros(8))
        elif platform == "robotwin2":
            np.testing.assert_array_equal(obs["images"]["cam_high"], expected)
        else:
            np.testing.assert_array_equal(next(iter(obs["images"].values())), expected)
            command = policy.robot_action(action)
            assert ("j0" if platform == "yam" else "delta_position.x") in command
            assert all(isinstance(value, float) for value in command.values())
        policy.reset()
        policy.select_action(data)
        assert inner.instructions == ["pick up the object", "pick up the object"]
    assert inner.closed


def test_float_images_roundtrip_and_invalid_external_scheduling(tmp_path):
    cfg = config("libero")
    inner = RecordingPolicy(7)
    with LongWAMPolicy(cfg, policy=inner) as policy:
        data = batch(cfg)
        for key in cfg.image_features:
            data[key] = data[key].float() / 255
        policy.select_action(data)
        np.testing.assert_array_equal(inner.observations[0]["images"]["image"],
                                      torch.arange(60).reshape(3, 4, 5).permute(1, 2, 0).numpy())
        changed = dict(data, task=["another task"])
        with pytest.raises(ValueError, match="reset"):
            policy.select_action(changed)
        with pytest.raises(ValueError, match="RTC"):
            policy.select_action(data, inference_delay=1)
        with pytest.raises(NotImplementedError, match="select_action"):
            policy.predict_action_chunk(data)
        with pytest.raises(NotImplementedError, match="checkpoint"):
            policy.save_pretrained(tmp_path / "wrapper")


@pytest.mark.parametrize("problem", ["batch", "image", "state", "missing"])
def test_bad_features_rejected_before_advancing_runtime(problem):
    cfg = config("yam")
    inner = RecordingPolicy(14)
    with LongWAMPolicy(cfg, policy=inner) as policy:
        data = batch(cfg)
        if problem == "batch":
            data["observation.state"] = torch.zeros(2, 14)
        elif problem == "image":
            data[next(iter(cfg.image_features))] = torch.full((1, 3, 4, 5), -1.)
        elif problem == "state":
            data["observation.state"][0, 0] = float("nan")
        else:
            del data["observation.state"]
        with pytest.raises(ValueError):
            policy.select_action(data)
        assert not inner.observations


@pytest.mark.parametrize("platform", ["yam", "franka"])
@pytest.mark.parametrize("execution", ["sync", "async"])
def test_standard_interface_preserves_spawned_worker_handoff(platform, execution):
    from longwam.runtime.worker import SpawnedInferenceProcess
    from longwam.runtime.execution import SyncPolicy, AsyncPolicy, SyncSchedule, Schedule

    cfg = config(platform)
    cfg.runtime_settings.update(execution=execution, action_horizon=8, execute_steps=6,
                                trigger_stride=4 if execution == "async" else None,
                                streaming_vae=False, hardware="reference", optimization={})
    worker = SpawnedInferenceProcess(fake_robot_initializer, cfg.runtime_settings,
                                     name="lerobot-boundary-test", startup_timeout_s=30)
    schedule = Schedule(8, 6, 4) if execution == "async" else SyncSchedule(8, 6)
    inner = (AsyncPolicy if execution == "async" else SyncPolicy)(worker, schedule)
    with LongWAMPolicy(cfg, policy=inner) as policy:
        for step in range(25):
            data = batch(cfg)
            data["observation.state"].fill_(step)
            action = policy.select_action(data)
            torch.testing.assert_close(action, torch.full((1, 14 if platform == "yam" else 7), step / 100))
        policy.reset()
        action = policy.select_action(batch(cfg))
        torch.testing.assert_close(action, torch.zeros(1, 14 if platform == "yam" else 7))


def test_from_runtime_uses_checkpoint_features_without_loading_weights(monkeypatch):
    from omegaconf import OmegaConf
    import longwam.inference as inference
    import longwam.runtime.policy as runtime

    run = OmegaConf.create({"data": {"train": {"action_chunk": 32, "shape_meta": {
        "images": [{"key": key, "shape": [3, 4, 5], "raw_shape": [3, 8, 10]}
                   for key in ("image", "wrist_image")],
        "state": [{"shape": 8, "raw_shape": 8}],
        "action": [{"shape": 7, "raw_shape": 7}],
    }}}})
    monkeypatch.setattr(inference, "load_run_config", lambda *args, **kwargs: run)
    inner = RecordingPolicy(7)
    monkeypatch.setattr(runtime, "_create_execution", lambda settings, **kwargs: inner)
    with LongWAMPolicy.from_runtime({"benchmark": "libero", "model_config": "unused",
                                     "execute_steps": 24, "device": "cpu"}) as policy:
        assert policy.config.image_features["observation.images.image2"].shape == (3, 8, 10)
        assert policy.settings.action_horizon == 32
        assert policy.select_action(batch(policy.config)).shape == (1, 7)


@pytest.mark.parametrize("platform", ["yam", "franka"])
def test_robot_driver_bridge_uses_named_scalar_commands(platform):
    cfg = config(platform)
    with LongWAMPolicy(cfg, policy=RecordingPolicy(14 if platform == "yam" else 7)) as policy:
        raw = yam_observation() if platform == "yam" else franka_observation()
        command = policy.select_robot_action(raw, task="pick")
        assert len(command) == (14 if platform == "yam" else 7)
        assert all(isinstance(value, float) for value in command.values())


def test_only_one_public_control_factory():
    import longwam.runtime as runtime
    assert callable(runtime.create_policy)
    for obsolete in ("create_robot_policy", "create_lerobot_policy", "RobotPolicy"):
        assert not hasattr(runtime, obsolete)


def test_robotwin_components_keep_preprocessing_and_action_units(tmp_path, monkeypatch):
    import longwam.inference as inference
    from longwam.runtime.robotwin import RobotWinInference
    from longwam.runtime.settings import normalize_settings
    from longwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    data = OmegaConf.load(Path(__file__).parents[2] / "configs/data/robotwin2.yaml").train
    run = OmegaConf.create({"data": {"train": data}})
    stats = {category: {"default": {
        "global_mean": [0.25] * 14, "global_std": [0.5] * 14,
        "global_min": [0] * 14, "global_max": [1] * 14,
    }} for category in ("state", "action")}
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    calls = []
    def infer(**kwargs):
        calls.append(kwargs)
        return {"action": torch.zeros(32, 14)}
    model = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32,
                            vae=SimpleNamespace(temporal_downsample_factor=4),
                            num_clean_frames=4, num_imagine_frames=2, infer_joint_ar=infer,
                            encode_prompt=lambda prompt: (torch.zeros(1, 2, 4096), torch.ones(1, 2)))
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: model)
    monkeypatch.delenv("LONGWAM_VIDEO_LORA_PATH", raising=False)
    settings = normalize_settings({"benchmark": "robotwin2", "checkpoint": "unused",
                                   "stats": str(path), "device": "cpu",
                                   "action_horizon": 32, "execute_steps": 24})
    component = RobotWinInference(run, settings)
    observation = {"joint_action": {"vector": np.full(14, 0.25)}, "observation": {
        key: {"rgb": np.full((8, 10, 3), value, np.uint8)}
        for key, value in zip(("head_camera", "left_camera", "right_camera"), (0, 128, 255))
    }}
    frame = component._build_robotwin_image_tensor(observation)
    assert frame.shape == (1, 3, 384, 320)
    torch.testing.assert_close(frame[0, :, 0, 0], torch.full((3,), -1.))
    torch.testing.assert_close(frame[0, :, -1, -1], torch.ones(3))
    actions = component._infer_action_chunk(observation, "pick", obs_frame_buffer=[frame[0]])
    np.testing.assert_allclose(actions, 0.25)
    torch.testing.assert_close(calls[0]["proprio"], torch.zeros(1, 14))
    assert calls[0]["obs_window"].shape[2] == 13


def test_libero_diagnostic_uses_shared_scheduler_and_records_every_step(monkeypatch):
    pytest.importorskip("libero")
    from longwam.benchmarks.libero import eval_libero_single as evaluator
    from longwam.runtime.execution import SyncPolicy
    import longwam.runtime.backends as backends

    predictions, frames, selected = [], [], []
    monkeypatch.setattr(backends, "prepare_backend", lambda *args, **kwargs: None)
    def image(obs, *args, **kwargs):
        return torch.full((1, 3, 4, 5), float(obs["control"])), None, None
    monkeypatch.setattr(evaluator, "_obs_to_model_input", image)
    def predict(obs, instruction, model, processor, cfg, **kwargs):
        predictions.append(obs["control"])
        frames.append([int(frame[0, 0, 0]) for frame in kwargs["obs_frame_buffer"]])
        actions = np.repeat(np.arange(obs["control"], obs["control"] + 4)[:, None], 7, axis=1)
        return actions, {}, None
    monkeypatch.setattr(evaluator, "_predict_action_chunk", predict)
    monkeypatch.setattr(evaluator, "get_libero_image", lambda obs: {})
    original = SyncPolicy.act
    def act(self, observation):
        selected.append(observation["control"])
        return original(self, observation)
    monkeypatch.setattr(SyncPolicy, "act", act)
    class Env:
        def reset(self): pass
        def set_init_state(self, state):
            self.control = 0
            return {"control": 0}
        def step(self, action):
            self.control += 1
            return {"control": self.control}, 0, self.control == 7, {}
    cfg = OmegaConf.create({"EVALUATION": {"task_suite_name": "libero_spatial",
                                           "replan_steps": 3, "num_steps_wait": 2},
                            "data": {"train": {"past_obs_size": 4, "action_video_freq_ratio": 1}}})
    success, _, _, _ = evaluator.run_single_episode(
        Env(), None, "pick", SimpleNamespace(torch_dtype=torch.float32), None, cfg, 0,
        action_horizon=4, input_w=5, input_h=4, model_device="cpu", show_progress=False,
        record_replay=False, max_steps_override=10,
    )
    assert success
    assert selected == [2, 3, 4, 5, 6]
    assert predictions == [2, 5]
    assert frames == [[2], [2, 3, 4, 5]]


def test_robotwin_native_callback_delegates_to_standard_policy():
    from longwam.benchmarks.robotwin.longwam_policy import deploy_policy
    observed, executed = [], []
    model = SimpleNamespace(select_action=lambda batch: observed.append(batch) or torch.ones(1, 14))
    env = SimpleNamespace(get_instruction=lambda: "pick",
                          take_action=lambda action, **kwargs: executed.append((action, kwargs)))
    obs = {"joint_action": {"vector": np.zeros(14)}, "observation": {
        key: {"rgb": np.zeros((4, 5, 3), np.uint8)}
        for key in ("head_camera", "left_camera", "right_camera")
    }}
    deploy_policy.eval(env, model, obs)
    assert observed[0]["task"] == ["pick"]
    np.testing.assert_array_equal(executed[0][0], np.ones(14))
    assert executed[0][1] == {"action_type": "qpos"}


def test_libero_standard_bridge_preserves_native_pose_and_camera_orientation():
    pytest.importorskip("libero")
    from longwam.benchmarks.libero.adapter import policy_batch
    from longwam.benchmarks.libero.libero_utils import quat2axisangle
    from longwam.runtime.lerobot import LongWAMPolicy
    quaternion = np.asarray([0.1, -0.2, 0.3, -0.8], np.float32)
    quaternion /= np.linalg.norm(quaternion)
    raw = {"agentview_image": np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
           "robot0_eye_in_hand_image": np.arange(60, 120, dtype=np.uint8).reshape(4, 5, 3),
           "robot0_eef_pos": np.array([0.1, 0.2, 0.3], np.float32),
           "robot0_eef_quat": quaternion,
           "robot0_gripper_qpos": np.array([0.01, 0.02], np.float32)}
    inner = RecordingPolicy(7)
    with LongWAMPolicy(config("libero"), policy=inner) as policy:
        policy.select_action(policy_batch(raw, "pick"))
    converted = inner.observations[0]
    for model_key, native in (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image")):
        np.testing.assert_array_equal(converted["images"][model_key], raw[native][::-1, ::-1])
    np.testing.assert_allclose(converted["state"][3:6],
                               quat2axisangle(raw["robot0_eef_quat"]), atol=1e-6)


def test_generic_lerobot_policy_needs_no_platform_adapter():
    cfg = config('libero')
    cfg.runtime_settings['benchmark'] = None
    cfg.camera_keys = {'camera_a': 'image', 'camera_b': 'image2'}
    cfg.input_features['observation.state'] = PolicyFeature(FeatureType.STATE, (6,))
    cfg.output_features['action'] = PolicyFeature(FeatureType.ACTION, (5,))
    inner = RecordingPolicy(5)
    with LongWAMPolicy(cfg, policy=inner) as policy:
        assert policy.select_action(batch(cfg)).shape == (1, 5)
        assert set(inner.observations[0]['images']) == {'camera_a', 'camera_b'}
        assert policy.adapter is None
