# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM deployment example.
# Licensed under the Apache License, Version 2.0; see LICENSE at the repository root.

"""G1 inference contract and fail-closed robot demos; no hardware connections."""
from types import SimpleNamespace
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

from longwam.paths import repository_root
from longwam.runtime import deploy
from longwam.runtime.robots import G1Adapter
from longwam.runtime.robots.g1 import CAMERAS, JOINT_NAMES
from longwam.runtime.robots.g1_driver import G1BridgeDriver


@pytest.mark.parametrize("side,dim", [("both", 16), ("right", 8)])
def test_g1_wire_and_named_observations_match_without_padding(side, dim):
    adapter = G1Adapter(control_side=side)
    images = {name: np.full((4, 5, 3), i * 50, np.uint8)
              for i, name in enumerate(adapter.camera_keys.values())}
    raw = {**dict(zip(JOINT_NAMES[side], range(dim))), **images}
    packed = adapter.observation(raw)
    np.testing.assert_array_equal(packed["state"], np.arange(dim))
    wire = {"observation/state": list(range(dim)), **{
        "observation/" + key: images[value] for key, value in adapter.camera_keys.items()}}
    other = adapter.observation(wire)
    for key in packed["images"]:
        np.testing.assert_array_equal(packed["images"][key], other["images"][key])
    assert adapter.action(np.arange(dim)) == dict(zip(JOINT_NAMES[side], map(float, range(dim))))
    with pytest.raises(ValueError):
        adapter.observation(dict(wire, **{"observation/state": np.zeros(24 - dim)}))
    with pytest.raises(ValueError):
        adapter.action(np.full(dim, np.nan))
    assert not any("Hip" in name or "Waist" in name for name in adapter.action(np.zeros(dim)))


def data_config(side):
    dim = len(JOINT_NAMES[side])
    return OmegaConf.create({
        "concat_multi_camera": "robotwin" if side == "both" else "robotwin_right",
        "video_size": [384, 320],
        "shape_meta": {"images": [{"key": key, "shape": [3, 240, 320],
                                   "raw_shape": [3, 480, 640]} for key in CAMERAS[side]],
                       "state": [{"key": "default", "shape": dim, "raw_shape": dim}],
                       "action": [{"key": "default", "shape": dim, "raw_shape": dim}]}})


@pytest.mark.parametrize("side", ["both", "right"])
def test_g1_checkpoint_validation_rejects_mode_and_camera_swaps(side):
    adapter = G1Adapter(control_side=side)
    data = data_config(side)
    adapter.validate_checkpoint(data)
    data.shape_meta.images = list(reversed(data.shape_meta.images))
    with pytest.raises(ValueError, match="camera order"):
        adapter.validate_checkpoint(data)
    with pytest.raises(ValueError, match="requires .*D"):
        adapter.validate_checkpoint(data_config("right" if side == "both" else "both"))
    data = data_config(side)
    data.concat_multi_camera = "horizontal"
    with pytest.raises(ValueError, match="layout"):
        adapter.validate_checkpoint(data)
    data = data_config(side)
    data.processor = {"shape_meta": OmegaConf.to_container(data.shape_meta)}
    data.processor.shape_meta.images = list(reversed(data.processor.shape_meta.images))
    with pytest.raises(ValueError, match="camera order"):
        adapter.validate_checkpoint(data)


def test_g1_standard_lerobot_entrypoint(monkeypatch):
    pytest.importorskip("lerobot")
    import longwam.inference
    from longwam.runtime.lerobot import LongWAMPolicy
    from longwam.runtime import policy as execution
    data = data_config("right")
    data.action_chunk = 32
    calls = []
    inner = SimpleNamespace(
        reset=lambda task: None, close=lambda: None, metrics={},
        act=lambda obs: calls.append(obs) or np.arange(8, dtype=np.float32))
    monkeypatch.setattr(longwam.inference, "load_run_config",
                        lambda *args, **kwargs: SimpleNamespace(data=SimpleNamespace(train=data)))
    monkeypatch.setattr(execution, "_create_execution", lambda *args, **kwargs: inner)
    with LongWAMPolicy.from_runtime({
        "robot": "g1", "adapter_options": {"control_side": "right"},
        "model_config": "unused", "device": "cpu",
    }) as policy:
        raw = {"state": np.zeros(8), "cam_high": np.zeros((4, 5, 3), np.uint8),
               "cam_right_wrist": np.zeros((4, 5, 3), np.uint8)}
        assert tuple(policy.select_robot_action(raw, task="my task")) == JOINT_NAMES["right"]
        assert calls[-1]["state"].shape == (8,)


def driver_config():
    return {"robot_host": "robot.example", "verified_end_effector": "dex1",
            "teleop_home_mode": True, "agv": {"enabled": False}}


class FakeInterface:
    def __init__(self, *args, **kwargs):
        self.events = []
        self.kwargs = kwargs
    def connect(self, **kwargs):
        self.events.append(("connect", kwargs))
    def observation(self):
        return {key: 0.0 for key in JOINT_NAMES["both"]}
    def start(self, **kwargs):
        self.events.append("start")
    def send_action(self, value):
        self.events.append(("send", value))
        return value
    def abort_hold(self):
        self.events.append("hold")
        return True
    def close(self, **kwargs):
        self.events.append(("close", kwargs))


@pytest.mark.parametrize("side", ["both", "right"])
def test_optional_bridge_read_only_arm_limits_and_hold(side, monkeypatch):
    monkeypatch.setitem(sys.modules, "ial_g1d.robot.model",
                        SimpleNamespace(G1DModelInterface=FakeInterface))
    driver = G1BridgeDriver(adapter=G1Adapter(control_side=side), deployment=driver_config(),
                            max_arm_delta=0.1, max_gripper_delta=0.2)
    driver.connect()
    interface = driver.interface
    assert interface.events == [("connect", {"read_only": True})]
    assert interface.kwargs["camera_control_mode"] == "dual_arm"  # left head eye
    with pytest.raises(PermissionError):
        driver.send_action(dict.fromkeys(JOINT_NAMES[side], 0.0))
    driver.arm(confirm_safety_area_clear=True)
    good = dict.fromkeys(JOINT_NAMES[side], 0.05)
    assert driver.send_action(good) == good
    bad = dict(good)
    bad[JOINT_NAMES[side][0]] = 1.0
    with pytest.raises(ValueError, match="limit"):
        driver.send_action(bad)
    driver.disconnect()
    assert interface.events[-2:] == ["hold", ("close", {"require_safe_exit": False})]
    assert driver.interface is None


def test_bridge_rejects_agv_missing_calibration_and_unsafe_keys():
    adapter = G1Adapter(control_side="right")
    with pytest.raises(ValueError, match="Disable AGV"):
        G1BridgeDriver(adapter=adapter, deployment=dict(driver_config(), agv={"enabled": True}))
    driver = G1BridgeDriver(adapter=adapter, deployment=driver_config())
    driver.interface = FakeInterface()
    with pytest.raises(ValueError, match="Calibrate"):
        driver.arm(confirm_safety_area_clear=True)


class RecordingDriver:
    def __init__(self):
        self.events = []
    def connect(self, **kwargs):
        assert kwargs == {"read_only": True}
        self.events.append("connect")
    def get_observation(self):
        return {}
    def arm(self, **kwargs):
        assert kwargs["confirm_safety_area_clear"]
        self.events.append("arm")
    def send_action(self, command):
        self.events.append("send")
    def stop(self):
        self.events.append("stop")
    def disconnect(self):
        self.events.append("disconnect")


@pytest.mark.parametrize("motion", [False, True])
def test_shared_runner_warms_before_arming_and_defaults_to_no_writes(motion, monkeypatch):
    driver = RecordingDriver()
    policy = SimpleNamespace(reset=lambda: None, select_robot_action=lambda *a, **kw:
                             driver.events.append("infer") or {"joint": 0.0})
    monkeypatch.setattr(deploy.time, "sleep", lambda value: None)
    result = deploy.run_demo(policy, driver, task="my task", control_hz=30, max_steps=2,
                             enable_motion=motion, confirm_safety_area_clear=motion,
                             max_action_age_s=1.0)
    assert driver.events[:2] == ["connect", "infer"]
    assert result["sent_actions"] == (2 if motion else 0)
    if motion:
        assert driver.events[2] == "arm"
        assert driver.events[-2:] == ["stop", "disconnect"]
    else:
        assert "arm" not in driver.events and "send" not in driver.events


def test_runner_rejects_stale_action_and_always_stops(monkeypatch):
    driver = RecordingDriver()
    policy = SimpleNamespace(reset=lambda: None, select_robot_action=lambda *a, **kw: {})
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(deploy.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="stale"):
        deploy.run_demo(policy, driver, task="my task", control_hz=30, max_steps=1,
                        enable_motion=True, confirm_safety_area_clear=True, max_action_age_s=0.5)
    assert "send" not in driver.events
    assert driver.events[-2:] == ["stop", "disconnect"]


def test_partial_arm_failure_still_holds_and_disconnects():
    driver = RecordingDriver()
    def fail(**kwargs):
        driver.events.append("partial-arm")
        raise RuntimeError("startup acknowledgement lost")
    driver.arm = fail
    policy = SimpleNamespace(reset=lambda: None, select_robot_action=lambda *a, **kw: {})
    with pytest.raises(RuntimeError, match="acknowledgement"):
        deploy.run_demo(policy, driver, task="my task", control_hz=30, max_steps=1,
                        enable_motion=True, confirm_safety_area_clear=True, max_action_age_s=0.5)
    assert driver.events[-3:] == ["partial-arm", "stop", "disconnect"]


def test_failed_hold_does_not_skip_disconnect():
    driver = RecordingDriver()
    def fail():
        raise RuntimeError("hold acknowledgement lost")
    driver.stop = fail
    policy = SimpleNamespace(reset=lambda: None, select_robot_action=lambda *a, **kw: {})
    with pytest.raises(RuntimeError, match="hold acknowledgement"):
        deploy.run_demo(policy, driver, task="my task", control_hz=1000, max_steps=1,
                        enable_motion=True, confirm_safety_area_clear=True, max_action_age_s=1.0)
    assert driver.events[-1] == "disconnect"


@pytest.mark.parametrize("robot", ["g1", "yam", "franka"])
def test_deployment_templates_have_no_checkpoint_default(robot, capsys):
    path = repository_root() / f"configs/deploy/{robot}.yaml"
    cfg = deploy.read_deployment(path)
    assert cfg.policy.robot == robot
    assert all(cfg.policy[key] is None for key in ("checkpoint", "model_config", "stats"))
    assert deploy.main(["--config", str(path), "--dry-run"]) == 0
    assert "Plan only" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        deploy.main(["--config", str(path), "--enable-motion"])
