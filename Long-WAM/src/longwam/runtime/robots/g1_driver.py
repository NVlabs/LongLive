# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation; G1 deployment contracts referenced from the source below.
# Source: https://github.com/kaiknower/Long-WAM-G1-Dynamic-Task-Deploy @ a0eda8d2f269635dfef87f7ce3f34cf94a85b777
# Changes: Inference-only adapter for the shared Long-WAM runtime; no upstream runtime or training code copied.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

"""Optional bridge to an operator-installed ial_g1d driver, not a second runtime.

The robot PC retains its SDK, camera services, calibration and watchdog.
The workstation runs this driver and the shared Long-WAM inference scheduler.
"""
import math
from pathlib import Path

from .g1 import G1Adapter, JOINT_NAMES


class G1BridgeDriver:
    def __init__(self, *, adapter, deployment=None, deployment_file=None,
                 max_arm_delta=None, max_gripper_delta=None):
        if not isinstance(adapter, G1Adapter):
            raise ValueError("G1 bridge requires the G1 adapter")
        if deployment_file:
            if deployment is not None:
                raise ValueError("Choose deployment or deployment_file, not both")
            import yaml
            with Path(deployment_file).expanduser().open() as stream:
                contents = yaml.safe_load(stream)
            if not isinstance(contents, dict):
                raise ValueError("The bridge YAML must contain a deployment mapping")
            deployment = contents.get("deployment", contents)
        if not isinstance(deployment, dict) or not deployment.get("robot_host"):
            raise ValueError("Supply driver.options.deployment with your robot_host and camera settings")
        if deployment.get("verified_end_effector") != "dex1":
            raise ValueError("This example requires a calibrated Dex1 end effector")
        if deployment.get("agv", {}).get("enabled", False):
            raise ValueError("Disable AGV: this example only exposes arm/Dex1 policy commands")
        if adapter.action_names != JOINT_NAMES[adapter.control_side]:
            raise ValueError("The ial_g1d bridge requires its canonical action_names")
        if adapter.state_names != JOINT_NAMES[adapter.control_side]:
            raise ValueError("The ial_g1d bridge requires its canonical state_names")
        self.adapter = adapter
        self.deployment = dict(deployment)
        self.limits = (max_arm_delta, max_gripper_delta)
        self.interface = None
        self.motion_started = False

    def connect(self, *, read_only=True):
        if not read_only:
            raise ValueError("Connect read-only first; motion requires a separate arm() call")
        if self.interface is not None:
            raise RuntimeError("Driver is already connected")
        try:
            from ial_g1d.robot.model import G1DModelInterface
        except ImportError as exc:
            raise ImportError("Install your reviewed ial_g1d driver separately; see infra/README.md") from exc
        self.interface = G1DModelInterface(
            self.deployment,
            control_mode="dual_arm" if self.adapter.control_side == "both" else "right_arm",
            # color_0 is the left head eye for BOTH checkpoint contracts.
            camera_control_mode="dual_arm",
            camera_names=tuple(self.adapter.camera_keys.values()),
        )
        try:
            self.interface.connect(read_only=True)
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self):
        if self.interface is None:
            raise RuntimeError("Connect the driver before reading observations")
        # The external driver rejects stale joint/camera samples.
        return self.interface.observation()

    def arm(self, *, confirm_safety_area_clear=False):
        if not confirm_safety_area_clear:
            raise PermissionError("Explicit operator safety confirmation is required")
        if self.interface is None:
            raise RuntimeError("Connect read-only before arming")
        if self.deployment.get("teleop_home_mode") is not True:
            raise ValueError("Use the reviewed fixed-column / hold-on-exit bridge mode")
        if any(v is None or isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in self.limits):
            raise ValueError("Calibrate positive max_arm_delta and max_gripper_delta before motion")
        # Mark before startup: it may move even if the network acknowledgement fails.
        self.motion_started = True
        self.interface.start(confirm_safety_area_clear=True)

    def send_action(self, command):
        if not self.motion_started:
            raise PermissionError("Driver is read-only; call arm() explicitly")
        names = JOINT_NAMES[self.adapter.control_side]
        if set(command) != set(names):
            raise ValueError("G1 command keys must exactly match the selected arms/Dex1")
        measured = self.get_observation()
        for name in names:
            target, current = float(command[name]), float(measured[name])
            limit = self.limits[1 if "gripper" in name else 0]
            if not math.isfinite(target) or not math.isfinite(current):
                raise ValueError(f"Nonfinite G1 command/state: {name}")
            if abs(target - current) > limit:
                raise ValueError(f"G1 target exceeds the calibrated feedback-relative limit: {name}")
        # No action queue, interpolation, smoothing, or second async scheduler.
        return self.interface.send_action(command)

    def stop(self):
        if self.interface is not None and self.motion_started:
            # Never issue the source client's home/exit trajectory automatically.
            self.interface.abort_hold()
            self.motion_started = False

    def disconnect(self):
        if self.interface is None:
            return
        try:
            self.stop()
        finally:
            try:
                self.interface.close(require_safe_exit=False)
            finally:
                self.interface = None


def create_driver(*, adapter, **options):
    return G1BridgeDriver(adapter=adapter, **options)
