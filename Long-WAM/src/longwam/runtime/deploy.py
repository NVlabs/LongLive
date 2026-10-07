# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM deployment example.
# Licensed under the Apache License, Version 2.0; see LICENSE at the repository root.

"""Portable, finite robot demo using the shared policy; no hardware writes by default."""
import argparse
import importlib
import json
import math
from pathlib import Path
import time

from omegaconf import OmegaConf

from .settings import normalize_settings
from .robots import make_adapter


def read_deployment(path, overrides=()):
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.driver.options, False)
    OmegaConf.set_struct(cfg.policy.adapter_options, False)
    OmegaConf.set_struct(cfg.policy.optimization, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.policy = normalize_settings(cfg.policy)
    if cfg.policy.get("robot") not in {"yam", "franka", "g1"}:
        raise ValueError("Deployment requires a yam, franka or g1 adapter")
    if type(cfg.max_steps) is not int or cfg.max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    return cfg


def _positive(value, name):
    if value is None or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Set a finite positive {name} for your deployment")
    return float(value)


def run_demo(policy, driver, *, task, control_hz, max_steps, enable_motion=False,
             confirm_safety_area_clear=False, max_action_age_s=None):
    """Driver owns calibrated hardware safety; runtime owns all action scheduling."""
    period = 1 / _positive(control_hz, "control_hz")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Supply the instruction used to train your task")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if enable_motion and not confirm_safety_area_clear:
        raise PermissionError("Motion also requires --confirm-safety-area-clear")
    if enable_motion:
        _positive(max_action_age_s, "max_action_age_s")
    attempted_motion = False
    last_command = None
    try:
        driver.connect(read_only=True)
        policy.reset()
        # Cold compilation and first inference happen before any motion is armed.
        policy.select_robot_action(driver.get_observation(), task=task)
        if enable_motion:
            attempted_motion = True
            driver.arm(confirm_safety_area_clear=True)
        policy.reset()  # Initialization may have changed the pose.
        for _ in range(max_steps):
            start = time.monotonic()
            observation = driver.get_observation()
            command = policy.select_robot_action(observation, task=task)
            if enable_motion:
                if time.monotonic() - start > max_action_age_s:
                    raise TimeoutError("Action is stale; stopping without sending it")
                driver.send_action(command)
            last_command = command
            time.sleep(max(0, period - (time.monotonic() - start)))
    finally:
        try:
            if attempted_motion:
                driver.stop()
        finally:
            driver.disconnect()
    return {"control_steps": max_steps, "sent_actions": max_steps if enable_motion else 0,
            "last_command": last_command}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--robot", choices=("yam", "franka", "g1"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config only: no model, network or hardware imports")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-safety-area-clear", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    cfg = read_deployment(args.config, overrides)
    if args.robot and cfg.policy.robot != args.robot:
        parser.error("The configuration robot does not match the selected target")
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("Plan only. Blank paths/calibration must be supplied before deployment.")
        return 0
    if args.enable_motion and not args.confirm_safety_area_clear:
        parser.error("--enable-motion requires --confirm-safety-area-clear")
    _positive(cfg.control_hz, "control_hz")
    if not cfg.task or not str(cfg.task).strip():
        parser.error("Supply task= with the instruction used for your own checkpoint")
    if args.enable_motion:
        _positive(cfg.max_action_age_s, "max_action_age_s")
    for key in ("checkpoint", "model_config", "stats"):
        value = cfg.policy.get(key)
        if not value or not Path(value).expanduser().is_file():
            parser.error(f"Supply policy.{key}= with your own local file")
    if not cfg.driver.factory or ":" not in cfg.driver.factory:
        parser.error("Supply driver.factory=your_module:create_driver")
    adapter = make_adapter(cfg.policy.robot, cfg.policy.adapter_options)
    module, name = cfg.driver.factory.rsplit(":", 1)
    factory = getattr(importlib.import_module(module), name)
    driver = factory(adapter=adapter, **OmegaConf.to_container(cfg.driver.options, resolve=True))
    from . import create_policy
    with create_policy(cfg.policy) as policy:
        result = run_demo(
            policy, driver, task=cfg.task, control_hz=cfg.control_hz, max_steps=cfg.max_steps,
            enable_motion=args.enable_motion,
            confirm_safety_area_clear=args.confirm_safety_area_clear,
            max_action_age_s=cfg.max_action_age_s,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
