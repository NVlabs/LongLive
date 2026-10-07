# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/settings.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
# Changes: Added Unitree G1 inference/deployment integration while retaining the shared runtime.

"""Execution settings shared by evaluation and direct policy construction."""

from omegaconf import OmegaConf

from .execution import Schedule


def normalize_settings(settings):
    cfg = OmegaConf.merge(
        {
            "execution": "sync",
            "action_horizon": None,
            "execute_steps": None,
            "trigger_stride": None,
            "streaming_vae": False,
            "hardware": "reference",
            "optimization": {},
            "device": "cuda",
            "seed": 0,
            "num_inference_steps": 10,
            "text_cache": None,
            "policy_options": {},
            "replan_steps": 24,
        },
        settings,
    )
    if not OmegaConf.is_dict(cfg.optimization):
        raise TypeError("optimization must be a mapping")
    if cfg.execution not in {"sync", "async"}:
        raise ValueError("execution must be sync or async (pure async, without blending)")
    if cfg.hardware not in {"reference", "rtx5090", "spark", "thor"}:
        raise ValueError("hardware must be reference, rtx5090, spark or thor")
    if type(cfg.streaming_vae) is not bool:
        raise TypeError("streaming_vae must be true or false")
    robot = cfg.get("robot")
    if robot is not None and robot not in {"yam", "franka", "g1"}:
        raise ValueError("robot must be yam, franka or g1")
    supported = cfg.get("benchmark") in {None, "libero", "robotwin2", "domino"} or robot is not None
    if cfg.get("benchmark") == "domino" and (
        cfg.execution != "sync" or cfg.hardware != "reference"
        or cfg.streaming_vae or cfg.optimization
    ):
        raise ValueError("Domino currently uses the shared sync reference runtime")
    if not supported and (
        cfg.execution != "sync"
        or cfg.hardware != "reference"
        or cfg.streaming_vae
        or cfg.optimization
    ):
        raise ValueError(
            "Async, streaming VAE and edge backends currently support "
            "LIBERO/RoboTwin and YAM/Franka/G1 adapters"
        )
    if cfg.hardware == "reference" and cfg.optimization:
        raise ValueError("Reference backend takes no optimization overrides")
    if not supported:
        return cfg
    execute = cfg.get("execute_steps")
    if execute is None:
        execute = cfg.replan_steps
    if type(execute) is not int or execute <= 0:
        raise ValueError("execute_steps must be a positive integer")
    cfg.execute_steps = cfg.replan_steps = execute
    horizon = cfg.get("action_horizon")
    if horizon is not None and (type(horizon) is not int or horizon < execute):
        raise ValueError("action_horizon must be an integer >= execute_steps")
    if cfg.hardware != "reference" and horizon is not None and horizon != 32:
        raise ValueError("Edge profiles require action_horizon=32")
    if cfg.execution == "async":
        if cfg.trigger_stride is None:
            cfg.trigger_stride = (execute + 1) // 2
        Schedule(horizon if horizon is not None else execute, execute, cfg.trigger_stride)
    elif cfg.trigger_stride is not None:
        raise ValueError("trigger_stride is only used by execution=async")
    options = cfg.get("policy_options", {})
    if options.get("use_action_ensembler", False):
        raise ValueError("The shared runtime does not use action blending")
    if cfg.streaming_vae and options.get("tiled", False):
        raise ValueError("Streaming VAE does not support tiled encoding")
    return cfg


def read_config(path, overrides=()):
    """Read the same evaluation config used by the public CLI."""
    from longwam.evaluation import read_settings

    return read_settings(path, overrides)
