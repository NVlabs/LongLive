# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/robots/__init__.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
# Changes: Added Unitree G1 inference/deployment integration while retaining the shared runtime.

"""Robot data converters, independent of device connections and model loading."""
from .yam import YAMAdapter
from .franka import FrankaAdapter
from .g1 import G1Adapter


def make_adapter(robot, options):
    adapters = {"yam": YAMAdapter, "franka": FrankaAdapter, "g1": G1Adapter}
    if robot not in adapters:
        raise ValueError(f"Unknown robot: {robot}")
    return adapters[robot](**dict(options))
