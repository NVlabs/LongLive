# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/__init__.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Shared model construction, training and inference execution runtime."""

from .factory import (
    _normalize_mixed_precision,
    _mixed_precision_to_model_dtype,
    create_wan22_model,
    create_longwam_base,
    create_longwam_joint,
    _load_longlive_video_expert_weights,
    create_longwam,
    create_longwam_idm,
    build_datasets,
    _resolve_train_device,
    run_training,
    run_inference,
    LONG_LIVE_VIDEO_LEGACY_STATE_CONTAINER,
    LONG_LIVE_VIDEO_ROBOT_S_STATE_CONTAINER,
    _ROBOT_S_VIDEO_STATE_PREFIX,
)


def create_policy(settings, *, camera_keys=None):
    """The single public control-policy factory: returns a LeRobot PreTrainedPolicy."""
    from .lerobot import LongWAMPolicy

    return LongWAMPolicy.from_runtime(settings, camera_keys=camera_keys)
