# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/robots/_common.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Validation shared by robot observation and command converters."""
import numpy as np

def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result.copy()


def _images(observation, camera_keys):
    source = observation.get("images", observation)
    result = {}
    for name, key in camera_keys.items():
        value = source[key] if key in source else source[f"observation.images.{key}"]
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera {key} must be RGB uint8 [height, width, 3]")
        if min(image.shape[:2]) == 0:
            raise ValueError(f"Camera {key} is empty")
        result[name] = np.ascontiguousarray(image)
    return result


def _names(names, size, label):
    if names is None:
        raise ValueError(f"Explicit {label} are required in checkpoint vector order")
    names = tuple(names)
    if len(names) != size or len(set(names)) != size or any(
        not isinstance(key, str) or not key for key in names
    ):
        raise ValueError(f"{label} must contain {size} distinct nonempty names")
    return names
