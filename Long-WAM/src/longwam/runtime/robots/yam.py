# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/robots/yam.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""YAM checkpoint observation and action conversion; hardware stays in the driver."""
from ._common import _vector, _images, _names


class YAMAdapter:
    """ABC-130k top/wrist mosaic and 14D state/action in checkpoint order."""

    state_dim = action_dim = 14

    def __init__(self, *, action_names, state_names=None, camera_keys=None):
        self.action_names = _names(action_names, 14, "action_names")
        self.state_names = None if state_names is None else _names(state_names, 14, "state_names")
        self.camera_keys = dict(
            camera_keys or {key: key for key in ("top", "left_wrist", "right_wrist")}
        )
        if set(self.camera_keys) != {"top", "left_wrist", "right_wrist"}:
            raise ValueError("YAM requires top, left_wrist and right_wrist cameras")

    def observation(self, raw):
        if "observation.state" in raw:
            state = raw["observation.state"]
        elif self.state_names is not None:
            state = [raw[key] for key in self.state_names]
        else:
            state = raw["state"] if "state" in raw else raw["observation.state"]
        return {"images": _images(raw, self.camera_keys), "state": _vector(state, 14, "YAM state")}

    def action(self, values):
        values = _vector(values, 14, "YAM action")
        return dict(zip(self.action_names, map(float, values)))
