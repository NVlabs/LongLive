# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/datasets/lerobot/transforms/misc.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

from typing import List

import torch


class WrapStateAngle:
    def __init__(self, keys: List[str]):
        self.keys = keys

    @staticmethod
    def _wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, batch):
        for k in self.keys:
            batch["state"][k] = self._wrap(batch["state"][k])
        return batch

    def backward(self, batch):
        return batch
