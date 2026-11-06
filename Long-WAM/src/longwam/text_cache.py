# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/datasets/lerobot/robot_video_dataset.py
# Changes: Shared prompt and cache-path helpers separated from training dataset imports.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Training-compatible prompt and text-cache names without data-stack imports."""

import hashlib
import os


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
TEXT_CACHE_TAG = "wan22ti2v5b"


def get_text_cache_path(cache_dir, prompt: str, context_len: int):
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(
        cache_dir,
        f"{hashed}.t5_len{int(context_len)}.{TEXT_CACHE_TAG}.pt",
    )
