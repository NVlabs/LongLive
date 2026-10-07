# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: wan_5b/distributed/util.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/wan_5b/distributed/util.py
# Source: https://github.com/Wan-Video/Wan2.2
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# Adopted from https://github.com/Wan-Video/Wan2.2
# SPDX-License-Identifier: Apache-2.0

"""Wan distributed utility compatibility layer.

LongLive-specific SP/DP group routing and autograd all-to-all live in
``wan_5b.distributed.sp_training``. This module keeps Wan2.2's public import
paths intact for the rest of the codebase.
"""

import torch.distributed as dist

from .sp_training import (
    all_gather,
    all_to_all,
    all_to_all_with_grad,
    gather_forward,
    get_data_parallel_group,
    get_sp_rank,
    get_sp_world_size,
    set_data_parallel_group,
    set_sequence_parallel_group,
)


def init_distributed_group():
    """Initialize the default distributed group when it is not yet ready."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def get_rank():
    return get_sp_rank()


def get_world_size():
    return get_sp_world_size()


__all__ = [
    "all_gather",
    "all_to_all",
    "all_to_all_with_grad",
    "gather_forward",
    "get_data_parallel_group",
    "get_rank",
    "get_world_size",
    "init_distributed_group",
    "set_data_parallel_group",
    "set_sequence_parallel_group",
]
