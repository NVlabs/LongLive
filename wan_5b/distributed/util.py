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
