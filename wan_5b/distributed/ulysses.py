# Adopted from https://github.com/Wan-Video/Wan2.2
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist

from ..modules.attention import flash_attention
from .sp_training import distributed_flex_attention
from .util import all_to_all


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")

    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )

    return all_to_all(x, scatter_dim=1, gather_dim=2)


__all__ = ["distributed_attention", "distributed_flex_attention"]
