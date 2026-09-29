# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Native PyTorch fused SDPA backend used in the reproduced recipes."""
import torch


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                   softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), dtype=torch.bfloat16):
    """Native PyTorch fused attention with support for ragged lengths.

    Slice each sample instead of constructing a quadratic padding mask. CUDA
    math fallback is disabled so a missing fused kernel fails before an OOM.
    """
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn import functional as F

    if tuple(window_size) != (-1, -1):
        raise NotImplementedError('SDPA backend requires global attention')
    outputs = []
    for i in range(q.shape[0]):
        nq = q.shape[1] if q_lens is None else int(q_lens[i])
        nk = k.shape[1] if k_lens is None else int(k_lens[i])
        if not 0 < nq <= q.shape[1] or not 0 < nk <= k.shape[1]:
            raise ValueError('Invalid SDPA sequence length')
        if causal and nq != nk:
            raise NotImplementedError('Unequal causal lengths use different FA/SDPA alignment')
        qi = q[i:i+1, :nq].transpose(1, 2).to(dtype)
        ki = k[i:i+1, :nk].transpose(1, 2).to(dtype)
        vi = v[i:i+1, :nk].transpose(1, 2).to(dtype)
        if q_scale is not None:
            qi = qi * q_scale
        context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                               SDPBackend.EFFICIENT_ATTENTION]) if q.is_cuda else nullcontext()
        with context:
            out = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        out = out.transpose(1, 2).to(q.dtype)
        outputs.append(F.pad(out, (0, 0, 0, 0, 0, q.shape[1] - nq)))
    return torch.cat(outputs, dim=0)


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16, version=None):
    return sdpa_attention(q, k, v, q_lens, k_lens, dropout_p,
                          softmax_scale, q_scale, causal, window_size, dtype)


attention = flash_attention
