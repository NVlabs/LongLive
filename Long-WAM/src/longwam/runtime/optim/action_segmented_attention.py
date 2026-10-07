# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/action_segmented_attention.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Shape-dispatched segmented action attention with optional split-KV parallelism.

One implementation covers LIBERO and RoboTwin history lengths. Triton
specializes scalar launch parameters; callers use the same tensor-only entry.
Video and action K/V remain separate allocations throughout attention.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BATCH = 1
_HEADS = 24
_QUERY = 32
_ACTION_KV = 32
_HEAD_DIM = 128
_HIDDEN = 3072
# M+k = 6/9/15/27 latent frames. LIBERO has 98 spatial tokens per
# frame; RoboTwin's three-camera 384x320 mosaic has 120. Both use the
# same split-KV kernel and length-derived launch rule, without KV copies.
_SUPPORTED_VIDEO_KV = (588, 720, 882, 1080, 1470, 1800, 2646, 3240)


@triton.jit
def _action_segmented_attention_kernel(
    q_ptr,
    kv_ptr,
    vv_ptr,
    ka_ptr,
    va_ptr,
    mask_ptr,
    out_ptr,
    partial_ptr,
    stats_ptr,
    sq: tl.constexpr,
    skv: tl.constexpr,
    svv: tl.constexpr,
    ska: tl.constexpr,
    sva: tl.constexpr,
    smask: tl.constexpr,
    sout: tl.constexpr,
    VIDEO_KV: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    head = tl.program_id(1)
    split = tl.program_id(2)
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, 128)
    nd = tl.arange(0, BLOCK_N)
    q = tl.load(q_ptr + m[:, None] * sq + head * 128 + d[None, :], m[:, None] < 32, 0)
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start in tl.range(split * CHUNK, (split + 1) * CHUNK, BLOCK_N):
        n = start + nd
        valid = n < VIDEO_KV + 32
        video = n < VIDEO_KV
        # Select addresses first: only one K and one V tile need shared memory.
        kp = tl.where(
            video[None, :], kv_ptr + n[None, :] * skv, ka_ptr + (n[None, :] - VIDEO_KV) * ska
        )
        k = tl.load(kp + head * 128 + d[:, None], valid[None, :], 0)
        scores = tl.dot(q, k) * (128**-0.5 * 1.4426950408889634)
        valid_pair = (m[:, None] < 32) & valid[None, :]
        allowed = tl.load(mask_ptr + m[:, None] * smask + n[None, :], valid_pair, 0).to(tl.int1)
        scores = tl.where(valid_pair & allowed, scores, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(scores, 1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(scores - safe_max[:, None])
        scale = tl.exp2(row_max - safe_max)
        row_sum = row_sum * scale + tl.sum(p, 1)
        acc = acc * scale[:, None]
        vp = tl.where(
            video[:, None], vv_ptr + n[:, None] * svv, va_ptr + (n[:, None] - VIDEO_KV) * sva
        )
        v = tl.load(vp + head * 128 + d[None, :], valid[:, None], 0)
        acc += tl.dot(p.to(tl.bfloat16), v)
        row_max = new_max
    if SPLITS == 1:
        out = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
        tl.store(out_ptr + m[:, None] * sout + head * 128 + d[None, :], out, m[:, None] < 32)
    else:
        row = (split * 24 + head) * 32 + m
        tl.store(partial_ptr + row[:, None] * 128 + d[None, :], acc, m[:, None] < 32)
        tl.store(stats_ptr + row * 2, row_max, m < 32)
        tl.store(stats_ptr + row * 2 + 1, row_sum, m < 32)


@triton.jit
def _action_segmented_attention_merge_kernel(
    partial_ptr,
    stats_ptr,
    out_ptr,
    sout: tl.constexpr,
    SPLITS: tl.constexpr,
):
    head = tl.program_id(1)
    m = tl.program_id(0) * 4 + tl.arange(0, 4)
    splits = tl.arange(0, SPLITS)
    d = tl.arange(0, 128)
    row = (splits[:, None] * 24 + head) * 32 + m[None, :]
    maxima = tl.load(stats_ptr + row * 2)
    sums = tl.load(stats_ptr + row * 2 + 1)
    global_max = tl.max(maxima, axis=0)
    safe_max = tl.where(global_max == -float("inf"), 0.0, global_max)
    scales = tl.exp2(maxima - safe_max[None, :])
    denominator = tl.sum(sums * scales, axis=0)
    acc = tl.load(partial_ptr + row[:, :, None] * 128 + d[None, None, :])
    numerator = tl.sum(acc * scales[:, :, None], axis=0)
    result = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(out_ptr + m[:, None] * sout + head * 128 + d[None, :], result)


def supports_action_segmented_attention(q, k_video, v_video, k_action, v_action, mask) -> bool:
    return bool(
        q.device.type == "cuda"
        and q.dtype == torch.bfloat16
        and tuple(q.shape) == (1, 32, 3072)
        and k_video.ndim == 3
        and k_video.shape[1] in _SUPPORTED_VIDEO_KV
        and tuple(k_video.shape) == (1, k_video.shape[1], 3072)
        and tuple(v_video.shape) == tuple(k_video.shape)
        and tuple(k_action.shape) == (1, 32, 3072)
        and tuple(v_action.shape) == tuple(k_action.shape)
        and tuple(mask.shape) in ((32, k_video.shape[1] + 32), (1, 32, k_video.shape[1] + 32))
        and mask.dtype == torch.bool
        and all(t.device == q.device for t in (k_video, v_video, k_action, v_action, mask))
        and all(t.dtype == q.dtype for t in (k_video, v_video, k_action, v_action))
        and all(t.stride(-1) == 1 for t in (q, k_video, v_video, k_action, v_action, mask))
    )


def _launch(q, k_video, v_video, k_action, v_action, mask, *, block_n, splits, stages):
    output = torch.empty_like(q)
    if splits > 1:
        partial = torch.empty((splits, 24, 32, 128), device=q.device, dtype=torch.float32)
        stats = torch.empty((splits, 24, 32, 2), device=q.device, dtype=torch.float32)
    else:
        partial = stats = output
    length = k_video.shape[1]
    chunk = triton.cdiv(length + 32, splits * block_n) * block_n
    _action_segmented_attention_kernel[(2, 24, splits)](
        q,
        k_video,
        v_video,
        k_action,
        v_action,
        mask,
        output,
        partial,
        stats,
        q.stride(1),
        k_video.stride(1),
        v_video.stride(1),
        k_action.stride(1),
        v_action.stride(1),
        mask.stride(-2),
        output.stride(1),
        VIDEO_KV=length,
        SPLITS=splits,
        BLOCK_M=16,
        BLOCK_N=block_n,
        CHUNK=chunk,
        num_warps=4,
        num_stages=stages,
    )
    if splits > 1:
        _action_segmented_attention_merge_kernel[(8, 24)](
            partial,
            stats,
            output,
            output.stride(1),
            SPLITS=splits,
            num_warps=4,
            num_stages=1,
        )
    return output


def launch_parameters(video_kv_length: int, block_n: int = 64) -> tuple[int, int]:
    """Bound split parallelism while targeting three KV tiles per partition.

    This is a length-derived launch rule, not a context-to-kernel lookup table.
    The RTX 5090 sweep selected up to eight partitions; fewer partitions benefit
    from a second pipeline stage because their per-partition loops are longer.
    """
    splits = min(8, triton.next_power_of_2(triton.cdiv(video_kv_length + 32, 3 * block_n)))
    return splits, 2 if splits <= 4 else 1


def action_segmented_attention(q, k_video, v_video, k_action, v_action, mask, *, block_n=64):
    """Infer context geometry and launch the shared implementation automatically."""
    if not supports_action_segmented_attention(q, k_video, v_video, k_action, v_action, mask):
        raise ValueError("Inputs do not satisfy the segmented attention contract.")
    if block_n not in (32, 64, 128):
        raise ValueError("BLOCK_N must be 32, 64 or 128.")
    splits, stages = launch_parameters(k_video.shape[1], block_n)
    return _launch(
        q, k_video, v_video, k_action, v_action, mask, block_n=block_n, splits=splits, stages=stages
    )
