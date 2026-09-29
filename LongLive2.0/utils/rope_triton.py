# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Triton RoPE kernel for causal_rope_apply.

iter-42: replaces the complex<double> × view_as_complex × view_as_real chain
(33 ms / 1.3% of profile + feeding elementwise muls) with a single Triton
kernel. Internal precision is fp32 — bf16 outputs cannot resolve any precision
loss from fp32 vs fp64 arithmetic at this stage. cos / sin lookup tables come
from the complex128 freqs split into real / imag floats up-front (one-shot per
freqs_i cache entry).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# iter-45 (REVERTED): tried @triton.autotune over (BLOCK_N∈{4,8,16},
# num_warps∈{2,4,8}, num_stages∈{1,2,3}) = 27 configs. Result was FLAT vs
# iter-42 fixed BLOCK_N=8: median tied (-0.1%), total +0.4% (autotune warmup
# bled into p1/p2 p90). The original BLOCK_N=8 / default warps was already
# near-optimal for the (N=24, D_half=64) shape, autotune found no better.
# Reverted to fixed config — same kernel as iter-42.
#
# iter-46: kernel now accepts FULL x[i] of shape [S_total, N, D] and a
# runtime `seq_len` — for rows s < seq_len it applies rotation, for
# s >= seq_len it copies through. This subsumes the `torch.cat([rotated,
# x[i, seq_len:]])` step (1 fewer kernel + 1 fewer alloc per call). Also
# skips the upstream `.contiguous()` because we no longer slice x.
@triton.jit
def _rope_apply_kernel(
    x_ptr,         # [S_total, N, D] bf16  (D is even, pairs are (a,b)=(2d, 2d+1))
    cos_ptr,       # [seq_len, D/2] fp32 (valid only for s < seq_len)
    sin_ptr,       # [seq_len, D/2] fp32
    out_ptr,       # [S_total, N, D] bf16
    SEQ_LEN, N, D_half,
    x_stride_s, x_stride_n,
    o_stride_s, o_stride_n,
    cs_stride_s,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)  # over the D/2 pairs

    n_mask = offs_n < N
    d_mask = offs_d < D_half

    x_row_base = pid_s * x_stride_s
    o_row_base = pid_s * o_stride_s
    a_offs = x_row_base + offs_n[:, None] * x_stride_n + (2 * offs_d)[None, :]
    b_offs = a_offs + 1
    mask = n_mask[:, None] & d_mask[None, :]
    a = tl.load(x_ptr + a_offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_offs, mask=mask, other=0.0).to(tl.float32)

    a_out_offs = o_row_base + offs_n[:, None] * o_stride_n + (2 * offs_d)[None, :]
    b_out_offs = a_out_offs + 1

    if pid_s < SEQ_LEN:
        cs_base = pid_s * cs_stride_s
        cos = tl.load(cos_ptr + cs_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + cs_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        # Rotate: (a + bi) * (cos + sin i) = (a*cos - b*sin) + (a*sin + b*cos) i
        out_a = a * cos[None, :] - b * sin[None, :]
        out_b = a * sin[None, :] + b * cos[None, :]
        tl.store(out_ptr + a_out_offs, out_a, mask=mask)
        tl.store(out_ptr + b_out_offs, out_b, mask=mask)
    else:
        # passthrough copy for the unrotated tail
        tl.store(out_ptr + a_out_offs, a, mask=mask)
        tl.store(out_ptr + b_out_offs, b, mask=mask)


def _split_complex_to_cos_sin(freqs_complex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complex128 freqs to (cos_f32, sin_f32) — once per cache entry."""
    # freqs_complex shape: (S, 1, D/2). Squeeze the middle 1.
    if freqs_complex.dim() == 3 and freqs_complex.size(1) == 1:
        freqs_complex = freqs_complex.squeeze(1)
    cos = freqs_complex.real.to(torch.float32).contiguous()
    sin = freqs_complex.imag.to(torch.float32).contiguous()
    return cos, sin


def rope_apply_triton(
    x: torch.Tensor,           # [S_total, N, D] bf16 (or fp16/fp32)
    cos_f32: torch.Tensor,     # [seq_len, D/2] fp32
    sin_f32: torch.Tensor,     # [seq_len, D/2] fp32
    seq_len: int | None = None,
) -> torch.Tensor:
    """Apply rotary embedding via Triton kernel.

    iter-46: when `seq_len < x.size(0)`, the kernel rotates the first
    `seq_len` rows and copies through rows `[seq_len:]`. This replaces the
    `cat([rotated, x[i, seq_len:]])` pattern in `causal_rope_apply` with a
    single kernel + single allocation. `seq_len=None` (default) means rotate
    all rows (equivalent to iter-42 behavior).

    Returns a tensor of the same shape and dtype as `x`.
    """
    assert x.dim() == 3, f"expected x.shape == (S, N, D), got {x.shape}"
    S_total, N, D = x.shape
    assert D % 2 == 0
    D_half = D // 2
    if seq_len is None:
        seq_len = S_total
    assert seq_len <= S_total
    assert cos_f32.shape == (seq_len, D_half), \
        f"cos_f32 expected ({seq_len},{D_half}), got {cos_f32.shape}"
    assert sin_f32.shape == (seq_len, D_half)

    out = torch.empty_like(x)

    BLOCK_N = 8
    BLOCK_D = triton.next_power_of_2(D_half)
    grid = (S_total, triton.cdiv(N, BLOCK_N))

    _rope_apply_kernel[grid](
        x, cos_f32, sin_f32, out,
        seq_len, N, D_half,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        cos_f32.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out
