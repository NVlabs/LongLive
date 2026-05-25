# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Triton fused adaLN-modulation kernel.

iter-43: replaces the
    (norm(x).unflatten(1, (F, frame_seqlen)) * (1 + e_scale) + e_shift).flatten(1, 2)
chain (LayerNorm + 2 broadcast elementwise ops per call, x2 per transformer block
for norm1/norm2) with a single Triton kernel. Each token does one fp32 pass:
mean/var reduce, normalize, multiply by (1+scale), add shift, cast back.

LayerNorm matches `WanLayerNorm` (nn.LayerNorm with elementwise_affine=False,
eps=1e-6, output cast back to input dtype).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# iter-44: autotune over num_warps + num_stages. Fixed BLOCK_C
# (next_power_of_2(C)) — varying it would change the reduce semantics. Provide
# enough configs to cover the (small_B, large_L) regime of inference chunks.
_ADALN_CONFIGS = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in (4, 8, 16)
    for ns in (1, 2, 3)
]


@triton.autotune(configs=_ADALN_CONFIGS, key=["C", "FRAME_SEQLEN"])
@triton.jit
def _adaln_modulate_kernel(
    x_ptr,        # [B, L, C]
    scale_ptr,    # [B, F, 1, C] (or any layout, indexed via strides)
    shift_ptr,    # [B, F, 1, C]
    out_ptr,      # [B, L, C]
    C,
    FRAME_SEQLEN,
    x_stride_b, x_stride_l,
    scale_stride_b, scale_stride_f,
    shift_stride_b, shift_stride_f,
    eps,
    ADD_ONE_TO_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_f = pid_l // FRAME_SEQLEN

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x_off = pid_b * x_stride_b + pid_l * x_stride_l + offs_c
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)

    inv_C = 1.0 / C
    mean = tl.sum(x, axis=0) * inv_C
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) * inv_C
    rstd = 1.0 / tl.sqrt(var + eps)
    # Match WanLayerNorm: nn.LayerNorm casts back to input dtype via .type_as(x)
    # before downstream ops. Round-trip through bf16 to keep numerics identical
    # to the eager path so latent diff stays in run-to-run noise floor.
    x_norm = (x_centered * rstd).to(tl.bfloat16).to(tl.float32)

    s_off = pid_b * scale_stride_b + pid_f * scale_stride_f + offs_c
    h_off = pid_b * shift_stride_b + pid_f * shift_stride_f + offs_c
    scale = tl.load(scale_ptr + s_off, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + h_off, mask=mask, other=0.0).to(tl.float32)

    if ADD_ONE_TO_SCALE:
        # Match eager: `1 + e_scale` happens in bf16 (autocast off at this site)
        scale = (scale + 1.0)
        scale = scale.to(tl.bfloat16).to(tl.float32)

    # Match eager: bf16 * bf16, bf16 + bf16
    prod = (x_norm * scale).to(tl.bfloat16).to(tl.float32)
    y = prod + shift

    tl.store(out_ptr + x_off, y, mask=mask)


def adaln_modulate_triton(
    x: torch.Tensor,        # [B, L, C] contiguous
    e_scale: torch.Tensor,  # [B, F, 1, C]
    e_shift: torch.Tensor,  # [B, F, 1, C]
    frame_seqlen: int,
    eps: float = 1e-6,
    add_one_to_scale: bool = True,
) -> torch.Tensor:
    """Fused (LayerNorm + (1+e_scale)*x + e_shift) over [B, F*frame_seqlen, C].

    Replaces the WanAttentionBlock norm1/norm2 + modulate pattern. Output dtype
    follows x.dtype; internal arithmetic is fp32. eps matches `WanLayerNorm`.
    """
    assert x.dim() == 3, f"x must be [B, L, C], got {x.shape}"
    B, L, C = x.shape
    assert L % frame_seqlen == 0, (L, frame_seqlen)
    F = L // frame_seqlen
    assert e_scale.dim() == 4 and e_scale.shape[0] == B and e_scale.shape[1] == F \
        and e_scale.shape[2] == 1 and e_scale.shape[3] == C, \
        f"e_scale {tuple(e_scale.shape)} vs expected {(B, F, 1, C)}"
    assert e_shift.shape == e_scale.shape

    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)

    grid = (B, L)
    # num_warps / num_stages picked by @triton.autotune (iter-44).
    _adaln_modulate_kernel[grid](
        x, e_scale, e_shift, out,
        C, frame_seqlen,
        x.stride(0), x.stride(1),
        e_scale.stride(0), e_scale.stride(1),
        e_shift.stride(0), e_shift.stride(1),
        eps,
        ADD_ONE_TO_SCALE=add_one_to_scale,
        BLOCK_C=BLOCK_C,
    )
    return out
