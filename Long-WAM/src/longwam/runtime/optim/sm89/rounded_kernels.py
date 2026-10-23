# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""BF16 pointwise kernels with every original intermediate rounding retained."""

import triton
import triton.language as tl


@triton.jit
def rounded_gate_kernel(
    X, G, R, Y, count, D: tl.constexpr, GS: tl.constexpr, GB: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // D, index % D
    gate_index = column if GB else row * GS + column
    x = tl.load(X + index, index < count, other=0).to(tl.float32)
    gate = tl.load(G + gate_index, index < count, other=0).to(tl.float32)
    residual = tl.load(R + index, index < count, other=0).to(tl.float32)
    product = (gate * residual).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    result = (x + product).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Y + index, result, index < count)


@triton.jit
def rounded_modulate_kernel(
    X,
    SHIFT,
    SCALE,
    Y,
    count,
    D: tl.constexpr,
    SHS: tl.constexpr,
    SHB: tl.constexpr,
    SCS: tl.constexpr,
    SCB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // D, index % D
    shift_index = column if SHB else row * SHS + column
    scale_index = column if SCB else row * SCS + column
    x = tl.load(X + index, index < count, other=0).to(tl.float32)
    shift = tl.load(SHIFT + shift_index, index < count, other=0).to(tl.float32)
    scale = tl.load(SCALE + scale_index, index < count, other=0).to(tl.float32)
    gain = (1.0 + scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    product = (x * gain).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    result = (product + shift).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Y + index, result, index < count)
