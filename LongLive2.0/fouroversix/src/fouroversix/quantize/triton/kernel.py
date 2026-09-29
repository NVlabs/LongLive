from __future__ import annotations

import torch
import triton
import triton.language as tl
from fouroversix.utils import DataType, RoundStyle, ScaleRule
from triton.tools.tensor_descriptor import TensorDescriptor

E2M1_MAX_VALUE = tl.constexpr(6)
E2M1_MAX_FOUR = tl.constexpr(4)
E4M3_MAX_VALUE = tl.constexpr(448)
E4M3_MAX_FOUROVERSIX = tl.constexpr(256)
SCALE_MEGABLOCK_SIZE = tl.constexpr(512)

DATA_TYPE_MXFP4 = tl.constexpr(DataType.mxfp4.value)
DATA_TYPE_NVFP4 = tl.constexpr(DataType.nvfp4.value)

ROUND_STYLE_NEAREST = tl.constexpr(RoundStyle.nearest.value)
ROUND_STYLE_STOCHASTIC = tl.constexpr(RoundStyle.stochastic.value)

SCALE_RULE_ABS_MAX = tl.constexpr(ScaleRule.abs_max.value)
SCALE_RULE_STATIC_4 = tl.constexpr(ScaleRule.static_4.value)
SCALE_RULE_STATIC_6 = tl.constexpr(ScaleRule.static_6.value)
SCALE_RULE_MAE = tl.constexpr(ScaleRule.mae.value)
SCALE_RULE_MSE = tl.constexpr(ScaleRule.mse.value)


@triton.jit
def rht_kernel(
    x_desc,
    h_desc,
    y_desc,
    # Meta-parameters
    # TODO(jack): Update RHT kernel to support unpadded dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
) -> None:
    HAD_BLOCK_SIZE: tl.constexpr = h_desc.block_shape[0]

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load H [B, B]
    h_block = h_desc.load([0, 0])

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    if not TRANSPOSE:
        x_block = x_desc.load([m_block_offset, n_block_offset])
    else:
        x_block = x_desc.load([n_block_offset, m_block_offset]).T

    y_block = tl.dot(
        x_block.reshape(
            BLOCK_SIZE_M * BLOCK_SIZE_N // HAD_BLOCK_SIZE,
            HAD_BLOCK_SIZE,
        ).to(tl.bfloat16),
        h_block,
    ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

    y_desc.store([m_block_offset, n_block_offset], y_block)


@triton.jit
def block_scaled_fp4_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FP4_FORMAT: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    SCALE_RULE: tl.constexpr,
) -> None:
    E2M1_MAX_ALLOWED_VALUE: tl.constexpr = (
        E2M1_MAX_VALUE if SCALE_RULE == SCALE_RULE_STATIC_6 else E2M1_MAX_FOUR
    )

    if FP4_FORMAT == DATA_TYPE_MXFP4:
        x_scale_blocks = x_block.reshape(128, 4, 32)
        x_scales_hp = tl.max(x_scale_blocks.abs(), axis=-1) / E2M1_MAX_ALLOWED_VALUE
        x_scales_e8m0_u32 = x_scales_hp.cast(tl.uint32, bitcast=True)

        # Use the 8-bit exponent as the scale factor
        x_scales_e8m0 = ((x_scales_e8m0_u32 >> 23) & 0xFF).to(tl.uint8)

        # Add one in order to round up
        x_scales = tl.where(
            (x_scales_e8m0_u32 & 0x7FFFFF) == 0,
            x_scales_e8m0,
            x_scales_e8m0 + 1,
        )

        # Convert the rounded-up scale factor back to a 32-bit float
        x_scales_hp = (x_scales.cast(tl.uint32) << 23).cast(x_block.dtype, bitcast=True)

        if BLOCK_SCALE_2D:
            x_scales_hp = (
                tl.max(
                    x_scales_hp.reshape(4, 32, 4).permute(0, 2, 1),
                    axis=-1,
                )
                .expand_dims(0)
                .broadcast_to(4, 32, 4)
                .permute(1, 0, 2)
                .reshape(128, 4)
            )

        (x_block_scaled_b1, x_block_scaled_b2) = (
            (x_scale_blocks / x_scales_hp.expand_dims(2))
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
            .split()
        )
    elif FP4_FORMAT == DATA_TYPE_NVFP4:
        x_amax = tl.load(x_amax_ptr)
        x_scale_blocks = x_block.reshape(128, 4, 16)

        if x_amax == 0:
            x_scales_hp = tl.full((128, 4), 0, dtype=tl.float32)
        else:
            encode_scale = tl.div_rn(E2M1_MAX_ALLOWED_VALUE * E4M3_MAX_VALUE, x_amax)
            x_scales_hp = (
                tl.div_rn(tl.max(x_scale_blocks.abs(), axis=-1), E2M1_MAX_ALLOWED_VALUE)
                * encode_scale
            )

        if BLOCK_SCALE_2D:
            x_scales_hp = (
                tl.max(
                    x_scales_hp.reshape(8, 16, 4).permute(0, 2, 1),
                    axis=-1,
                )
                .expand_dims(0)
                .broadcast_to(16, 8, 4)
                .permute(1, 0, 2)
                .reshape(128, 4)
            )

        x_scales = x_scales_hp.to(tl.float8e4nv)

        decode_scale = tl.div_rn(
            1,
            tl.div_rn(E2M1_MAX_ALLOWED_VALUE * E4M3_MAX_VALUE, x_amax),
        )
        (x_block_scaled_b1, x_block_scaled_b2) = (
            tl.where(
                x_scales.expand_dims(2).to(x_amax.dtype) != 0,
                x_scale_blocks
                * tl.div_rn(1, decode_scale * x_scales.to(x_amax.dtype).expand_dims(2)),
                0,
            )
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
            .split()
        )

    if ROUND_STYLE == ROUND_STYLE_NEAREST:
        x_e2m1 = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
                }
                """,
            constraints="=r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
    elif ROUND_STYLE == ROUND_STYLE_STOCHASTIC:
        rbits = tl.randint(
            tl.load(rbits_ptr),
            tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N // 2
            + tl.arange(0, BLOCK_SIZE_N // 2)[None, :],
        ).cast(tl.uint32, bitcast=True)

        x_e2m1 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            cvt.rs.satfinite.e2m1x4.f32 tmp0, {$6, $2, $5, $1}, $9;
            cvt.rs.satfinite.e2m1x4.f32 tmp1, {$8, $4, $7, $3}, $10;
            mov.b32 $0, {tmp0, tmp1};
            }
            """,
            constraints="=r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2, rbits],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    return x_e2m1, x_scales.reshape(4, 32, 4).permute(1, 0, 2).ravel()


@triton.jit
def nvfp4_fouroversix_quantization_kernel(  # noqa: C901, PLR0915
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    SCALE_RULE: tl.constexpr,
) -> None:
    x_amax = tl.load(x_amax_ptr)
    x_scale_blocks = x_block.reshape(128, 4, 16)

    if x_amax == 0:
        x_scales_hp = tl.full((128, 4), 0, dtype=tl.float32)
    else:
        encode_scale = tl.div_rn(E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX, x_amax)
        x_scales_hp = (
            tl.div_rn(tl.max(x_scale_blocks.abs(), axis=-1), E2M1_MAX_VALUE)
            * encode_scale
        )

    if BLOCK_SCALE_2D:
        x_scales_hp = (
            tl.max(
                x_scales_hp.reshape(8, 16, 4).permute(0, 2, 1),
                axis=-1,
            )
            .expand_dims(0)
            .broadcast_to(16, 8, 4)
            .permute(1, 0, 2)
            .reshape(128, 4)
        )

    x_scales_6 = x_scales_hp.to(tl.float8e4nv)
    x_scales_4 = (x_scales_hp * 1.5).to(tl.float8e4nv)

    decode_scale = tl.div_rn(
        1,
        tl.div_rn(E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX, x_amax),
    )

    (x_block_scaled_6_b1, x_block_scaled_6_b2) = (
        tl.where(
            x_scales_6.expand_dims(2).to(x_amax.dtype) != 0,
            x_scale_blocks
            * tl.div_rn(1, decode_scale * x_scales_6.to(x_amax.dtype).expand_dims(2)),
            0,
        )
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )

    (x_block_scaled_4_b1, x_block_scaled_4_b2) = (
        tl.where(
            x_scales_4.expand_dims(2).to(x_amax.dtype) != 0,
            x_scale_blocks
            * tl.div_rn(1, decode_scale * x_scales_4.to(x_amax.dtype).expand_dims(2)),
            0,
        )
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )

    if ROUND_STYLE == ROUND_STYLE_NEAREST:
        (x_e2m1_6, x_e2m1_4, x_fp16x2_6, x_fp16x2_4) = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b8 byte0, byte1, byte2, byte3;

                cvt.rn.satfinite.e2m1x2.f32 byte0, $28, $20;
                cvt.rn.f16x2.e2m1x2 $4, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $29, $21;
                cvt.rn.f16x2.e2m1x2 $5, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $30, $22;
                cvt.rn.f16x2.e2m1x2 $6, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $31, $23;
                cvt.rn.f16x2.e2m1x2 $7, byte3;
                mov.b32 $0, {byte0, byte1, byte2, byte3};

                cvt.rn.satfinite.e2m1x2.f32 byte0, $32, $24;
                cvt.rn.f16x2.e2m1x2 $8, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $33, $25;
                cvt.rn.f16x2.e2m1x2 $9, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $34, $26;
                cvt.rn.f16x2.e2m1x2 $10, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $35, $27;
                cvt.rn.f16x2.e2m1x2 $11, byte3;
                mov.b32 $1, {byte0, byte1, byte2, byte3};

                cvt.rn.satfinite.e2m1x2.f32 byte0, $44, $36;
                cvt.rn.f16x2.e2m1x2 $12, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $45, $37;
                cvt.rn.f16x2.e2m1x2 $13, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $46, $38;
                cvt.rn.f16x2.e2m1x2 $14, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $47, $39;
                cvt.rn.f16x2.e2m1x2 $15, byte3;
                mov.b32 $2, {byte0, byte1, byte2, byte3};

                cvt.rn.satfinite.e2m1x2.f32 byte0, $48, $40;
                cvt.rn.f16x2.e2m1x2 $16, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $49, $41;
                cvt.rn.f16x2.e2m1x2 $17, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $50, $42;
                cvt.rn.f16x2.e2m1x2 $18, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $51, $43;
                cvt.rn.f16x2.e2m1x2 $19, byte3;
                mov.b32 $3, {byte0, byte1, byte2, byte3};
                }
                """,
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[
                x_block_scaled_6_b1,
                x_block_scaled_6_b2,
                x_block_scaled_4_b1,
                x_block_scaled_4_b2,
            ],
            dtype=(tl.uint8, tl.uint8, tl.uint32, tl.uint32),
            is_pure=True,
            pack=8,
        )
    elif ROUND_STYLE == ROUND_STYLE_STOCHASTIC:
        rbits = tl.randint(
            tl.load(rbits_ptr),
            tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N // 2
            + tl.arange(0, BLOCK_SIZE_N // 2)[None, :],
        ).cast(tl.uint32, bitcast=True)

        (x_e2m1_6, x_e2m1_4, x_fp16x2_6, x_fp16x2_4) = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b16 tmp0, tmp1;
                .reg .b8 byte0, byte1;

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$29, $21, $28, $20}, $52;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $4, byte0;
                cvt.rn.f16x2.e2m1x2 $5, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$31, $23, $30, $22}, $53;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $6, byte0;
                cvt.rn.f16x2.e2m1x2 $7, byte1;
                mov.b32 $0, {tmp0, tmp1};

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$33, $25, $32, $24}, $54;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $8, byte0;
                cvt.rn.f16x2.e2m1x2 $9, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$35, $27, $34, $26}, $55;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $10, byte0;
                cvt.rn.f16x2.e2m1x2 $11, byte1;
                mov.b32 $1, {tmp0, tmp1};

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$45, $37, $44, $36}, $56;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $12, byte0;
                cvt.rn.f16x2.e2m1x2 $13, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$47, $39, $46, $38}, $57;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $14, byte0;
                cvt.rn.f16x2.e2m1x2 $15, byte1;
                mov.b32 $2, {tmp0, tmp1};

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$49, $41, $48, $40}, $58;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $16, byte0;
                cvt.rn.f16x2.e2m1x2 $17, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$51, $43, $50, $42}, $59;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $18, byte0;
                cvt.rn.f16x2.e2m1x2 $19, byte1;
                mov.b32 $3, {tmp0, tmp1};
                }
                """,
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[
                x_block_scaled_6_b1,
                x_block_scaled_6_b2,
                x_block_scaled_4_b1,
                x_block_scaled_4_b2,
                rbits,
            ],
            dtype=(tl.uint8, tl.uint8, tl.uint32, tl.uint32),
            is_pure=True,
            pack=8,
        )

    x_fp16_6_lo = (
        (x_fp16x2_6 & 0xFFFF)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_fp16_6_hi = (
        (x_fp16x2_6 >> 16)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_hp_6 = tl.join(x_fp16_6_lo, x_fp16_6_hi).reshape(128, 4, 16)

    x_dequantized_6 = tl.div_rn(
        x_hp_6 * x_scales_6.to(x_amax.dtype).expand_dims(2) * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
    )

    x_fp16_4_lo = (
        (x_fp16x2_4 & 0xFFFF)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_fp16_4_hi = (
        (x_fp16x2_4 >> 16)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_hp_4 = tl.join(x_fp16_4_lo, x_fp16_4_hi).reshape(128, 4, 16)

    x_dequantized_4 = tl.div_rn(
        x_hp_4 * x_scales_4.to(x_amax.dtype).expand_dims(2) * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
    )

    diff_6 = x_dequantized_6 - x_scale_blocks
    diff_4 = x_dequantized_4 - x_scale_blocks

    if SCALE_RULE == SCALE_RULE_ABS_MAX:
        six_error = tl.max(tl.abs(diff_6), axis=-1)
        four_error = tl.max(tl.abs(diff_4), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MAE:
        six_error = tl.sum(tl.abs(diff_6), axis=-1)
        four_error = tl.sum(tl.abs(diff_4), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MSE:
        six_error = tl.sum(diff_6 * diff_6, axis=-1)
        four_error = tl.sum(diff_4 * diff_4, axis=-1)

    if BLOCK_SCALE_2D:
        six_error = six_error.reshape(8, 16, 4).permute(0, 2, 1)
        four_error = four_error.reshape(8, 16, 4).permute(0, 2, 1)

        if SCALE_RULE == SCALE_RULE_ABS_MAX:
            six_error = tl.max(six_error, axis=-1)
            four_error = tl.max(four_error, axis=-1)
        elif SCALE_RULE == SCALE_RULE_MAE or SCALE_RULE == SCALE_RULE_MSE:
            six_error = tl.sum(six_error, axis=-1)
            four_error = tl.sum(four_error, axis=-1)

        six_error = (
            six_error.expand_dims(0)
            .broadcast_to(16, 8, 4)
            .permute(1, 0, 2)
            .reshape(128, 4)
        )
        four_error = (
            four_error.expand_dims(0)
            .broadcast_to(16, 8, 4)
            .permute(1, 0, 2)
            .reshape(128, 4)
        )

    x_e2m1 = tl.where(
        (four_error < six_error).expand_dims(2),
        x_e2m1_4.reshape(128, 4, 8),
        x_e2m1_6.reshape(128, 4, 8),
    ).reshape(128, 32)

    x_scales = (
        tl.where(four_error < six_error, x_scales_4, x_scales_6)
        .reshape(4, 32, 4)
        .permute(1, 0, 2)
        .ravel()
    )

    return x_e2m1, x_scales


@triton.jit
def fp4_quantization_kernel(
    x_desc,
    x_amax_ptr,
    x_e2m1_desc,
    x_sf_desc,
    rbits_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    FP4_FORMAT: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    SCALE_RULE: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    # Load [B, B] block from A or A^T
    if not TRANSPOSE:
        x_block = x_desc.load([m_block_offset, n_block_offset])
    else:
        x_block = x_desc.load([n_block_offset, m_block_offset]).T

    x_block = x_block.to(tl.float32)

    if SCALE_RULE == SCALE_RULE_STATIC_6 or SCALE_RULE == SCALE_RULE_STATIC_4:
        x_e2m1, x_scales = block_scaled_fp4_quantization_kernel(
            x_block,
            x_amax_ptr,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            FP4_FORMAT,
            ROUND_STYLE,
            BLOCK_SCALE_2D,
            SCALE_RULE,
        )
    else:
        x_e2m1, x_scales = nvfp4_fouroversix_quantization_kernel(
            x_block,
            x_amax_ptr,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ROUND_STYLE,
            BLOCK_SCALE_2D,
            SCALE_RULE,
        )

    e2m1_n_block_offset = pid_n * BLOCK_SIZE_N // 2
    x_e2m1_desc.store([m_block_offset, e2m1_n_block_offset], x_e2m1)

    scale_block_offset = (pid_m * tl.num_programs(1) + pid_n) * SCALE_MEGABLOCK_SIZE
    x_sf_desc.store([scale_block_offset], x_scales)


def quantize_to_fp4(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    fp4_format: DataType = DataType.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: ScaleRule = ScaleRule.mse,
    block_scale_2d: bool = False,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if transpose:
        N, M = x.shape
    else:
        M, N = x.shape

    block_size_m = 128
    block_size_n = 4 * fp4_format.block_size()
    scale_dtype = torch.float8_e4m3fn if fp4_format == DataType.nvfp4 else torch.uint8

    if x_amax is None:
        x_amax = (
            x.abs().max().float()
            if fp4_format == DataType.nvfp4
            else torch.ones(1, device=x.device, dtype=torch.float32)
        )

    padded_m = M + (block_size_m - M % block_size_m) % block_size_m
    padded_n = N + (block_size_n - N % block_size_n) % block_size_n

    x_e2m1 = torch.empty((padded_m, padded_n // 2), device=x.device, dtype=torch.uint8)
    x_sf = torch.empty(
        padded_m * padded_n // fp4_format.block_size(),
        device=x.device,
        dtype=scale_dtype,
    )

    grid = lambda _: (  # noqa: E731
        padded_m // block_size_m,
        padded_n // block_size_n,
    )

    x_desc = TensorDescriptor.from_tensor(
        x,
        block_shape=[
            block_size_m if not transpose else block_size_n,
            block_size_n if not transpose else block_size_m,
        ],
    )
    x_e2m1_desc = TensorDescriptor.from_tensor(
        x_e2m1,
        block_shape=[block_size_m, block_size_n // 2],
    )
    x_sf_desc = TensorDescriptor.from_tensor(
        x_sf,
        block_shape=[SCALE_MEGABLOCK_SIZE.value],
    )

    if had is not None:
        had_block_size = had.shape[0]

        if M % had_block_size != 0:
            msg = (
                f"The first dimension of A ({M}) must be divisible by the width of H "
                f"({had_block_size})"
            )
            raise ValueError(msg)
        if N % had_block_size != 0:
            msg = (
                f"The second dimension of A ({N}) must be divisible by the width of H "
                f"({had_block_size})"
            )
            raise ValueError(msg)
        if had.shape[0] != had.shape[1]:
            msg = "H must be a square matrix"
            raise ValueError(msg)
        if (had.shape[0] & (had.shape[0] - 1)) != 0:
            msg = "H must have dimensions that are a power of two"
            raise ValueError(msg)

        x_rht = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

        h_desc = TensorDescriptor.from_tensor(
            had,
            block_shape=[had_block_size, had_block_size],
        )
        x_rht_desc = TensorDescriptor.from_tensor(
            x_rht,
            block_shape=[block_size_m, block_size_n],
        )

        rht_kernel[grid](
            x_desc,
            h_desc,
            x_rht_desc,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            TRANSPOSE=transpose,
        )

        transpose = False
        x_amax = x_rht.abs().max().float()

    rbits = (
        torch.randint(0, torch.iinfo(torch.int32).max, (1,), device=x.device)
        if round_style == RoundStyle.stochastic
        else None
    )

    fp4_quantization_kernel[grid](
        x_rht_desc if had is not None else x_desc,
        x_amax,
        x_e2m1_desc,
        x_sf_desc,
        rbits,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        TRANSPOSE=transpose,
        FP4_FORMAT=fp4_format.value,
        ROUND_STYLE=round_style.value,
        BLOCK_SCALE_2D=block_scale_2d,
        SCALE_RULE=scale_rule.value,
    )

    if fp4_format == DataType.mxfp4:
        x_sf = x_sf.view(torch.float8_e8m0fnu)

    return x_e2m1, x_sf, x_amax
