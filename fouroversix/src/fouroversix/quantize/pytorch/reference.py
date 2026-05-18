from __future__ import annotations

import torch
from fouroversix.quantize.utils import to_blocked
from fouroversix.utils import DataType, RoundStyle, ScaleRule

E2M1_MAX_VALUE = 6
E2M1_MAX_FOUR = 4
E4M3_MAX_VALUE = 448
E4M3_MAX_FOUROVERSIX = 256


def fake_quantize_to_e2m1(
    x: torch.Tensor,
    *,
    round_style: RoundStyle = RoundStyle.nearest,
) -> torch.Tensor:
    if round_style == RoundStyle.nearest:
        step1 = torch.round(2 * x.abs()) / 2
        step2 = torch.round(x.abs())
        step3 = 2 * torch.round(x.abs() / 2)
    elif round_style == RoundStyle.stochastic:
        rbits = torch.rand_like(x.abs()) - 0.5
        step1 = torch.round(2 * x.abs() + rbits) / 2
        step2 = torch.round(x.abs() + rbits)
        step3 = 2 * torch.round(x.abs() / 2 + rbits)
        step3[step3 > E2M1_MAX_VALUE] = E2M1_MAX_VALUE

    mask1 = x.abs() < 2  # noqa: PLR2004
    mask2 = x.abs() < 4  # noqa: PLR2004

    return x.sign() * (
        step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)
    )


def quantize_bf16_to_unpacked_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16

    bx = x.view(torch.int16)
    s = (bx >> 15) & 0x1
    e = (bx >> 7) & 0xFF
    m = bx & 0x7F
    is_zero = (e == 0) & (m == 0)

    # Default mantissa bit (for 1.5, 3.0, 6.0)
    m = (m >> 6) & 1
    is_half = (e == 126) & (m == 0)  # noqa: PLR2004
    m = torch.where(is_half, torch.tensor(1, dtype=torch.int16, device=x.device), m)

    # Exponent mapping
    # exp=126 -> E=0 (subnormals)
    # exp=127 -> E=1
    # exp=128 -> E=2
    # exp=129 -> E=3
    e = e - 126
    e = torch.where(is_zero, torch.tensor(0, dtype=torch.int16, device=x.device), e)

    # Zero always M=0
    m = torch.where(is_zero, torch.tensor(0, dtype=torch.int16, device=x.device), m)

    code = (s << 3) | (e << 1) | m
    return code.to(torch.uint8)


def pack_unpacked_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8

    dim = 1
    size_along_dim = x.size(dim)
    new_size_along_dim = (size_along_dim + 1) // 2

    # If the size is odd, we pad the data along dim with zeros at the end
    if size_along_dim % 2 != 0:
        pad_sizes = [0] * (2 * x.ndim)
        pad_index = (x.ndim - dim - 1) * 2 + 1
        pad_sizes[pad_index] = 1
        x = torch.nn.functional.pad(x, pad_sizes, mode="constant", value=0)

    new_shape = list(x.shape)
    new_shape[dim] = new_size_along_dim
    new_shape.insert(dim + 1, 2)  # packed dimension of length 2
    x = x.reshape(*new_shape)

    low = x.select(dim + 1, 0)
    high = x.select(dim + 1, 1)
    return (high << 4) | low


def quantize_to_mxfp4(
    x_scale_blocks: torch.Tensor,
    *,
    scale_rule: ScaleRule = ScaleRule.mse,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert scale_rule in {ScaleRule.static_6, ScaleRule.static_4}

    x_scales_hp = (
        x_scale_blocks.abs().max(axis=-1).values / scale_rule.max_allowed_e2m1_value()
    )

    x_scales_e8m0_u32 = x_scales_hp.view(torch.int32)

    # Use the 8-bit exponent as the scale factor
    x_scales_e8m0 = ((x_scales_e8m0_u32 >> 23) & 0xFF).to(torch.uint8)

    # Add one in order to round up
    x_scales = torch.where(
        (x_scales_e8m0_u32 & 0x7FFFFF) == 0,
        x_scales_e8m0,
        x_scales_e8m0 + 1,
    )

    # Convert the rounded-up scale factor back to a 32-bit float
    x_scales_hp = (x_scales.to(torch.int32) << 23).view(torch.float32)

    x_block_scaled = x_scale_blocks / x_scales_hp.unsqueeze(1)

    return x_block_scaled, x_scales.view(torch.float8_e8m0fnu)


def quantize_to_nvfp4(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: ScaleRule,
    scale_expansion_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_amax == 0:
        x_scales_hp = torch.zeros(
            *x_scale_blocks.shape[:-1],
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    else:
        encode_scale = (
            torch.tensor(
                scale_rule.max_allowed_e2m1_value()
                * scale_rule.max_allowed_e4m3_value(),
                dtype=x_amax.dtype,
                device=x_amax.device,
            )
            / x_amax
        )
        x_scales_hp = (
            x_scale_blocks.abs().max(axis=-1).values
            / torch.tensor(
                scale_rule.max_allowed_e2m1_value(),
                dtype=x_amax.dtype,
                device=x_amax.device,
            )
            * encode_scale
        )

    if scale_expansion_factor is not None:
        x_scales_hp = x_scales_hp * scale_expansion_factor

    x_scales = x_scales_hp.to(torch.float8_e4m3fn)

    decode_scale = 1 / (
        torch.tensor(
            scale_rule.max_allowed_e2m1_value() * scale_rule.max_allowed_e4m3_value(),
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
        / x_amax
    )
    x_block_scaled = torch.where(
        x_scales.unsqueeze(1) != 0,
        x_scale_blocks * (1 / (decode_scale * x_scales.to(x_amax.dtype).unsqueeze(1))),
        0,
    )

    return x_block_scaled, x_scales


def select_fouroversix(
    x_scale_blocks: torch.Tensor,
    x_block_scaled_6: torch.Tensor,
    scales_6: torch.Tensor,
    x_block_scaled_4: torch.Tensor,
    scales_4: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: ScaleRule = ScaleRule.mse,
    round_style: RoundStyle = RoundStyle.nearest,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fake_quantized_6 = fake_quantize_to_e2m1(
        x_block_scaled_6,
        round_style=round_style,
    )
    x_fake_quantized_4 = fake_quantize_to_e2m1(
        x_block_scaled_4,
        round_style=round_style,
    )

    x_dequantized_6 = (
        x_fake_quantized_6.to(x_amax.dtype)
        * scales_6.unsqueeze(1).to(x_amax.dtype)
        * x_amax
        / torch.tensor(
            E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    )
    x_dequantized_4 = (
        x_fake_quantized_4.to(x_amax.dtype)
        * scales_4.unsqueeze(1).to(x_amax.dtype)
        * x_amax
        / torch.tensor(
            E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    )

    if scale_rule == ScaleRule.abs_max:
        x_error_4 = (x_dequantized_4 - x_scale_blocks).abs().max(axis=-1).values
        x_error_6 = (x_dequantized_6 - x_scale_blocks).abs().max(axis=-1).values
    elif scale_rule == ScaleRule.mae:
        x_error_4 = (x_dequantized_4 - x_scale_blocks).abs().sum(axis=-1)
        x_error_6 = (x_dequantized_6 - x_scale_blocks).abs().sum(axis=-1)
    elif scale_rule == ScaleRule.mse:
        x_error_4 = ((x_dequantized_4 - x_scale_blocks) ** 2).sum(axis=-1)
        x_error_6 = ((x_dequantized_6 - x_scale_blocks) ** 2).sum(axis=-1)

    select_4 = (x_error_4 < x_error_6).unsqueeze(1)
    x_fake_quantized = torch.where(
        select_4,
        x_fake_quantized_4.reshape(x_scale_blocks.shape[0], -1),
        x_fake_quantized_6.reshape(x_scale_blocks.shape[0], -1),
    )
    scales = torch.where(
        select_4,
        scales_4.reshape(-1, 1),
        scales_6.reshape(-1, 1),
    )

    return x_fake_quantized, scales


def quantize_to_fp4(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    block_scale_2d: bool = False,
    fp4_format: DataType = DataType.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: ScaleRule = ScaleRule.mse,
    transpose: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]
):
    if transpose:
        x = x.T

    if had is not None:
        x = (x.reshape(-1, had.shape[0]) @ had).reshape_as(x)

    if x_amax is None:
        x_amax = (
            torch.ones(1, device=x.device, dtype=x.dtype)
            if fp4_format == DataType.mxfp4
            else x.abs().max().float()
        )

    if block_scale_2d:
        assert x.ndim == 2  # noqa: PLR2004
        assert x.shape[1] % fp4_format.block_size() == 0

        x_scale_blocks = (
            x.reshape(
                -1,
                fp4_format.block_size(),
                x.shape[1] // fp4_format.block_size(),
                fp4_format.block_size(),
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, fp4_format.block_size() ** 2)
            .float()
        )
    else:
        x_scale_blocks = x.reshape(-1, fp4_format.block_size()).float()

    x_fake_quantized = None

    if fp4_format == DataType.mxfp4:
        x_block_scaled, scales = quantize_to_mxfp4(
            x_scale_blocks,
            scale_rule=scale_rule,
        )
    elif fp4_format == DataType.nvfp4 and scale_rule in {
        ScaleRule.static_6,
        ScaleRule.static_4,
    }:
        x_block_scaled, scales = quantize_to_nvfp4(
            x_scale_blocks,
            x_amax,
            scale_rule=scale_rule,
        )
    elif fp4_format == DataType.nvfp4:  # Four over six
        x_block_scaled_6, scales_6 = quantize_to_nvfp4(
            x_scale_blocks,
            x_amax,
            scale_rule=scale_rule,
        )
        x_block_scaled_4, scales_4 = quantize_to_nvfp4(
            x_scale_blocks,
            x_amax,
            scale_rule=scale_rule,
            scale_expansion_factor=1.5,
        )
        x_fake_quantized, scales = select_fouroversix(
            x_scale_blocks,
            x_block_scaled_6,
            scales_6,
            x_block_scaled_4,
            scales_4,
            x_amax,
            scale_rule=scale_rule,
            round_style=round_style,
        )
    else:
        msg = f"Invalid FP4 format: {fp4_format}"
        raise ValueError(msg)

    if x_fake_quantized is None:
        x_fake_quantized = fake_quantize_to_e2m1(
            x_block_scaled,
            round_style=round_style,
        )

    if block_scale_2d:
        x_fake_quantized = x_fake_quantized.reshape(
            -1,
            x.shape[1] // fp4_format.block_size(),
            fp4_format.block_size(),
            fp4_format.block_size(),
        ).permute(0, 2, 1, 3)

        scales = (
            scales.reshape(
                1,
                x.shape[0] // fp4_format.block_size(),
                x.shape[1] // fp4_format.block_size(),
            )
            .broadcast_to(
                fp4_format.block_size(),
                x.shape[0] // fp4_format.block_size(),
                x.shape[1] // fp4_format.block_size(),
            )
            .permute(1, 0, 2)
        )

    x_quantized = pack_unpacked_fp4(
        quantize_bf16_to_unpacked_fp4(x_fake_quantized.bfloat16().reshape_as(x)),
    )

    reshaped_scales = to_blocked(
        scales.reshape(
            x.shape[0],
            x.shape[1] // fp4_format.block_size(),
        ),
    )

    return x_quantized, reshaped_scales, x_amax
