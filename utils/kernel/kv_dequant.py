# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch

try:
    from . import longlive_kv_dequant_cuda  # noqa: F401
except ImportError:
    import longlive_kv_dequant_cuda  # noqa: F401


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.float32:
        return 2
    raise ValueError(f"Unsupported fused KV dequant dtype: {dtype}")


def scale_rule_to_fp4_limits(scale_rule) -> tuple[float, float]:
    """Return the dequant denominator limits used by FourOverSix ScaleRule."""
    if hasattr(scale_rule, "max_allowed_e2m1_value") and hasattr(
        scale_rule, "max_allowed_e4m3_value",
    ):
        return (
            float(scale_rule.max_allowed_e2m1_value()),
            float(scale_rule.max_allowed_e4m3_value()),
        )

    normalized = str(scale_rule).lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    normalized = normalized.strip().strip("\"'")

    if normalized == "static_4":
        return 4.0, 448.0
    if normalized == "static_6":
        return 6.0, 448.0
    if normalized in {"mse", "mae", "l1_norm", "abs_max"}:
        return 6.0, 256.0

    raise ValueError(f"Unsupported FP4 scale_rule: {scale_rule}")


def dequantize_kv_cache_fp4(
    values: list[torch.Tensor],
    scale_factors: list[torch.Tensor],
    amax: list[torch.Tensor],
    *,
    num_heads: int,
    block_token_size: int,
    dtype: torch.dtype,
    e2m1_max: float | None = None,
    e4m3_max: float | None = None,
    scale_rule=None,
) -> torch.Tensor:
    """Dequantize multiple AR KV-cache chunks with one CUDA launch."""
    if e2m1_max is None or e4m3_max is None:
        if scale_rule is None:
            raise ValueError(
                "Either e2m1_max/e4m3_max or scale_rule must be provided.",
            )
        e2m1_max, e4m3_max = scale_rule_to_fp4_limits(scale_rule)

    return torch.ops.longlive_kernels.dequantize_kv_cache_fp4.default(
        values,
        scale_factors,
        amax,
        num_heads,
        block_token_size,
        _dtype_to_code(dtype),
        e2m1_max,
        e4m3_max,
    )
