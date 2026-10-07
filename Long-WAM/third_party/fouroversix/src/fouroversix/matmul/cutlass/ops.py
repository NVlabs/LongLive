# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/mit-han-lab/fouroversix @ 4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb :: src/fouroversix/matmul/cutlass/ops.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: third_party/fouroversix/src/fouroversix/matmul/cutlass/ops.py
# End Long-WAM attribution.

import torch

import fouroversix._C  # noqa: F401


def gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake("fouroversix::gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt")
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake(
    "fouroversix::gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120",
)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake("fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt")
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake(
    "fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120",
)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake("fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt")
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.float16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake(
    "fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120",
)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.float16, device=a.device)
