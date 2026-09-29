import functools

import torch
from fouroversix.matmul.backend import MatmulBackendBase
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import BLACKWELL_SM_IDS, SM_100, SM_120, DataType


class CUTLASSMatmulBackend(MatmulBackendBase):
    """
    The CUTLASS matrix multiplication backend. Uses CUTLASS kernels to perform fast
    FP4 matrix multiplication. Requires a Blackwell GPU.
    """

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        """Return True if the CUTLASS backend is available on the current machine."""

        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] not in BLACKWELL_SM_IDS
        ):
            return False

        try:
            import fouroversix._C  # noqa: F401
        except ModuleNotFoundError:
            return False

        return True

    @classmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> bool:
        """
        Return True if the CUTLASS backend supports the given inputs and output data
        type.
        """

        if not super().is_supported(input, other, out_dtype=out_dtype):
            return False

        return input.device.type == "cuda"

    @classmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication (`a @ b.T`) between two quantized tensors using
        the CUTLASS backend.
        """

        from .ops import (
            gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
            gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120,
            gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
            gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120,
            gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
            gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120,
        )

        out_shape = (input.original_shape[0], other.original_shape[0])

        if input.dtype == DataType.mxfp4:
            alpha = torch.ones(
                1,
                device=input.values.device,
                dtype=torch.float32,
            )
        elif input.dtype == DataType.nvfp4:
            alpha = (
                (input.amax * other.amax)
                / (
                    input.scale_rule.max_allowed_e2m1_value()
                    * input.scale_rule.max_allowed_e4m3_value()
                    * other.scale_rule.max_allowed_e2m1_value()
                    * other.scale_rule.max_allowed_e4m3_value()
                )
            ).to(torch.float32)

        gemm_fns = {
            (
                SM_100,
                DataType.mxfp4,
                DataType.bfloat16,
            ): gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
            (
                SM_120,
                DataType.mxfp4,
                DataType.bfloat16,
            ): gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120,
            (
                SM_100,
                DataType.nvfp4,
                DataType.bfloat16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
            (
                SM_120,
                DataType.nvfp4,
                DataType.bfloat16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120,
            (
                SM_100,
                DataType.nvfp4,
                DataType.float16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
            (
                SM_120,
                DataType.nvfp4,
                DataType.float16,
            ): gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120,
        }

        gemm_fn = gemm_fns.get(
            (torch.cuda.get_device_capability()[0], input.dtype, out_dtype),
        )

        if gemm_fn is None:
            msg = (
                "No gemm function found for the given device capability and "
                f"out_dtype: {torch.cuda.get_device_capability()[0]}, {out_dtype}"
            )
            raise ValueError(msg)

        out = gemm_fn(
            input.values,
            other.values,
            input.scale_factors,
            other.scale_factors,
            alpha,
        )

        if out_shape is not None and out.shape != out_shape:
            out = out[: out_shape[0], : out_shape[1]]

        return out
