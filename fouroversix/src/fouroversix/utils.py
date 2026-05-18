from enum import Enum

import torch

SM_100 = 10
SM_110 = 11
SM_120 = 12

BLACKWELL_SM_IDS = {SM_100, SM_110, SM_120}


class DataType(str, Enum):
    """Data types."""

    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    mxfp4 = "mxfp4"
    nvfp4 = "nvfp4"

    def block_size(self) -> int | None:
        """Return the block size if this a block-scaled format, or `None` otherwise."""

        return {
            DataType.mxfp4: 32,
            DataType.nvfp4: 16,
        }.get(self)

    def scale_dtype(self) -> torch.dtype | None:
        """Return the scale dtype if this a block-scaled format, or `None` otherwise."""

        return {
            DataType.mxfp4: torch.float8_e8m0fnu,
            DataType.nvfp4: torch.float8_e4m3fn,
        }.get(self)

    def torch_dtype(self) -> torch.dtype | None:
        """
        Return the corresponding torch.dtype if one is available, or `None`
        otherwise.
        """

        return {
            DataType.bfloat16: torch.bfloat16,
            DataType.float16: torch.float16,
            DataType.float32: torch.float32,
        }.get(self)


class MatmulBackend(str, Enum):
    """
    Backends for matrix multiplication with FP4.

    - `cutlass`: CUTLASS implementation. This requires a Blackwell GPU.
    - `pytorch`: PyTorch implementation which first dequantizes the input tensors to
        FP32 and then performs an FP32 matrix multiplication.
    """

    cutlass = "cutlass"
    pytorch = "pytorch"


class QuantizeBackend(str, Enum):
    """
    Backends for quantizing a tensor to NVFP4 or MXFP4.

    - `cuda`: CUDA implementation. Requires a Blackwell GPU, and currently only supports
        the forward pass for PTQ (no stochastic rounding, no transposed matrices, no
        RHT, no 2D block scaling).
    - `pytorch`: PyTorch implementation.
    - `triton`: Triton implementation. Requires a Blackwell GPU.
    """

    cuda = "cuda"
    pytorch = "pytorch"
    transformer_engine = "transformer_engine"
    triton = "triton"


class RoundStyle(str, Enum):
    """
    Rounding styles for quantization.

    - `nearest`: Round to the nearest FP4 value.
    - `stochastic`: Round to the nearest FP4 value after applying random noise to each
        value.
    """

    nearest = "nearest"
    stochastic = "stochastic"


class ScaleRule(str, Enum):
    """
    Block scale selection rules for NVFP4 quantization.

    - `abs_max`: Between 4 and 6, select the block scale that minimizes the maximum
        absolute quantization error.
    - `static_4`: Select 4 for all blocks.
    - `static_6`: Select 6 for all blocks (normal NVFP4 quantization).
    - `mae`: Between 4 and 6, select the block scale that minimizes the mean absolute
        quantization error.
    - `mse`: Between 4 and 6, select the block scale that minimizes the mean squared
        quantization error.
    """

    abs_max = "abs_max"
    mae = "mae"
    mse = "mse"
    static_4 = "static_4"
    static_6 = "static_6"

    def cuda_id(self) -> int:
        """ID for the rule in the CUDA implementation."""

        return {
            ScaleRule.abs_max: 4,
            ScaleRule.mae: 2,
            ScaleRule.mse: 3,
            ScaleRule.static_4: 1,
            ScaleRule.static_6: 0,
        }[self]

    def is_static(self) -> bool:
        """Return True if the rule is static, False otherwise."""
        return self in {ScaleRule.static_4, ScaleRule.static_6}

    def max_allowed_e2m1_value(self) -> int:
        """Return the maximum allowed E2M1 value for the rule."""
        return 4 if self == ScaleRule.static_4 else 6

    def max_allowed_e4m3_value(self) -> int:
        """Return the maximum allowed E4M3 value for the rule."""
        return 448 if self in {ScaleRule.static_6, ScaleRule.static_4} else 256
