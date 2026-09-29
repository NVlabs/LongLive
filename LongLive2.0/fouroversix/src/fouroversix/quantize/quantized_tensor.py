from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fouroversix.utils import DataType, ScaleRule

from .utils import to_blocked


def from_blocked(a: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = orig_shape
    return (
        a.view(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(-1, cols // 4, 128, 4)
        .transpose(1, 2)
        .reshape(rows, cols)
    )


def convert_e2m1_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    sign = (x >> 3) & 0x1
    exponent = (x >> 1) & 0x3
    mantissa = x & 0x1

    # Make adjustments
    new_exponent = torch.where(
        (exponent == 0) & (mantissa == 0),
        0,
        (exponent + 6) & 0xF,
    )
    new_mantissa = torch.where(exponent == 0, 0, mantissa << 2)

    return ((sign << 7) | (new_exponent << 3) | new_mantissa).view(torch.float8_e4m3fn)


def unpack_packed_fp4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if to_dtype == torch.float8_e4m3fn:
        convert_function = convert_e2m1_to_fp8_e4m3
    else:
        msg = f"Unsupported dtype: {to_dtype}"
        raise ValueError(msg)

    high = (x >> 4) & 0xF
    low = x & 0xF

    return torch.stack(
        [convert_function(low), convert_function(high)],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2)


@dataclass
class QuantizedTensor:
    """A quantized tensor."""

    values: torch.Tensor
    scale_factors: torch.Tensor
    amax: torch.Tensor

    dtype: DataType
    original_shape: tuple[int, int]
    scale_rule: ScaleRule

    padded_shape: tuple[int, int]

    def __init__(
        self,
        values: torch.Tensor,
        scale_factors: torch.Tensor,
        amax: torch.Tensor,
        dtype: DataType,
        original_shape: tuple[int, int],
        scale_rule: ScaleRule,
        padded_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str):
            dtype = DataType(dtype)

        if isinstance(original_shape, torch.Size):
            original_shape = tuple(original_shape)

        if isinstance(scale_rule, str):
            scale_rule = ScaleRule(scale_rule)

        if isinstance(padded_shape, torch.Size):
            padded_shape = tuple(padded_shape)

        self.dtype = dtype
        self.original_shape = original_shape
        self.scale_rule = scale_rule
        self.padded_shape = padded_shape

        if self.padded_shape is None:
            rows_div = 128

            # The scale factor layout requires 4 blocks along the K dimension for both
            # MXFP4 and NVFP4. See:
            # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
            cols_div = 4 * dtype.block_size()

            self.padded_shape = (
                original_shape[0]
                + (rows_div - original_shape[0] % rows_div) % rows_div,
                original_shape[1]
                + (cols_div - original_shape[1] % cols_div) % cols_div,
            )

            expected_packed_elements = self.padded_shape[0] * self.padded_shape[1] // 2
            expected_scale_factors = expected_packed_elements * 2 // dtype.block_size()

            if values.numel() != expected_packed_elements:
                values = F.pad(
                    values,
                    (
                        0,
                        # Divide by 2 because these are packed values
                        self.padded_shape[1] // 2 - values.shape[1],
                        0,
                        self.padded_shape[0] - values.shape[0],
                    ),
                )

            # If the scale factors are 1D, we assume that they are already in the
            # correct layout for Blackwell. See:
            # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
            if (
                scale_factors.ndim > 1
                and scale_factors.numel() != expected_scale_factors
            ):
                scale_factors = F.pad(
                    scale_factors,
                    (
                        0,
                        (
                            self.padded_shape[1] // dtype.block_size()
                            - scale_factors.shape[1]
                        ),
                        0,
                        self.padded_shape[0] - scale_factors.shape[0],
                    ),
                    value=0 if dtype == DataType.nvfp4 else 1,
                )

                scale_factors = to_blocked(scale_factors)

            if values.numel() != expected_packed_elements:
                msg = (
                    f"Expected {expected_packed_elements} e2m1 values, got "
                    f"{values.numel()}"
                )
                raise ValueError(msg)

            if scale_factors.numel() != expected_scale_factors:
                msg = (
                    f"Expected {expected_scale_factors} scale factors, got "
                    f"{scale_factors.numel()}"
                )
                raise ValueError(msg)
        # preprocess scale_factors
        # padded_shape = self.padded_shape
        # scales_2d = from_blocked(
        #     scale_factors,
        #     (padded_shape[0], padded_shape[1] // 16),
        # )

        # # 2. 正确计算 global scale: amax / (max_e2m1 * max_e4m3)
        # global_scale = amax / (6.0 * 256.0)
        # global_scale = amax / (
        #     self.scale_rule.max_allowed_e2m1_value()
        #     * self.scale_rule.max_allowed_e4m3_value()
        # )


        self.values = values
        self.scale_factors = scale_factors
        self.amax = amax

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return a high-precision tensor with the dequantized values (PyTorch impl)."""

        values = unpack_packed_fp4(self.values).to(dtype)
        scales = from_blocked(
            self.scale_factors,
            (
                self.padded_shape[0],
                self.padded_shape[1] // self.dtype.block_size(),
            ),
        )

        result = values * scales.to(dtype).repeat_interleave(
            self.dtype.block_size(),
            -1,
        )

        if self.dtype == DataType.nvfp4 and self.amax is not None:
            result = (
                result.to(torch.float32)
                * self.amax
                / (
                    self.scale_rule.max_allowed_e2m1_value()
                    * self.scale_rule.max_allowed_e4m3_value()
                )
            ).to(dtype)

        if result.shape != self.original_shape:
            result = result[: self.original_shape[0], : self.original_shape[1]]

        return result

    def dequantize_triton(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return a high-precision tensor with the dequantized values (Triton kernel)."""
        from utils.nvfp4_kernel import fp4_dequantize

        block_size = self.dtype.block_size()

        scales_2d = from_blocked(
            self.scale_factors,
            (self.padded_shape[0], self.padded_shape[1] // block_size),
        )

        global_scale = self.amax / (
            self.scale_rule.max_allowed_e2m1_value()
            * self.scale_rule.max_allowed_e4m3_value()
        )

        result = fp4_dequantize(
            self.values,
            scales_2d,
            global_scale,
            block_size=block_size,
            dtype=dtype,
        )

        if result.shape != self.original_shape:
            result = result[: self.original_shape[0], : self.original_shape[1]]

        return result

    @property
    def device(self) -> torch.device:
        """Get device of the values in this tensor."""
        return self.values.device
