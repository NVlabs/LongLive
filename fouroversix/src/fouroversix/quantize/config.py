from dataclasses import dataclass

from fouroversix.utils import DataType, QuantizeBackend, RoundStyle, ScaleRule


@dataclass
class QuantizationConfig:
    """
    Configuration to use when quantizing a tensor.

    Args:
        backend (QuantizeBackend): The backend to use for quantization. If no backend is
            provided, a backend will be selected automatically based on the available
            GPU and the specified options.
        block_scale_2d (bool): If True, scale factors will be computed across 16x16
            chunks of the input rather than 1x16 chunks. This is useful to apply to the
            weight matrix during training, so that W and W.T will be equivalent after
            quantization.
        dtype (DataType): The data type to quantize to.
        rht (bool): If True, the random Hadamard transform will be applied to the input
            prior to quantization.
        round_style (RoundStyle): The rounding style to apply during quantization.
        scale_rule (ScaleRule): The scaling rule to use during quantization.
        transpose (bool): If True, the output will be a quantized version of the
            transposed input. This may be helpful for certain operations during training
            as `fp4_matmul` requires that both tensors are provided in row-major format.

    """

    backend: QuantizeBackend | None = None
    block_scale_2d: bool = False
    dtype: DataType = DataType.nvfp4
    rht: bool = False
    round_style: RoundStyle = RoundStyle.nearest
    scale_rule: ScaleRule = ScaleRule.mse
    transpose: bool = False

    def __post_init__(self) -> None:
        """Convert string values to enums."""

        if isinstance(self.backend, str):
            self.backend = QuantizeBackend(self.backend)

        if isinstance(self.dtype, str):
            self.dtype = DataType(self.dtype)

        if isinstance(self.round_style, str):
            self.round_style = RoundStyle(self.round_style)

        if isinstance(self.scale_rule, str):
            self.scale_rule = ScaleRule(self.scale_rule)
