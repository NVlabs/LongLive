import torch
import torch.nn.functional as F
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix

from .reference import quantize_to_fp4


class PyTorchQuantizeBackend(QuantizeBackendBase):
    """
    The PyTorch quantization backend. Supports all quantization options, and can be run
    on non-Blackwell GPUs, but is slow. Should be used primarily as a reference.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the PyTorch backend is available on the current machine."""
        return True

    @classmethod
    def is_supported(
        cls,
        x: torch.Tensor,  # noqa: ARG003
        config: QuantizationConfig,  # noqa: ARG003
    ) -> bool:
        """
        Return True if the PyTorch backend supports the given input and quantization
        configuration.
        """

        return True

    @classmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the PyTorch backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        input_shape = (x.shape[1], x.shape[0]) if config.transpose else x.shape

        rows_div = 128
        cols_div = 4 * config.dtype.block_size()

        if input_shape[0] % rows_div != 0 or input_shape[1] % cols_div != 0:
            x = F.pad(
                x,
                (
                    0,
                    (
                        cols_div - (input_shape[1] % cols_div)
                        if input_shape[1] % cols_div > 0
                        else 0
                    ),
                    0,
                    (
                        rows_div - (input_shape[0] % rows_div)
                        if input_shape[0] % rows_div > 0
                        else 0
                    ),
                ),
            )

        if x.device.type == "meta":
            values = torch.zeros(
                input_shape[0],
                input_shape[1] // 2,
                device=x.device,
                dtype=torch.uint8,
            )
            scale_factors = torch.zeros(
                input_shape[0] * input_shape[1] // config.dtype.block_size(),
                device=x.device,
                dtype=(
                    torch.uint8
                    if config.dtype.scale_dtype() == torch.float8_e8m0fnu
                    else config.dtype.scale_dtype()
                ),
            )
            amax = torch.zeros(1, device=x.device, dtype=torch.float32)
        else:
            values, scale_factors, amax = quantize_to_fp4(
                x,
                had=get_rht_matrix() if config.rht else None,
                fp4_format=config.dtype,
                round_style=config.round_style,
                scale_rule=config.scale_rule,
                block_scale_2d=config.block_scale_2d,
                transpose=config.transpose,
            )

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            input_shape,
            config.scale_rule,
        )
