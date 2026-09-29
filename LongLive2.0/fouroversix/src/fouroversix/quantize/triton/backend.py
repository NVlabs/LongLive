import functools

import torch
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import BLACKWELL_SM_IDS, SM_100, RoundStyle


class TritonQuantizeBackend(QuantizeBackendBase):
    """
    The Triton quantization backend. Supports all parameters required for efficient
    NVFP4 training, including stochastic rounding, the random Hadamard transform,
    transposed inputs, and 2D block scaling. Requires a Blackwell GPU.
    """

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        """Return True if the Triton backend is available on the current machine."""
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] in BLACKWELL_SM_IDS
        )

    @classmethod
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the Triton backend supports the given input and quantization
        configuration.
        """

        if not super().is_supported(x, config):
            return False

        if config.round_style == RoundStyle.stochastic:
            return torch.cuda.get_device_capability()[0] == SM_100

        return x.device.type == "cuda"

    @classmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the Triton backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        from .kernel import quantize_to_fp4

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
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
        )
