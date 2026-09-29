import functools

import torch
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.utils import BLACKWELL_SM_IDS, DataType, RoundStyle


class CUDAQuantizeBackend(QuantizeBackendBase):
    """
    The CUDA quantization backend. Supports basic quantization options (no 2D block
    scaling, no stochastic rounding, no random Hadamard transform). As a result, it can
    be used for inference, but not training. Requires a Blackwell GPU.
    """

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        """Return True if the CUDA backend is available on the current machine."""

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
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the CUDA backend supports the given input and quantization
        configuration.
        """

        if not super().is_supported(x, config):
            return False

        return (
            x.device.type == "cuda"
            and x.dtype in {torch.float16, torch.bfloat16}
            and config.round_style == RoundStyle.nearest
            and config.dtype == DataType.nvfp4
            and not config.transpose
        )

    @classmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the CUDA backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        from .ops import quantize_to_fp4

        values, scale_factors, amax = quantize_to_fp4(
            x,
            config.dtype == DataType.nvfp4,
            config.round_style == RoundStyle.nearest,
            config.rht,
            config.block_scale_2d,
            config.transpose,
            config.scale_rule.cuda_id(),
            0,
        )

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
        )