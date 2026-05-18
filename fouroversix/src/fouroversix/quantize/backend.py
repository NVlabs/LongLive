from abc import ABC, abstractmethod

import torch
from fouroversix.utils import DataType, ScaleRule

from .config import QuantizationConfig
from .quantized_tensor import QuantizedTensor


class QuantizeBackendBase(ABC):
    """Base class for all quantization backends."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the backend is available on the current machine."""
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @classmethod
    @abstractmethod
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the backend supports the given input and quantization
        configuration.
        """

        if not cls.is_available():
            return False

        if x.ndim != 2:  # noqa: PLR2004
            return False

        if config.dtype not in {DataType.mxfp4, DataType.nvfp4}:
            return False

        if config.dtype == DataType.mxfp4 and config.scale_rule not in {
            ScaleRule.static_6,
            ScaleRule.static_4,
        }:
            msg = (
                "MXFP4 quantization only supports the `static_6` and `static_4` scale "
                "rules"
            )
            raise ValueError(msg)

        return True

    @classmethod
    @abstractmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)
