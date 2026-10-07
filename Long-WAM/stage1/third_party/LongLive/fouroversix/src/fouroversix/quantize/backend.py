# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/quantize/backend.py
# Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/quantize/backend.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

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
