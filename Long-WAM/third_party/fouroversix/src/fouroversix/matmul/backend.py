# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/mit-han-lab/fouroversix @ 4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb :: src/fouroversix/matmul/backend.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: third_party/fouroversix/src/fouroversix/matmul/backend.py
# End Long-WAM attribution.

from abc import ABC, abstractmethod

import torch
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import DataType


class MatmulBackendBase(ABC):
    """Base class for all matrix multiplication backends."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the backend is available on the current machine."""
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @classmethod
    @abstractmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> bool:
        """Return True if the backend supports the given inputs and output data type."""

        if not cls.is_available():
            return False

        if input.dtype != other.dtype:
            msg = "Both inputs must have the same dtype"
            raise ValueError(msg)

        if input.original_shape[1] != other.original_shape[1]:
            msg = (
                "The first input must be in row-major layout, the second input must be"
                "in column-major layout, and both inputs must have the same inner "
                "dimension"
            )
            raise ValueError(msg)

        return True

    @classmethod
    @abstractmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication (`a @ b.T`) between two quantized tensors using
        the backend.
        """
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)
