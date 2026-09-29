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
