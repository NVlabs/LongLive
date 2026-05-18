
import torch
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import DataType

from .backend import MatmulBackendBase


class PyTorchMatmulBackend(MatmulBackendBase):
    """
    The PyTorch matrix multiplication backend. Dequantizes both inputs to FP32 and
    performs an FP32 matrix multiplication in order to simulate an NVFP4 matrix
    multiplication which accumulates in FP32. Slow, but can be run on any GPU.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the PyTorch backend is available on the current machine."""
        return True

    @classmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> torch.Tensor:
        """Perform a matrix multiplication (`a @ b.T`) between two quantized tensors."""

        out_shape = (input.original_shape[0], other.original_shape[0])

        out = torch.matmul(
            input.dequantize(dtype=torch.float32),
            other.dequantize(dtype=torch.float32).T,
        ).to(out_dtype.torch_dtype())

        if out.shape != out_shape:
            out = out[: out_shape[0], : out_shape[1]]

        return out
