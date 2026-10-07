# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/matmul/pytorch.py
# Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/matmul/pytorch.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.


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
