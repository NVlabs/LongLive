# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/__init__.py
# Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/__init__.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

from importlib.metadata import version

from .matmul import fp4_matmul
from .model import (
    FourOverSixLinear,
    ModelQuantizationConfig,
    ModuleQuantizationConfig,
    QuantizedModule,
    quantize_model,
)
from .quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from .utils import DataType, MatmulBackend, QuantizeBackend, RoundStyle, ScaleRule
from .weight_conversions import WeightConversions

__version__ = version("fouroversix")

__all__ = [
    "DataType",
    "FourOverSixLinear",
    "MatmulBackend",
    "ModelQuantizationConfig",
    "ModuleQuantizationConfig",
    "QuantizationConfig",
    "QuantizeBackend",
    "QuantizedModule",
    "QuantizedTensor",
    "RoundStyle",
    "ScaleRule",
    "WeightConversions",
    "fp4_matmul",
    "quantize_model",
    "quantize_to_fp4",
]
