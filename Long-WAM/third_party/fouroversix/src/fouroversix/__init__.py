# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/mit-han-lab/fouroversix @ 4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb :: src/fouroversix/__init__.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: third_party/fouroversix/src/fouroversix/__init__.py
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
