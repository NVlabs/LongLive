# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""TorchAO FP8 post-training quantization helpers."""

import torch
import torch.nn as nn


# Small conditioning/output projections are both numerically sensitive and too
# small to amortize dynamic activation quantization on H100.
_BF16_MODULES = {
    "text_embedding.0",
    "text_embedding.2",
    "time_embedding.0",
    "time_embedding.2",
    "time_projection.1",
    "head.head",
}


def quantize_model_fp8(model: nn.Module, *, verbose: bool = False) -> int:
    """Quantize compatible BF16 linear layers to row-wise dynamic FP8 in-place."""
    if not torch.cuda.is_available():
        raise RuntimeError("TorchAO FP8 inference requires a CUDA GPU.")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Move the BF16 model to CUDA before applying FP8 quantization.")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("TorchAO FP8 inference requires compute capability 8.9 or newer.")

    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            PerRow,
            quantize_,
        )
    except ImportError as exc:
        raise ImportError(
            "FP8 inference requires a TorchAO version compatible with the installed PyTorch."
        ) from exc

    quantized_names = []
    skipped_names = []

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if fqn in _BF16_MODULES:
            skipped_names.append(fqn)
            return False
        if module.weight.dtype != torch.bfloat16:
            raise TypeError(f"FP8 layer {fqn!r} must be BF16, got {module.weight.dtype}.")
        out_features, in_features = module.weight.shape
        if in_features % 16 or out_features % 16:
            skipped_names.append(fqn)
            return False
        quantized_names.append(fqn)
        return True

    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        filter_fn=filter_fn,
    )
    if verbose:
        print(
            f"[FP8] TorchAO W8A8 quantized {len(quantized_names)} linear layers "
            f"with row-wise scaling; kept {len(skipped_names)} layers in BF16"
        )
    return len(quantized_names)
