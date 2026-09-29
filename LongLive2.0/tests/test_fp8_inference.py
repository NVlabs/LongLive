# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from utils.fp8 import quantize_model_fp8
from utils.torch_compile_utils import SafeCompiledCallable


def test_fp8_quantization_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        quantize_model_fp8(nn.Linear(16, 16, dtype=torch.bfloat16))


def test_strict_compile_wrapper_does_not_shadow_torch(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)

    compiled = SafeCompiledCallable(
        lambda value: value + 1,
        name="test",
        mode=None,
        suppress_errors=False,
    )

    assert compiled(1) == 2
