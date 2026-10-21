# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Model-local substitution of original BF16 gate/modulation operations.

LayerNorm, Linear/LoRA, attention, schedules and original Python block bodies
remain unchanged. Only the named pointwise function resolves to a fused kernel.
Every BF16 intermediate is explicitly rounded, and implicit FMA is disabled.
"""

from functools import update_wrapper, lru_cache
import ast
import inspect
import textwrap
import types


def original_gate(x, gate, residual):
    return x + gate * residual


def original_modulate(x, shift, scale):
    return x * (1 + scale) + shift


@lru_cache(maxsize=8)
def verify_original_formula(function, expression):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = tree.body[0].body
    expected = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
    if (
        len(body) != 1
        or not isinstance(body[0], ast.Return)
        or ast.dump(body[0].value, include_attributes=False) != expected
    ):
        raise ValueError("Original pointwise formula changed; audit the new source before fusion")


def row_layout(value, x):
    if value.ndim not in (1, 2, 3) or value.shape[-1] != x.shape[-1] or value.stride(-1) != 1:
        raise ValueError("Unsupported pointwise broadcast layout")
    if value.ndim == 3 and value.shape[0] != 1:
        raise ValueError("Rounded pointwise operations require batch one")
    rows = 1 if value.ndim == 1 else int(value.shape[-2])
    if rows not in (1, int(x.shape[-2])):
        raise ValueError("Pointwise broadcast must be per-channel or per-token")
    return 0 if rows == 1 else value.stride(-2), rows == 1


def eligible(x, *values):
    import torch

    if (
        not all(
            value.device == x.device and value.dtype == torch.bfloat16 for value in (x, *values)
        )
        or x.device.type != "cuda"
    ):
        return False
    if torch.is_grad_enabled():
        raise RuntimeError("Rounded BF16 fusion is inference-only")
    if (
        x.ndim not in (2, 3)
        or (x.ndim == 3 and x.shape[0] != 1)
        or x.stride(-1) != 1
        or x.stride(-2) != x.shape[-1]
    ):
        raise ValueError("Rounded BF16 fusion requires dense token rows and batch one")
    return True


def rounded_gate(x, gate, residual):
    if not eligible(x, gate, residual):
        return original_gate(x, gate, residual)
    import torch
    import triton
    from .rounded_kernels import rounded_gate_kernel

    if (
        residual.shape != x.shape
        or residual.stride(-1) != 1
        or residual.stride(-2) != residual.shape[-1]
    ):
        raise ValueError("Gate residual rows must match the original token rows")
    stride, broadcast = row_layout(gate, x)
    out = torch.empty_like(x)
    rounded_gate_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        gate,
        residual,
        out,
        x.numel(),
        D=x.shape[-1],
        GS=stride,
        GB=broadcast,
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def rounded_modulate(x, shift, scale):
    if not eligible(x, shift, scale):
        return original_modulate(x, shift, scale)
    import torch
    import triton
    from .rounded_kernels import rounded_modulate_kernel

    shift_stride, shift_broadcast = row_layout(shift, x)
    scale_stride, scale_broadcast = row_layout(scale, x)
    out = torch.empty_like(x)
    rounded_modulate_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        shift,
        scale,
        out,
        x.numel(),
        D=x.shape[-1],
        SHS=shift_stride,
        SHB=shift_broadcast,
        SCS=scale_stride,
        SCB=scale_broadcast,
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def replace_modulate_lookup(original, substitute):
    """Clone only the function environment; never modify shared module globals."""
    function = original.__func__ if isinstance(original, types.MethodType) else original
    if "modulate" not in function.__code__.co_names:
        raise ValueError("Frozen block no longer resolves the original modulate helper")
    verify_original_formula(function.__globals__["modulate"], "x * (1 + scale) + shift")
    namespace = dict(function.__globals__, modulate=substitute)
    clone = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    update_wrapper(clone, function)
    return (
        types.MethodType(clone, original.__self__)
        if isinstance(original, types.MethodType)
        else clone
    )


class RoundedPointwise:
    def __init__(self, model):
        if model.training or model.joint_denoise:
            raise ValueError("Rounded pointwise policy requires sequential evaluation")
        self.model = model
        self.originals = []
        self.gates = self.blocks = 0
        for expert in (model.video_expert, model.action_expert):
            for block in expert.blocks:
                gate_function = block.gate.forward.__func__
                verify_original_formula(gate_function, "x + gate * residual")
                if any(True for _ in block.gate.parameters()):
                    raise ValueError("Do not substitute a gate with learned parameters")
                self.originals.append((block, "forward", block.forward))
                block.forward = replace_modulate_lookup(block.forward, rounded_modulate)
                self.originals.append((block.gate, "forward", block.gate.forward))
                block.gate.forward = rounded_gate
                self.gates += 1
                self.blocks += 1
        for name in ("_build_expert_attention_io", "_apply_expert_post_block"):
            original = getattr(model.mot, name)
            self.originals.append((model.mot, name, original))
            setattr(model.mot, name, replace_modulate_lookup(original, rounded_modulate))

    def telemetry(self):
        return {
            "gate_modules": self.gates,
            "block_forward_environments": self.blocks,
            "mot_function_environments": 2,
            "implicit_fp_fusion": False,
            "rounding": "Explicit rtne BF16 rounding after every original add/multiply",
            "other_dtypes": "Original Torch operations; no casting to BF16",
            "scope": "Model-local callables; shared module globals and parameter bindings unchanged",
        }
