# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/model_quant.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from longwam.utils.logging_config import get_logger

from .config import InferenceOptimizationConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExpertQuantizationReport:
    target: str
    eligible_modules: tuple[str, ...]
    materialized_modules: tuple[str, ...]
    master_weight_bytes: int
    quantized_weight_bytes: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_modules"] = list(self.eligible_modules)
        payload["materialized_modules"] = list(self.materialized_modules)
        return payload


def _is_core_dit_linear(name: str) -> bool:
    """Return whether an expert Linear belongs to the core block stack."""
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit():
        return False
    suffix = ".".join(parts[2:])
    if suffix in {
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    }:
        return True
    return False


def core_linear_inventory(expert: nn.Module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    eligible: list[str] = []
    excluded: list[str] = []
    for name, module in expert.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_core_dit_linear(name):
            eligible.append(name)
        else:
            excluded.append(name)
    return tuple(sorted(eligible)), tuple(sorted(excluded))


def _materialize_quantized_weights(
    model: nn.Module,
    *,
    target_device: torch.device,
) -> tuple[tuple[str, ...], int, int]:
    materialized_modules: list[str] = []
    master_weight_bytes = 0
    quantized_weight_bytes = 0

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not hasattr(module, "parameters_to_quantize") or not hasattr(
                module,
                "get_quantized_parameters",
            ):
                continue

            parameters_to_quantize = getattr(module, "parameters_to_quantize", ())
            if callable(parameters_to_quantize):
                parameters_to_quantize = parameters_to_quantize()
            did_materialize = False

            for parameter_name in parameters_to_quantize:
                parameter = getattr(module, parameter_name, None)
                if parameter is None:
                    continue
                if isinstance(parameter, nn.Parameter):
                    parameter_tensor = parameter.detach()
                elif isinstance(parameter, torch.Tensor):
                    parameter_tensor = parameter
                else:
                    continue

                master_weight_bytes += parameter_tensor.numel() * parameter_tensor.element_size()
                quantized_parameters = module.get_quantized_parameters(
                    parameter_name,
                    parameter_tensor,
                )
                for quantized_name, quantized_tensor in quantized_parameters.items():
                    if not isinstance(quantized_tensor, torch.Tensor):
                        continue
                    existing = getattr(module, quantized_name, None)
                    dst_dtype = (
                        existing.dtype
                        if isinstance(existing, torch.Tensor)
                        else quantized_tensor.dtype
                    )
                    quantized_tensor = quantized_tensor.to(
                        device=target_device,
                        dtype=dst_dtype,
                    )
                    setattr(module, quantized_name, quantized_tensor)
                    quantized_weight_bytes += (
                        quantized_tensor.numel() * quantized_tensor.element_size()
                    )

                if isinstance(getattr(module, parameter_name, None), nn.Parameter):
                    module.register_parameter(parameter_name, None)
                else:
                    setattr(module, parameter_name, None)
                did_materialize = True

            if did_materialize:
                for cached_name in (
                    "_quantized_weight",
                    "_quantized_weight_transposed",
                    "_quantized_weights",
                ):
                    if hasattr(module, cached_name):
                        delattr(module, cached_name)
                if hasattr(module, "config") and hasattr(
                    module.config,
                    "keep_master_weights",
                ):
                    module.config.keep_master_weights = False
                # Build the lightweight QuantizedTensor view before torch.compile.
                # Otherwise FourOverSix reconstructs shape metadata with GPU
                # `.tolist()` inside the first compiled forward.
                quantized_weight_fn = getattr(module, "quantized_weight", None)
                if callable(quantized_weight_fn):
                    quantized_weight_fn()
                quantized_weight_transposed_fn = getattr(
                    module,
                    "quantized_weight_transposed",
                    None,
                )
                if callable(quantized_weight_transposed_fn):
                    quantized_weight_transposed_fn()
                materialized_modules.append(module_name)

    return (
        tuple(sorted(materialized_modules)),
        master_weight_bytes,
        quantized_weight_bytes,
    )


def quantize_experts(
    model: nn.Module,
    config: InferenceOptimizationConfig,
) -> tuple[ExpertQuantizationReport, ...]:
    """Replace and materialize selected expert Linear layers as W4A4."""
    try:
        from fouroversix import ModelQuantizationConfig, quantize_model
    except ImportError as exc:
        raise ImportError(
            "FourOverSix is required for model_quant. Install it in the isolated "
            "Long-WAM quantization environment."
        ) from exc

    reports: list[ExpertQuantizationReport] = []
    target_device = torch.device(model.device)
    if target_device.type != "cuda":
        raise ValueError("FourOverSix model quantization requires the model on CUDA.")

    for target in config.model_quant_targets:
        attr = f"{target}_expert"
        expert = getattr(model, attr, None)
        if expert is None:
            raise ValueError(f"Selected model_quant target {target!r} has no {attr}.")

        eligible, excluded = core_linear_inventory(expert)
        if not eligible:
            raise RuntimeError(f"No eligible core Linear modules found for {target}.")

        quant_config = ModelQuantizationConfig(
            scale_rule=config.model_quant_scale_rule,
            activation_scale_rule=config.model_quant_activation_scale_rule,
            weight_scale_rule=config.model_quant_weight_scale_rule,
            quantize_backend=config.model_quant_quantize_backend,
            matmul_backend=config.model_quant_matmul_backend,
            keep_master_weights=False,
            modules_to_not_convert=list(excluded),
        )
        quantize_model(expert, quant_config)
        materialized, master_bytes, quantized_bytes = _materialize_quantized_weights(
            expert,
            target_device=target_device,
        )
        if set(materialized) != set(eligible):
            missing = sorted(set(eligible) - set(materialized))
            extra = sorted(set(materialized) - set(eligible))
            raise RuntimeError(
                f"{target} FourOverSix materialization inventory mismatch: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )

        report = ExpertQuantizationReport(
            target=target,
            eligible_modules=eligible,
            materialized_modules=materialized,
            master_weight_bytes=master_bytes,
            quantized_weight_bytes=quantized_bytes,
        )
        reports.append(report)
        logger.info(
            "[NVFP4][model] target=%s modules=%d master=%.3f GiB packed=%.3f GiB",
            target,
            len(materialized),
            master_bytes / (1024**3),
            quantized_bytes / (1024**3),
        )

    return tuple(reports)
