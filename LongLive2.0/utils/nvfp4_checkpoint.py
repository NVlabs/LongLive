# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


NVFP4_CHECKPOINT_FORMAT = "longlive_generator_nvfp4"
TE_NVFP4_CHECKPOINT_FORMAT = "longlive_generator_te_nvfp4"
NVFP4_CHECKPOINT_VERSION = 1


def is_nvfp4_state_dict(state_dict: object) -> bool:
    """Return True when a state dict contains materialized FourOverSix NVFP4 buffers."""
    if not isinstance(state_dict, Mapping):
        return False
    return any(str(key).endswith("quantized_weight_values") for key in state_dict)


def is_te_nvfp4_checkpoint(checkpoint: object) -> bool:
    """Return True for checkpoints saved with TransformerEngine module state."""
    return (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("checkpoint_format") == TE_NVFP4_CHECKPOINT_FORMAT
    )


def unwrap_generator_state_dict(checkpoint: object, use_ema: bool = False) -> object:
    """Extract the generator state dict from common LongLive checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    if "generator" in checkpoint or "generator_ema" in checkpoint:
        ema_key = "generator_ema" if use_ema and "generator_ema" in checkpoint else "generator"
        return checkpoint[ema_key]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def clean_fsdp_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove FSDP wrapper prefixes used by some EMA checkpoints."""
    return {str(key).replace("_fsdp_wrapped_module.", ""): value for key, value in state_dict.items()}


def build_model_quantization_config(config, keep_master_weights: bool = False):
    from utils.quant import ModelQuantizationConfig

    quant_cfg = ModelQuantizationConfig(
        scale_rule=getattr(config, "model_quant_scale_rule", "static_6"),
        quantize_backend=getattr(config, "model_quant_backend", None),
        activation_scale_rule=getattr(
            config,
            "model_quant_activation_scale_rule",
            getattr(config, "model_quant_scale_rule", "static_6"),
        ),
        weight_scale_rule=getattr(config, "model_quant_weight_scale_rule", None),
        gradient_scale_rule=getattr(config, "model_quant_gradient_scale_rule", None),
    )
    quant_cfg.keep_master_weights = keep_master_weights
    return quant_cfg


def _maybe_to_dict(value):
    if value is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return dict(value)


def quantize_model_for_fouroversix_nvfp4(model: nn.Module, config, *, keep_master_weights: bool = False, verbose: bool = True):
    """Replace eligible modules with FourOverSix NVFP4 modules using the runtime config."""
    from utils.quant import quantize_model_with_filter

    return quantize_model_with_filter(
        model,
        quant_config=build_model_quantization_config(config, keep_master_weights=keep_master_weights),
        filtered_modules=getattr(config, "model_quant_filtered_modules", None),
        use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
        cast_model_to_bf16=True,
        materialize_for_inference=False,
        use_transformer_engine=False,
        verbose=verbose,
    )


def quantize_model_for_transformer_engine_nvfp4(
    model: nn.Module,
    config,
    *,
    keep_master_weights: bool = False,
    verbose: bool = True,
):
    """Replace eligible modules with TransformerEngine NVFP4 wrappers."""
    from utils.quant import quantize_model_with_filter

    use_transformer_engine = True
    te_inference_only = bool(getattr(config, "model_quant_te_inference_only", use_transformer_engine))
    te_low_precision_weights = bool(getattr(config, "model_quant_te_low_precision_weights", te_inference_only))
    te_fallback_to_fouroversix = bool(getattr(config, "model_quant_te_fallback_to_fouroversix", False))

    return quantize_model_with_filter(
        model,
        quant_config=build_model_quantization_config(config, keep_master_weights=keep_master_weights),
        filtered_modules=getattr(config, "model_quant_filtered_modules", None),
        use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
        cast_model_to_bf16=True,
        materialize_for_inference=False,
        use_transformer_engine=True,
        te_inference_only=te_inference_only,
        te_low_precision_weights=te_low_precision_weights,
        te_recipe_kwargs=_maybe_to_dict(getattr(config, "model_quant_te_recipe_kwargs", None)),
        te_module_kwargs=_maybe_to_dict(getattr(config, "model_quant_te_module_kwargs", None)),
        te_fallback_to_fouroversix=te_fallback_to_fouroversix,
        verbose=verbose,
    )


def drop_fouroversix_master_weights(model: nn.Module) -> list[str]:
    """Drop high-precision master weights after loading materialized NVFP4 buffers."""
    materialized_modules = []
    for module_name, module in model.named_modules():
        if not hasattr(module, "parameters_to_quantize"):
            continue

        parameters_to_quantize = getattr(module, "parameters_to_quantize", ())
        if callable(parameters_to_quantize):
            parameters_to_quantize = parameters_to_quantize()
        if not parameters_to_quantize:
            continue

        dropped_any = False
        for parameter_name in parameters_to_quantize:
            if isinstance(getattr(module, parameter_name, None), nn.Parameter):
                module.register_parameter(parameter_name, None)
                dropped_any = True
            elif hasattr(module, parameter_name):
                setattr(module, parameter_name, None)
                dropped_any = True

        if not dropped_any:
            continue
        for cache_name in ("_quantized_weight", "_quantized_weight_transposed", "_quantized_weights"):
            if hasattr(module, cache_name):
                delattr(module, cache_name)
        if hasattr(module, "config") and hasattr(module.config, "keep_master_weights"):
            module.config.keep_master_weights = False
        materialized_modules.append(module_name)
    return materialized_modules


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU state dict suitable for torch.save."""
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}
