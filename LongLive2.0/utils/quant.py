# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import importlib
import inspect
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any
import warnings

import torch
import torch.nn as nn
from fouroversix import (
    DataType,
    ModelQuantizationConfig,
    QuantizationConfig,
    QuantizedTensor,
    RoundStyle,
    ScaleRule,
    quantize_model,
    quantize_to_fp4,
)
from fouroversix.quantize.quantized_tensor import from_blocked

from utils.nvfp4_kernel import fp4_dequantize

_FUSED_KV_DEQUANT_DISABLED = False
_FUSED_KV_DEQUANT_WARNED = False

QUANTIZATION_TYPE = {
    "weight": "weight",
    "activation": "activation",
    "kv": "kv",
}

DEFAULT_GENERATOR_FILTERED_MODULES = [
    "text_embedding.0",
    "text_embedding.2",
    "patch_embedding",
    "time_projection.1",
    "time_embedding.0",
    "time_embedding.2",
    "head.head",
    "head.modulation",
    "re:.*norm_k$",
    "re:.*norm_q$",
    "re:.*norm1$",
    "re:.*norm2$",
    "re:.*norm3$"
]

DEFAULT_REAL_SCORE_FILTERED_MODULES = list(DEFAULT_GENERATOR_FILTERED_MODULES)
DEFAULT_FAKE_SCORE_FILTERED_MODULES = list(DEFAULT_GENERATOR_FILTERED_MODULES)
DEFAULT_FILTERED_MODULES = list(DEFAULT_GENERATOR_FILTERED_MODULES)

FILTER_PROFILE_ALIASES = {
    "generator": "generator",
    "student": "generator",
    "real_score": "real_score",
    "teacher": "real_score",
    "fake_score": "fake_score",
    "critic": "fake_score",
}


@dataclass
class LongLiveQuantizationConfig(QuantizationConfig):
    type: str = "weight"

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.type, str):
            raise TypeError("Quantization type must be a string.")
        if self.type not in QUANTIZATION_TYPE:
            allowed = ", ".join(QUANTIZATION_TYPE.keys())
            raise ValueError(f"Unknown quantization type '{self.type}'. Expected one of: {allowed}.")

        self.type = QUANTIZATION_TYPE[self.type]


def _resolve_modules_to_not_convert(
    model: nn.Module,
    filtered_modules: list[str] | None,
) -> list[str]:
    if not filtered_modules:
        return []

    exact_names = set()
    regex_patterns = []
    for pattern in filtered_modules:
        if not isinstance(pattern, str):
            raise TypeError("Each filtered module pattern must be a string.")
        if pattern.startswith("re:"):
            regex_patterns.append(re.compile(pattern[3:]))
        else:
            exact_names.add(pattern)

    resolved = []
    for module_name, _ in model.named_modules():
        if not module_name:
            continue
        if module_name in exact_names or any(
            regex.search(module_name) for regex in regex_patterns
        ):
            resolved.append(module_name)

    return sorted(set(resolved))


def _get_default_filtered_modules(filter_profile: str | None) -> list[str]:
    if filter_profile is None:
        return list(DEFAULT_FILTERED_MODULES)

    normalized = FILTER_PROFILE_ALIASES.get(filter_profile, filter_profile)
    if normalized == "generator":
        return list(DEFAULT_GENERATOR_FILTERED_MODULES)
    if normalized == "real_score":
        return list(DEFAULT_REAL_SCORE_FILTERED_MODULES)
    if normalized == "fake_score":
        return list(DEFAULT_FAKE_SCORE_FILTERED_MODULES)

    allowed = ", ".join(sorted(FILTER_PROFILE_ALIASES))
    raise ValueError(
        f"Unknown filter_profile '{filter_profile}'. Expected one of: {allowed}.",
    )


def _warn_for_te_config_mismatch(model_quant_config: ModelQuantizationConfig) -> None:
    config_entries = [("default", model_quant_config)]
    module_overrides = getattr(model_quant_config, "module_config_overrides", None) or {}
    config_entries.extend(sorted(module_overrides.items()))

    mismatched_rules = []
    for module_name, module_config in config_entries:
        if getattr(module_config, "dtype", DataType.nvfp4) != DataType.nvfp4:
            raise NotImplementedError(
                "TransformerEngine replacement currently only supports NVFP4."
            )

        for attr_name in (
            "scale_rule",
            "activation_scale_rule",
            "weight_scale_rule",
            "gradient_scale_rule",
        ):
            rule = getattr(module_config, attr_name, None)
            if rule is not None and rule != ScaleRule.static_6:
                mismatched_rules.append(f"{module_name}:{attr_name}={rule}")

    if mismatched_rules:
        preview = ", ".join(mismatched_rules[:8])
        if len(mismatched_rules) > 8:
            preview += ", ..."
        warnings.warn(
            "TransformerEngine NVFP4 path maps to `NVFP4BlockScaling` and does not "
            "replicate FourOverSix non-`static_6` scale rules exactly. "
            f"Mismatched config entries: {preview}",
            stacklevel=3,
        )


def _build_te_recipe(module_config: Any, te_recipe_kwargs: dict[str, Any] | None = None):
    recipe_module = importlib.import_module("transformer_engine.common.recipe")
    NVFP4BlockScaling = recipe_module.NVFP4BlockScaling

    recipe_kwargs = {
        "disable_2d_quantization": not getattr(module_config, "weight_scale_2d", False),
        "disable_stochastic_rounding": (
            getattr(module_config, "gradient_round_style", RoundStyle.nearest)
            != RoundStyle.stochastic
        ),
        # FourOverSix only uses RHT in specific training paths, so keep TE conservative
        # by default and let callers override via `te_recipe_kwargs`.
        "disable_rht": True,
    }
    if te_recipe_kwargs:
        recipe_kwargs.update(te_recipe_kwargs)
    return NVFP4BlockScaling(**recipe_kwargs)


class TransformerEngineLinear(nn.Module):
    """A lightweight wrapper that routes a linear layer through TransformerEngine."""

    def __init__(
        self,
        module: nn.Linear,
        module_name: str,
        module_config: Any,
        inference_only: bool = False,
        low_precision_weights: bool = False,
        te_recipe_kwargs: dict[str, Any] | None = None,
        te_module_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        try:
            te = importlib.import_module("transformer_engine.pytorch")
        except ImportError as exc:
            raise ImportError(
                "TransformerEngine is not installed, but `use_transformer_engine=True` "
                "was requested."
            ) from exc

        if module.weight.device.type != "cuda":
            raise ValueError(
                "TransformerEngine replacement requires CUDA modules. "
                f"Module `{module_name}` is on `{module.weight.device}`."
            )

        self.module_name = module_name
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.inference_only = inference_only
        self.low_precision_weights = low_precision_weights
        self._te = te
        self._recipe = _build_te_recipe(
            module_config=module_config,
            te_recipe_kwargs=te_recipe_kwargs,
        )

        module_kwargs = dict(te_module_kwargs or {})
        module_kwargs.setdefault("device", module.weight.device)
        module_kwargs.setdefault("params_dtype", module.weight.dtype)
        module_kwargs.setdefault("name", module_name)

        fp8_model_init_fn = getattr(te, "fp8_model_init", None)
        if self.low_precision_weights and fp8_model_init_fn is None:
            warnings.warn(
                "TransformerEngine low-precision parameter init requested, but "
                "`fp8_model_init` is unavailable. Falling back to regular TE parameter "
                "storage for this inference path.",
                stacklevel=2,
            )
            self.low_precision_weights = False

        model_init_context = (
            fp8_model_init_fn(
                enabled=True,
                recipe=self._recipe,
                preserve_high_precision_init_val=False,
            )
            if self.low_precision_weights
            else nullcontext()
        )
        with model_init_context:
            self.linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                **module_kwargs,
            )

        self._load_from_linear(module)

        if self.inference_only:
            self.linear.requires_grad_(False)
            self.train(False)
        else:
            self.linear.weight.requires_grad_(module.weight.requires_grad)
            if self.linear.bias is not None and module.bias is not None:
                self.linear.bias.requires_grad_(module.bias.requires_grad)
            self.train(module.training)

    def _copy_tensor_into_parameter(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
    ) -> None:
        source = source.detach().to(device=destination.device)
        try:
            destination.copy_(source)
            return
        except Exception:
            pass

        destination.copy_(source.to(dtype=destination.dtype))

    def _load_from_linear(self, module: nn.Linear) -> None:
        with torch.no_grad():
            try:
                self._copy_tensor_into_parameter(self.linear.weight, module.weight)
                if module.bias is not None and self.linear.bias is not None:
                    self._copy_tensor_into_parameter(self.linear.bias, module.bias)
                return
            except Exception as copy_exc:
                state_dict = {
                    "weight": module.weight.detach().to(device=self.linear.weight.device),
                }
                if module.bias is not None:
                    state_dict["bias"] = module.bias.detach().to(
                        device=self.linear.weight.device,
                    )
                incompatible_keys = self.linear.load_state_dict(state_dict, strict=False)
                missing_keys = [
                    key
                    for key in getattr(incompatible_keys, "missing_keys", [])
                    if key != "_extra_state"
                ]
                unexpected_keys = list(getattr(incompatible_keys, "unexpected_keys", []))
                if missing_keys or unexpected_keys:
                    raise RuntimeError(
                        "Failed to load weights into TransformerEngine linear "
                        f"`{self.module_name}`. missing_keys={missing_keys}, "
                        f"unexpected_keys={unexpected_keys}"
                    ) from copy_exc

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def _autocast_context(self):
        autocast_fn = getattr(self._te, "autocast", None)
        if autocast_fn is not None:
            return autocast_fn(enabled=True, recipe=self._recipe)
        fp8_autocast_fn = getattr(self._te, "fp8_autocast", None)
        if fp8_autocast_fn is None:
            raise AttributeError(
                "TransformerEngine does not expose `autocast` or `fp8_autocast`."
            )
        return fp8_autocast_fn(enabled=True, fp8_recipe=self._recipe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with self._autocast_context():
            return self.linear(input)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"inference_only={self.inference_only}, "
            f"low_precision_weights={self.low_precision_weights}, "
            "backend=transformer_engine"
        )


def quantize_model_with_optional_te(
    model: nn.Module,
    model_quant_config: ModelQuantizationConfig,
    *,
    use_transformer_engine: bool = False,
    te_inference_only: bool = False,
    te_low_precision_weights: bool | None = None,
    te_recipe_kwargs: dict[str, Any] | None = None,
    te_module_kwargs: dict[str, Any] | None = None,
    te_fallback_to_fouroversix: bool = False,
    **kwargs,
) -> list[str]:
    """
    Quantize a model with FourOverSix by default, or replace `nn.Linear` with
    TransformerEngine wrappers when `use_transformer_engine=True`.
    """
    if not use_transformer_engine:
        quantize_model(model, model_quant_config, **kwargs)
        return []

    if te_low_precision_weights is None:
        te_low_precision_weights = te_inference_only

    if kwargs:
        if te_fallback_to_fouroversix:
            warnings.warn(
                "Additional kwargs passed to `quantize_model_with_optional_te` will "
                "only be forwarded to the fallback FourOverSix pass after "
                f"TransformerEngine replacement: {sorted(kwargs)}",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Additional kwargs passed to `quantize_model` are ignored in the "
                f"TransformerEngine path: {sorted(kwargs)}",
                stacklevel=2,
            )

    _warn_for_te_config_mismatch(model_quant_config)

    replaced_modules = []
    for module_name, module in list(model.named_modules()):
        if (
            module_name == ""
            or module_name in model_quant_config.modules_to_not_convert
            or not isinstance(module, nn.Linear)
        ):
            continue

        model.set_submodule(
            module_name,
            TransformerEngineLinear(
                module=module,
                module_name=module_name,
                module_config=model_quant_config.get_module_config(module_name),
                inference_only=te_inference_only,
                low_precision_weights=te_low_precision_weights,
                te_recipe_kwargs=te_recipe_kwargs,
                te_module_kwargs=te_module_kwargs,
            ),
        )
        replaced_modules.append(module_name)

    if te_fallback_to_fouroversix:
        quantize_model(model, model_quant_config, **kwargs)

    return replaced_modules


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _materialize_transformer_engine_weights_for_inference(
    model: nn.Module,
    target_device: torch.device | str | None = None,
    cache_transposed_weights: bool = False,
) -> tuple[list[str], int, int]:
    del cache_transposed_weights

    materialized_modules = []
    master_weight_bytes = 0
    quantized_weight_bytes = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, TransformerEngineLinear):
            continue

        if target_device is not None:
            module.to(device=torch.device(target_device))

        quantized_weight_bytes += _tensor_nbytes(module.weight)
        quantized_weight_bytes += _tensor_nbytes(module.bias)
        materialized_modules.append(module_name)

    return materialized_modules, master_weight_bytes, quantized_weight_bytes


def _materialize_mixed_quantized_weights_for_inference(
    model: nn.Module,
    target_device: torch.device | str | None = None,
    cache_transposed_weights: bool = False,
) -> tuple[list[str], int, int]:
    te_modules, te_master_bytes, te_quantized_bytes = (
        _materialize_transformer_engine_weights_for_inference(
            model,
            target_device=target_device,
            cache_transposed_weights=cache_transposed_weights,
        )
    )
    f46_modules, f46_master_bytes, f46_quantized_bytes = (
        _materialize_quantized_weights_for_inference(
            model,
            target_device=target_device,
            cache_transposed_weights=cache_transposed_weights,
        )
    )

    return (
        sorted(set(te_modules + f46_modules)),
        te_master_bytes + f46_master_bytes,
        te_quantized_bytes + f46_quantized_bytes,
    )


def _materialize_quantized_weights_for_inference(
    model: nn.Module,
    target_device: torch.device | str | None = None,
    cache_transposed_weights: bool = False,
) -> tuple[list[str], int, int]:
    """
    Materialize quantized weights and drop master weights.

    Optionally cache an additional transposed quantized layout for training paths that
    still require dgrad after the master weight is deleted (e.g. NVFP4 + LoRA).

    This function expects modules replaced by `fouroversix.quantize_model`.
    """
    materialized_modules = []
    master_weight_bytes = 0
    quantized_weight_bytes = 0

    for module_name, module in model.named_modules():
        if not hasattr(module, "parameters_to_quantize") or not hasattr(
            module, "get_quantized_parameters",
        ):
            continue

        parameters_to_quantize = getattr(module, "parameters_to_quantize", ())
        if callable(parameters_to_quantize):
            parameters_to_quantize = parameters_to_quantize()
        if not parameters_to_quantize:
            continue

        did_materialize = False
        for parameter_name in parameters_to_quantize:
            parameter = getattr(module, parameter_name, None)
            if parameter is None:
                continue

            if isinstance(parameter, nn.Parameter):
                parameter_tensor = parameter.data
            elif isinstance(parameter, torch.Tensor):
                parameter_tensor = parameter
            else:
                continue

            master_weight_bytes += parameter_tensor.numel() * parameter_tensor.element_size()
            get_quantized_parameters = module.get_quantized_parameters
            if (
                cache_transposed_weights
                and "include_transposed" in inspect.signature(
                    get_quantized_parameters,
                ).parameters
            ):
                quantized_params = get_quantized_parameters(
                    parameter_name,
                    parameter_tensor,
                    include_transposed=True,
                )
            else:
                quantized_params = get_quantized_parameters(
                    parameter_name,
                    parameter_tensor,
                )

            for quantized_name, quantized_tensor in quantized_params.items():
                if not isinstance(quantized_tensor, torch.Tensor):
                    continue

                existing = getattr(module, quantized_name, None)
                dst_dtype = (
                    existing.dtype
                    if isinstance(existing, torch.Tensor)
                    else quantized_tensor.dtype
                )
                if target_device is not None:
                    dst_device = torch.device(target_device)
                elif isinstance(existing, torch.Tensor):
                    dst_device = existing.device
                else:
                    dst_device = quantized_tensor.device

                quantized_tensor = quantized_tensor.to(
                    device=dst_device,
                    dtype=dst_dtype,
                )
                setattr(module, quantized_name, quantized_tensor)
                quantized_weight_bytes += (
                    quantized_tensor.numel() * quantized_tensor.element_size()
                )

            # Drop high-precision master weight once quantized weights are materialized.
            if isinstance(getattr(module, parameter_name, None), nn.Parameter):
                module.register_parameter(parameter_name, None)
            else:
                setattr(module, parameter_name, None)
            did_materialize = True

        if did_materialize:
            if hasattr(module, "_quantized_weight"):
                delattr(module, "_quantized_weight")
            if hasattr(module, "_quantized_weight_transposed"):
                delattr(module, "_quantized_weight_transposed")
            if hasattr(module, "_quantized_weights"):
                delattr(module, "_quantized_weights")
            if hasattr(module, "config") and hasattr(module.config, "keep_master_weights"):
                module.config.keep_master_weights = False
            materialized_modules.append(module_name)

    return materialized_modules, master_weight_bytes, quantized_weight_bytes


def quantize_model_with_filter(
    model: nn.Module,
    quant_config: ModelQuantizationConfig | dict | None = None,
    filtered_modules: list[str] | None = None,
    filter_profile: str | None = None,
    use_default_filtered_modules: bool = False,
    cast_model_to_bf16: bool = True,
    materialize_for_inference: bool = False,
    materialize_target_device: torch.device | str | None = None,
    use_transformer_engine: bool = False,
    te_inference_only: bool = False,
    te_low_precision_weights: bool | None = None,
    te_recipe_kwargs: dict[str, Any] | None = None,
    te_module_kwargs: dict[str, Any] | None = None,
    te_fallback_to_fouroversix: bool = False,
    verbose: bool = True,
    **kwargs,
) -> tuple[nn.Module, list[str]]:
    """
    Quantize model with FourOverSix and optionally skip selected modules.

    `filtered_modules` supports:
    - Exact module names, e.g. "head.head"
    - Regex patterns prefixed with "re:", e.g. "re:.*norm1$"

    `filter_profile` selects which built-in filtered module profile to use when
    `use_default_filtered_modules=True`. Supported values:
    "generator"/"student" and "real_score"/"teacher".
    """
    if quant_config is None:
        model_quant_config = ModelQuantizationConfig()
    elif isinstance(quant_config, dict):
        model_quant_config = ModelQuantizationConfig(**quant_config)
    elif isinstance(quant_config, ModelQuantizationConfig):
        model_quant_config = deepcopy(quant_config)
    else:
        raise TypeError(
            "quant_config must be ModelQuantizationConfig, dict, or None.",
        )

    patterns = list(filtered_modules or [])
    if use_default_filtered_modules:
        patterns = _get_default_filtered_modules(filter_profile) + patterns

    matched_modules = _resolve_modules_to_not_convert(model, patterns)
    modules_to_not_convert = set(model_quant_config.modules_to_not_convert or [])
    modules_to_not_convert.update(matched_modules)
    model_quant_config.modules_to_not_convert = sorted(modules_to_not_convert)

    if cast_model_to_bf16:
        model.to(torch.bfloat16)

    resolved_te_low_precision_weights = (
        te_inference_only if te_low_precision_weights is None else te_low_precision_weights
    )

    te_replaced_modules = quantize_model_with_optional_te(
        model,
        model_quant_config,
        use_transformer_engine=use_transformer_engine,
        te_inference_only=te_inference_only,
        te_low_precision_weights=resolved_te_low_precision_weights,
        te_recipe_kwargs=te_recipe_kwargs,
        te_module_kwargs=te_module_kwargs,
        te_fallback_to_fouroversix=te_fallback_to_fouroversix,
        **kwargs,
    )

    if materialize_for_inference:
        materialize_fn = _materialize_quantized_weights_for_inference
        if use_transformer_engine and te_fallback_to_fouroversix:
            materialize_fn = _materialize_mixed_quantized_weights_for_inference
        elif use_transformer_engine:
            materialize_fn = _materialize_transformer_engine_weights_for_inference

        materialized_modules, master_bytes, quantized_bytes = materialize_fn(
            model,
            target_device=materialize_target_device,
        )
        if verbose:
            print(
                "[quantize_model_with_filter] "
                f"materialized_modules={len(materialized_modules)}, "
                f"master_weight={master_bytes / (1024 ** 3):.3f} GiB, "
                f"quantized_weight={quantized_bytes / (1024 ** 3):.3f} GiB",
            )

    if verbose:
        profile_label = filter_profile or "default"
        print(
            "[quantize_model_with_filter] "
            f"profile={profile_label}, "
            f"matched={len(matched_modules)}, "
            f"total_excluded={len(model_quant_config.modules_to_not_convert)}",
        )
        if use_transformer_engine:
            print(
                "[quantize_model_with_filter] "
                f"transformer_engine_replaced={len(te_replaced_modules)}, "
                f"inference_only={te_inference_only}, "
                f"low_precision_weights={resolved_te_low_precision_weights}, "
                f"fallback_to_fouroversix={te_fallback_to_fouroversix}",
            )

    return model, matched_modules


def _dequantize_kv_cache_fused_cuda(kv_list, max_blocks, num_heads, block_token_size, dtype):
    global _FUSED_KV_DEQUANT_DISABLED, _FUSED_KV_DEQUANT_WARNED

    if _FUSED_KV_DEQUANT_DISABLED or max_blocks <= 0:
        return None

    first_qt = kv_list[0]
    if first_qt.values.device.type != "cuda":
        return None

    try:
        from utils.kernel.kv_dequant import dequantize_kv_cache_fp4

        blocks = kv_list[:max_blocks]
        values = [qt.values for qt in blocks]
        scale_factors = [qt.scale_factors for qt in blocks]
        amax = [qt.amax for qt in blocks]

        return dequantize_kv_cache_fp4(
            values,
            scale_factors,
            amax,
            num_heads=num_heads,
            block_token_size=block_token_size,
            dtype=dtype,
            scale_rule=first_qt.scale_rule,
        )
    except Exception as exc:  # pragma: no cover - exercised only when extension is stale/missing
        _FUSED_KV_DEQUANT_DISABLED = True
        if not _FUSED_KV_DEQUANT_WARNED:
            warnings.warn(
                "Fused CUDA KV-cache dequantization is unavailable; falling back to "
                f"the Triton per-block path. Reason: {exc}",
                stacklevel=2,
            )
            _FUSED_KV_DEQUANT_WARNED = True
        return None


def dequantize_kv_cache(kv_list, max_blocks, num_heads, block_token_size, dtype, device):
    """
    Dequantize list of QuantizedTensor to a contiguous bf16 tensor.
    kv_list[block_idx] -> QuantizedTensor(block_token_size * num_heads, 128)
    Returns: [1, max_blocks * block_token_size, num_heads, 128]
    """
    fused_result = _dequantize_kv_cache_fused_cuda(
        kv_list, max_blocks, num_heads, block_token_size, dtype,
    )
    if fused_result is not None:
        return fused_result

    total_tokens = max_blocks * block_token_size
    result = torch.zeros([1, total_tokens, num_heads, 128], dtype=dtype, device=device)
    for block_idx in range(max_blocks):
        t_start = block_idx * block_token_size
        t_end = t_start + block_token_size
        # deq = kv_list[block_idx].dequantize(dtype)
        # triton fp4_dequantize
        qt = kv_list[block_idx]
        padded_shape = qt.padded_shape
        scales_2d = from_blocked(
            qt.scale_factors,
            (padded_shape[0], padded_shape[1] // 16),
        )
        global_scale = qt.amax / (
            qt.scale_rule.max_allowed_e2m1_value()
            * qt.scale_rule.max_allowed_e4m3_value()
        )
        deq = fp4_dequantize(
            kv_list[block_idx].values,
            scales_2d,
            global_scale,
            block_size=16,
            dtype=dtype,
        )
        result[0, t_start:t_end, :, :] = deq.view(block_token_size, num_heads, 128)
    return result

def clone_quantized_tensor(qt):
    """Clone a QuantizedTensor by cloning its internal tensors."""
    return QuantizedTensor(
        values=qt.values.clone(),
        scale_factors=qt.scale_factors.clone(),
        amax=qt.amax.clone() if qt.amax is not None else None,
        dtype=qt.dtype,
        original_shape=qt.original_shape,
        scale_rule=qt.scale_rule,
        padded_shape=qt.padded_shape,
    )


def copy_quantized_into(slot: QuantizedTensor, src: QuantizedTensor) -> None:
    """In-place copy a QuantizedTensor's data into a pre-allocated slot.

    Keeps the slot's `values`/`scale_factors`/`amax` buffers persistent
    (their addresses don't change) so cudagraph allocator does not see them
    as fresh outputs that can be reused across step boundaries. Used by the
    quantized KV cache rolling/insert paths.
    """
    slot.values.copy_(src.values)
    slot.scale_factors.copy_(src.scale_factors)
    if src.amax is not None and slot.amax is not None:
        slot.amax.copy_(src.amax)


def k_smooth(k: torch.Tensor) -> torch.Tensor:
    return k - k.mean(dim=-1, keepdim=True)

def quantize_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, S, H, D = k.shape
    # B is always 1
    # S is the number of tokens
    # H is the number of heads
    # D is the dimension of the key and value

    config = QuantizationConfig(scale_rule="mse", backend="cuda")
    # per head quantization
    for head in range(H):
        k_head = k[:, :, head, :]
        v_head = v[:, :, head, :]
        k_head = k_smooth(k_head)
        v_head = v_head
        k_head = quantize_to_fp4(k_head, config)
        v_head = quantize_to_fp4(v_head, config)
        k[:, :, head, :] = k_head
        v[:, :, head, :] = v_head
    return k, v
