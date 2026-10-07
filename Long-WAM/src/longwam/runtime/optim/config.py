# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/config.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


_SCALE_RULES = {"abs_max", "mae", "mse", "static_4", "static_6"}
_MODEL_TARGETS = {"video", "action"}
_COMPILE_TARGETS = {
    "observation_vae",
    "video_cross_kv_prefill",
    "video_clean_prefix_prefill",
    "video_denoiser",
    "video_kv_prefill",
    "action_cross_kv_prefill",
    "action_denoiser",
}
_TORCH_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
_KV_BACKENDS = {"auto", "cuda", "triton", "pytorch"}
_VIDEO_ROPE_GRID = (6, 7, 14)


def _tuple_of_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a string or sequence of strings, got {type(value)}")
    return tuple(str(item) for item in value)


def _compile_mode(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    if normalized == "auto":
        return "auto"
    raise ValueError(f"torch_compile must be true, false, or 'auto', got {value!r}")


@dataclass(frozen=True)
class InferenceOptimizationConfig:
    """Resolved inference-only optimization contract.

    Defaults are deliberately disabled so existing model configurations keep the
    exact BF16/eager behavior.
    """

    model_quant: bool = False
    model_quant_backend: str = "fouroversix"
    model_quant_targets: tuple[str, ...] = ("video", "action")
    model_quant_scale_rule: str = "mse"
    model_quant_activation_scale_rule: str = "mse"
    model_quant_weight_scale_rule: str = "mse"
    model_quant_quantize_backend: str = "cuda"
    model_quant_matmul_backend: str = "cutlass"

    kv_quant: bool = False
    kv_quant_scale_rule: str = "mse"
    kv_quant_backend: str = "cuda"
    kv_quant_quantize_backend: str = "cuda"
    kv_quant_allow_fallback: bool = True
    kv_quant_require_extension: bool = False

    resident_rope: bool = False
    video_rope_grid_size: tuple[int, int, int] = _VIDEO_ROPE_GRID

    torch_compile: bool | str = False
    torch_compile_targets: tuple[str, ...] = (
        "observation_vae",
        "video_denoiser",
        "video_kv_prefill",
        "action_cross_kv_prefill",
        "action_denoiser",
    )
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str | None = "max-autotune-no-cudagraphs"
    torch_compile_target_modes: tuple[tuple[str, str], ...] = ()
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
    torch_compile_suppress_errors: bool = True
    torch_compile_static_inputs: bool = False
    torch_compile_mark_static_addresses: bool = False
    observation_vae_channels_last: bool = False
    observation_vae_cudnn_benchmark_limit: int = 0
    observation_vae_internal_spatial_padding: bool = False
    observation_vae_first_frame_conv2d: bool = False
    video_cross_kv_cache: bool = False
    action_cross_kv_cache: bool = False
    action_video_kv_resident_cache: bool = False
    action_video_kv_direct_sink: bool = False
    action_gemm_autotune: bool = False
    action_packed_qkv: bool = False
    action_packed_cross_kv: bool = False
    action_segmented_attention: bool = False
    video_fp32_rope: bool = False
    action_fp32_rope: bool = False
    video_cudnn_self_sdpa: bool = False
    video_flex_group_causal_sdpa: bool = False
    video_nvfp4_bias_epilogue: bool = False
    video_self_qkv_shared_quant: bool = False
    video_self_qkv_direct_shared_quant: bool = False
    video_self_qkv_concat_gemm: bool = False
    video_self_qkv_batched_gemm: bool = False
    video_cross_kv_shared_quant: bool = False
    video_clean_prefix_wavefront: bool = False
    context_token_bucket: int | None = None

    required_compute_capability: str | None = "12.0"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "InferenceOptimizationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) and not hasattr(value, "items"):
            raise TypeError(f"inference_optimization must resolve to a mapping, got {type(value)}")

        raw = {str(key): item for key, item in value.items()}
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown inference_optimization fields: {unknown}")

        if "model_quant_targets" in raw:
            raw["model_quant_targets"] = _tuple_of_strings(
                raw["model_quant_targets"],
                field="model_quant_targets",
            )
        if "torch_compile_targets" in raw:
            raw["torch_compile_targets"] = _tuple_of_strings(
                raw["torch_compile_targets"],
                field="torch_compile_targets",
            )
        if "torch_compile_target_modes" in raw:
            modes = raw["torch_compile_target_modes"]
            if not isinstance(modes, Mapping) and not hasattr(modes, "items"):
                raise TypeError("torch_compile_target_modes must be a mapping")
            raw["torch_compile_target_modes"] = tuple(
                (str(target), str(mode)) for target, mode in modes.items()
            )
        if "video_rope_grid_size" in raw:
            grid = raw["video_rope_grid_size"]
            if not isinstance(grid, (list, tuple)) or len(grid) != 3:
                raise TypeError("video_rope_grid_size must be a three-element sequence")
            raw["video_rope_grid_size"] = tuple(int(item) for item in grid)
        if "torch_compile" in raw:
            raw["torch_compile"] = _compile_mode(raw["torch_compile"])

        config = cls(**raw)
        config.validate()
        return config

    @property
    def enabled(self) -> bool:
        return bool(
            self.model_quant
            or self.kv_quant
            or self.resident_rope
            or self.torch_compile
            or self.video_cross_kv_cache
            or self.action_cross_kv_cache
            or self.action_video_kv_resident_cache
            or self.action_video_kv_direct_sink
            or self.action_gemm_autotune
            or self.action_packed_qkv
            or self.action_packed_cross_kv
            or self.action_segmented_attention
            or self.observation_vae_first_frame_conv2d
            or self.video_fp32_rope
            or self.action_fp32_rope
            or self.video_cudnn_self_sdpa
            or self.video_flex_group_causal_sdpa
            or self.video_nvfp4_bias_epilogue
            or self.video_self_qkv_shared_quant
            or self.video_self_qkv_direct_shared_quant
            or self.video_self_qkv_batched_gemm
            or self.video_cross_kv_shared_quant
            or self.video_clean_prefix_wavefront
            or self.context_token_bucket is not None
        )

    def validate(self) -> None:
        if self.context_token_bucket is not None and (
            isinstance(self.context_token_bucket, bool)
            or not isinstance(self.context_token_bucket, int)
        ):
            raise TypeError("context_token_bucket must be an integer")
        if self.video_clean_prefix_wavefront:
            if not self.torch_compile:
                raise ValueError("video_clean_prefix_wavefront requires torch_compile")
            required_wavefront_targets = {
                "video_clean_prefix_prefill",
                "video_denoiser",
                "video_kv_prefill",
            }
            missing = required_wavefront_targets.difference(self.torch_compile_targets)
            if missing:
                raise ValueError(
                    "video_clean_prefix_wavefront requires "
                    "video_clean_prefix_prefill, video_denoiser, and "
                    "video_kv_prefill in torch_compile_targets"
                )
            if not self.model_quant or "video" not in self.model_quant_targets:
                raise ValueError("video_clean_prefix_wavefront requires video model_quant")
            if self.kv_quant:
                raise ValueError("video_clean_prefix_wavefront requires BF16 KV storage")
        if self.context_token_bucket is not None and self.context_token_bucket <= 0:
            raise ValueError("context_token_bucket must be positive")
        if self.model_quant_backend != "fouroversix":
            raise ValueError(
                "model_quant_backend currently supports only 'fouroversix', "
                f"got {self.model_quant_backend!r}"
            )
        invalid_targets = set(self.model_quant_targets) - _MODEL_TARGETS
        if invalid_targets:
            raise ValueError(f"Unsupported model_quant_targets: {sorted(invalid_targets)}")
        invalid_compile_targets = set(self.torch_compile_targets) - _COMPILE_TARGETS
        if invalid_compile_targets:
            raise ValueError(
                f"Unsupported torch_compile_targets: {sorted(invalid_compile_targets)}"
            )
        target_modes = dict(self.torch_compile_target_modes)
        if len(target_modes) != len(self.torch_compile_target_modes):
            raise ValueError("torch_compile_target_modes contains duplicate targets")
        invalid_mode_targets = set(target_modes) - set(self.torch_compile_targets)
        if invalid_mode_targets:
            raise ValueError(
                "torch_compile_target_modes must only override enabled targets: "
                f"{sorted(invalid_mode_targets)}"
            )
        invalid_modes = {
            target: mode
            for target, mode in target_modes.items()
            if mode not in _TORCH_COMPILE_MODES
        }
        if invalid_modes:
            raise ValueError(f"Unsupported per-target torch.compile modes: {invalid_modes}")
        if len(self.video_rope_grid_size) != 3 or any(
            int(item) <= 0 for item in self.video_rope_grid_size
        ):
            raise ValueError("video_rope_grid_size must contain three positive integers")
        if self.torch_compile_static_inputs and not self.torch_compile:
            raise ValueError("torch_compile_static_inputs requires torch_compile to be enabled")
        if self.torch_compile_mark_static_addresses and not self.torch_compile_static_inputs:
            raise ValueError(
                "torch_compile_mark_static_addresses requires torch_compile_static_inputs"
            )
        if self.observation_vae_channels_last and not self.torch_compile:
            raise ValueError("observation_vae_channels_last requires torch_compile to be enabled")
        if (
            self.observation_vae_channels_last
            and "observation_vae" not in self.torch_compile_targets
        ):
            raise ValueError(
                "observation_vae_channels_last requires observation_vae in torch_compile_targets"
            )
        if (
            isinstance(self.observation_vae_cudnn_benchmark_limit, bool)
            or not isinstance(self.observation_vae_cudnn_benchmark_limit, int)
            or self.observation_vae_cudnn_benchmark_limit < 0
        ):
            raise ValueError("observation_vae_cudnn_benchmark_limit must be a nonnegative integer")
        if (
            self.observation_vae_cudnn_benchmark_limit > 0
            and not self.observation_vae_channels_last
        ):
            raise ValueError(
                "observation_vae_cudnn_benchmark_limit requires observation_vae_channels_last"
            )
        if self.observation_vae_internal_spatial_padding and not self.observation_vae_channels_last:
            raise ValueError(
                "observation_vae_internal_spatial_padding requires observation_vae_channels_last"
            )
        if (
            self.observation_vae_first_frame_conv2d
            and not self.observation_vae_internal_spatial_padding
        ):
            raise ValueError(
                "observation_vae_first_frame_conv2d requires "
                "observation_vae_internal_spatial_padding"
            )
        if self.video_cross_kv_cache and not self.torch_compile:
            raise ValueError("video_cross_kv_cache requires torch_compile to be enabled")
        if self.video_cross_kv_cache and not {
            "video_cross_kv_prefill",
            "video_denoiser",
            "video_kv_prefill",
        }.issubset(self.torch_compile_targets):
            raise ValueError(
                "video_cross_kv_cache requires video_cross_kv_prefill, "
                "video_denoiser, and video_kv_prefill in torch_compile_targets"
            )
        if self.action_cross_kv_cache and not self.torch_compile:
            raise ValueError("action_cross_kv_cache requires torch_compile to be enabled")
        if self.action_cross_kv_cache and not {
            "action_cross_kv_prefill",
            "action_denoiser",
        }.issubset(self.torch_compile_targets):
            raise ValueError(
                "action_cross_kv_cache requires action_cross_kv_prefill and "
                "action_denoiser in torch_compile_targets"
            )
        if self.action_video_kv_resident_cache and not self.torch_compile:
            raise ValueError("action_video_kv_resident_cache requires torch_compile to be enabled")
        if (
            self.action_video_kv_resident_cache
            and "action_denoiser" not in self.torch_compile_targets
        ):
            raise ValueError(
                "action_video_kv_resident_cache requires action_denoiser in torch_compile_targets"
            )
        if self.action_video_kv_resident_cache and self.kv_quant:
            raise ValueError("action_video_kv_resident_cache currently requires BF16 video K/V")
        if self.action_video_kv_direct_sink and not self.action_video_kv_resident_cache:
            raise ValueError("action_video_kv_direct_sink requires action_video_kv_resident_cache")
        if (
            self.action_video_kv_direct_sink
            and "video_kv_prefill" not in self.torch_compile_targets
        ):
            raise ValueError(
                "action_video_kv_direct_sink requires video_kv_prefill in torch_compile_targets"
            )
        if self.action_video_kv_direct_sink:
            video_kv_mode = dict(self.torch_compile_target_modes).get(
                "video_kv_prefill",
                self.torch_compile_mode,
            )
            if video_kv_mode not in {"reduce-overhead", "max-autotune"}:
                raise ValueError(
                    "action_video_kv_direct_sink requires a CUDA-Graph-enabled "
                    "video_kv_prefill compile mode"
                )
        if self.action_gemm_autotune and not self.torch_compile:
            raise ValueError("action_gemm_autotune requires torch_compile to be enabled")
        if self.action_gemm_autotune and "action_denoiser" not in self.torch_compile_targets:
            raise ValueError(
                "action_gemm_autotune requires action_denoiser in torch_compile_targets"
            )
        if self.action_packed_qkv and not self.torch_compile:
            raise ValueError("action_packed_qkv requires torch_compile to be enabled")
        if self.action_packed_qkv and "action_denoiser" not in self.torch_compile_targets:
            raise ValueError("action_packed_qkv requires action_denoiser in torch_compile_targets")
        if self.action_packed_cross_kv and not self.torch_compile:
            raise ValueError("action_packed_cross_kv requires torch_compile to be enabled")
        if (
            self.action_packed_cross_kv
            and "action_cross_kv_prefill" not in self.torch_compile_targets
        ):
            raise ValueError(
                "action_packed_cross_kv requires action_cross_kv_prefill in torch_compile_targets"
            )
        if self.video_fp32_rope and not self.resident_rope:
            raise ValueError("video_fp32_rope requires resident_rope to be enabled")
        if self.video_fp32_rope and not self.torch_compile:
            raise ValueError("video_fp32_rope requires torch_compile to be enabled")
        if self.video_fp32_rope and not {
            "video_denoiser",
            "video_kv_prefill",
        }.issubset(self.torch_compile_targets):
            raise ValueError(
                "video_fp32_rope requires video_denoiser and video_kv_prefill "
                "in torch_compile_targets"
            )
        if self.action_fp32_rope and not self.resident_rope:
            raise ValueError("action_fp32_rope requires resident_rope to be enabled")
        if self.action_fp32_rope and not self.torch_compile:
            raise ValueError("action_fp32_rope requires torch_compile to be enabled")
        if self.action_fp32_rope and "action_denoiser" not in self.torch_compile_targets:
            raise ValueError("action_fp32_rope requires action_denoiser in torch_compile_targets")
        if self.video_cudnn_self_sdpa and not self.torch_compile:
            raise ValueError("video_cudnn_self_sdpa requires torch_compile to be enabled")
        if self.video_cudnn_self_sdpa and not {
            "video_denoiser",
            "video_kv_prefill",
        }.issubset(self.torch_compile_targets):
            raise ValueError(
                "video_cudnn_self_sdpa requires video_denoiser and "
                "video_kv_prefill in torch_compile_targets"
            )
        if self.video_flex_group_causal_sdpa and not self.torch_compile:
            raise ValueError("video_flex_group_causal_sdpa requires torch_compile to be enabled")
        if self.video_flex_group_causal_sdpa and not {
            "video_denoiser",
            "video_kv_prefill",
        }.issubset(self.torch_compile_targets):
            raise ValueError(
                "video_flex_group_causal_sdpa requires video_denoiser and "
                "video_kv_prefill in torch_compile_targets"
            )
        if self.video_flex_group_causal_sdpa and self.video_cudnn_self_sdpa:
            raise ValueError("video FlexAttention and cuDNN self-SDPA are mutually exclusive")
        if self.video_nvfp4_bias_epilogue and (
            not self.model_quant or "video" not in self.model_quant_targets
        ):
            raise ValueError("video_nvfp4_bias_epilogue requires video model_quant")
        if self.video_self_qkv_shared_quant and (
            not self.model_quant or "video" not in self.model_quant_targets
        ):
            raise ValueError("video_self_qkv_shared_quant requires video model_quant")
        if self.video_self_qkv_shared_quant and not self.torch_compile:
            raise ValueError("video_self_qkv_shared_quant requires torch_compile to be enabled")
        if self.video_self_qkv_shared_quant and not {
            "video_denoiser",
            "video_kv_prefill",
        }.issubset(self.torch_compile_targets):
            raise ValueError(
                "video_self_qkv_shared_quant requires video_denoiser and "
                "video_kv_prefill in torch_compile_targets"
            )
        if self.video_self_qkv_direct_shared_quant and not self.video_self_qkv_shared_quant:
            raise ValueError(
                "video_self_qkv_direct_shared_quant requires video_self_qkv_shared_quant"
            )
        if self.video_self_qkv_concat_gemm and not self.video_self_qkv_shared_quant:
            raise ValueError("video_self_qkv_concat_gemm requires video_self_qkv_shared_quant")
        if self.video_self_qkv_batched_gemm and not self.video_self_qkv_shared_quant:
            raise ValueError("video_self_qkv_batched_gemm requires video_self_qkv_shared_quant")
        if self.video_self_qkv_batched_gemm and self.video_self_qkv_concat_gemm:
            raise ValueError("video self QKV batched and concatenated GEMMs are mutually exclusive")
        if self.video_cross_kv_shared_quant and not self.video_cross_kv_cache:
            raise ValueError("video_cross_kv_shared_quant requires video_cross_kv_cache")
        if self.video_cross_kv_shared_quant and (
            not self.model_quant or "video" not in self.model_quant_targets
        ):
            raise ValueError("video_cross_kv_shared_quant requires video model_quant")
        if self.action_segmented_attention and not self.torch_compile:
            raise ValueError("action_segmented_attention requires torch_compile to be enabled")
        if self.action_segmented_attention and "action_denoiser" not in self.torch_compile_targets:
            raise ValueError(
                "action_segmented_attention requires action_denoiser in torch_compile_targets"
            )
        for field in (
            "model_quant_scale_rule",
            "model_quant_activation_scale_rule",
            "model_quant_weight_scale_rule",
            "kv_quant_scale_rule",
        ):
            rule = getattr(self, field)
            if rule not in _SCALE_RULES:
                raise ValueError(f"{field} must be one of {sorted(_SCALE_RULES)}, got {rule!r}")
        if self.model_quant_quantize_backend not in {"cuda", "triton", "pytorch"}:
            raise ValueError("model_quant_quantize_backend must be cuda, triton, or pytorch")
        if self.model_quant_matmul_backend not in {"cutlass", "pytorch"}:
            raise ValueError("model_quant_matmul_backend must be cutlass or pytorch")
        if self.kv_quant_backend not in _KV_BACKENDS:
            raise ValueError(
                f"kv_quant_backend must be one of {sorted(_KV_BACKENDS)}, "
                f"got {self.kv_quant_backend!r}"
            )
        if self.kv_quant_quantize_backend not in {"cuda", "triton", "pytorch"}:
            raise ValueError("kv_quant_quantize_backend must be cuda, triton, or pytorch")
        if self.kv_quant_require_extension and self.kv_quant_backend not in {
            "auto",
            "cuda",
        }:
            raise ValueError(
                "kv_quant_require_extension requires kv_quant_backend='cuda' or 'auto'"
            )
        _compile_mode(self.torch_compile)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_quant_targets"] = list(self.model_quant_targets)
        payload["torch_compile_targets"] = list(self.torch_compile_targets)
        payload["torch_compile_target_modes"] = dict(self.torch_compile_target_modes)
        payload["video_rope_grid_size"] = list(self.video_rope_grid_size)
        return payload
