# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/prepare.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch

from longwam.utils.logging_config import get_logger

from .compile import (
    CompileTargetState,
    _configure_observation_vae_channels_last,
    _configure_observation_vae_cudnn_benchmark,
    _configure_observation_vae_internal_spatial_padding,
    _configure_observation_vae_first_frame_conv2d,
    configure_torch_compile,
    enable_edge_inductor_gemm_templates,
)
from .config import InferenceOptimizationConfig
from .model_quant import ExpertQuantizationReport, quantize_experts

logger = get_logger(__name__)


@dataclass
class InferenceOptimizationState:
    prepared: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    model_quant: list[dict[str, Any]] = field(default_factory=list)
    kv_quant_enabled: bool = False
    resident_rope: dict[str, Any] = field(default_factory=dict)
    packed_action_qkv_layers: int = 0
    packed_action_cross_kv_layers: int = 0
    video_nvfp4_bias_epilogue_modules: int = 0
    shared_video_self_qkv_quantization_layers: int = 0
    direct_shared_video_self_qkv_quantization_layers: int = 0
    concatenated_video_self_qkv_gemm_layers: int = 0
    batched_video_self_qkv_gemm_layers: int = 0
    shared_video_cross_kv_quantization_layers: int = 0
    compile_targets: dict[str, CompileTargetState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compile_targets"] = {
            name: state.to_dict() for name, state in self.compile_targets.items()
        }
        return payload


def _actual_model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device(getattr(model, "device", "cpu"))


def _validate_runtime(
    model: torch.nn.Module,
    config: InferenceOptimizationConfig,
) -> torch.device:
    device = _actual_model_device(model)
    if config.enabled and device.type != "cuda":
        raise ValueError(f"Inference optimizations require a CUDA-resident model, got {device}.")
    if config.enabled and config.required_compute_capability:
        expected = tuple(int(part) for part in str(config.required_compute_capability).split("."))
        if len(expected) != 2:
            raise ValueError(
                "required_compute_capability must have major.minor form, "
                f"got {config.required_compute_capability!r}"
            )
        actual = torch.cuda.get_device_capability(device)
        if actual != expected:
            raise RuntimeError(
                "Inference optimization GPU mismatch: "
                f"required CC {expected[0]}.{expected[1]}, "
                f"got {actual[0]}.{actual[1]} ({torch.cuda.get_device_name(device)})."
            )
    return device


def prepare_inference_optimizations(
    model: torch.nn.Module,
    config: InferenceOptimizationConfig | Mapping[str, Any] | None = None,
) -> InferenceOptimizationState:
    """Apply post-checkpoint, post-LoRA inference transformations once."""
    if getattr(model, "_inference_optimization_prepared", False):
        return model._inference_optimization_state

    if config is None:
        config = getattr(model, "_inference_optimization_config", None)
    resolved = InferenceOptimizationConfig.from_mapping(config)
    device = _validate_runtime(model, resolved)
    state = InferenceOptimizationState(config=resolved.to_dict())
    if not resolved.enabled:
        state.prepared = True
        model._inference_optimization_prepared = True
        model._inference_optimization_state = state
        return state

    if model.training:
        raise RuntimeError("prepare_inference_optimizations() must run after model.eval().")
    model.requires_grad_(False)
    model._context_token_bucket = resolved.context_token_bucket

    if resolved.action_gemm_autotune:
        enabled = enable_edge_inductor_gemm_templates(device)
        if not enabled:
            capability = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                "action_gemm_autotune is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        model._edge_action_gemm_autotune_enabled = True

    if resolved.action_packed_qkv:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "action_packed_qkv is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot"):
            raise ValueError("action_packed_qkv requires model.mot")
        state.packed_action_qkv_layers = model.mot.configure_expert_packed_qkv("action")

    if resolved.action_packed_cross_kv:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "action_packed_cross_kv is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot"):
            raise ValueError("action_packed_cross_kv requires model.mot")
        state.packed_action_cross_kv_layers = model.mot.configure_expert_packed_cross_kv("action")

    if resolved.action_segmented_attention:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "action_segmented_attention is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot") or not hasattr(model, "action_expert"):
            raise ValueError(
                "action_segmented_attention requires model.mot and model.action_expert"
            )
        model.mot.action_segmented_attention = True
        model._action_segmented_attention_enabled = True

    if resolved.video_cross_kv_cache:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "video_cross_kv_cache is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError("video_cross_kv_cache requires model.mot and model.video_expert")
        model._video_cross_kv_cache_enabled = True

    if resolved.action_cross_kv_cache:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "action_cross_kv_cache is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot") or not hasattr(model, "action_expert"):
            raise ValueError("action_cross_kv_cache requires model.mot and model.action_expert")
        model._action_cross_kv_cache_enabled = True

    if resolved.action_video_kv_resident_cache:
        capability = torch.cuda.get_device_capability(device)
        if capability != (11, 0):
            raise RuntimeError("action_video_kv_resident_cache is validated only on Thor CC 11.0")
        if not hasattr(model, "mot") or not hasattr(model, "action_expert"):
            raise ValueError(
                "action_video_kv_resident_cache requires model.mot and model.action_expert"
            )
        model._action_video_kv_resident_cache_enabled = True

    if resolved.video_fp32_rope:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "video_fp32_rope is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError("video_fp32_rope requires model.mot and model.video_expert")
        model.mot.configure_expert_rope_compute_dtype(
            "video",
            torch.float32,
            direct_output=True,
        )
        model._video_fp32_rope_enabled = True

    if resolved.action_fp32_rope:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "action_fp32_rope is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot") or not hasattr(model, "action_expert"):
            raise ValueError("action_fp32_rope requires model.mot and model.action_expert")
        model.mot.configure_expert_rope_compute_dtype(
            "action",
            torch.float32,
            direct_output=True,
        )
        model._action_fp32_rope_enabled = True

    if resolved.video_cudnn_self_sdpa:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "video_cudnn_self_sdpa is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not torch.backends.cudnn.is_available():
            raise RuntimeError("video_cudnn_self_sdpa requires cuDNN")
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError("video_cudnn_self_sdpa requires model.mot and model.video_expert")
        model.mot.configure_expert_self_sdpa_backend("video", "cudnn")
        model._video_cudnn_self_sdpa_enabled = True

    if resolved.video_flex_group_causal_sdpa:
        capability = torch.cuda.get_device_capability(device)
        if capability != (11, 0):
            raise RuntimeError("video_flex_group_causal_sdpa is validated only on Thor CC 11.0")
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError(
                "video_flex_group_causal_sdpa requires model.mot and model.video_expert"
            )
        if model.video_expert.video_attention_mask_mode != "per_frame_causal":
            raise ValueError(
                "video_flex_group_causal_sdpa requires per_frame_causal video attention"
            )
        from longwam.models.wan22.wan_video_dit import (
            build_flex_group_causal_mask,
        )

        f, h, w = (int(value) for value in resolved.video_rope_grid_size)
        tokens_per_frame = h * w
        sequence_length = f * tokens_per_frame
        if (f, h, w) != (6, 7, 14):
            raise RuntimeError("video_flex_group_causal_sdpa is validated only for grid [6,7,14]")
        block_mask = build_flex_group_causal_mask(
            device=device,
            batch_size=1,
            num_heads=int(model.video_expert.num_heads),
            sequence_length=sequence_length,
            tokens_per_frame=tokens_per_frame,
        )
        if resolved.video_clean_prefix_wavefront:
            clean_sequence_length = int(model.num_clean_frames) * tokens_per_frame
            future_sequence_length = sequence_length - clean_sequence_length
            if (clean_sequence_length, future_sequence_length) != (392, 196):
                raise RuntimeError(
                    "Thor clean-prefix wavefront FlexAttention is validated only "
                    "for Primary clean/future token lengths 392/196"
                )
            model.mot._video_clean_prefix_sdpa_block_mask = build_flex_group_causal_mask(
                device=device,
                batch_size=1,
                num_heads=int(model.video_expert.num_heads),
                sequence_length=clean_sequence_length,
                tokens_per_frame=tokens_per_frame,
            )
            model.mot._video_future_sdpa_block_mask = build_flex_group_causal_mask(
                device=device,
                batch_size=1,
                num_heads=int(model.video_expert.num_heads),
                sequence_length=sequence_length,
                tokens_per_frame=tokens_per_frame,
                query_length=future_sequence_length,
                query_start=clean_sequence_length,
            )
        model.mot.configure_expert_self_sdpa_backend(
            "video",
            "flex-group-causal",
            block_mask=block_mask,
        )
        model._video_flex_group_causal_sdpa_enabled = True

    if resolved.video_clean_prefix_wavefront:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "video_clean_prefix_wavefront is unvalidated on compute capability "
                f"{capability[0]}.{capability[1]}"
            )
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError(
                "video_clean_prefix_wavefront requires model.mot and model.video_expert"
            )
        if model.video_expert.video_attention_mask_mode != "per_frame_causal":
            raise ValueError(
                "video_clean_prefix_wavefront requires per_frame_causal video attention"
            )
        model._video_clean_prefix_wavefront_enabled = True

    reports: tuple[ExpertQuantizationReport, ...] = ()
    if resolved.model_quant:
        reports = quantize_experts(model, resolved)
        state.model_quant = [report.to_dict() for report in reports]
        torch.cuda.empty_cache()

    if resolved.video_nvfp4_bias_epilogue:
        capability = torch.cuda.get_device_capability(device)
        if capability != (11, 0):
            raise RuntimeError("video_nvfp4_bias_epilogue is validated only on Thor CC 11.0")
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError("video_nvfp4_bias_epilogue requires model.mot and model.video_expert")
        state.video_nvfp4_bias_epilogue_modules = model.mot.configure_expert_nvfp4_bias_epilogue(
            "video"
        )
        model._video_nvfp4_bias_epilogue_enabled = True

    if resolved.video_self_qkv_shared_quant:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError("video_self_qkv_shared_quant requires a supported edge GPU")
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError(
                "video_self_qkv_shared_quant requires model.mot and model.video_expert"
            )
        state.shared_video_self_qkv_quantization_layers = (
            model.mot.configure_expert_shared_self_qkv_quantization("video")
        )
        model._video_self_qkv_shared_quant_enabled = True

    if resolved.video_self_qkv_direct_shared_quant:
        capability = torch.cuda.get_device_capability(device)
        if capability != (11, 0):
            raise RuntimeError(
                "video_self_qkv_direct_shared_quant is validated only on Thor CC 11.0"
            )
        state.direct_shared_video_self_qkv_quantization_layers = (
            model.mot.configure_expert_direct_shared_self_qkv_quantization("video")
        )
        model._video_self_qkv_direct_shared_quant_enabled = True

    if resolved.video_self_qkv_concat_gemm:
        capability = torch.cuda.get_device_capability(device)
        if capability != (11, 0):
            raise RuntimeError("video_self_qkv_concat_gemm is validated only on Thor CC 11.0")
        state.concatenated_video_self_qkv_gemm_layers = (
            model.mot.configure_expert_concatenated_self_qkv_gemm("video")
        )
        model._video_self_qkv_concat_gemm_enabled = True

    if resolved.video_self_qkv_batched_gemm:
        capability = torch.cuda.get_device_capability(device)
        if capability != (11, 0):
            raise RuntimeError("video_self_qkv_batched_gemm is validated only on Thor CC 11.0")
        state.batched_video_self_qkv_gemm_layers = model.mot.configure_expert_batched_self_qkv_gemm(
            "video"
        )
        model._video_self_qkv_batched_gemm_enabled = True

    if resolved.video_cross_kv_shared_quant:
        capability = torch.cuda.get_device_capability(device)
        if capability not in {(11, 0), (12, 0), (12, 1)}:
            raise RuntimeError("video_cross_kv_shared_quant requires a supported edge GPU")
        if not hasattr(model, "mot") or not hasattr(model, "video_expert"):
            raise ValueError(
                "video_cross_kv_shared_quant requires model.mot and model.video_expert"
            )
        state.shared_video_cross_kv_quantization_layers = (
            model.mot.configure_expert_shared_cross_kv_quantization("video")
        )
        model._video_cross_kv_shared_quant_enabled = True

    if resolved.kv_quant:
        raise ValueError("Published runtime keeps KV storage in BF16")

    if resolved.resident_rope:
        action_expert = getattr(model, "action_expert", None)
        video_expert = getattr(model, "video_expert", None)
        if action_expert is None or video_expert is None:
            raise ValueError("resident_rope requires model.action_expert and model.video_expert")
        action_expert.configure_action_rope_mode("gpu_resident")
        action_expert.prepare_action_rope_cache(
            device,
            storage_dtype=(torch.float32 if resolved.action_fp32_rope else torch.float64),
        )
        video_expert.configure_video_rope_mode("gpu_resident")
        video_expert.prepare_video_rope_cache(
            device=device,
            grid_size=resolved.video_rope_grid_size,
            storage_dtype=(torch.float32 if resolved.video_fp32_rope else torch.float64),
        )
        torch.cuda.synchronize(device)
        state.resident_rope = {
            "action": action_expert.action_rope_descriptor(),
            "video": video_expert.video_rope_descriptor(),
        }

    state.compile_targets = configure_torch_compile(model, resolved)
    if resolved.observation_vae_channels_last:
        conv2d, conv3d, deterministic = _configure_observation_vae_channels_last(
            model,
            device,
        )
        vae_state = state.compile_targets["observation_vae"]
        vae_state.channels_last_conv2d = conv2d
        vae_state.channels_last_conv3d = conv3d
        vae_state.cudnn_deterministic = deterministic
        if resolved.observation_vae_internal_spatial_padding:
            vae_state.internal_spatial_padding_conv3d = (
                _configure_observation_vae_internal_spatial_padding(
                    model,
                    device,
                )
            )
        if resolved.observation_vae_first_frame_conv2d:
            (
                vae_state.first_frame_conv2d,
                vae_state.first_frame_conv2d_bytes,
            ) = _configure_observation_vae_first_frame_conv2d(model, device)
        if resolved.observation_vae_cudnn_benchmark_limit > 0:
            vae_state.cudnn_benchmark = _configure_observation_vae_cudnn_benchmark(
                device,
                resolved.observation_vae_cudnn_benchmark_limit,
            )
            vae_state.cudnn_benchmark_limit = resolved.observation_vae_cudnn_benchmark_limit
    state.prepared = True
    model._inference_optimization_prepared = True
    model._inference_optimization_state = state
    logger.info(
        "[inference_optimization] prepared device=%s model_quant=%s "
        "kv_quant=%s packed_action_qkv_layers=%s resident_rope=%s compile=%s",
        device,
        [report.target for report in reports],
        state.kv_quant_enabled,
        state.packed_action_qkv_layers,
        bool(state.resident_rope),
        sorted(state.compile_targets),
    )
    return state
