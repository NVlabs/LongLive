# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/backends/__init__.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Explicit platform profiles for the shared inference implementation."""

CAPABILITIES = {"rtx5090": (12, 0), "spark": (12, 1), "thor": (11, 0)}


def prepare_backend(model, hardware="reference", overrides=None):
    """Apply transformations after strict checkpoint loading and optional LoRA.

    Hardware profiles choose implementations, not model shape or denoising mode.
    Callers can inspect model.inference_optimization_telemetry() for dispatch.
    """
    import torch
    from ..optim.config import InferenceOptimizationConfig

    overrides = dict(overrides or {})
    if hardware == "reference":
        if overrides:
            raise ValueError("Reference backend takes no optimization overrides")
        model.configure_inference_optimizations(None)
        return model.prepare_inference_optimizations()
    if hardware not in CAPABILITIES:
        raise ValueError(f"Choose hardware from reference or {tuple(CAPABILITIES)}")
    if torch.device(model.device).type != "cuda":
        raise ValueError("Edge profiles require a CUDA model")
    actual = torch.cuda.get_device_capability(model.device)
    if actual != CAPABILITIES[hardware]:
        raise ValueError(f"{hardware} requires CC {CAPABILITIES[hardware]}, got {actual}")
    options = {
        "model_quant": True,
        "model_quant_targets": ["video"],
        "resident_rope": True,
        "required_compute_capability": ".".join(map(str, actual)),
        "torch_compile": True,
        "torch_compile_mode": "default",
        "torch_compile_suppress_errors": False,
        "torch_compile_static_inputs": True,
        "torch_compile_mark_static_addresses": False,
        "torch_compile_targets": [
            "observation_vae",
            "video_cross_kv_prefill",
            "video_denoiser",
            "video_kv_prefill",
            "action_cross_kv_prefill",
            "action_denoiser",
        ],
        "observation_vae_channels_last": True,
        "observation_vae_internal_spatial_padding": True,
        "observation_vae_first_frame_conv2d": True,
        "observation_vae_cudnn_benchmark_limit": 40 if hardware == "thor" else 20,
        "video_cross_kv_cache": True,
        "action_cross_kv_cache": True,
        "action_gemm_autotune": True,
        "action_segmented_attention": True,
        "video_fp32_rope": True,
        "video_self_qkv_shared_quant": True,
        "video_cross_kv_shared_quant": True,
    }
    if hardware == "thor":
        options.update(
            torch_compile_target_modes={"video_kv_prefill": "reduce-overhead"},
            action_video_kv_resident_cache=True,
            action_video_kv_direct_sink=True,
            video_flex_group_causal_sdpa=True,
            video_nvfp4_bias_epilogue=True,
        )
    else:
        options.update(
            action_packed_qkv=hardware == "spark",
            action_packed_cross_kv=True,
            action_fp32_rope=True,
            video_cudnn_self_sdpa=True,
        )
    if hardware == "rtx5090":
        options["observation_vae_first_frame_conv2d"] = False
        options["video_clean_prefix_wavefront"] = True
        options["torch_compile_targets"].append("video_clean_prefix_prefill")
    if getattr(model, "joint_denoise", False):
        # CoD uses the release model's joint solver. Sequential cross-KV and
        # video-prefill transformations must not silently replace that regime.
        options = {
            k: v
            for k, v in options.items()
            if k
            in {
                "model_quant",
                "model_quant_targets",
                "resident_rope",
                "required_compute_capability",
                "torch_compile",
                "torch_compile_mode",
                "torch_compile_suppress_errors",
                "torch_compile_static_inputs",
                "torch_compile_mark_static_addresses",
            }
        }
        options["torch_compile_targets"] = ["observation_vae"]
        sequential = {
            "video_clean_prefix_wavefront",
            "action_video_kv_resident_cache",
            "action_video_kv_direct_sink",
            "video_cross_kv_cache",
            "action_cross_kv_cache",
        }
        if any(overrides.get(k) for k in sequential):
            raise ValueError(
                "Sequential IDM cache optimizations cannot be applied to joint denoising"
            )
    options.update(overrides)
    if options.get("kv_quant"):
        raise ValueError("Published runtime keeps KV storage in BF16")
    model.configure_inference_optimizations(
        InferenceOptimizationConfig.from_mapping(options)
    )
    return model.prepare_inference_optimizations()
