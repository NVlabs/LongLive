# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Small helpers for release inference examples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from einops import rearrange
from torchvision.io import write_video

from utils.nvfp4_checkpoint import (
    clean_fsdp_state_dict_keys,
    drop_fouroversix_master_weights,
    is_nvfp4_state_dict,
    is_te_nvfp4_checkpoint,
    quantize_model_for_fouroversix_nvfp4,
    quantize_model_for_transformer_engine_nvfp4,
    unwrap_generator_state_dict,
)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_generator_checkpoint(generator, checkpoint_path: str, *, use_ema: bool = False, strict: bool | None = None):
    """Load a LongLive generator checkpoint into ``generator``."""
    checkpoint = _torch_load(checkpoint_path)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)
    if strict is None:
        strict = not use_ema
    return generator.load_state_dict(state_dict, strict=strict)


def place_vae_for_streaming(pipeline, config) -> torch.device | None:
    """Move ``pipeline.vae`` to ``config.vae_device`` for streaming-pipeline decode.

    Only acts when both ``streaming_vae`` and ``vae_device`` are set; otherwise
    leaves the VAE on whatever device the rest of the pipeline already uses.
    Mirrors the relocation done in ``inference.py`` so that quick-start scripts
    can opt in to the streaming-pipeline VAE simply by enabling those config
    fields.
    """
    if not bool(getattr(config, "streaming_vae", False)):
        return None
    vae_device_str = getattr(config, "vae_device", None)
    if not vae_device_str:
        return None

    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    return vae_device


def setup_nvfp4_pipeline(
    pipeline,
    config,
    device: torch.device | str,
    *,
    verbose: bool = False,
):
    """Configure ``pipeline`` for NVFP4 inference from a merged generator checkpoint.

    Handles both supported NVFP4 backends:

    * ``model_quant_use_transformer_engine=True`` -> a BF16 generator checkpoint
      that gets wrapped with TransformerEngine NVFP4 modules and materialized
      after moving to ``device``.
    * ``model_quant_use_transformer_engine=False`` -> a pre-materialized
      FourOverSix NVFP4 state dict that is loaded directly into the
      already-quantized architecture.

    This helper assumes the generator checkpoint is fully merged (no LoRA
    adapter), which matches the released NVFP4 weights.
    """
    if not bool(getattr(config, "model_quant", False)):
        raise ValueError("setup_nvfp4_pipeline requires model_quant=true in the config.")

    generator_ckpt = getattr(config, "generator_ckpt", None)
    if not generator_ckpt:
        raise ValueError("checkpoints.generator_ckpt is required for NVFP4 inference.")

    use_te = bool(getattr(config, "model_quant_use_transformer_engine", False))
    device = torch.device(device)

    checkpoint = _torch_load(generator_ckpt)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=bool(getattr(config, "use_ema", False)))

    if is_te_nvfp4_checkpoint(checkpoint):
        raise ValueError(
            "Detected a TransformerEngine module state_dict export (no longer supported). "
            "Re-export with `--backend transformer_engine` (merged BF16) or `--backend fouroversix`."
        )

    is_prequantized = is_nvfp4_state_dict(state_dict)

    if is_prequantized:
        if use_te:
            raise ValueError(
                "generator_ckpt is a materialized NVFP4 (FourOverSix) checkpoint; set "
                "model_quant_use_transformer_engine: false."
            )
        pipeline.generator.model, _ = quantize_model_for_fouroversix_nvfp4(
            pipeline.generator.model,
            config=config,
            keep_master_weights=False,
            verbose=verbose,
        )
        drop_fouroversix_master_weights(pipeline.generator.model)
        pipeline.generator.load_state_dict(state_dict, strict=True)

        pipeline.text_encoder.to(dtype=torch.bfloat16)
        pipeline.vae.to(dtype=torch.bfloat16)
    else:
        pipeline.generator.load_state_dict(state_dict, strict=True)

        if use_te:
            pipeline.generator.model, _ = quantize_model_for_transformer_engine_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=verbose,
            )
            te_fallback = bool(getattr(config, "model_quant_te_fallback_to_fouroversix", False))
            if te_fallback:
                from utils.quant import _materialize_mixed_quantized_weights_for_inference as materialize_fn
            else:
                from utils.quant import _materialize_transformer_engine_weights_for_inference as materialize_fn
        else:
            pipeline.generator.model, _ = quantize_model_for_fouroversix_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=verbose,
            )
            from utils.quant import _materialize_quantized_weights_for_inference as materialize_fn

        pipeline.to(dtype=torch.bfloat16)
        materialize_fn(pipeline.generator.model, target_device=device)

    pipeline.generator.to(device=device)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    place_vae_for_streaming(pipeline, config)

    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = False
    return pipeline


def prepare_single_prompt_inputs(
    config,
    prompt: str,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    generator: torch.Generator | None = None,
):
    """Create the per-block prompt list and latent noise for one text prompt."""
    num_frames = int(getattr(config, "num_output_frames", config.image_or_video_shape[1]))
    frames_per_block = int(getattr(config, "num_frame_per_block", 1))
    if num_frames % frames_per_block != 0:
        raise ValueError(f"num_frames={num_frames} must be divisible by num_frame_per_block={frames_per_block}")

    latent_shape = list(config.image_or_video_shape[2:])
    if len(latent_shape) != 3:
        raise ValueError(f"Expected latent shape [C, H, W], got {latent_shape}")

    num_blocks = num_frames // frames_per_block
    prompts = [[prompt] * num_blocks for _ in range(batch_size)]
    noise = torch.randn(
        [batch_size, num_frames, *latent_shape],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return noise, prompts


def video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    """Convert a generated video tensor from [T, C, H, W] or [1, T, C, H, W] to uint8 THWC."""
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("video_to_uint8 expects a single sample when a batch dimension is present.")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims, got shape={tuple(video.shape)}")
    if video.shape[1] in (1, 3):
        video = rearrange(video, "t c h w -> t h w c")
    return (255.0 * video.cpu()).clamp(0, 255).to(torch.uint8)


def save_video(video: torch.Tensor, output_path: str | os.PathLike, *, fps: int = 24) -> None:
    """Save a generated LongLive video tensor as an mp4 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output_path), video_to_uint8(video), fps=fps)
