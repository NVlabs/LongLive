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
from collections.abc import Mapping
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


def _load_lora_state_dict(lora_ckpt_path: str) -> Mapping[str, torch.Tensor]:
    """Load a LoRA checkpoint, unwrapping ``generator_lora`` when present."""
    checkpoint = _torch_load(lora_ckpt_path)
    if isinstance(checkpoint, Mapping) and "generator_lora" in checkpoint:
        return checkpoint["generator_lora"]
    return checkpoint


def apply_and_merge_lora(
    pipeline,
    config,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    """Wrap ``pipeline.generator.model`` with a LoRA adapter, load weights, and merge.

    The merged module ends up structurally identical to the original generator
    (``nn.Linear`` layers carrying the base + LoRA delta), which is what NVFP4
    quantization needs as its starting point.

    Returns ``True`` when LoRA was applied and merged, ``False`` when the config
    did not request a LoRA adapter.
    """
    adapter_cfg = getattr(config, "adapter", None)
    lora_ckpt = getattr(config, "lora_ckpt", None)
    if adapter_cfg is None or not lora_ckpt:
        return False

    import peft
    from utils.lora_utils import configure_lora_for_model

    if device is not None:
        pipeline.generator.to(device=torch.device(device), dtype=dtype)
    else:
        pipeline.generator.to(dtype=dtype)

    if verbose:
        print(f"[LoRA] Wrapping generator with adapter config: {adapter_cfg}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=adapter_cfg,
        is_main_process=verbose,
    )

    if verbose:
        print(f"[LoRA] Loading LoRA weights from: {lora_ckpt}")
    lora_state = _load_lora_state_dict(lora_ckpt)
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)  # type: ignore[arg-type]

    if verbose:
        print("[LoRA] Merging LoRA delta into base weights (merge_and_unload)...")
    pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
    pipeline.generator.model.eval().requires_grad_(False)
    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = True
    return True


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
    """Configure ``pipeline`` for NVFP4 inference from a generator checkpoint.

    Handles both supported NVFP4 backends:

    * ``model_quant_use_transformer_engine=True`` -> a BF16 generator checkpoint
      that gets wrapped with TransformerEngine NVFP4 modules and materialized
      after moving to ``device``.
    * ``model_quant_use_transformer_engine=False`` -> either a BF16 generator
      checkpoint that gets quantized with FourOverSix at load time, or a
      pre-materialized FourOverSix NVFP4 state dict loaded directly into the
      already-quantized architecture.

    Optional LoRA support (BF16 base only): when ``config.adapter`` and
    ``config.lora_ckpt`` are both set, the LoRA adapter is loaded on the BF16
    base generator, merged via ``merge_and_unload``, and the resulting weights
    are then quantized — so the same yaml can swap between TE and FourOverSix
    backends without pre-merging the LoRA checkpoint.

    For materialized FourOverSix checkpoints LoRA cannot be applied (the master
    weights have already been quantized away); ``lora_ckpt``/``adapter`` are
    ignored in that case with a printed warning.
    """
    if not bool(getattr(config, "model_quant", False)):
        raise ValueError("setup_nvfp4_pipeline requires model_quant=true in the config.")

    generator_ckpt = getattr(config, "generator_ckpt", None)
    if not generator_ckpt:
        raise ValueError("checkpoints.generator_ckpt is required for NVFP4 inference.")

    use_te = bool(getattr(config, "model_quant_use_transformer_engine", False))
    device = torch.device(device)
    use_ema = bool(getattr(config, "use_ema", False))

    checkpoint = _torch_load(generator_ckpt)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)

    if is_te_nvfp4_checkpoint(checkpoint):
        raise ValueError(
            "Detected a TransformerEngine module state_dict export (no longer supported). "
            "Re-export with `--backend transformer_engine` (merged BF16) or `--backend fouroversix`."
        )

    is_prequantized = is_nvfp4_state_dict(state_dict)
    has_lora_request = bool(getattr(config, "adapter", None)) and bool(getattr(config, "lora_ckpt", None))

    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = False

    if is_prequantized:
        if has_lora_request and verbose:
            print(
                "[NVFP4] generator_ckpt is a materialized FourOverSix NVFP4 checkpoint; "
                "ignoring lora_ckpt/adapter because the master weights are already quantized. "
                "Use a BF16 base checkpoint if you need to load a LoRA on top."
            )
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
        load_strict = not use_ema
        pipeline.generator.load_state_dict(state_dict, strict=load_strict)

        if has_lora_request:
            # Apply + merge LoRA on the BF16 base before quantization. Move the
            # generator to CUDA first so the TE wrapper (which requires CUDA
            # modules) can later replace the merged Linear layers in-place.
            apply_and_merge_lora(
                pipeline,
                config,
                device=device,
                dtype=torch.bfloat16,
                verbose=verbose,
            )

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
