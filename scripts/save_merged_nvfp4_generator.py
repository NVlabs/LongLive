#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Save a merged LoRA generator as a reusable checkpoint."""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import normalize_config
from utils.lora_utils import configure_lora_for_model
from utils.nvfp4_checkpoint import (
    NVFP4_CHECKPOINT_FORMAT,
    NVFP4_CHECKPOINT_VERSION,
    clean_fsdp_state_dict_keys,
    cpu_state_dict,
    is_nvfp4_state_dict,
    quantize_model_for_fouroversix_nvfp4,
    unwrap_generator_state_dict,
)
from utils.quant import _materialize_quantized_weights_for_inference
from utils.wan_5b_wrapper import WanDiffusionWrapper


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_generator_checkpoint(generator: WanDiffusionWrapper, checkpoint_path: str, use_ema: bool) -> None:
    checkpoint = _torch_load(checkpoint_path)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if is_nvfp4_state_dict(state_dict):
        raise ValueError(
            f"{checkpoint_path} is already a materialized NVFP4 checkpoint; "
            "use it directly as checkpoints.generator_ckpt."
        )
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)
    incompatible = generator.load_state_dict(state_dict, strict=not use_ema)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        print(f"[Warning] Missing generator keys while loading base checkpoint: {missing[:8]} ...")
    if unexpected:
        print(f"[Warning] Unexpected generator keys while loading base checkpoint: {unexpected[:8]} ...")


def _load_lora_state(lora_ckpt_path: str):
    checkpoint = _torch_load(lora_ckpt_path)
    if isinstance(checkpoint, dict) and "generator_lora" in checkpoint:
        return checkpoint["generator_lora"]
    return checkpoint


def _merge_lora(generator: WanDiffusionWrapper, config, lora_ckpt_path: str) -> WanDiffusionWrapper:
    if not getattr(config, "adapter", None):
        raise ValueError("LoRA merge was requested, but config.adapter is missing.")
    if not lora_ckpt_path:
        raise ValueError("LoRA merge was requested, but no lora_ckpt was provided.")

    print(f"Applying LoRA config: {config.adapter}")
    generator.model = configure_lora_for_model(
        generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )

    import peft

    print(f"Loading LoRA weights: {lora_ckpt_path}")
    peft.set_peft_model_state_dict(generator.model, _load_lora_state(lora_ckpt_path))  # type: ignore[arg-type]
    print("Merging LoRA into generator...")
    generator.model = generator.model.merge_and_unload(safe_merge=True)
    return generator


def _metadata(
    config,
    args: argparse.Namespace,
    *,
    backend: str,
    matched_modules: list[str],
    materialized_modules: list[str],
):
    checkpoint_format = (
        "longlive_generator_merged_bf16"
        if backend == "transformer_engine"
        else NVFP4_CHECKPOINT_FORMAT
    )
    quant_format = "bf16" if backend == "transformer_engine" else "nvfp4"
    quant_backend = "transformer_engine_runtime" if backend == "transformer_engine" else "fouroversix"
    return {
        "checkpoint_format": checkpoint_format,
        "checkpoint_version": NVFP4_CHECKPOINT_VERSION,
        "source_generator_ckpt": args.generator_ckpt,
        "source_lora_ckpt": args.lora_ckpt,
        "merged_lora": bool(args.lora_ckpt and not args.no_merge_lora),
        "model_name": getattr(config.model_kwargs, "model_name", None),
        "quantization": {
            "format": quant_format,
            "backend": quant_backend,
            "materialized": backend == "fouroversix",
            "dtype": "bfloat16",
            "scale_rule": getattr(config, "model_quant_scale_rule", "static_6"),
            "activation_scale_rule": getattr(config, "model_quant_activation_scale_rule", None),
            "weight_scale_rule": getattr(config, "model_quant_weight_scale_rule", None),
            "gradient_scale_rule": getattr(config, "model_quant_gradient_scale_rule", None),
            "te_inference_only": bool(
                getattr(config, "model_quant_te_inference_only", backend == "transformer_engine")
            ),
            "te_low_precision_weights": bool(
                getattr(config, "model_quant_te_low_precision_weights", backend == "transformer_engine")
            ),
            "te_fallback_to_fouroversix": bool(
                getattr(config, "model_quant_te_fallback_to_fouroversix", False)
            ),
            "matched_filtered_modules": matched_modules,
            "materialized_modules": materialized_modules,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge generator LoRA weights and save either packed FourOverSix NVFP4 or TE-ready bf16."
    )
    parser.add_argument("--config_path", required=True, help="Inference yaml that contains model/adapter/quant settings.")
    parser.add_argument("--output_path", required=True, help="Path to save the generator .pt file.")
    parser.add_argument("--generator_ckpt", default=None, help="Override checkpoints.generator_ckpt from the yaml.")
    parser.add_argument("--lora_ckpt", default=None, help="Override checkpoints.lora_ckpt from the yaml.")
    parser.add_argument("--device", default="cuda:0", help="Device used for quantization, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--backend",
        choices=("fouroversix", "transformer_engine"),
        default="fouroversix",
        help=(
            "fouroversix saves packed/materialized NVFP4. "
            "transformer_engine saves merged bf16 for TE runtime quantization."
        ),
    )
    parser.add_argument("--no_merge_lora", action="store_true", help="Quantize the base generator without merging LoRA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = normalize_config(OmegaConf.load(args.config_path))
    args.generator_ckpt = args.generator_ckpt or getattr(config, "generator_ckpt", None)
    args.lora_ckpt = args.lora_ckpt or getattr(config, "lora_ckpt", None)

    if not args.generator_ckpt:
        raise ValueError("Missing generator checkpoint. Set checkpoints.generator_ckpt or pass --generator_ckpt.")
    config.model_quant_use_transformer_engine = args.backend == "transformer_engine"

    device = torch.device(args.device)
    print(f"Building generator on CPU: {config.model_kwargs}")
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
    generator.eval().requires_grad_(False)

    print(f"Loading base generator checkpoint: {args.generator_ckpt}")
    _load_generator_checkpoint(generator, args.generator_ckpt, use_ema=bool(getattr(config, "use_ema", False)))

    should_merge_lora = bool(getattr(config, "merge_lora", False)) and not args.no_merge_lora
    if should_merge_lora:
        generator = _merge_lora(generator, config, args.lora_ckpt)
    else:
        print("Skipping LoRA merge; quantizing the loaded generator as-is.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Moving generator to {device} and casting to bfloat16...")
    generator.to(device=device, dtype=torch.bfloat16)
    materialized_modules = []
    if args.backend == "transformer_engine":
        matched_modules = []
        print(
            "Saving merged bf16 weights for TransformerEngine runtime quantization. "
            "TransformerEngine state_dict is not a packed NVFP4 storage format."
        )
    else:
        generator.model, matched_modules = quantize_model_for_fouroversix_nvfp4(
            generator.model,
            config=config,
            keep_master_weights=False,
            verbose=True,
        )

        print("Materializing NVFP4 weights and dropping bf16 master weights...")
        materialized_modules, master_bytes, quantized_bytes = _materialize_quantized_weights_for_inference(
            generator.model,
            target_device=device,
        )
        print(
            "[NVFP4] "
            f"materialized_modules={len(materialized_modules)}, "
            f"master_weight={master_bytes / (1024 ** 3):.3f} GiB, "
            f"quantized_weight={quantized_bytes / (1024 ** 3):.3f} GiB"
        )

    print("Copying checkpoint tensors to CPU for saving...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "generator": cpu_state_dict(generator),
        **_metadata(
            config,
            args,
            backend=args.backend,
            matched_modules=matched_modules,
            materialized_modules=materialized_modules,
        ),
    }
    torch.save(checkpoint, output_path)
    size_gib = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Saved {args.backend} generator checkpoint to {output_path} ({size_gib:.3f} GiB)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
