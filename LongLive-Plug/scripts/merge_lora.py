# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Merge Few-Step and CFG PEFT LoRAs into native Wan-named safetensors weights.

The input may contain multiple DiT shards; the output is one safetensors file.
Only ordinary linear LoRA A/B weights with alpha/r scaling are supported.
"""
import argparse
import math
from pathlib import Path
import re

import torch
from safetensors.torch import load_file, save_file


def adapter_updates(path, weight, alpha):
    if not math.isfinite(weight) or (alpha is not None and not math.isfinite(alpha)):
        raise ValueError("Adapter weight and alpha must be finite")
    state = load_file(str(path), device="cpu")
    pairs = {}
    for key, value in state.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", key)
        if match is None:
            raise ValueError(f"Unsupported adapter key: {key}")
        name, side = match.groups()
        name = name.removeprefix("base_model.model.") + ".weight"
        pair = pairs.setdefault(name, {})
        if side in pair:
            raise ValueError(f"Duplicate LoRA {side}: {name}")
        pair[side] = value
    if not pairs:
        raise ValueError(f"Empty adapter: {path}")
    result = []
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete LoRA pair: {name}")
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid LoRA dimensions: {name}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Nonfinite adapter: {name}")
        scale = weight * (alpha / a.shape[0] if alpha is not None else 1.0)
        result.append((name, a, b, scale))
    return result


@torch.no_grad()
def merge(state, adapters):
    """Validate all targets first, then accumulate each weight's deltas in FP32."""
    grouped = {}
    for updates in adapters:
        for name, a, b, scale in updates:
            if name not in state:
                raise ValueError(f"Missing downstream weight: {name}; convert key names first")
            target = state[name]
            if not target.is_floating_point() or target.dtype not in (
                torch.float32, torch.float16, torch.bfloat16, torch.float64
            ):
                raise ValueError(f"Expected unquantized floating-point weight: {name}")
            if tuple(target.shape) != (b.shape[0], a.shape[1]):
                raise ValueError(f"Shape mismatch for downstream weight: {name}")
            grouped.setdefault(name, []).append((a, b, scale))
    for name, updates in grouped.items():
        value = state[name].float().clone()
        for a, b, scale in updates:
            value.addmm_(b.float(), a.float(), beta=1, alpha=scale)
        value = value.to(state[name].dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite merged weight: {name}")
        state[name] = value.contiguous()
    return len(grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, nargs="+", required=True,
                        help="Downstream DiT safetensors file(s), excluding text encoder/VAE")
    parser.add_argument("--few-step", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--few-step-weight", type=float, default=1.0)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--few-step-alpha", type=float,
                        help="Training alpha; default assumes alpha equals rank")
    parser.add_argument("--cfg-alpha", type=float,
                        help="Training alpha; default assumes alpha equals rank")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".safetensors":
        parser.error("--output must end in .safetensors")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    updates = [adapter_updates(args.few_step, args.few_step_weight, args.few_step_alpha),
               adapter_updates(args.cfg, args.cfg_weight, args.cfg_alpha)]
    state = {}
    for path in args.base:
        shard = load_file(str(path), device="cpu")
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(f"Duplicate keys across base shards: {sorted(duplicate)[:5]}")
        state.update(shard)
    count = merge(state, updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output), metadata={"format": "pt"})
    print(f"Merged {count} downstream weights into {args.output}")


if __name__ == "__main__":
    main()
