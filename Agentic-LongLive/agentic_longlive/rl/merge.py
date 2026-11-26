# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge a PEFT LoRA adapter into the Qwen3-8B safetensors shards (W += alpha / r * B @ A, stored in bf16), so the
planner can be served as a plain model with EAGLE-3 speculative decoding.

  python -m agentic_longlive.rl.merge --base Qwen/Qwen3-8B --adapter outputs/rl/grpo_s0/adapter_step0080 \\
      --out outputs/rl/planner_merged

``--base`` is a local snapshot directory or a Hugging Face model id (resolved through the HF cache).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def resolve_base(base: str) -> Path:
    path = Path(base)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base))


def merge(base: Path, adapter: Path, out: Path, device: str = "cpu") -> int:
    import torch
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    pairs = {}
    for k, v in lora.items():
        # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight -> model.layers.0.self_attn.q_proj.weight
        name = k.replace("base_model.model.", "", 1)
        for tag in (".lora_A.weight", ".lora_B.weight"):
            if name.endswith(tag):
                target = name[: -len(tag)] + ".weight"
                pairs.setdefault(target, {})[tag[1:7]] = v
    merged = 0
    for f in sorted(base.glob("*.safetensors")):
        tensors = load_file(str(f))
        for name in list(tensors):
            if name in pairs:
                a = pairs[name]["lora_A"].to(device, torch.float32)
                b = pairs[name]["lora_B"].to(device, torch.float32)
                w = tensors[name].to(device, torch.float32)
                tensors[name] = (w + scale * (b @ a)).to(torch.bfloat16).cpu()
                merged += 1
        save_file(tensors, str(out / f.name), metadata={"format": "pt"})
    for f in base.iterdir():
        if not f.name.endswith(".safetensors") and f.is_file():
            shutil.copy(f.resolve(), out / f.name)
    print(f"merged {merged} / {len(pairs)} LoRA targets (scale {scale}) into {out}")
    if merged != len(pairs):
        raise SystemExit("not every LoRA target was found in the base weights")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="base model snapshot directory or HF id")
    ap.add_argument("--adapter", required=True, help="PEFT adapter directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    merge(resolve_base(args.base), Path(args.adapter), Path(args.out), args.device)


if __name__ == "__main__":
    main()
