#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Modified from the author-provided longlive2.0-robot inference demo.
# Source: longlive2.0-robot/inference.py, SHA256 21f4a911a3e61d2f72680681a083901569a0c2b6ddc7fb5d6debd24896e81a29
# Source: https://github.com/NVlabs/LongLive :: utils/inference_utils.py; trainer/diffusion.py
# Changes: Bundled core/assets paths, complete examples, custom inputs and output guards.
# Licensed under the Apache License, Version 2.0; see LICENSE and THIRD_PARTY_NOTICES.md.

"""First-frame image + instruction -> MP4 using the bundled LongLive 2.0 core."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "third_party/LongLive"
EXAMPLES = ROOT / "examples/robot_s"
CONFIG = ROOT / "configs/inference_robot.yaml"
FPS, HEIGHT, WIDTH = 24, 704, 1280


def load_examples():
    return json.loads((EXAMPLES / "examples.json").read_text(encoding="utf-8"))["examples"]


def check_image(path, sha256=None):
    from PIL import Image

    path = Path(path).expanduser().resolve()
    if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Example image changed: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != (WIDTH, HEIGHT) or image.mode != "RGB":
            raise ValueError(f"Expected a {WIDTH}x{HEIGHT} RGB first frame: {path}")
    return path


def select_jobs(args):
    if args.image is not None:
        if args.example is not None or not args.prompt or not args.prompt.strip():
            raise ValueError("--image requires --prompt and cannot be combined with --example")
        frames = args.latent_frames if args.latent_frames is not None else 32
        seed = args.seed if args.seed is not None else 20260711
        if frames < 8 or frames % 8:
            raise ValueError("--latent-frames must be a positive multiple of 8")
        if not 0 <= seed < 2**63:
            raise ValueError("--seed must be in [0, 2**63)")
        jobs = [dict(name="custom", image=str(check_image(args.image)),
                     prompt=args.prompt, latent_frames=frames, raw_frames=4 * frames - 3,
                     noise_seed=seed)]
    else:
        if args.prompt is not None or args.latent_frames is not None or args.seed is not None:
            raise ValueError("--prompt, --latent-frames and --seed require --image")
        selected = args.example or "short_00"
        jobs = [dict(row) for row in load_examples() if selected in ("all", row["name"])]
        if not jobs:
            raise ValueError("Unknown --example; use --list")
        for job in jobs:
            job["image"] = str(check_image(EXAMPLES / job["image"], job["image_sha256"]))
    return jobs


def load_config(path):
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    # This entry point is the original full-resolution BF16 recipe.
    model = config.model_kwargs
    if (model.model_name != "Wan2.2-TI2V-5B" or not model.initialize_from_config
            or model.num_frame_per_block != 8
            or list(config.image_or_video_shape) != [1, 32, 48, 44, 80]):
        raise ValueError("Use the compatible Wan2.2 Robot inference architecture")
    if not config.inference.independent_first_frame:
        raise ValueError("First-frame conditioning must stay enabled")
    if config.inference.streaming_vae or config.inference.async_vae:
        raise ValueError("This reference demo uses batch VAE decoding")
    steps = config.inference.sampling_steps
    guidance = config.inference.guidance_scale
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("sampling_steps must be a positive integer")
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    return config


def generator_state(checkpoint):
    """Select online weights, never silently switch to EMA or partially load."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a generator state dictionary")
    if "generator" in checkpoint:
        state = checkpoint["generator"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        state = checkpoint["model"]
    elif "generator_ema" in checkpoint:
        raise ValueError("EMA-only checkpoint: supply the online generator")
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Empty or invalid generator state dictionary")
    clean = {str(key).replace("_fsdp_wrapped_module.", ""): value for key, value in state.items()}
    if len(clean) != len(state):
        raise ValueError("Duplicate generator keys after FSDP unwrapping")
    return clean


def load_weights(generator, checkpoint):
    state = generator_state(checkpoint)
    if set(state) == set(generator.model.state_dict()):
        state = {"model." + key: value for key, value in state.items()}
    generator.load_state_dict(state, strict=True)


def build_pipeline(config, checkpoint, assets_root, device):
    import torch

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Inference requires CUDA; use --dry-run for CPU-only checks")
    torch.cuda.set_device(device)
    # Wrappers resolve wan_models/ against cwd, like stage1/run.py.
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    try:
        os.chdir(assets_root)
        sys.path.insert(0, str(CORE))
        from pipeline import CausalDiffusionInferencePipeline
        from utils.config import normalize_config

        with torch.no_grad():
            pipe = CausalDiffusionInferencePipeline(normalize_config(config), device=device)
            pipe.to(dtype=torch.bfloat16)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            load_weights(pipe.generator, state)
            del state
            pipe.to(device=device).eval().requires_grad_(False)
        return pipe
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def generate_one(pipe, job, device):
    import numpy as np
    from PIL import Image
    import torch

    pipe.clear_cache()
    try:
        with torch.no_grad():
            with Image.open(job["image"]) as image:
                pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
            # Preserve FP32 normalization -> FP16 -> BF16 from the original data path.
            pixels = (pixels.float() / 255.0 * 2.0 - 1.0).to(torch.float16)
            pixels = pixels[None, :, None].to(device=device, dtype=torch.bfloat16)
            latent = pipe.vae.encode_to_latent(pixels).to(device=device, dtype=torch.bfloat16)
            frames = job["latent_frames"]
            rng = torch.Generator(device=device).manual_seed(job["noise_seed"])
            noise = torch.randn((1, frames, 48, 44, 80), device=device,
                                dtype=torch.bfloat16, generator=rng)
            prompts = [[job["prompt"]] * (frames // 8)]
            return pipe.inference(noise=noise, text_prompts=prompts, initial_latent=latent)
    finally:
        pipe.clear_cache()


def save_video(path, frames):
    """Publish one complete MP4 without overwriting existing output."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".", suffix=".mp4") as temp:
        imageio.mimwrite(temp.name, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
        os.link(temp.name, path)  # Atomic no-clobber publish on the same filesystem.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--generator", type=Path, dest="checkpoint")
    parser.add_argument("--assets-root", type=Path, help="Contains wan_models/Wan2.2-TI2V-5B/")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--example", help="Bundled name, or all; default: short_00")
    parser.add_argument("--image", type=Path, help="Your 1280x704 RGB image")
    parser.add_argument("--prompt", help="Instruction for --image")
    parser.add_argument("--latent-frames", type=int, help="Custom image only; default: 32")
    parser.add_argument("--seed", type=int, help="Custom image only; default: 20260711")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video-demo"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/images; no weights or GPU")
    args = parser.parse_args(argv)
    if args.list:
        for item in load_examples():
            print(f"{item['name']:10}  {item['raw_frames']:3} frames  {item['prompt']}")
        return 0
    try:
        jobs = select_jobs(args)
        config = load_config(args.config.expanduser().resolve())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.expanduser().resolve()
    if args.dry_run:
        from omegaconf import OmegaConf
        print(json.dumps({"config": OmegaConf.to_container(config, resolve=True),
                          "output_dir": str(output_dir), "jobs": jobs}, indent=2))
        return 0
    if args.checkpoint is None or not args.checkpoint.expanduser().is_file():
        parser.error("Supply --checkpoint with your local online-generator .pt file")
    if args.assets_root is None:
        parser.error("Supply --assets-root with the Wan base-model assets directory")
    checkpoint = args.checkpoint.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    base = assets_root / "wan_models/Wan2.2-TI2V-5B"
    for name in ("config.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth", "google/umt5-xxl"):
        if not (base / name).exists():
            parser.error(f"Missing base-model asset: {base / name}")
    for job in jobs:
        if (output_dir / f"{job['name']}.mp4").exists():
            parser.error(f"Output exists: {job['name']}.mp4; choose a new output directory")
    pipe = build_pipeline(config, checkpoint, assets_root, args.device)
    for job in jobs:
        print(f"Generating {job['name']}: {job['prompt']}", flush=True)
        video = generate_one(pipe, job, args.device)
        frames = video[0].permute(0, 2, 3, 1).mul(255).clamp(0, 255).byte().cpu().numpy()
        if frames.shape != (job["raw_frames"], HEIGHT, WIDTH, 3):
            raise RuntimeError(f"Unexpected decoded shape: {frames.shape}")
        path = output_dir / f"{job['name']}.mp4"
        save_video(path, frames)
        print(f"Saved {path}", flush=True)
        del video, frames
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
