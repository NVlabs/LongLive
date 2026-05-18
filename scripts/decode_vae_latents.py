# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""
Decode saved latent files (.pt) to pixel video and write .mp4 using the specified VAE.

Supports multi-GPU via torchrun:
    torchrun --nproc_per_node=8 scripts/decode_vae_latents.py --input_dir /path/to/latents
Single-GPU also works:
    python scripts/decode_vae_latents.py --input_dir /path/to/latents

Uses the Wan2.2-TI2V-5B VAE.
"""
import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torchvision.io import write_video
from tqdm import tqdm


def init_distributed():
    """Initialize distributed process group if launched via torchrun. Returns (rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def get_vae():
    """Return the Wan2.2-TI2V-5B VAE wrapper."""
    from utils.wan_5b_wrapper import WanVAEWrapper
    return WanVAEWrapper()


def decode_latent_to_video(vae, latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Decode latent to pixel video; same logic as pipeline/causal_diffusion_inference.py.

    Args:
        vae: Wan2.2-TI2V-5B VAE wrapper
        latent: shape (batch, T, C, H, W) or (T, C, H, W)
        device, dtype: device and dtype for computation

    Returns:
        video: shape (batch, T, C, H, W), range [0, 1]
    """
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(device=device, dtype=dtype)
    video = vae.decode_to_pixel(latent)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video


def main():
    parser = argparse.ArgumentParser(description="Decode saved latents to video.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .pt latent files (e.g. latents_rank00_idx000000.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save decoded .mp4 videos. Default: input_dir/decoded_videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="FPS for output video (default: 16).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="latents_*.pt",
        help="Glob pattern for latent files (default: latents_*.pt).",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    is_main = (rank == 0)

    output_dir = args.output_dir or os.path.join(args.input_dir, "decoded_videos")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    vae = get_vae()
    vae = vae.to(device=device, dtype=dtype).eval()

    search_path = os.path.join(args.input_dir, args.pattern)
    pt_files = sorted(glob.glob(search_path))
    if not pt_files:
        if is_main:
            print(f"No files matching '{search_path}' found. Exiting.")
        return

    # Shard files across ranks
    pt_files_local = pt_files[rank::world_size]
    if is_main:
        print(f"Found {len(pt_files)} latent file(s), {world_size} GPU(s), ~{len(pt_files_local)} per GPU.")

    pbar = tqdm(pt_files_local, desc=f"[Rank {rank}] Decoding", disable=(not is_main))
    for pt_path in pbar:
        basename = os.path.splitext(os.path.basename(pt_path))[0]
        video_name = basename.replace("latents_", "video_", 1) if basename.startswith("latents_") else f"video_{basename}"
        out_path = os.path.join(output_dir, f"{video_name}.mp4")

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            if isinstance(data, torch.Tensor):
                latent = data
            elif isinstance(data, dict):
                latent = next(iter(data.values()))
                if not isinstance(latent, torch.Tensor):
                    print(f"[Rank {rank}] Skip {pt_path}: dict value is not a tensor.")
                    continue
            else:
                print(f"[Rank {rank}] Skip {pt_path}: unsupported type {type(data)}.")
                continue
        except Exception as e:
            print(f"[Rank {rank}] Failed to load {pt_path}: {e}")
            continue

        video = decode_latent_to_video(vae, latent, device, dtype)
        video_frames = video[0].cpu()
        video_uint8 = (video_frames * 255.0).clamp(0, 255).to(torch.uint8)
        video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
        write_video(out_path, video_uint8, fps=args.fps)

    if world_size > 1:
        dist.barrier()
    if is_main:
        print(f"Done. Decoded {len(pt_files)} latent(s) to {output_dir}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
