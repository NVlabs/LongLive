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
Decode saved latent files (.pt) to pixel video using a local LightVAE loader.

This mirrors `scripts/decode_vae_latents.py`, but only depends on the current
repo's 5B VAE code plus a LightVAE checkpoint such as `MG-LightVAE_v2.pth`.

Examples:
    python scripts/decode_lightvae_latents.py \
        --input_dir /path/to/latents \
        --vae_path /path/to/MG-LightVAE_v2.pth

    torchrun --nproc_per_node=8 scripts/decode_lightvae_latents.py \
        --input_dir /path/to/latents \
        --ckpt_dir /path/to/lightvae_ckpts \
        --vae_type mg_lightvae
"""

import argparse
import glob
import os
import sys
from typing import Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
import torch.distributed as dist
from torchvision.io import write_video
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from utils.lightvae_5b_wrapper import LightVAE5BWrapper


def init_distributed():
    """Initialize distributed process group if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def decode_latent_to_video(
    vae, latent: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Decode latent to pixel video.

    Args:
        vae: VAE wrapper exposing `decode_to_pixel(latent)`
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


def _normalize_requested_vae_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "wan":
        return "wan2.2"
    if normalized in {"wan2.2", "mg_lightvae", "mg_lightvae_v2"}:
        return normalized
    raise ValueError(
        f"Unsupported --vae_type '{value}'. "
        "Expected one of: wan2.2, wan, mg_lightvae, mg_lightvae_v2."
    )


def _parse_lightvae_pruning_rate(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"", "auto", "none"}:
        return None
    return float(lowered)


def _resolve_vae_paths(
    *,
    ckpt_dir: Optional[str],
    vae_path: Optional[str],
    requested_vae_type: str,
    lightvae_pruning_rate: Optional[str],
):
    pruning_rate = _parse_lightvae_pruning_rate(lightvae_pruning_rate)
    ckpt_dir = os.path.abspath(ckpt_dir) if ckpt_dir else None

    if vae_path is not None:
        resolved_vae_path = os.path.abspath(vae_path)
        if requested_vae_type == "wan2.2" and pruning_rate is None:
            pruning_rate = 0.0
    else:
        if ckpt_dir is None:
            raise ValueError("Either --ckpt_dir or --vae_path must be provided.")
        if requested_vae_type == "mg_lightvae":
            resolved_vae_path = os.path.join(ckpt_dir, "MG-LightVAE.pth")
            if pruning_rate is None:
                pruning_rate = 0.5
        elif requested_vae_type == "mg_lightvae_v2":
            resolved_vae_path = os.path.join(ckpt_dir, "MG-LightVAE_v2.pth")
            if pruning_rate is None:
                pruning_rate = 0.75
        else:
            resolved_vae_path = os.path.join(ckpt_dir, "Wan2.2_VAE.pth")
            if pruning_rate is None:
                pruning_rate = 0.0

    return resolved_vae_path, pruning_rate


def _load_latent_tensor(pt_path: str):
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu")

    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, dict):
        latent = next(iter(data.values()))
        if isinstance(latent, torch.Tensor):
            return latent
        raise TypeError(f"First dict value is not a tensor: {type(latent)}")
    raise TypeError(f"Unsupported latent file payload type: {type(data)}")


def _parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported --dtype '{dtype_name}'. "
            "Expected one of: float32, float16, bfloat16."
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Decode saved latents to video with a local LightVAE loader."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .pt latent files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save decoded .mp4 videos. Default: input_dir/decoded_lightvae_videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="FPS for output video (default: 24).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="latents_*.pt",
        help="Glob pattern for latent files (default: latents_*.pt).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory containing `MG-LightVAE*.pth` or `Wan2.2_VAE.pth`.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=(
            "Explicit LightVAE/Wan2.2 checkpoint path. "
            "Use this when you only have a single VAE checkpoint file."
        ),
    )
    parser.add_argument(
        "--matrix_game_root",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--teacher_vae_path",
        "--lightvae_encoder_path",
        dest="teacher_vae_path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default="mg_lightvae_v2",
        choices=["wan2.2", "wan", "mg_lightvae", "mg_lightvae_v2"],
        help=(
            "VAE variant to use. "
            "`mg_lightvae` maps to MG-LightVAE.pth, "
            "`mg_lightvae_v2` maps to MG-LightVAE_v2.pth."
        ),
    )
    parser.add_argument(
        "--lightvae_pruning_rate",
        type=str,
        default=None,
        help=(
            "Override pruning rate. Use `auto`/`none` to let Wan2_2_VAE infer it "
            "when an explicit LightVAE checkpoint is provided."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Decode dtype (default: bfloat16).",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    output_dir = args.output_dir or os.path.join(
        args.input_dir, "decoded_lightvae_videos"
    )
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    dtype = _parse_dtype(args.dtype)

    requested_vae_type = _normalize_requested_vae_type(args.vae_type)
    resolved_vae_path, resolved_pruning_rate = _resolve_vae_paths(
        ckpt_dir=args.ckpt_dir,
        vae_path=args.vae_path,
        requested_vae_type=requested_vae_type,
        lightvae_pruning_rate=args.lightvae_pruning_rate,
    )

    vae = LightVAE5BWrapper(
        vae_path=resolved_vae_path,
        pruning_rate=resolved_pruning_rate,
        device=device,
        dtype=dtype,
    ).eval()

    if is_main:
        print(
            "Using local VAE-only loader: "
            f"requested={requested_vae_type}, "
            f"pruning={vae.pruning_rate}, "
            f"vae_path={vae.vae_path}",
            flush=True,
        )

    search_path = os.path.join(args.input_dir, args.pattern)
    pt_files = sorted(glob.glob(search_path))
    if not pt_files:
        if is_main:
            print(f"No files matching '{search_path}' found. Exiting.")
        return

    pt_files_local = pt_files[rank::world_size]
    if is_main:
        print(
            f"Found {len(pt_files)} latent file(s), {world_size} GPU(s), "
            f"~{len(pt_files_local)} per GPU.",
            flush=True,
        )

    pbar = tqdm(pt_files_local, desc=f"[Rank {rank}] Decoding", disable=(not is_main))
    for pt_path in pbar:
        basename = os.path.splitext(os.path.basename(pt_path))[0]
        if basename.startswith("latents_"):
            video_name = basename.replace("latents_", "video_", 1)
        else:
            video_name = f"video_{basename}"
        out_path = os.path.join(output_dir, f"{video_name}.mp4")

        try:
            latent = _load_latent_tensor(pt_path)
        except Exception as exc:
            print(f"[Rank {rank}] Failed to load {pt_path}: {exc}", flush=True)
            continue

        try:
            video = decode_latent_to_video(vae, latent, device, dtype)
            video_frames = video[0].cpu()
            video_uint8 = (video_frames * 255.0).clamp(0, 255).to(torch.uint8)
            video_uint8 = video_uint8.permute(0, 2, 3, 1)
            write_video(out_path, video_uint8, fps=args.fps)
        except Exception as exc:
            print(f"[Rank {rank}] Failed to decode {pt_path}: {exc}", flush=True)

    if world_size > 1:
        dist.barrier()
    if is_main:
        print(f"Done. Decoded {len(pt_files)} latent(s) to {output_dir}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
