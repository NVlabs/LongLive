# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute SP chunk-halo VAE frame windows for a training config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import normalize_config, wan_default_config

DEFAULT_SP_VAE_HALO_LATENTS = 28


def latent_range_to_raw_window(
    latent_start: int,
    latent_end: int,
    *,
    temporal_compression_ratio: int,
) -> tuple[int, int, int]:
    """Map a latent range to the raw-frame window needed by Wan VAE encode."""
    if latent_end <= latent_start:
        raise ValueError(f"latent_end must be > latent_start, got {latent_start}, {latent_end}")
    ratio = int(temporal_compression_ratio)
    if latent_start == 0:
        return 0, 1 + ratio * (latent_end - 1), 0
    return ratio * (latent_start - 1), 1 + ratio * (latent_end - 1), 1


def compute_chunk_halo_metas(
    *,
    total_latent_frames: int,
    total_raw_frames: int,
    sp_size: int,
    halo_latents: int,
    temporal_compression_ratio: int,
) -> list[dict[str, int]]:
    if sp_size <= 0:
        raise ValueError("sp_size must be positive")
    if halo_latents < 0:
        raise ValueError("halo_latents must be non-negative")
    if total_latent_frames % sp_size != 0:
        raise ValueError(
            f"total_latent_frames={total_latent_frames} must be divisible by sp_size={sp_size}"
        )

    local_latent_frames = total_latent_frames // sp_size
    metas = []
    for sp_rank in range(sp_size):
        keep_start = sp_rank * local_latent_frames
        keep_end = keep_start + local_latent_frames
        halo_start = max(0, keep_start - halo_latents)
        raw_start, raw_end, pseudo_prefix_latents = latent_range_to_raw_window(
            halo_start,
            keep_end,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        raw_start = max(0, raw_start)
        raw_end = min(total_raw_frames, raw_end)
        drop_latents = pseudo_prefix_latents + (keep_start - halo_start)
        metas.append(
            {
                "sp_rank": sp_rank,
                "keep_start": keep_start,
                "keep_end": keep_end,
                "halo_start": halo_start,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "raw_frames": raw_end - raw_start,
                "drop_latents": drop_latents,
                "local_latent_frames": local_latent_frames,
            }
        )
    return metas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_ar.yaml", help="Training config path")
    parser.add_argument("--sp-size", type=int, default=None, help="Override sequence_parallel_size")
    parser.add_argument("--halo-latents", type=int, default=None, help="Override vae_halo_latents")
    parser.add_argument("--latent-frames", type=int, default=None, help="Override image_or_video_shape[1]")
    parser.add_argument("--raw-frames", type=int, default=None, help="Override total raw frame count")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = normalize_config(OmegaConf.load(args.config))
    model_name = cfg.model_kwargs.model_name
    temporal_ratio = int(wan_default_config[model_name]["temporal_compression_ratio"])
    total_latent_frames = int(
        args.latent_frames if args.latent_frames is not None else cfg.image_or_video_shape[1]
    )
    total_raw_frames = int(
        args.raw_frames
        if args.raw_frames is not None
        else ((total_latent_frames - 1) * temporal_ratio + 1)
    )
    sp_size = int(
        args.sp_size if args.sp_size is not None else getattr(cfg, "sequence_parallel_size", 1)
    )
    halo_latents = int(
        args.halo_latents
        if args.halo_latents is not None
        else getattr(cfg, "vae_halo_latents", DEFAULT_SP_VAE_HALO_LATENTS)
    )

    metas = compute_chunk_halo_metas(
        total_latent_frames=total_latent_frames,
        total_raw_frames=total_raw_frames,
        sp_size=sp_size,
        halo_latents=halo_latents,
        temporal_compression_ratio=temporal_ratio,
    )
    payload = {
        "config": str(Path(args.config)),
        "model_name": model_name,
        "temporal_compression_ratio": temporal_ratio,
        "total_latent_frames": total_latent_frames,
        "total_raw_frames": total_raw_frames,
        "sp_size": sp_size,
        "halo_latents": halo_latents,
        "rank_metas": metas,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(
        f"config={payload['config']} model={model_name} "
        f"latent_frames={total_latent_frames} raw_frames={total_raw_frames} "
        f"sp_size={sp_size} halo_latents={halo_latents}"
    )
    print("rank  keep_latents  halo_start  raw_window  raw_frames  drop_latents")
    for meta in metas:
        keep_latents = f"[{meta['keep_start']},{meta['keep_end']})".ljust(12)
        raw_window = f"[{meta['raw_start']},{meta['raw_end']})".ljust(11)
        print(
            f"{meta['sp_rank']:>4}  {keep_latents}  {meta['halo_start']:>10}  "
            f"{raw_window}  {meta['raw_frames']:>10}  {meta['drop_latents']:>12}"
        )


if __name__ == "__main__":
    main()
