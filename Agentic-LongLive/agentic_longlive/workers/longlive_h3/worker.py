# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongLive-H3 worker: renders the jobs of one file queue (:mod:`..protocol`) on one GPU.

Each job is one shot (default 8 s, 192 frames at 24 fps, 1344x768, video + audio MP4). Layouts:

  refs + ref_seconds   sink: picture j at text_len + 40 * ref_seconds[j], generated media shifted 40 * video_start_seconds
  refs, no ref_seconds ref2va: picture j at text_len + j, generated media shifted by the number of pictures
  no refs              text to video+audio (shifted by 40 * video_start_seconds if that is > 0)

``ref_masks`` (one PNG or null per picture) keeps only the foreground tokens of a picture: a 32x32-pixel token is kept
when at least 10% of its mask is foreground. Prompts and pictures are encoded by the text-encoder service
(:mod:`.text_encoder`, ``--te``). Several workers (one per GPU) can share one queue and one text-encoder service.

    python -m agentic_longlive.workers.longlive_h3.worker --weights WEIGHTS --te TE_DIR --queue QUEUE --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .. import protocol
from . import text_encoder as te

TICKS_PER_SECOND = 40.0  # H3's rotary clock (= core.TICKS_PER_SECOND)
FPS = 24
WIDTH, HEIGHT = 1344, 768
PICTURE_VAE_SEED = 20261020  # seed of the VAE encoding of memory pictures


def memory_layout(n_refs: int, ref_seconds: Optional[Sequence[float]],
                  video_start_seconds: Optional[float]) -> Tuple[List[float], float, str]:
    """(picture offsets after the text, start of the generated media after the text, layout name), in rotary ticks."""
    if n_refs:
        if ref_seconds:
            if video_start_seconds is None:
                raise ValueError("sink layout without video_start_seconds")
            offsets = [TICKS_PER_SECOND * float(s) for s in ref_seconds]
            shift = TICKS_PER_SECOND * float(video_start_seconds)
            if len(offsets) != n_refs or offsets != sorted(offsets) or offsets[-1] >= shift:
                raise ValueError(f"bad sink layout {offsets} / {shift}")
            return offsets, shift, "sink"
        return [float(j) for j in range(n_refs)], float(n_refs), "ref2va"
    if float(video_start_seconds or 0.0) > 0:
        return [], TICKS_PER_SECOND * float(video_start_seconds), "shift"
    return [], 0.0, "t2va"


def token_keep(mask_path, width: int, height: int, cell: int = 32, min_frac: float = 0.1) -> np.ndarray:
    """(height // cell, width // cell) bool grid: tokens whose cell of the cover-resized mask is >= min_frac foreground."""
    m = Image.open(mask_path).convert("L")
    scale = max(width / m.width, height / m.height)
    size = (max(width, round(m.width * scale)), max(height, round(m.height * scale)))
    m = m.resize(size, Image.BILINEAR)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    a = np.asarray(m.crop((left, top, left + width, top + height)), dtype=np.float32) / 255.0
    gh, gw = height // cell, width // cell
    return a[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3)) >= min_frac


def memory_keep(masks: Optional[Sequence[Optional[str]]], count: int, width: int, height: int):
    """Bool tensor over the count * tokens memory rows, or None to keep every row. A picture without a mask, or whose
    mask keeps no token, keeps all of its rows."""
    import torch

    if not any(masks or []):
        return None
    per = (height // 32) * (width // 32)
    keep = torch.ones(count * per, dtype=torch.bool)
    for j, path in enumerate(list(masks)[:count]):
        if path:
            k = torch.from_numpy(token_keep(path, width, height).reshape(-1))
            if k.any():
                keep[j * per:(j + 1) * per] = k
    return keep


def shot_frames(duration_seconds: Optional[float], default: int) -> int:
    """Frames of a shot of ``duration_seconds`` (must be 17k + 5 at 24 fps, e.g. 8 s = 192 frames)."""
    if duration_seconds is None:
        return default
    frames = round(float(duration_seconds) * FPS)
    if (frames - 5) % 17 or abs(frames - float(duration_seconds) * FPS) > 1e-6:
        raise ValueError(f"LongLive-H3 renders 17k + 5 frames at {FPS} fps; {duration_seconds} s is not such a length")
    return frames


class LongLiveH3Worker:
    """The transformer, its KV runtime and the VAEs on one GPU; ``handle(job)`` renders one shot."""

    def __init__(self, base: str, transformer: str, te_dir: str, device: str = "cuda:0", frames: int = 192):
        import torch

        from . import core

        self.torch, self.core = torch, core
        self.te = Path(te_dir)
        self.frames = frames
        self.dev = torch.device(device)
        torch.cuda.set_device(self.dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        t0 = time.time()
        self.rt = core.KVRuntime(core.load_transformer(transformer, self.dev), self.dev)
        self.vaes = core.VAEs(base, self.dev)
        self.info = {"worker": "longlive_h3", "transformer": transformer, "frames": frames, "device": str(self.dev),
                     "load_s": round(time.time() - t0, 1), "model_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}

    def handle(self, job: Dict[str, Any]) -> Dict[str, Any]:
        torch, core = self.torch, self.core
        if job.get("keyframe"):
            raise ValueError("LongLive-H3 does not take keyframe jobs (use the FastH3-Omni worker)")
        if job.get("ref_scale") not in (None, 1, 1.0):
            raise ValueError("LongLive-H3 encodes memory pictures at the full video canvas (ref_scale must be 1)")
        g = core.Geometry(shot_frames(job.get("duration_seconds"), self.frames), WIDTH, HEIGHT)
        start = time.time()
        t: Dict[str, Any] = {}
        try:
            torch.cuda.reset_peak_memory_stats()
            ts = time.perf_counter()
            enc = te.encode(self.te, {"kind": "text", "prompt": job["prompt"]})
            t["te_s"] = time.perf_counter() - ts
            refs = [str(p) for p in job.get("refs") or []]
            offsets, shift, t["layout"] = memory_layout(len(refs), job.get("ref_seconds"), job.get("video_start_seconds"))
            mem = None
            if refs:
                ts = time.perf_counter()
                lats = [te.encode(self.te, {"kind": "image", "path": p, "seed": PICTURE_VAE_SEED,
                                            "width": g.width, "height": g.height})["latent"] for p in refs]
                mem = torch.cat([x[0] for x in lats], dim=1)  # (24, n, lh, lw)
                t["ref_encode_s"] = time.perf_counter() - ts
            pe = enc["prompt_embeds"]
            tags = enc.get("text_token_tags")
            tags = tags.reshape(-1) if tags is not None else None
            keep = memory_keep(job.get("ref_masks"), len(refs), g.width, g.height) if refs else None
            video, audio = core.generate(self.rt, g, pe, int(job["seed"]), mem_latents=mem, mem_offsets=offsets,
                                         shift=shift, text_tags=tags, timing=t, mem_keep=keep)
            ts = time.perf_counter()
            frames, wav, rate = self.vaes.decode(video, audio, g)
            t["decode_s"] = time.perf_counter() - ts
            t["gen_s"] = time.time() - start
            t["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            t["text_len"] = int(pe.shape[1])
            t["n_refs"] = len(refs)
            ts = time.perf_counter()
            self._save(frames, wav, rate, Path(job["out"]))
            t["save_s"] = time.perf_counter() - ts
            return t
        except Exception:
            self.rt.free()
            torch.cuda.empty_cache()
            raise

    @staticmethod
    def _save(frames, wav, rate, out: Path) -> None:
        from diffusers.utils.export_utils import encode_video

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.mp4")
        encode_video(video=frames, fps=FPS, output_path=str(tmp), audio=wav, audio_sample_rate=rate)
        os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="weights directory written by scripts/workers/setup_longlive_h3.sh")
    ap.add_argument("--base", help="MiniMax-H3 base directory (VAEs); default <weights>/MiniMax-H3")
    ap.add_argument("--transformer", help="directory holding transformer/ (LongLive-H3); default <weights>/LongLive-H3")
    ap.add_argument("--te", required=True, help="text-encoder service directory (its --queue)")
    ap.add_argument("--queue", required=True, help="job queue directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=192, help="frames per shot when a job has no duration_seconds")
    args = ap.parse_args()
    if not args.weights and not (args.base and args.transformer):
        raise SystemExit("give --weights (setup_longlive_h3.sh layout) or both --base and --transformer")
    base = args.base or str(Path(args.weights) / "MiniMax-H3")
    transformer = args.transformer or str(Path(args.weights) / "LongLive-H3")
    worker = LongLiveH3Worker(base, transformer, args.te, args.device, args.frames)
    print("waiting for the text encoder service", args.te, flush=True)
    te.wait_ready(args.te)
    info = dict(worker.info, pid=os.getpid())
    print("ready", json.dumps(info), flush=True)
    protocol.serve(Path(args.queue), worker.handle, ready=info)


if __name__ == "__main__":
    main()
