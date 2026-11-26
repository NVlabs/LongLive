# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MiniMax-H3 text / picture encoder service for the LongLive-H3 workers (one GPU, any number of workers).

The MiniMax-H3 text encoder is large, so one process keeps it resident and serves every worker on the machine through
a small file queue of its own (separate from the job protocol):

  <dir>/req/<key>.json   {"kind": "text", "prompt": str} or
                         {"kind": "image", "path": str, "seed": int, "width": int, "height": int}
  <dir>/out/<key>.pt     result (``torch.save`` dict): text -> prompt_embeds, text_token_tags; image -> latent
  <dir>/err/<key>.txt    traceback when the request failed

``key`` is the SHA-256 of the request, so repeated requests (the same prompt, the same memory picture) are served from
``out/``; results unused for ``--keep-s`` seconds are deleted. Text is encoded with diffusers' MiniMaxH3TextEncoderStep;
a memory picture is cover-resized to the video canvas and encoded with diffusers' ``encode_vae_condition`` (H3's
conditioning recipe).

    python -m agentic_longlive.workers.longlive_h3.text_encoder --weights WEIGHTS --queue TE_DIR --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def cover_resize(path, width, height) -> np.ndarray:
    """Scale the picture to cover width x height (bicubic) and center-crop it; (height, width, 3) uint8."""
    image = Image.open(path).convert("RGB")
    scale = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    image = image.resize(size, Image.BICUBIC)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.uint8)


def request_key(req: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()


def encode(te_dir, req: Dict[str, Any], timeout: float = 1800.0) -> Dict[str, Any]:
    """Client side: submit ``req`` to the service at ``te_dir`` (or reuse its cached result) and load the result."""
    import torch

    te_dir = Path(te_dir)
    key = request_key(req)
    out, err = te_dir / "out" / f"{key}.pt", te_dir / "err" / f"{key}.txt"
    if not out.exists():
        tmp = te_dir / "req" / f".{key}.{time.time_ns()}.tmp"
        tmp.write_text(json.dumps(req))
        os.replace(tmp, te_dir / "req" / f"{key}.json")
    start = time.monotonic()
    while not out.exists():
        if err.exists():
            raise RuntimeError("text encoder service failed: " + err.read_text()[-1500:])
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"text encoder service timed out on a {req.get('kind')} request ({te_dir})")
        time.sleep(0.05)
    for _ in range(50):
        try:
            res = torch.load(out, map_location="cpu")
            out.touch()  # keep it from the service's cleanup
            return res
        except Exception:  # noqa: BLE001 -- racing the cleanup; retry
            time.sleep(0.1)
    raise RuntimeError(f"cannot read the text encoder result {out}")


def wait_ready(te_dir, poll_s: float = 2.0) -> None:
    while not (Path(te_dir) / "ready").exists():
        time.sleep(poll_s)


class Encoders:
    """The resident MiniMax-H3 text encoder and video VAE (encoder side) on one GPU."""

    def __init__(self, base: str, device: str):
        import torch
        from diffusers import AutoencoderKLMiniMaxH3
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep, encode_vae_condition

        self.torch, self.encode_vae_condition = torch, encode_vae_condition
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.pipe = MiniMaxH3TextEncoderStep().init_pipeline(base)
        self.pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=base, local_files_only=True)
        self.pipe.text_encoder.eval().requires_grad_(False)
        self.pipe.to(self.device)
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(base, subfolder="vae", dtype=torch.float32, local_files_only=True,
                                                          device_map={"": str(self.device)}).eval().requires_grad_(False)

    def __call__(self, req: Dict[str, Any]) -> Dict[str, Any]:
        torch = self.torch
        with torch.inference_mode():
            if req["kind"] == "text":
                enc = self.pipe(prompt=req["prompt"], output=["prompt_embeds", "text_token_tags"])
                return {k: v.detach().cpu() for k, v in enc.items()}
            if req["kind"] == "image":
                arr = cover_resize(req["path"], int(req.get("width", 1344)), int(req.get("height", 768)))
                pix = torch.from_numpy(arr).permute(2, 0, 1)[None, :, None].to(self.device)
                return {"latent": self.encode_vae_condition(self.vae, pix, IMAGE_MEAN, IMAGE_STD, int(req["seed"]))}
        raise ValueError(f"unknown request kind {req.get('kind')!r}")


def serve(root: Path, encoders: Encoders, keep_s: float = 3600.0) -> None:
    import torch

    (root / "ready").write_text(json.dumps({"time": time.time(), "pid": os.getpid()}))
    print("text encoder ready", flush=True)
    last_gc = 0.0
    while True:
        reqs = sorted((root / "req").glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if not reqs:
            if time.time() - last_gc > 300:
                last_gc = time.time()
                for p in (root / "out").glob("*.pt"):
                    try:
                        if time.time() - p.stat().st_mtime > keep_s:
                            p.unlink()
                    except FileNotFoundError:
                        pass
            time.sleep(0.05)
            continue
        for path in reqs:
            key = path.stem
            try:
                req = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 -- partially visible file; retry next round
                continue
            t0 = time.perf_counter()
            try:
                res = encoders(req)
                res["seconds"] = time.perf_counter() - t0
                tmp = root / "out" / f".{key}.tmp"
                torch.save(res, tmp)
                os.replace(tmp, root / "out" / f"{key}.pt")
            except Exception:  # noqa: BLE001 -- reported to the worker
                (root / "err" / f"{key}.txt").write_text(traceback.format_exc())
            path.unlink(missing_ok=True)


def base_dir(args) -> str:
    if args.base:
        return args.base
    if not args.weights:
        raise SystemExit("give --weights (setup_longlive_h3.sh layout) or --base")
    return str(Path(args.weights) / "MiniMax-H3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="weights directory written by scripts/workers/setup_longlive_h3.sh")
    ap.add_argument("--base", help="MiniMax-H3 base directory (text encoder, VAEs); default <weights>/MiniMax-H3")
    ap.add_argument("--queue", required=True, help="service directory, passed to the workers as --te")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keep-s", type=float, default=3600.0, help="delete results unused for this long")
    args = ap.parse_args()
    root = Path(args.queue)
    for sub in ("req", "out", "err"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "ready").unlink(missing_ok=True)  # workers wait until this service has loaded
    serve(root, Encoders(base_dir(args), args.device), keep_s=args.keep_s)


if __name__ == "__main__":
    main()
