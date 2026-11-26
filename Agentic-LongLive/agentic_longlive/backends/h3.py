# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 generators (LongLive-H3, FastH3-Omni) with sink memory, through a worker queue.

The generator runs in its own process and environment (``agentic_longlive.workers``); this client builds each shot's
job and waits for the video. The memory mechanism is the one of the LongLive-2.0 backend, mapped to H3's clock
(40 rotary ticks per second): the routed memory pictures sit at ``ref_seconds`` (0 / 0.25 / 0.5 s) and the shot starts
after a cut at ``video_start_seconds`` (2 s). Scene routing (:mod:`..agent.router`) and entity cards are shared with
LongLive-2.0. Two H3-specific details:

  * prompt: H3 takes an audio-video description per shot (story variant ``prompt_variant``, default ``h3``). Memory
    pictures enter it as short shots before a cut (``[Shot k] ... as established by Picture k``), followed by the
    revealed shot; entity cards are added to the shot description, before the soundscape. The scene router reads
    this H3 description (without cards) as the next shot's prompt;
  * foreground G: G shows its members on a plain studio backdrop. Its backdrop tokens are dropped (a mask computed
    when G is committed), so the backdrop does not enter the video as a place.

params:
  queue_dir            worker queue (shared directory)
  prompt_variant       story variant with the H3 prompt of each shot (default h3; without it the raw prompt is used)
  ref_seconds          memory picture times (default [0.0, 0.25, 0.5]); video_start_seconds (2.0)
  g_masks              drop G's backdrop tokens (workers that support ``ref_masks``): matte (default, also true) =
                       BiRefNet matting OR backdrop key, as in the paper; key = backdrop key only (no extra model);
                       false = whole G. mask_device: device of the matting model (default cpu)
  scene_routing / route_llm_url / route_llm_model / route_llm_style / route_decisions   as for LongLive-2.0
  fps (24), timeout_s (3600), ready_timeout_s (3600)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agent.router import SceneRouter, load_decisions
from ..memory.snapshot import MemoryImage, MemorySnapshot
from ..workers import protocol
from .base import ShotResult, VideoBackend, concat_shots

H3_SHOT_PREFIX = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "


def _timestamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _phrase(text: str) -> str:
    """A memory caption as a noun phrase that reads after "Live-action, cinematic, " or "the camera cuts to "."""
    text = " ".join(str(text or "").split()).rstrip(" .") or "the scene"
    first, _, rest = text.partition(" ")
    return (first.lower() + " " + rest).strip() if first in ("A", "An", "The") else text


def members_only(caption: str) -> str:
    """G's caption without the sentence about its plain studio backdrop (the backdrop is not passed with G)."""
    return re.sub(r"\s*[^.]*\b(?:studio background|studio backdrop|plain backdrop|plain background)\b[^.]*\.", "",
                  caption).strip()


def memory_prompt(prompt: str, captions: List[str], ref_seconds: List[float], video_start: float) -> str:
    """Memory pictures as short shots before a cut to the generated shot (H3 prompt format only)."""
    if not captions or not prompt.startswith(H3_SHOT_PREFIX):
        return prompt
    body, sep, tail = prompt[len(H3_SHOT_PREFIX):].partition("\n\n")
    header = "How the reference pictures align with the target video — " + "; ".join(
        f"Picture {k} (from Shot {k}) aligns with the {t:.2f}-second mark of the target video"
        for k, t in enumerate(ref_seconds, 1)) + "."
    shots = []
    for k, (caption, t) in enumerate(zip(captions, ref_seconds), 1):
        lead = "Live-action, cinematic, " if k == 1 else f"At {_timestamp(t)}, the camera cuts to "
        shots.append(f"[Shot {k}] {lead}{_phrase(caption)}, as established by Picture {k}.")
    shots.append(f"[Shot {len(captions) + 1}] At {_timestamp(video_start)}, the camera cuts to {body}")
    return header + "\n\nintegrated_multimodal_description: " + " ".join(shots) + sep + tail


def with_cards(h3_prompt: str, card_text: str) -> str:
    """Entity cards at the end of the H3 shot description (the paragraph before the soundscape)."""
    if not card_text:
        return h3_prompt
    head, sep, tail = h3_prompt.partition("\n\n")
    return head.rstrip() + " " + card_text.strip() + sep + tail


class H3Backend(VideoBackend):
    name = "h3"

    def __init__(self, cfg, run, out_dir, tracer):
        super().__init__(cfg, run, out_dir, tracer)
        params = self.params
        self.queue = Path(params.get("queue_dir") or (out_dir / "h3_queue")).resolve()
        self.ref_seconds = [float(x) for x in params.get("ref_seconds", [0.0, 0.25, 0.5])]
        self.video_start = float(params.get("video_start_seconds", 2.0))
        self.fps = float(params.get("fps", 24))
        mode = params.get("g_masks", "matte")
        self.mask_mode = "matte" if mode is True else (str(mode) if mode else "")
        if self.mask_mode not in ("", "matte", "key"):
            raise ValueError(f"g_masks must be matte, key or false, not {mode!r}")
        self.use_masks = bool(self.mask_mode)
        self.mask_device = str(params.get("mask_device", "cpu"))
        self.router: Optional[SceneRouter] = None
        if self.injection != "none" and str(params.get("scene_routing", "llm")) == "llm":
            decisions = None
            if params.get("route_decisions"):
                decisions = load_decisions(params["route_decisions"], Path(run.story).stem)
            self.router = SceneRouter(
                url=str(params.get("route_llm_url") or os.environ.get("AGL_ROUTE_URL") or "http://127.0.0.1:31001/v1"),
                model=str(params.get("route_llm_model") or os.environ.get("AGL_ROUTE_MODEL")
                          or "Qwen/Qwen3.6-35B-A3B-FP8"),
                style=str(params.get("route_llm_style", "sglang")), decisions=decisions)
        self._blocks: List[MemoryImage] = []
        self._route: Optional[Dict[str, Any]] = None
        self._masks: Dict[str, str] = {}
        self._mask_lock = threading.Lock()

    def start(self, setup) -> None:
        protocol.ensure(self.queue)
        deadline = time.monotonic() + float(self.params.get("ready_timeout_s", 3600))
        while not (self.queue / "ready").exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no generator worker on {self.queue} (start one, see docs/generators.md)")
            time.sleep(2.0)
        self.tracer.log("h3_worker_ready", queue=str(self.queue), info=json.loads((self.queue / "ready").read_text()))

    # ------------------------------------------------------------------ memory
    def prefill(self, snapshot: MemorySnapshot, on_aux: bool = True):
        """Nothing to prefill on this side; G's foreground mask is computed when G is committed."""
        if self.injection == "none":
            return None
        if self.use_masks and snapshot.g is not None:
            self._g_mask(snapshot.g.path)
        return snapshot

    def _g_mask(self, path: str) -> str:
        from PIL import Image

        from .fgmask import g_mask

        with self._mask_lock:
            if path not in self._masks:
                t0 = time.monotonic()
                mask = g_mask(Image.open(path), self.mask_mode, self.mask_device)
                root = self.out_dir / "g_masks"
                root.mkdir(parents=True, exist_ok=True)
                dst = root / (hashlib.sha1(Path(path).read_bytes()).hexdigest() + ".png")
                Image.fromarray(mask).save(dst)
                self._masks[path] = str(dst.resolve())
                self.tracer.log("g_mask", path=path, mask=str(dst), fg=round(float(mask.mean()) / 255, 4),
                                seconds=round(time.monotonic() - t0, 4))
            return self._masks[path]

    def _select(self, shot: int, snapshot: Optional[MemorySnapshot], prompt: str) -> List[MemoryImage]:
        """Scene routing at the boundary: [G, M_{i-2}, M_{i-1}], [G, M_loc] or [G]."""
        self._route = None
        if snapshot is None or self.injection == "none":
            return []
        blocks = list(snapshot.blocks)
        if self.router is None or shot == 0 or not snapshot.m:
            return blocks
        m_entries = [{"scene": m.scene, "prompt": m.prompt, "watermark": int(m.watermark), "path": m.path,
                      "caption": m.caption, "members": list(m.members), "round": int(m.round), "score": m.score}
                     for m in snapshot.m]
        scenes = {r.name: r.description for r in snapshot.references if r.kind == "scene" and r.name}
        self.router.register(shot, m_entries, scenes)
        decision = self.router.route(shot, prompt, m_entries[-1])
        entry = decision.pop("entry", None)
        g = [snapshot.g] if snapshot.g is not None else []
        if decision["decision"] == "recall":
            m = MemoryImage(role="M", path=entry["path"], caption=entry.get("caption", ""),
                            members=tuple(entry.get("members") or ()), watermark=int(entry.get("watermark", -1)),
                            round=int(entry.get("round", 0)), score=entry.get("score"), scene=entry["scene"],
                            prompt=entry.get("prompt", ""))
            decision["recall"] = {"scene": m.scene, "watermark": m.watermark, "path": m.path}
            blocks = g + [m]
        elif decision["decision"] == "drop":
            blocks = g
        decision["blocks"] = [b.role for b in blocks]
        self._route = decision
        self.tracer.log("route", shot=shot, **decision)
        return blocks

    def prepare_prompt(self, shot: int, prompt: str, snapshot: Optional[MemorySnapshot],
                       variant: Optional[str] = None) -> str:
        self._blocks = self._select(shot, snapshot, variant or prompt)
        base = with_cards(variant or prompt, self.text_suffix)  # cards go into the shot's own description
        if not self._blocks:
            return base
        captions = [members_only(b.caption) if (b.role == "G" and self.use_masks) else b.caption
                    for b in self._blocks]
        return memory_prompt(base, captions, self.ref_seconds[:len(self._blocks)], self.video_start)

    # --------------------------------------------------------------- generate
    def _freeze(self, shot: int, blocks: List[MemoryImage]) -> List[str]:
        """Copy the injected pictures next to the shot (the agent may later replace files of the same name)."""
        frozen = self.out_dir / "memory_blocks" / f"shot_{shot:03d}"
        frozen.mkdir(parents=True, exist_ok=True)
        paths = []
        for k, block in enumerate(blocks):
            dst = frozen / f"{k}_{block.role}{Path(block.path).suffix or '.png'}"
            if not dst.exists():
                shutil.copyfile(block.path, dst)
            paths.append(str(dst.resolve()))
        return paths

    def generate(self, shot: int, prompt: str, memory: Any, seed: int) -> ShotResult:
        blocks, self._blocks = self._blocks, []
        refs = self._freeze(shot, blocks) if blocks else []
        out = (self.out_dir / "shots" / f"shot_{shot:03d}.mp4").resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        job: Dict[str, Any] = {"id": f"{shot:03d}_{uuid.uuid4().hex[:8]}", "prompt": prompt, "refs": refs,
                               "seed": int(seed), "out": str(out), "ref_seconds": [], "video_start_seconds": 0.0}
        if refs:
            job.update(ref_seconds=self.ref_seconds[:len(refs)], video_start_seconds=self.video_start)
            if self.use_masks:
                masks = [self._g_mask(b.path) if b.role == "G" else None for b in blocks]
                if any(masks):
                    job["ref_masks"] = masks
        start = time.monotonic()
        record = protocol.submit(self.queue, job, timeout_s=float(self.params.get("timeout_s", 3600)))
        frames, last = self._probe(out)
        result = ShotResult(shot, str(out), frames, self.fps, last_frame=last,
                            timings={"kv_prepare": 0.0, "video": time.monotonic() - start},
                            meta={"worker_seconds": record.get("seconds"), "worker_timing": record.get("timing"),
                                  "memory_blocks": len(refs), "memory_roles": [b.role for b in blocks],
                                  "memory_paths": [b.path for b in blocks], "ref_seconds": job["ref_seconds"],
                                  "route": self._route, "ref_masks": job.get("ref_masks")})
        self.shots.append(result)
        return result

    @staticmethod
    def _probe(path: Path):
        try:
            import imageio.v2 as imageio
            from PIL import Image

            reader = imageio.get_reader(str(path))
            count = reader.count_frames()
            last = path.with_name(path.stem + "_last.jpg")
            Image.fromarray(reader.get_data(count - 1)).save(last, quality=92)
            reader.close()
            return count, str(last)
        except Exception:  # noqa: BLE001 -- preview only
            return 0, None

    def finish(self) -> Dict[str, Any]:
        return concat_shots(self.out_dir, self.shots)
