# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video generator interface.

Memory injection (``backend.injection``):
* ``sink`` - the routed memory pictures [G, M_{i-2}, M_{i-1}] enter the generator's attention sink at the start of
  every shot (VAE-encoded K/V for LongLive-2.0, pictures on the sink clock for MiniMax-H3 generators);
* ``none`` - the generator runs without external memory (LongLive-2.0: its built-in multi-shot memory).
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import BackendConfig, RunConfig
from ..memory.snapshot import MemorySnapshot
from ..runtime.trace import Tracer
from ..story import StorySetup


@dataclass
class ShotResult:
    shot: int
    video_path: Optional[str]
    num_frames: int
    fps: float
    last_frame: Optional[str] = None
    # Seconds: kv_prepare (moving/prefilling memory on the critical path) and video (denoising + decode).
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class VideoBackend(ABC):
    name = "base"

    def __init__(self, cfg: BackendConfig, run: RunConfig, out_dir: Path, tracer: Tracer):
        self.cfg = cfg
        self.run = run
        self.params = dict(cfg.params or {})
        self.out_dir = out_dir
        self.tracer = tracer
        self.injection = cfg.injection if run.memory.mode != "none" else "none"
        self.shots: List[ShotResult] = []
        # entity cards of the revealed prompt (set by the session before prepare_prompt; empty without cards)
        self.text_suffix = ""

    def start(self, setup: StorySetup) -> None:
        """Load models. Called once before the first shot (not counted in shot timing)."""

    def prepare_prompt(self, shot: int, prompt: str, snapshot: Optional[MemorySnapshot],
                       variant: Optional[str] = None) -> str:
        """Backend-specific prompt for the revealed shot (may only use this shot's prompt and memory)."""
        return variant or prompt

    def prefill(self, snapshot: MemorySnapshot, on_aux: bool = True) -> Any:
        """Turn a memory snapshot into a backend handle (e.g. sink K/V). Must not depend on future prompts."""
        return snapshot

    def memory_stats(self) -> Dict[str, Any]:
        """Accelerator memory in use (logged with every prefill); empty when not applicable."""
        return {}

    @abstractmethod
    def generate(self, shot: int, prompt: str, memory: Any, seed: int) -> ShotResult:
        ...

    def flush(self) -> None:
        """Block until every shot file returned so far is fully written."""

    def finish(self) -> Dict[str, Any]:
        """Assemble the full video; returns paths/metadata."""
        return {}

    def close(self) -> None:
        pass


def concat_shots(out_dir: Path, shots: List[ShotResult]) -> Dict[str, Any]:
    """Concatenate per-shot MP4s (same codec) into <out_dir>/video.mp4 with ffmpeg."""
    videos = [s for s in shots if s.video_path and str(s.video_path).endswith(".mp4")]
    if not videos:
        return {}
    listing = out_dir / "shots" / "concat.txt"
    listing.write_text("".join(f"file '{Path(s.video_path).name}'\n" for s in videos))
    full = out_dir / "video.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(full)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"video": None, "error": str(exc)}
    frames = sum(s.num_frames for s in videos)
    fps = videos[0].fps or 24.0
    return {"video": str(full), "frames": frames, "seconds": round(frames / fps, 2)}


def build_backend(cfg: BackendConfig, run: RunConfig, out_dir: Path, tracer: Tracer) -> VideoBackend:
    if cfg.name == "longlive":
        from .longlive import LongLiveBackend
        return LongLiveBackend(cfg, run, out_dir, tracer)
    if cfg.name == "h3":
        from .h3 import H3Backend
        return H3Backend(cfg, run, out_dir, tracer)
    if cfg.name == "mock":
        from .mock import MockBackend
        return MockBackend(cfg, run, out_dir, tracer)
    raise ValueError(f"unknown backend {cfg.name!r} (longlive | h3 | mock)")
