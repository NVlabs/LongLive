# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU stand-in for a video generator: renders the prompt and the injected memory into frames.

Used by tests and dry runs to exercise the asynchronous runtime with realistic timing
(``params.shot_seconds`` simulates generation time, ``params.prefill_seconds`` the prefill).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..memory.snapshot import MemorySnapshot
from .base import ShotResult, VideoBackend


class MockBackend(VideoBackend):
    name = "mock"

    def prefill(self, snapshot: MemorySnapshot, on_aux: bool = True) -> Dict[str, Any]:
        time.sleep(float(self.params.get("prefill_seconds", 0.0)))
        blocks = snapshot.blocks if self.injection == "sink" else []
        return {"version": snapshot.version, "watermark": snapshot.watermark,
                "blocks": [(b.role, b.path) for b in blocks]}

    def generate(self, shot: int, prompt: str, memory: Optional[Dict[str, Any]], seed: int) -> ShotResult:
        from PIL import Image, ImageDraw

        start = time.monotonic()
        time.sleep(float(self.params.get("shot_seconds", 0.0)))
        out = self.out_dir / "shots"
        out.mkdir(parents=True, exist_ok=True)
        frame = Image.new("RGB", (320, 176), (20 + 10 * (shot % 20), 30, 60))
        draw = ImageDraw.Draw(frame)
        draw.text((6, 6), f"shot {shot} seed {seed}", fill=(255, 255, 255))
        draw.text((6, 22), prompt[:48], fill=(255, 255, 255))
        if memory:
            draw.text((6, 38), f"memory v{memory['version']} w{memory['watermark']}", fill=(255, 255, 0))
            for k, (role, path) in enumerate(memory["blocks"]):
                try:
                    with Image.open(path) as tile:
                        frame.paste(tile.convert("RGB").resize((96, 54)), (6 + 100 * k, 110))
                except OSError:
                    pass
        path = out / f"shot_{shot:03d}.png"
        frame.save(path)
        result = ShotResult(shot, str(path), 1, 1.0, last_frame=str(path),
                            timings={"kv_prepare": 0.0, "video": time.monotonic() - start},
                            meta={"memory": None if not memory else {k: memory[k] for k in ("version", "watermark")},
                                  "prompt": prompt})
        self.shots.append(result)
        return result
