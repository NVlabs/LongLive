# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ffmpeg helpers: frames for the observer, the hand-off frame between segments, and joining segments.

A segment is an 8 s MP4 at 24 fps (192 frames) whose first frame is the keyframe it was started from.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Sequence

from PIL import Image

OBS_IDX = (40, 110, 191)            # v4 observer: early, middle and end frame of a segment
DENSE12 = (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 191)   # hybrid2 hide observer (v4 observer: 64, 128, 191)


def frames(mp4: Path, out_dir: Path, idx: Sequence[int]) -> List[Path]:
    """Extract the frames with the given indices as PNG (``out_dir/f<idx>.png``); returns the paths in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(mp4), "-vf", f"select='{sel}'",
                    "-vsync", "0", str(out_dir / "f_%03d.png")], check=True)
    got = sorted(out_dir.glob("f_*.png"))
    paths = []
    for g, i in zip(got, idx):
        q = out_dir / f"f{i:03d}.png"
        g.replace(q)
        paths.append(q)
    return paths


def load_frames(mp4: Path, out_dir: Path, idx: Sequence[int]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in frames(mp4, out_dir, idx)]


def last_frame(mp4: Path, out_dir: Path) -> Path:
    """The hand-off frame (``out_dir/last.png``) that the next segment starts from.

    ffmpeg seeks to 0.1 s before the end and keeps the last frame it decodes. This is the hand-off used for the paper's
    robot results; for a 192-frame segment it yields frame 190, i.e. 1/24 s before the end (an exact last frame would
    use ``select='eq(n,N-1)'`` with the counted frame number).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "last.png"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-sseof", "-0.1", "-i", str(mp4), "-update", "1",
                    "-frames:v", "1", str(p)], check=True)
    return p


def concat(seg_mp4s: Sequence[Path], out_mp4: Path, crf: int = 20) -> Path:
    """Join segments into one video. Segment k > 0 starts with its keyframe (the previous segment's hand-off frame),
    so its first frame is dropped."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for k in range(len(seg_mp4s)):
        trim = "trim=start_frame=1,setpts=PTS-STARTPTS" if k else "setpts=PTS-STARTPTS"
        parts.append(f"[{k}:v]{trim}[v{k}]")
    n = len(seg_mp4s)
    fc = ";".join(parts) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    for m in seg_mp4s:
        cmd += ["-i", str(m)]
    cmd += ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4.with_suffix(".tmp.mp4"))]
    subprocess.run(cmd, check=True)
    out_mp4.with_suffix(".tmp.mp4").replace(out_mp4)
    return out_mp4
