# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame sampling for the LongStory-Online metrics.

Every metric reads a fixed set of frames from each shot video:

* shot judge and protagonist pair judge: 4 frames at 4 / 35 / 65 / 97% of the shot, resized to 768 px wide (JPEG q90);
* place (revisit) pair judge: 2 frames at 50 / 95% of each shot, 640 px wide (JPEG q88);
* VBench imaging quality: 16 uniformly spaced frames (centre of 16 equal segments);
* MIR (StoryEngine protocol): 4 evenly spaced frames including the first and the last.

Frames are decoded with ``decord`` when it is installed, otherwise with imageio-ffmpeg (frames are read in increasing
order, so the ffmpeg reader never seeks backwards). Both return RGB uint8 arrays [H, W, 3].
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

JUDGE_FRACTIONS = (0.04, 0.35, 0.65, 0.97)
JUDGE_WIDTH, JUDGE_QUALITY = 768, 90
PAIR_FRACTIONS = (0.5, 0.95)
PAIR_WIDTH, PAIR_QUALITY = 640, 88
IMAGING_FRAMES = 16
MIR_FRAMES = 4


def fraction_index(n: int, fraction: float) -> int:
    return min(n - 1, max(0, round(fraction * (n - 1))))


def uniform_indices(n: int, count: int) -> List[int]:
    """Centre of ``count`` equal segments (the 16 frames used for VBench imaging quality)."""
    return [min(n - 1, round((k + 0.5) * n / count)) for k in range(count)]


def evenly_spaced(n: int, count: int) -> List[int]:
    """``count`` indices from the first to the last frame (numpy.linspace, rounded), de-duplicated."""
    if n <= 1:
        return [0]
    step = (n - 1) / (count - 1)
    return sorted({int(round(k * step)) for k in range(count)})


class Video:
    """Frame count, frame rate and random access to the frames of one shot video."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        try:
            import decord  # optional, faster

            self._decord = decord.VideoReader(self.path, num_threads=2)
            self.n = len(self._decord)
            self.fps = float(self._decord.get_avg_fps() or 24.0)
        except ImportError:
            import imageio.v2 as imageio

            self._decord = None
            reader = imageio.get_reader(self.path, "ffmpeg")
            try:
                self.fps = float(reader.get_meta_data().get("fps") or 24.0)
                self.n = int(reader.count_frames())
            finally:
                reader.close()
        if self.n <= 0:
            raise ValueError(f"no frames in {self.path}")

    @property
    def duration(self) -> float:
        return round(self.n / self.fps, 3)

    def frames(self, indices: Sequence[int]):
        """RGB uint8 frames (list of numpy arrays) at the given indices, in the given order."""
        wanted = sorted(set(int(i) for i in indices))
        got: Dict[int, object] = {}
        if self._decord is not None:
            batch = self._decord.get_batch(wanted)
            array = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.numpy()
            got = {i: array[k] for k, i in enumerate(wanted)}
        else:
            import imageio.v2 as imageio
            import numpy as np

            reader = imageio.get_reader(self.path, "ffmpeg")
            try:
                for i in wanted:  # increasing order: sequential reads, no backward seek
                    got[i] = np.asarray(reader.get_data(i))[:, :, :3]
            finally:
                reader.close()
        return [got[int(i)] for i in indices]

    def at(self, fractions: Sequence[float]):
        return self.frames([fraction_index(self.n, f) for f in fractions])


def to_pil(frames, width: Optional[int] = None):
    """numpy frames -> PIL images, downscaled (bicubic) to ``width`` if wider."""
    from PIL import Image

    out = []
    for frame in frames:
        image = Image.fromarray(frame)
        if width and image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.BICUBIC)
        out.append(image)
    return out


def judge_frame_paths(eval_dir: Path, shot: int) -> List[Path]:
    return [Path(eval_dir) / "judge_frames" / f"shot_{shot:03d}_{k}.jpg" for k in range(len(JUDGE_FRACTIONS))]


def save_judge_frames(video: str, eval_dir: Path, shot: int) -> List[Path]:
    """The shot judge's 4 frames, also reused by the protagonist pair judge."""
    paths = judge_frame_paths(eval_dir, shot)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, image in zip(paths, to_pil(Video(video).at(JUDGE_FRACTIONS), width=JUDGE_WIDTH)):
        image.save(path, quality=JUDGE_QUALITY)
    return paths


def save_pair_frames(video: str, eval_dir: Path, shot: int) -> List[Path]:
    """The place pair judge's 2 frames of a shot (cached: a shot can be in several pairs)."""
    paths = [Path(eval_dir) / "pair_frames" / f"shot_{shot:03d}_p{k}.jpg" for k in range(len(PAIR_FRACTIONS))]
    if all(p.is_file() for p in paths):
        return paths
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, image in zip(paths, to_pil(Video(video).at(PAIR_FRACTIONS), width=PAIR_WIDTH)):
        image.save(path, quality=PAIR_QUALITY)
    return paths
