# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VBench imaging quality (optional; needs ``torch``, ``torchvision`` and ``pyiqa``).

Same model and preprocessing as VBench's ``imaging_quality`` dimension: MUSIQ trained on SPAQ, frames whose longer
side exceeds 512 px resized to 512 (VBench mode ``longer``, no antialiasing), per-frame scores averaged and divided by
100. As in the paper, each shot is scored on 16 uniformly spaced frames (:func:`frames.uniform_indices`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

MUSIQ_SPAQ = "musiq_spaq_ckpt-358bb6af.pth"
MUSIQ_SPAQ_URL = f"https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/{MUSIQ_SPAQ}"


def default_cache() -> Path:
    return Path(os.environ.get("VBENCH_CACHE_DIR", str(Path.home() / ".cache" / "vbench")))


class ImagingQuality:
    """Callable: list of RGB uint8 frames [H, W, 3] -> imaging quality in [0, 1]."""

    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
        try:
            import torch
            from pyiqa.archs.musiq_arch import MUSIQ
            from torchvision import transforms
        except ImportError as exc:  # optional dependency
            raise ImportError("VBench imaging quality needs `pip install torch torchvision pyiqa`") from exc
        self.torch, self.transforms = torch, transforms
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(cache_dir or default_cache()) / "pyiqa_model" / MUSIQ_SPAQ
        if not ckpt.is_file():  # the checkpoint VBench downloads on first use
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(MUSIQ_SPAQ_URL, str(ckpt))
        self.model = MUSIQ(pretrained_model_path=str(ckpt)).to(self.device)
        self.model.eval()
        self.model.training = False

    def __call__(self, frames: Sequence) -> float:
        import numpy as np

        torch = self.torch
        images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # [T, C, H, W] uint8
        _, _, h, w = images.shape
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            images = self.transforms.Resize(size=(int(scale * h), int(scale * w)), antialias=False)(images)
        with torch.no_grad():
            scores = self.model((images.float() / 255.0).to(self.device)).flatten().tolist()
        return sum(scores) / len(scores) / 100.0
