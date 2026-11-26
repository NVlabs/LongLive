# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foreground masks for the global memory image G (its members on a plain light-grey studio backdrop).

mask = matting(G) OR key(G):
  * key: grey, bright, locally smooth pixels connected to the image border are backdrop (a magic-wand fill from the
    border); it keeps every member, including props and scenery objects, but can leak into white or light-grey clothing
    whose edge against the backdrop is weak;
  * matting (BiRefNet, salient objects): keeps people and animals whole, including light clothing, but can drop props.
The mask is stored at the picture's own resolution; each generator pools it to its own token grid (``token_keep``),
so one mask serves any H3 generator resolution. ``g_mask(image, "matte")`` is the paper's mask (matting OR key);
``"key"`` needs no model.
"""
from __future__ import annotations

import threading
from typing import Dict

import numpy as np
from PIL import Image

BIREFNET = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"  # the revision of the paper's masks
_MATTING: Dict[str, object] = {}
_MATTING_LOCK = threading.Lock()


def key_foreground(rgb: np.ndarray, grad: float = 5.0, chroma: float = 24.0, value: float = 110.0) -> np.ndarray:
    """True = foreground (not the plain backdrop). rgb uint8 (H, W, 3)."""
    from scipy import ndimage as ndi

    img = ndi.gaussian_filter(rgb.astype(np.float32), sigma=(1.5, 1.5, 0))
    h, w, _ = img.shape
    lum = img.mean(-1)
    g = np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1)) / 8.0
    c = img.max(-1) - img.min(-1)
    cand = (g < grad) & (c < chroma) & (lum > value)
    lab, _n = ndi.label(cand)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, border[border > 0])
    bg = ndi.binary_closing(bg, iterations=2) | bg
    fg = ndi.binary_opening(~bg, iterations=2)
    lab, n = ndi.label(fg)
    if n:
        sizes = ndi.sum(fg, lab, range(1, n + 1))
        fg[np.isin(lab, np.nonzero(sizes < 0.002 * h * w)[0] + 1)] = False
    holes = ndi.binary_fill_holes(fg) & ~fg
    lab, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab, range(1, n + 1))
        fg[np.isin(lab, np.nonzero(sizes < 0.003 * h * w)[0] + 1)] = True
    return fg


def combine(rgb: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = foreground): matting (soft, [0, 1]) OR key."""
    return (((matte >= 0.5) | key_foreground(rgb)) * 255).astype(np.uint8)


def birefnet_matte(image: Image.Image, device: str = "cpu") -> np.ndarray:
    """Salient-object matte in [0, 1] at the picture's resolution (BiRefNet at 1024x1024, float32)."""
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    with _MATTING_LOCK:
        if device not in _MATTING:
            model = AutoModelForImageSegmentation.from_pretrained(BIREFNET, revision=BIREFNET_REVISION,
                                                                  trust_remote_code=True)
            _MATTING[device] = model.to(device).eval().float()
        model = _MATTING[device]
        tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            matte = model(tf(image).unsqueeze(0).to(device))[-1].sigmoid().cpu()[0, 0]
    return np.asarray(transforms.functional.to_pil_image(matte).resize(image.size), dtype=np.float32) / 255.0


def g_mask(image: Image.Image, mode: str = "matte", device: str = "cpu") -> np.ndarray:
    """uint8 foreground mask of a G picture: ``matte`` = BiRefNet matting OR key (paper), ``key`` = key only."""
    image = image.convert("RGB")
    rgb = np.asarray(image)
    if mode == "key":
        return (key_foreground(rgb) * 255).astype(np.uint8)
    if mode != "matte":
        raise ValueError(f"unknown G mask mode {mode!r} (matte | key)")
    return combine(rgb, birefnet_matte(image, device))


def cover_resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center crop after a cover resize (as the H3 workers fit pictures to the canvas), for a float mask in [0, 1]."""
    im = Image.fromarray((mask * 255).astype(np.uint8))
    scale = max(width / im.width, height / im.height)
    size = (max(width, round(im.width * scale)), max(height, round(im.height * scale)))
    im = im.resize(size, Image.BILINEAR)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return np.asarray(im.crop((left, top, left + width, top + height)), dtype=np.float32) / 255.0


def token_keep(mask_path: str, width: int, height: int, cell: int = 32, min_frac: float = 0.1) -> np.ndarray:
    """(height // cell, width // cell) bool: tokens whose cell holds at least ``min_frac`` foreground."""
    m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    m = cover_resize_mask(m, width, height)
    gh, gw = height // cell, width // cell
    frac = m[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    return frac >= min_frac
