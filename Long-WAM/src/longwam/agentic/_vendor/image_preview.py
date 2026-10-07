# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/image_preview.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Teacher-only PNG attachments; never resize policy arrays or source images."""

from io import BytesIO
from pathlib import Path

from PIL import Image


def image_max_edge(value):
    if not str(value).isdigit():
        raise ValueError(
            "CODEX_IMAGE_MAX_EDGE must be a non-negative integer (0 disables resizing)"
        )
    return int(value)


def prepare_image(item, max_edge):
    """Return a copied descriptor and exact attachment bytes, saving any preview.

    ``path`` remains the full-resolution source for native view_image access.
    A separate preview is stored next to the immutable observation, so the
    existing artifact inventory and RPC log preserve what Codex actually saw.
    """
    max_edge = image_max_edge(max_edge)
    source = Path(item["path"])
    original = source.read_bytes()
    with Image.open(BytesIO(original)) as picture:
        if max_edge == 0 or max(picture.size) <= max_edge:
            return dict(item), original
        original_size = list(picture.size)
        picture.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        picture.save(buffer, format="PNG")
        payload = buffer.getvalue()
        size = list(picture.size)
    preview = source.parent / "codex_previews" / f"{source.stem}.max{max_edge}.png"
    preview.parent.mkdir(exist_ok=True)
    try:
        with preview.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if preview.read_bytes() != payload:
            raise ValueError(f"Existing Codex preview differs; refusing to overwrite {preview}")
    return dict(
        item,
        codex_preview=dict(
            path=str(preview), size=size, original_size=original_size, resampling="LANCZOS"
        ),
    ), payload
