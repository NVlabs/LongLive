# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for HTTP providers: credentials, image encoding."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

from ..config import ModelSpec
from .base import ProviderError


def secret(spec: ModelSpec, required: bool = True) -> Optional[str]:
    """API key from ``api_key_env``, or the contents of the file named by ``credentials_env``."""
    if spec.api_key_env and os.environ.get(spec.api_key_env):
        return os.environ[spec.api_key_env].strip()
    if spec.credentials_env and os.environ.get(spec.credentials_env):
        path = Path(os.environ[spec.credentials_env]).expanduser()
        if path.is_file():
            return path.read_text().strip()
    if required:
        raise ProviderError(f"{spec.provider}: set {spec.api_key_env or spec.credentials_env} to provide credentials")
    return None


def credentials_file(spec: ModelSpec) -> Path:
    if not spec.credentials_env or not os.environ.get(spec.credentials_env):
        raise ProviderError(f"{spec.provider}: set {spec.credentials_env or '<credentials_env>'} to a credentials file")
    path = Path(os.environ[spec.credentials_env]).expanduser()
    if not path.is_file():
        raise ProviderError(f"{spec.provider}: credentials file not found ({spec.credentials_env})")
    return path


def image_bytes(path: Path, max_side: int = 1024, fmt: str = "PNG") -> bytes:
    """Load an image and downscale it so that the longer side is at most ``max_side``."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
