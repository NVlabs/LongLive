# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-style image endpoints (GPT Image on Azure or OpenAI): generations and multi-image edits."""

from __future__ import annotations

import base64
import re
import threading
from pathlib import Path
from typing import List

import requests

from ..config import ModelSpec
from ._http import image_bytes, secret
from .base import ImageClient, ImageRefused, ImageResult, ProviderError
from .throttle import RetryAfter


class OpenAIImageClient(ImageClient):
    """OpenAI-style image API: GPT Image (Azure/OpenAI) or a self-hosted server with the same API (e.g. SGLang
    diffusion serving FLUX.2-klein / Qwen-Image).

    params: flavor (azure|openai|local), api_version (azure), quality, max_ref_side, max_refs, size (overrides the
    agent's image size), extra_fields (merged into the request, e.g. num_inference_steps, seed),
    base_urls (several servers; each request goes to the one with the fewest requests in flight).

    With reference images the ``edits`` endpoint is used (all references are sent as ``image[]``),
    otherwise ``generations``.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        params = spec.params or {}
        self.flavor = params.get("flavor", "azure")
        self.key = secret(spec, required=self.flavor != "local") or "EMPTY"
        bases = params.get("base_urls") or ([spec.base_url] if spec.base_url else [])
        if isinstance(bases, str):  # "http://a:1,http://b:2" (e.g. from an environment variable)
            bases = [b.strip() for b in bases.split(",") if b.strip()]
        bases = list(bases)
        if not bases:
            raise ProviderError("openai_image needs base_url or params.base_urls")
        if self.flavor == "azure":
            version = params.get("api_version", "2025-04-01-preview")
            self.servers = [{k: f"{b.rstrip('/')}/openai/deployments/{spec.model}/images/{k}?api-version={version}"
                             for k in ("generations", "edits")} for b in bases]
            self.headers = {"api-key": self.key}
        else:
            self.servers = [{k: f"{b.rstrip('/')}/images/{k}" for k in ("generations", "edits")} for b in bases]
            self.headers = {"Authorization": f"Bearer {self.key}"}
        self._busy = [0] * len(self.servers)
        self._lock = threading.Lock()

    def _acquire(self) -> int:
        with self._lock:
            k = min(range(len(self.servers)), key=lambda i: self._busy[i])
            self._busy[k] += 1
            return k

    def _generate(self, prompt: str, refs: List[Path], size: str) -> ImageResult:
        params = self.spec.params or {}
        size = params.get("size", size)
        fields = {"prompt": prompt, "size": size, "n": "1"}
        if self.flavor != "local":
            fields["quality"] = params.get("quality", "medium")
        if self.flavor != "azure":
            fields["model"] = self.spec.model
            fields["response_format"] = "b64_json"
        fields.update({k: str(v) for k, v in (params.get("extra_fields") or {}).items()})
        refs = refs[: int(params.get("max_refs", 16))]
        k = self._acquire()
        try:
            urls = self.servers[k]
            if refs:
                files = [("image[]", (f"ref{i}.png", image_bytes(r, params.get("max_ref_side", 1536)), "image/png"))
                         for i, r in enumerate(refs)]
                resp = requests.post(urls["edits"], data=fields, files=files, headers=self.headers,
                                     timeout=self.spec.timeout_s)
            else:
                body: dict = {**fields, "n": 1}
                for key in ("num_inference_steps", "seed"):
                    if key in body:
                        body[key] = int(body[key])
                for key in ("guidance_scale", "true_cfg_scale"):
                    if key in body:
                        body[key] = float(body[key])
                resp = requests.post(urls["generations"], json=body, headers=self.headers, timeout=self.spec.timeout_s)
        finally:
            with self._lock:
                self._busy[k] -= 1
        text = resp.text[:600]
        if resp.status_code == 429:
            match = re.search(r"retry after (\d+) second", text, re.I)
            raise RetryAfter("image 429", float(match.group(1)) + 2 if match else 30.0)
        if resp.status_code == 400 and ("safety" in text.lower() or "moderation" in text.lower()):
            raise ImageRefused(text)
        if resp.status_code >= 500:
            raise RuntimeError(f"image {resp.status_code}: {text}")
        if resp.status_code != 200:
            raise ProviderError(f"image {resp.status_code}: {text}")
        item = resp.json()["data"][0]
        if not item.get("b64_json") and item.get("url"):
            png = requests.get(item["url"], timeout=self.spec.timeout_s).content
        else:
            png = base64.b64decode(item["b64_json"])
        return ImageResult(png=png, model=self.spec.model,
                           meta={"endpoint": "edits" if refs else "generations", "refs": len(refs), "server": k,
                                 "size": size})
