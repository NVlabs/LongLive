# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat endpoints: self-hosted vLLM / SGLang servers, OpenAI, Azure OpenAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import requests

from ..config import ModelSpec
from ._http import b64, image_bytes, secret
from .base import Content, LLMClient, LLMResult, ProviderError
from .throttle import RetryAfter


class OpenAICompatClient(LLMClient):
    """params: azure_api_version (switches to Azure deployments), json_response_format (default true),
    max_image_side, extra_body (merged into the request, e.g. chat_template_kwargs)."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        if not spec.base_url:
            raise ProviderError("openai_compat needs base_url (e.g. http://127.0.0.1:8000/v1)")
        self.key = secret(spec, required=False) or "EMPTY"
        params = spec.params or {}
        self.azure_version = params.get("azure_api_version")
        if self.azure_version:
            self.url = (f"{spec.base_url.rstrip('/')}/openai/deployments/{spec.model}/chat/completions"
                        f"?api-version={self.azure_version}")
            self.headers = {"api-key": self.key}
        else:
            self.url = spec.base_url.rstrip("/") + "/chat/completions"
            self.headers = {"Authorization": f"Bearer {self.key}"}

    def _chat(self, system: str, content: Content, json_mode: bool, temperature: float, max_tokens: int) -> LLMResult:
        params = self.spec.params or {}
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, Path):
                url = "data:image/png;base64," + b64(image_bytes(item, params.get("max_image_side", 1024)))
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": str(item)})
        body: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not self.azure_version:
            body["model"] = self.spec.model
        if json_mode and params.get("json_response_format", True):
            body["response_format"] = {"type": "json_object"}
        body.update(params.get("extra_body", {}) or {})
        resp = requests.post(self.url, json=body, headers=self.headers, timeout=self.spec.timeout_s)
        if resp.status_code == 429:
            raise RetryAfter("chat 429", float(resp.headers.get("retry-after", 10)))
        if resp.status_code >= 500:
            raise RuntimeError(f"chat {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 200:
            raise ProviderError(f"chat {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        message = data["choices"][0]["message"]
        return LLMResult(text=message.get("content") or "", model=data.get("model", self.spec.model),
                         usage=data.get("usage", {}))
