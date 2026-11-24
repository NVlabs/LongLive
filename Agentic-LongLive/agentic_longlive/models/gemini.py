# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemini via Vertex AI (service-account credentials) or the Gemini API (API key)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from ..config import ModelSpec
from ._http import b64, credentials_file, image_bytes, secret
from .base import Content, LLMClient, LLMResult, ProviderError
from .throttle import RetryAfter

_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class _VertexToken:
    def __init__(self, path: Path):
        try:
            from google.oauth2 import service_account
            import google.auth.transport.requests as gtr
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError("Vertex AI needs `pip install google-auth`") from exc
        self._creds = service_account.Credentials.from_service_account_file(str(path), scopes=[_SCOPE])
        self._request = gtr.Request()
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if not self._creds.valid or self._creds.expiry is None or self._creds.expiry.timestamp() - time.time() < 300:
                self._creds.refresh(self._request)
            return self._creds.token


class GeminiClient(LLMClient):
    """params: backend (vertex|studio), project, location, max_image_side, thinking_budget."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        params = spec.params or {}
        self.backend = params.get("backend", "vertex")
        if self.backend == "vertex":
            self._token = _VertexToken(credentials_file(spec))
            project = params.get("project") or self._project_from_file(credentials_file(spec))
            location = params.get("location", "global")
            host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            self.url = (f"https://{host}/v1/projects/{project}/locations/{location}"
                        f"/publishers/google/models/{spec.model}:generateContent")
        elif self.backend == "studio":
            self._key = secret(spec)
            base = spec.base_url or "https://generativelanguage.googleapis.com/v1beta"
            self.url = f"{base}/models/{spec.model}:generateContent"
        else:
            raise ProviderError(f"unknown gemini backend {self.backend}")

    @staticmethod
    def _project_from_file(path: Path) -> str:
        import json

        return json.loads(path.read_text())["project_id"]

    def _chat(self, system: str, content: Content, json_mode: bool, temperature: float, max_tokens: int) -> LLMResult:
        params = self.spec.params or {}
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, Path):
                parts.append({"inline_data": {"mime_type": "image/png",
                                              "data": b64(image_bytes(item, params.get("max_image_side", 1024)))}})
            else:
                parts.append({"text": str(item)})
        generation: Dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            generation["responseMimeType"] = "application/json"
        if "thinking_budget" in params:
            generation["thinkingConfig"] = {"thinkingBudget": int(params["thinking_budget"])}
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}], "generationConfig": generation}
        headers = {"Content-Type": "application/json"}
        if self.backend == "vertex":
            headers["Authorization"] = f"Bearer {self._token.get()}"
        else:
            headers["x-goog-api-key"] = self._key
        resp = requests.post(self.url, json=body, headers=headers, timeout=self.spec.timeout_s)
        if resp.status_code == 429:
            raise RetryAfter("gemini 429", 20.0)
        if resp.status_code >= 500:
            raise RuntimeError(f"gemini {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 200:
            raise ProviderError(f"gemini {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            candidate = data["candidates"][0]
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"gemini returned no text: {str(data)[:300]}") from exc
        return LLMResult(text=text, model=data.get("modelVersion", self.spec.model), usage=data.get("usageMetadata", {}))
