# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-independent interfaces for language/vision models and image generators.

Every agent component talks to models only through :class:`LLMClient` (text or text+images in,
text/JSON out) and :class:`ImageClient` (prompt + optional reference images in, one image out).
Providers are chosen in the config (``agent.planner``, ``agent.checker``, ``agent.image``), so
API models (Gemini, GPT Image) and self-hosted models (vLLM/SGLang servers, diffusers pipelines)
can be swapped without touching the agent.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..config import ModelSpec
from .throttle import Throttle

# A user message is a list of text segments and image paths (rendered in order).
Content = Sequence[Union[str, Path]]


class ImageRefused(Exception):
    """The image model refused the request (e.g. safety filter). Callers keep the previous image."""


class ProviderError(Exception):
    """Non-retryable provider error (bad request, missing credentials)."""


@dataclass
class LLMResult:
    text: str
    data: Optional[Dict[str, Any]] = None  # parsed JSON when json_mode was requested
    model: str = ""
    latency_s: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResult:
    png: bytes
    model: str = ""
    latency_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def parse_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def parse_size(size: str) -> Tuple[int, int]:
    w, h = size.lower().split("x")
    return int(w), int(h)


class LLMClient(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.throttle = Throttle(spec.max_concurrency, spec.min_interval_s)

    @property
    def name(self) -> str:
        return f"{self.spec.provider}:{self.spec.model}"

    def chat(self, system: str, content: Content, json_mode: bool = False,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> LLMResult:
        params = self.spec.params or {}
        temperature = params.get("temperature", 0.2) if temperature is None else temperature
        max_tokens = params.get("max_tokens", 4096) if max_tokens is None else max_tokens
        last: Optional[Exception] = None
        for attempt in range(self.spec.max_retries + 1):
            start = time.monotonic()
            try:
                with self.throttle:
                    result = self._chat(system, content, json_mode, temperature, max_tokens)
                result.latency_s = time.monotonic() - start
                if json_mode and result.data is None:
                    result.data = parse_json(result.text)
                return result
            except (json.JSONDecodeError, KeyError, RuntimeError, OSError) as exc:  # malformed or transient
                last = exc
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
        raise RuntimeError(f"{self.name} failed after retries: {last}")

    @abstractmethod
    def _chat(self, system: str, content: Content, json_mode: bool,
              temperature: float, max_tokens: int) -> LLMResult:
        ...


class ImageClient(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.throttle = Throttle(spec.max_concurrency, spec.min_interval_s)

    @property
    def name(self) -> str:
        return f"{self.spec.provider}:{self.spec.model}"

    def generate(self, prompt: str, refs: Sequence[Union[str, Path]] = (), size: str = "1536x1024",
                 priority: int = 0) -> ImageResult:
        """``priority``: lower is served first when calls queue (0 critical path, 1 background)."""
        last: Optional[Exception] = None
        for attempt in range(self.spec.max_retries + 1):
            start = time.monotonic()
            try:
                with self.throttle.slot(priority):
                    result = self._generate(prompt, [Path(r) for r in refs], size)
                result.latency_s = time.monotonic() - start
                return result
            except ImageRefused:
                raise
            except (RuntimeError, OSError, KeyError) as exc:
                last = exc
                wait = getattr(exc, "retry_after", None) or min(60.0, 5.0 * (attempt + 1))
                time.sleep(wait)
        raise RuntimeError(f"{self.name} failed after retries: {last}")

    @abstractmethod
    def _generate(self, prompt: str, refs: List[Path], size: str) -> ImageResult:
        ...
