# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build LLM / image clients from config. Add a provider by registering one class here."""

from __future__ import annotations

from typing import Callable, Dict

from ..config import ModelSpec
from .base import ImageClient, LLMClient, ProviderError


def _gemini(spec):
    from .gemini import GeminiClient
    return GeminiClient(spec)


def _openai_compat(spec):
    from .openai_compat import OpenAICompatClient
    return OpenAICompatClient(spec)


def _mock_llm(spec):
    from .mock import MockLLM
    return MockLLM(spec)


def _openai_image(spec):
    from .openai_image import OpenAIImageClient
    return OpenAIImageClient(spec)


def _mock_image(spec):
    from .mock import MockImage
    return MockImage(spec)


LLM_PROVIDERS: Dict[str, Callable[[ModelSpec], LLMClient]] = {
    "gemini": _gemini,
    "openai_compat": _openai_compat,
    "mock": _mock_llm,
}

IMAGE_PROVIDERS: Dict[str, Callable[[ModelSpec], ImageClient]] = {
    "openai_image": _openai_image,
    "mock": _mock_image,
}


def build_llm(spec: ModelSpec) -> LLMClient:
    if spec.provider not in LLM_PROVIDERS:
        raise ProviderError(f"unknown LLM provider {spec.provider!r}; known: {sorted(LLM_PROVIDERS)}")
    return LLM_PROVIDERS[spec.provider](spec)


def build_image(spec: ModelSpec) -> ImageClient:
    if spec.provider not in IMAGE_PROVIDERS:
        raise ProviderError(f"unknown image provider {spec.provider!r}; known: {sorted(IMAGE_PROVIDERS)}")
    return IMAGE_PROVIDERS[spec.provider](spec)
