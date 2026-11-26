# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score candidate planner outputs in hindsight: apply to the stored state, build the memory view, judge, reward.

Judges are plain callables ``call(system, user) -> parsed JSON or None``:

* :func:`openai_judge`: a self-hosted OpenAI-compatible server (training judge, Qwen3.6-35B-A3B-FP8 on SGLang);
* :func:`gemini_judge`: Gemini through :mod:`agentic_longlive.models` (evaluation judge, a different model family).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from . import judge as J
from .common import Endpoints, chat, parse_json
from .judge import JudgeCall
from .reward import RewardConfig, reward
from .textenv import apply_output, memory_view

JUDGE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_JUDGE_URLS = "http://127.0.0.1:31000/v1,http://127.0.0.1:31001/v1"
GEMINI_MODEL = "gemini-3.8-flash"
# Name of the environment variable that holds the path of a Google service-account JSON (Vertex AI).
GEMINI_CREDENTIALS_ENV = "GEMINI_SERVICE_ACCOUNT"


def judge_urls_from_env(default: str = DEFAULT_JUDGE_URLS) -> str:
    """Judge endpoints from ``RL_JUDGE_URLS`` (comma-separated ``.../v1`` base URLs)."""
    return os.environ.get("RL_JUDGE_URLS", default)


def openai_judge(urls: str, model: str = JUDGE_MODEL) -> JudgeCall:
    eps = Endpoints(urls)

    def call(system: str, user: str):
        try:
            info = chat(eps, model, system, user, temperature=0.0, max_tokens=3000, timeout=600)
        except RuntimeError:
            return None
        return parse_json(info["texts"][0])
    return call


def gemini_client(model: str = GEMINI_MODEL, max_tokens: int = 4096, credentials_env: str = GEMINI_CREDENTIALS_ENV,
                  max_concurrency: int = 48, **params):
    """A Gemini LLM client (Vertex AI, temperature 0) from the agent's model registry."""
    from ..config import ModelSpec
    from ..models.registry import build_llm

    spec = ModelSpec(provider="gemini", model=model, credentials_env=credentials_env,
                     params={"backend": "vertex", "location": "global", "temperature": 0.0, "max_tokens": max_tokens,
                             **params}, max_concurrency=max_concurrency)
    return build_llm(spec)


def gemini_judge(model: str = GEMINI_MODEL, max_tokens: int = 4096,
                 credentials_env: str = GEMINI_CREDENTIALS_ENV) -> JudgeCall:
    client = gemini_client(model, max_tokens, credentials_env)

    def call(system: str, user: str):
        try:
            r = client.chat(system, [user], json_mode=True, temperature=0.0)
            return r.data if isinstance(r.data, dict) else parse_json(r.text)
        except Exception:  # noqa: BLE001  (a failed reply counts as a judge failure, never stops a batch)
            return None
    return call


class Scorer:
    """Thread-pooled hindsight scoring. ``image_layer`` is required for reward version 4 with ``image=True``."""

    def __init__(self, call: JudgeCall, cfg: Optional[RewardConfig] = None, workers: int = 64, tokenizer=None,
                 image_layer=None):
        self.call = call
        self.image_layer = image_layer
        self.cfg = cfg or RewardConfig()
        self.pool = ThreadPoolExecutor(workers)
        self.tok = tokenizer

    def n_tokens(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=False)["input_ids"])
        return max(1, len(text) // 4)

    def score_one(self, state: Dict[str, Any], output: str, n_tokens: Optional[int] = None,
                  cfg: Optional[RewardConfig] = None) -> Dict[str, Any]:
        """Reward parts of one candidate output for a decision state (``store_before``, ``shot``, ``targets``)."""
        cfg = cfg or self.cfg
        targets = state["targets"]
        try:
            store, plan, g, m, ok = apply_output(state["store_before"], output, state["shot"])
            mem = memory_view(store, plan, g, m)
        except Exception as exc:  # noqa: BLE001
            return {"total": cfg.format_fail, "format": 0.0, "error": repr(exc)[:200]}
        ntok = n_tokens if n_tokens is not None else self.n_tokens(output)
        if not ok:
            return {"total": cfg.format_fail, "format": 0.0, "n_tokens": ntok}
        j = J.judge(self.call, targets, mem)
        img = None
        if j is not None and cfg.version >= 4 and cfg.image:
            if self.image_layer is None:
                raise RuntimeError("reward version 4 with image=True needs an image layer (image_urls)")
            img = self.image_layer.audit(targets, store, plan, g, m)
        r = reward(targets, mem, j, ntok, ok, cfg, img=img)
        r["n_tokens"] = ntok
        r["judge"] = j
        if img is not None:
            r["img_audit"] = img
        return r

    def score_many(self, jobs: List[tuple]) -> List[Dict[str, Any]]:
        """jobs: (state, output[, n_tokens]) tuples."""
        futs = [self.pool.submit(self.score_one, *job) for job in jobs]
        return [f.result() for f in futs]
