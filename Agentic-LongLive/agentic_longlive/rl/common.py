# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: OpenAI-compatible chat over HTTP, JSON parsing, JSONL IO, round-robin endpoints.

All RL services (teacher / judge LLM, LoRA rollout server, image servers) are OpenAI-compatible HTTP endpoints, e.g.
SGLang servers. Endpoints are given as comma-separated base URLs ending in ``/v1``.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

_LOCK = threading.Lock()


def http_post(url: str, payload: Dict[str, Any], timeout: float = 900.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a model reply (code fences and surrounding prose tolerated), or None."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class Endpoints:
    """Round-robin over several OpenAI-compatible base URLs (``.../v1``)."""

    def __init__(self, urls: str | List[str]):
        self.urls = [u.strip().rstrip("/") for u in (urls.split(",") if isinstance(urls, str) else urls) if u.strip()]
        if not self.urls:
            raise ValueError("no endpoint URLs given")
        self._cycle = itertools.cycle(self.urls)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            return next(self._cycle)


def chat(endpoints: Endpoints | str, model: str, system: str, user: str, *, temperature: float = 0.2,
         max_tokens: int = 4096, n: int = 1, json_mode: bool = True, thinking: bool = False, top_p: float = 1.0,
         seed: Optional[int] = None, retries: int = 3, timeout: float = 900.0) -> Dict[str, Any]:
    """One chat request (``n`` samples); returns {"texts", "finish", "completion_tokens", "prompt_tokens", "latency",
    "url"}. Qwen3 thinking is off (``chat_template_kwargs.enable_thinking``), as in deployment."""
    eps = endpoints if isinstance(endpoints, Endpoints) else Endpoints(endpoints)
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens, "n": n, "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if seed is not None:
        body["seed"] = seed
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        url = eps.next()
        start = time.monotonic()
        try:
            out = http_post(url + "/chat/completions", body, timeout=timeout)
            choices = out["choices"]
            texts = [c["message"].get("content") or "" for c in choices]
            usage = out.get("usage") or {}
            return {"texts": texts, "finish": [c.get("finish_reason") for c in choices],
                    "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
                    "latency": time.monotonic() - start, "url": url}
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            last = exc
            time.sleep(min(30.0, 3.0 * (attempt + 1)))
    raise RuntimeError(f"chat failed after retries: {last}")


class LLMAdapter:
    """Duck-typed stand-in for :class:`agentic_longlive.models.base.LLMClient` (``chat`` returning an object with
    ``data``, ``text``, ``model`` and ``latency_s``), backed by :func:`chat`. Records token counts and latency of
    every call in ``calls`` so rollouts can log them per decision."""

    def __init__(self, endpoints: Endpoints | str, model: str, temperature: float = 0.2, max_tokens: int = 4096,
                 json_retries: int = 2):
        self.eps = endpoints if isinstance(endpoints, Endpoints) else Endpoints(endpoints)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_retries = json_retries
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return f"openai_compat:{self.model}"

    def chat(self, system: str, content, json_mode: bool = False, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None):
        user = content[0] if isinstance(content, (list, tuple)) else content
        temp = self.temperature if temperature is None else temperature
        data, text, total = None, "", 0.0
        info: Dict[str, Any] = {}
        for _ in range(self.json_retries + 1):
            info = chat(self.eps, self.model, system, str(user), temperature=temp,
                        max_tokens=max_tokens or self.max_tokens, json_mode=json_mode)
            text = info["texts"][0]
            total += info["latency"]
            data = parse_json(text) if json_mode else None
            if not json_mode or data is not None:
                break
        self.calls.append({"latency": total, "completion_tokens": info.get("completion_tokens"),
                           "prompt_tokens": info.get("prompt_tokens"), "parsed": data is not None or not json_mode})
        return SimpleNamespace(data=data if data is not None else ({} if json_mode else None), text=text,
                               model=self.model, latency_s=total, parsed=data is not None)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Rows of a JSONL file ([] if it does not exist); a partial last line (still being written) is skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(row: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(msg: str, path: Optional[str | Path] = None) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(path, "a") as fh:
            fh.write(line + "\n")
