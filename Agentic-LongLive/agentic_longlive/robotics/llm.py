# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-language client of the robot agent: one OpenAI-compatible chat call per question, JSON object out.

The request is kept exactly as for the paper's robot runs (Qwen3.6-35B-A3B on SGLang, thinking off): a single user
message with the images first (JPEG, longer side capped per call) and the question last, temperature 0. A reply without
a parsable JSON object is retried with twice the token budget and temperature 0.3 (a deterministic model would
otherwise return the same unparsable reply again).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import requests
from PIL import Image

Content = Sequence[Union[str, Path, Image.Image]]

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
# Turns off Qwen3 thinking on SGLang / vLLM chat templates.
DEFAULT_EXTRA_BODY: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def encode_image(im: Union[Path, str, Image.Image], max_side: int = 1024) -> str:
    """RGB JPEG (quality 92) whose longer side is at most ``max_side``, base64-encoded."""
    im = (Image.open(im) if not isinstance(im, Image.Image) else im).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def parse_reply(txt: str) -> Dict[str, Any]:
    """The outermost ``{...}`` of the reply (code fences removed). Raises AttributeError / ValueError if none parses."""
    txt = re.sub(r"```(?:json)?", "", txt)
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))


class AgentLLM:
    """OpenAI-compatible chat endpoint (SGLang, vLLM, or a hosted API).

    ``ask(content, max_tokens, max_side) -> (dict, seconds)`` is the only interface the agent uses, so tests can pass
    any object with that method.
    """

    def __init__(self, base_url: str, model: str = DEFAULT_MODEL, api_key_env: Optional[str] = None,
                 extra_body: Optional[Dict[str, Any]] = None, timeout_s: float = 180.0, attempts: int = 5,
                 retry_wait_s: float = 3.0):
        if not base_url:
            raise ValueError("AgentLLM needs base_url, e.g. http://127.0.0.1:31001/v1")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key_env and os.environ.get(api_key_env):
            self.headers["Authorization"] = "Bearer " + os.environ[api_key_env].strip()
        self.extra_body = DEFAULT_EXTRA_BODY if extra_body is None else extra_body
        self.timeout_s, self.attempts, self.retry_wait_s = timeout_s, attempts, retry_wait_s

    def ask(self, content: Content, max_tokens: int = 1500, max_side: int = 1024) -> Tuple[Dict[str, Any], float]:
        t = time.time()
        parts = []
        for c in content:
            if isinstance(c, (Image.Image, Path)):
                url = "data:image/jpeg;base64," + encode_image(c, max_side)
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": str(c)})
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens, **self.extra_body,
                "messages": [{"role": "user", "content": parts}]}
        err: Optional[Exception] = None
        for k in range(self.attempts):
            try:
                r = requests.post(self.url, data=json.dumps(body), headers=self.headers, timeout=self.timeout_s)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
                return parse_reply(txt), time.time() - t
            except Exception as e:  # noqa: BLE001 -- transport errors and malformed replies are both retried
                err = e
                if isinstance(e, (AttributeError, ValueError)):   # no JSON object in the reply (e.g. cut at max_tokens)
                    body["max_tokens"] = min(body["max_tokens"] * 2, 4000)
                    body["temperature"] = 0.3
                time.sleep(self.retry_wait_s * (k + 1))
        raise RuntimeError(f"agent LLM call failed: {err}")
