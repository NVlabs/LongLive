# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM scene routing (Method, "Scene routing").

M_{i-1} shows the previous shot's location, but the revealed prompt p_i may continue there, move to a new place or
return to an earlier one. When p_i is revealed, the agent's LLM reads it together with the names and descriptions of
the locations memory has drawn and answers with one of them or NEW (decoding constrained to these options):

  keep    the shot is at M_{i-1}'s location: inject the committed memory [G, M_{i-2}, M_{i-1}];
  recall  it returns to another known location: inject [G, M_loc], the newest M drawn there;
  drop    a new location: inject G only.

The router reads only revealed prompts and committed memory (no memory is built after p_i arrives), and it is the only
model call on the generator's critical path. A failed call behaves as ``keep``.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

ROUTE_SYSTEM = """You keep track of locations for a video generator that films a story one shot at a time. For every \
location that earlier shots were filmed in, the memory keeps a picture. You are given the known locations (name and \
description), the location of the previous shot, and the prompt of the next shot.

Decide where the next shot takes place:
- If it takes place in one of the known locations, whether it continues in the previous shot's location or returns to \
an earlier one, answer with the exact name of that location.
- If it takes place in a location that is not in the list, answer NEW.
Judge by the physical setting the camera shows, not by which characters or objects are mentioned."""


def _opening(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


class SceneRouter:
    """Location bank (newest committed M per location) and the LLM routing call.

    ``register`` is called at every shot boundary with the M images of the injected snapshot, so the bank only holds
    memory that existed before the prompt was revealed. ``decisions`` (optional, ``{shot: {"decision", "scene"}}``)
    replays the decisions of an earlier run instead of calling the LLM, e.g. to regenerate a recorded memory stream.
    """

    def __init__(self, url: str, model: str, style: str = "sglang", timeout_s: float = 60.0,
                 decisions: Optional[Dict[int, Dict[str, Any]]] = None):
        self.url = url.rstrip("/")
        self.model = model
        self.style = style  # sglang: EBNF-constrained decoding | vllm: guided choice
        self.timeout_s = timeout_s
        self.decisions = decisions
        self._bank: List[Tuple[int, Dict[str, Any]]] = []  # (shot it was injected at, M entry)
        self._paths: set = set()
        self.descriptions: Dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ bank
    def register(self, shot: int, m_entries, scene_descriptions: Optional[Dict[str, str]] = None) -> None:
        """Add the M images of the snapshot injected at ``shot`` (entries: scene, prompt, watermark, path, caption)."""
        with self._lock:
            for m in m_entries:
                if m.get("scene") and int(m.get("watermark", -1)) >= 0 and m["path"] not in self._paths:
                    self._paths.add(m["path"])
                    self._bank.append((shot, dict(m)))
            for name, text in (scene_descriptions or {}).items():
                self.descriptions[name] = text.split(":", 1)[1].strip() if text.startswith(name + ":") else text

    def bank_at(self, shot: int) -> Dict[str, Dict[str, Any]]:
        """Newest committed M per location that shot ``shot`` may use (built from prompts <= shot - 1)."""
        bank: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for used_at, m in self._bank:
                w = int(m.get("watermark", -1))
                if used_at > shot or w > shot - 1:
                    continue
                old = bank.get(m["scene"])
                if old is None or w >= int(old["watermark"]):
                    bank[m["scene"]] = m
        return bank

    # -------------------------------------------------------------- decision
    def route(self, shot: int, prompt: str, current: Dict[str, Any]) -> Dict[str, Any]:
        """Decide for the revealed ``prompt``; ``current`` is the newest injected M (scene, prompt, watermark, ...).
        Returns {"decision": keep|recall|drop, "entry": recalled M entry or None, plus call details}."""
        bank = self.bank_at(shot)
        if self.decisions is not None:
            info = dict(self.decisions.get(shot) or {"decision": "keep"})
            entry = bank.get(info.get("scene")) if info.get("decision") == "recall" else None
            if info.get("decision") == "recall" and entry is None:
                info.update(decision="keep", note="recalled location not in the bank")
            return {**info, "entry": entry, "replayed": True}
        scene = current.get("scene")
        if not scene:
            return {"decision": "keep", "entry": None, "choice": None, "note": "no location for M_{i-1}"}
        bank.setdefault(scene, {"scene": scene, "prompt": current.get("prompt", ""),
                                "watermark": int(current.get("watermark", -1)), "path": current.get("path", ""),
                                "caption": current.get("caption", "")})
        try:
            choice, info = self._ask(prompt, scene, bank)
        except Exception as exc:  # noqa: BLE001 -- a failed call behaves as without routing
            choice, info = scene, {"choice": None, "error": repr(exc)[:200]}
        if choice == "NEW":
            info.update(decision="drop", entry=None)
        elif choice in bank and choice != scene:
            info.update(decision="recall", entry=bank[choice], scene=choice)
        else:
            info.update(decision="keep", entry=None)
        return info

    def _ask(self, prompt: str, current: str, bank: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        lines = [f"- {name}: {self.descriptions.get(name) or _opening(e.get('prompt', ''))}"
                 for name, e in sorted(bank.items(), key=lambda kv: int(kv[1]["watermark"]))]
        user = ("Known locations:\n" + "\n".join(lines) + f"\n\nPrevious shot location: {current}\n\n"
                f"Next shot prompt:\n{prompt}\n\nWhere does the next shot take place? Answer with a known location name or NEW.")
        choices = list(bank) + ["NEW"]
        body: Dict[str, Any] = {"model": self.model, "temperature": 0.0, "max_tokens": 40,
                                "messages": [{"role": "system", "content": ROUTE_SYSTEM},
                                             {"role": "user", "content": user}],
                                "chat_template_kwargs": {"enable_thinking": False}}
        if self.style == "vllm":
            body["structured_outputs"] = {"choice": choices}
        else:
            literal = lambda c: '"' + c.replace("\\", "\\\\").replace('"', '\\"') + '"'  # noqa: E731
            body["ebnf"] = "root ::= " + " | ".join(literal(c) for c in choices)
        request = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        t0 = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            out = json.loads(response.read().decode())
        choice = (((out.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return choice, {"choice": choice, "latency_s": round(time.monotonic() - t0, 4), "n_known": len(bank),
                        "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens")}


def load_decisions(path, story_id: str) -> Dict[int, Dict[str, Any]]:
    """Recorded decisions: a JSON file {shot: {...}} or a directory with <story>.json."""
    from pathlib import Path

    path = Path(path)
    if path.is_dir():
        path = path / f"{story_id}.json"
    return {int(k): v for k, v in json.loads(path.read_text()).items()}
