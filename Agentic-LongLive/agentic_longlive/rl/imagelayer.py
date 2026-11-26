# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image term of reward version 4: render the G / M memory images that the deployed painter would compose from a
candidate memory and audit them against the hindsight targets of the next shot.

Rendering mirrors ``agent.painter`` in its text-only mode: ``prompts.compose_spec`` (role, the plan's G / M composition
prompt, member descriptions "name: identity. Current state: ...", and for M the location first) with FLUX.2-klein-4B,
a fixed seed and 768x432. The audit VLM (Qwen3.6-35B-A3B during training) sees both images and, for every recurring
entity of the NEXT shot, says whether it is recognisably depicted in G and in M and whether each persistent fact the
next shot needs is drawn correctly. ``reward.image_utility`` turns this into the video-calibrated utility.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.error
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent import prompts
from .common import Endpoints, http_post, parse_json

AUDIT_SYSTEM = """TASK: memory_image_audit
You check the memory images of a video-generation agent after the fact, using hidden knowledge of the NEXT shot. The
agent drew them without seeing the future; you only check what the images show. Be literal and consistent. Output one
JSON object and nothing else."""

AUDIT_USER = """After this text come {which}.
G is the cast image (main characters / objects), M is the current-scene image (location plus entities).

### ENTITIES OF THE NEXT SHOT (id: kind, name, unchanging look)
{ents}

### PERSISTENT STATE THE NEXT SHOT NEEDS
{facts}

Tasks:
1. "entities": for every entity id above, {{"G": bool, "M": bool}}: is that entity clearly depicted and recognisable
   (its unchanging look matches) in that image? A missing image counts as false. Similar-looking generic people or
   props that do not match the description count as false.
2. "facts": for every fact id above: "S" if an image that depicts the entity clearly shows the stated value, "C" if it
   clearly shows a different value, "M" if not visible or the entity is not depicted.
3. "quality": {{"G": 0-10, "M": 0-10}}: how clearly the depicted entities can be told apart and seen (crowded, tiny,
   merged or malformed figures score low; a missing image scores 0).
Return {{"entities": {{"e": {{"G": false, "M": true}}}}, "facts": {{"F1": "S"}}, "quality": {{"G": 7, "M": 6}}}}"""

SIZE = "768x432"
ROLES = ("G", "M")


class LRU:
    def __init__(self, cap: int = 4000):
        self.cap, self.d, self.lock = cap, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)


class Renderer:
    """FLUX.2-klein text-to-image over several OpenAI-image-API servers (least busy first), cached by prompt.
    ``save_dir``: keep every ``save_every``-th render (PNG + prompt) for inspection."""

    def __init__(self, urls: str, model: str = "black-forest-labs/FLUX.2-klein-4B", size: str = SIZE, steps: int = 4,
                 seed: int = 0, per_server: int = 2, save_dir: Optional[str] = None, save_every: int = 40):
        self.urls = [u.strip().rstrip("/") for u in urls.split(",") if u.strip()]
        if not self.urls:
            raise ValueError("no image server URLs given")
        self.model, self.size, self.steps, self.seed = model, size, steps, seed
        self.busy = [0] * len(self.urls)
        self.cv = threading.Condition()
        self.per_server = per_server
        self.cache = LRU()
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every, self.n = save_every, 0
        self.stats = {"renders": 0, "hits": 0, "fail": 0, "sec": 0.0}

    def _acquire(self) -> int:
        with self.cv:
            while True:
                k = min(range(len(self.urls)), key=lambda i: self.busy[i])
                if self.busy[k] < self.per_server:
                    self.busy[k] += 1
                    return k
                self.cv.wait(1.0)

    def _release(self, k: int) -> None:
        with self.cv:
            self.busy[k] -= 1
            self.cv.notify_all()

    def render(self, prompt: str) -> Optional[bytes]:
        key = hashlib.sha1(f"{prompt}|{self.size}|{self.seed}|{self.steps}".encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["hits"] += 1
            return hit
        body = {"model": self.model, "prompt": prompt, "size": self.size, "n": 1, "response_format": "b64_json",
                "num_inference_steps": self.steps, "seed": self.seed}
        for attempt in range(3):
            k = self._acquire()
            t0 = time.monotonic()
            try:
                out = http_post(self.urls[k] + "/images/generations", body, timeout=300)
                png = base64.b64decode(out["data"][0]["b64_json"])
            except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError, TimeoutError):
                png = None
            finally:
                self._release(k)
            if png:
                self.stats["renders"] += 1
                self.stats["sec"] += time.monotonic() - t0
                self.cache.put(key, png)
                self.n += 1
                if self.save_dir and self.n % self.save_every == 0:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    (self.save_dir / f"{key[:12]}.png").write_bytes(png)
                    (self.save_dir / f"{key[:12]}.txt").write_text(prompt)
                return png
            time.sleep(2 * (attempt + 1))
        self.stats["fail"] += 1
        return None


def compose_specs(store, plan, g: List[str], m: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """The G / M generation prompts of the deployed painter (text-only: no entity reference images)."""
    def desc(ids):
        out = []
        for i in ids:
            e = store.entities.get(i)
            if e is not None and e.kind != "scene":
                out.append(e.describe(store.split))
        return out

    g_names = desc(g)
    g_spec = prompts.compose_spec("G", plan.g_prompt or "The main characters together.", g_names) if g_names else None
    m_names = desc(m)
    scene = store.entities.get(store.scene) if store.scene else None
    if scene is not None:
        m_names.insert(0, "Location: " + scene.describe(store.split))
    m_spec = prompts.compose_spec("M", plan.m_prompt or "The current scene.", m_names) if m_names else None
    return g_spec, m_spec


def audit_prompt(targets: Dict[str, Any], rendered: Dict[str, bool]) -> Tuple[List[str], List[Dict[str, Any]], str]:
    """(next-shot entities, visual needed facts of those entities, audit user text) for one decision."""
    ents = [x["entity"] for x in targets["E"]]
    canon = targets["canon"]
    vis = [f for f in targets["F"] if f["visual"] and f["entity"] in ents]
    ent_text = "\n".join(f'- {e}: {canon.get(e, {}).get("kind", "object")}, "{canon.get(e, {}).get("name", e)}", '
                         f'{canon.get(e, {}).get("identity", "")[:200]}' for e in ents)
    fact_text = "\n".join(f'{f["id"]}: {canon.get(f["entity"], {}).get("name", f["entity"])} [{f["entity"]}].'
                          f'{f["attr"]} = "{f["value"]}"' for f in vis) or "(none)"
    which = " and ".join(f"IMAGE {r}" for r in ROLES if rendered[r]) + \
        ("" if all(rendered.values()) else " (the other image does not exist)")
    return ents, vis, AUDIT_USER.format(which=which, ents=ent_text, facts=fact_text)


def _score(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_audit(out: Dict[str, Any], ents: List[str], vis: List[Dict[str, Any]],
                rendered: Dict[str, bool]) -> Dict[str, Any]:
    """Normalise an audit reply: an entity counts as depicted only in an image that exists; labels default to "M"."""
    res_e = {}
    for e in ents:
        x = (out.get("entities") or {}).get(e) or {}
        res_e[e] = {r: bool(x.get(r)) and rendered[r] if isinstance(x, dict) else False for r in ROLES}
    res_f = {}
    for f in vis:
        b = str((out.get("facts") or {}).get(f["id"], "M") or "M").strip().upper()[:1]
        res_f[f["id"]] = b if b in ("S", "C", "M") else "M"
    q = out.get("quality") if isinstance(out.get("quality"), dict) else {}
    return {"entities": res_e, "facts": res_f, "quality": {r: _score(q.get(r)) for r in ROLES}}


def vlm_call(eps: Endpoints, model: str, system: str, text: str, images: List[Tuple[str, bytes]],
             max_tokens: int = 1500, timeout: float = 600) -> Optional[Dict[str, Any]]:
    """One OpenAI-compatible vision chat call (text, then each image preceded by its label); parsed JSON or None."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for label, png in images:
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
            "temperature": 0.0, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(3):
        try:
            out = http_post(eps.next() + "/chat/completions", body, timeout=timeout)
            data = parse_json(out["choices"][0]["message"].get("content") or "")
            if data is not None:
                return data
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError, TimeoutError):
            time.sleep(3 * (attempt + 1))
    return None


class ImageLayer:
    """Renders G / M for a candidate memory and audits them with a self-hosted VLM (reward version 4)."""

    def __init__(self, image_urls: str, vlm_urls: str, vlm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
                 image_model: str = "black-forest-labs/FLUX.2-klein-4B", save_dir: Optional[str] = None):
        self.renderer = Renderer(image_urls, model=image_model, save_dir=save_dir)
        self.eps = Endpoints(vlm_urls)
        self.model = vlm_model

    def audit(self, targets: Dict[str, Any], store, plan, g: List[str], m: List[str]) -> Optional[Dict[str, Any]]:
        """{"entities": {canon id: {"G": bool, "M": bool}}, "facts": {fid: S/C/M}, "quality": {...},
        "rendered": {...}}, or None when the audit failed."""
        if not targets["E"]:
            return {"entities": {}, "facts": {}, "quality": {}, "rendered": {}}
        g_spec, m_spec = compose_specs(store, plan, g, m)
        imgs: List[Tuple[str, bytes]] = []
        rendered = {}
        for role, spec in zip(ROLES, (g_spec, m_spec)):
            png = self.renderer.render(spec) if spec else None
            rendered[role] = png is not None
            if png:
                imgs.append((f"IMAGE {role}:", png))
        ents, vis, text = audit_prompt(targets, rendered)
        if not imgs:
            return {"entities": {e: {"G": False, "M": False} for e in ents}, "facts": {}, "quality": {"G": 0, "M": 0},
                    "rendered": rendered}
        out = vlm_call(self.eps, self.model, AUDIT_SYSTEM, text, imgs)
        if not isinstance(out, dict):
            return None
        return {**parse_audit(out, ents, vis, rendered), "rendered": rendered}
