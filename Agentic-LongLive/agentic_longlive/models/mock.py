# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline models for tests and dry runs (no network, no GPU)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .base import Content, ImageClient, ImageResult, LLMClient, LLMResult, parse_size

_PLACES = ("market", "kitchen", "balcony", "dining", "living", "garage", "workshop", "lab", "street", "garden",
           "office", "bathroom", "bedroom", "studio", "park", "beach", "station", "corridor", "hallway", "gym",
           "fab", "cleanroom", "desk", "room")
_STATE_WORDS = ("stain", "splash", "torn", "wet", "muddy", "broken", "cracked", "assembled", "installed", "black",
                "white", "red", "green", "blue")


def _section(text: str, title: str) -> str:
    match = re.search(rf"### {re.escape(title)}[^\n]*\n(.*?)(?=\n### |\Z)", text, re.S)
    return match.group(1).strip() if match else ""


class MockLLM(LLMClient):
    """Heuristic planner/checker that follows the agent's JSON contracts."""

    def _chat(self, system: str, content: Content, json_mode: bool, temperature: float, max_tokens: int) -> LLMResult:
        task = system.split("\n", 1)[0].replace("TASK:", "").strip()
        text = "\n".join(str(c) for c in content if not isinstance(c, Path))
        if task == "memory_plan":
            data = self._plan(text)
        elif task == "image_check":
            digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            score = 6 + digest % 4
            strict = float((self.spec.params or {}).get("pass_score", 6))  # >9: a checker that never passes
            data = {"pass": score >= strict, "score": score, "state_ok": True,
                    "issues": [] if score >= 8 else ["sharpen identity"]}
        elif task == "cast_split":
            # identity = the card up to "wearing"; the rest is the initial clothing
            cards = json.loads(_section(text, "CAST CARDS") or "[]")
            cast = []
            for card in cards:
                identity, _, worn = card["card"].partition(" wearing ")
                cast.append({"id": card["id"], "identity": identity.strip().rstrip(","),
                             "facts": {"clothing": worn.strip().rstrip(".")} if worn else {}})
            data = {"cast": cast}
        elif task == "h3_prompt":
            data = {"prompt": _section(text, "CAPTION")[:2000]}
        else:
            data = {"text": text[:200]}
        return LLMResult(text=json.dumps(data), data=data, model="mock")

    @staticmethod
    def _plan(text: str) -> Dict[str, Any]:
        table = json.loads(_section(text, "ENTITY TABLE") or "[]")
        prompt = _section(text, "NEW PROMPT").split("\n---\n")[-1]  # newest of possibly several prompts
        lower = prompt.lower()
        place = next((p for p in _PLACES if p in lower), None)
        scene = f"{place}_scene" if place else "scene_" + hashlib.sha256(prompt[:80].encode()).hexdigest()[:6]
        entities: List[Dict[str, Any]] = []
        characters = [row for row in table if row["kind"] == "character"]
        for row in characters:
            present = row["name"].split()[0].lower() in lower or row.get("protagonist", False)
            facts = []
            for word in _STATE_WORDS:
                if word in lower:
                    facts.append({"attr": "condition", "value": word, "persistent": True, "source": "explicit"})
                    break
            if "holding" in lower or "holds" in lower:
                facts.append({"attr": "holding", "value": "an item", "persistent": False, "source": "explicit"})
            entities.append({"id": row["id"], "kind": "character", "present": present, "facts": facts})
        if not any(row["id"] == scene for row in table):
            entities.append({"id": scene, "kind": "scene", "name": scene.replace("_", " "),
                             "description": f"the {scene.replace('_', ' ')}", "present": True, "facts": []})
        cast = [row["id"] for row in characters]
        return {
            "entities": entities,
            "scene": scene,
            "active": cast,
            "retire": [],
            "image_prompts": {},
            "memory": {"g_members": cast[:3], "m_members": cast[:4], "g_prompt": "The protagonists together.",
                       "m_prompt": f"The {scene.replace('_', ' ')} with the protagonists.",
                       "g_caption": "The protagonists in their current appearance.",
                       "m_caption": f"The {scene.replace('_', ' ')}."},
        }


class MockImage(ImageClient):
    """Solid-colour PNG whose colour depends on the prompt and references; text is drawn on top."""

    def _generate(self, prompt: str, refs: List[Path], size: str) -> ImageResult:
        import time

        from PIL import Image, ImageDraw

        params = self.spec.params or {}
        time.sleep(float(params.get("latency_s", 0.0)))  # simulate a slow image API
        if params.get("slow_if") and params["slow_if"] in prompt:  # e.g. one request stuck in API retries
            time.sleep(float(params.get("slow_s", 1.0)))
        width, height = parse_size(size)
        width, height = max(64, width // 4), max(36, height // 4)
        digest = hashlib.sha256((prompt + "|".join(map(str, refs))).encode()).digest()
        image = Image.new("RGB", (width, height), tuple(int(b) for b in digest[:3]))
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(re.findall(r".{1,40}", prompt[:200])):
            draw.text((6, 6 + 12 * i), line, fill=(255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return ImageResult(png=buffer.getvalue(), model="mock", meta={"refs": len(refs)})
