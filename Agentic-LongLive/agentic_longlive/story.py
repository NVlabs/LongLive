# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stories and the online prompt protocol.

A story is a session setup (title, style, cast cards) plus an ordered list of shot prompts.
Optional per-shot annotations (planned scene / state) are *hidden*: only evaluation and RL
rewards may read them. The agent never receives a :class:`Story`; it only receives
:class:`AgentView` objects created by :class:`PromptStream`, which reveals prompts one at a time
("commit memory first, then reveal the next prompt").
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXTRA_KEYS = ("voice", "music", "speech", "framing")


@dataclass(frozen=True)
class CastMember:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class StorySetup:
    """Information available before the first shot (like a live session's character cards)."""

    title: str
    style: str = ""
    cast: Tuple[CastMember, ...] = ()
    # Optional session-level direction known before shot 0 (e.g. voice, music, speech, framing) for
    # generators with audio or strict framing rules. Never shot-specific.
    extras: Dict[str, str] = field(default_factory=dict)


@dataclass
class Story:
    id: str
    setup: StorySetup
    prompts: List[str]
    # Hidden evaluation annotations, aligned with prompts (may be empty dicts).
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    # Optional backend-specific prompt variants, e.g. {"h3": [...]} written offline per shot.
    variants: Dict[str, List[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.prompts)

    @staticmethod
    def load(path: str | Path) -> "Story":
        path = Path(path)
        data = json.loads(path.read_text())
        if "shots" in data and data["shots"] and "prompt" in data["shots"][0]:
            return Story._from_unified(data)
        captions = path.with_name(path.stem + ".captions.json")
        if captions.is_file():
            return Story._from_outline(data, json.loads(captions.read_text()))
        raise ValueError(f"unrecognised story format: {path}")

    @staticmethod
    def _from_unified(data: Dict[str, Any]) -> "Story":
        setup = data.get("setup", {})
        cast = tuple(CastMember(c["id"], c.get("name", c["id"]), c["description"]) for c in setup.get("cast", []))
        shots = data["shots"]
        variants: Dict[str, List[str]] = {}
        for shot in shots:
            for key, value in (shot.get("variants") or {}).items():
                variants.setdefault(key, []).append(value)
        return Story(
            id=data["id"],
            setup=StorySetup(title=setup.get("title", data["id"]), style=setup.get("style", ""), cast=cast,
                             extras=dict(setup.get("extras") or {})),
            prompts=[s["prompt"] for s in shots],
            annotations=[s.get("annotation", {}) for s in shots],
            variants=variants,
        )

    @staticmethod
    def _from_outline(outline: Dict[str, Any], captions: Dict[str, Any]) -> "Story":
        """Legacy test-set pair: ``<case>.json`` (outline with planned state) + ``<case>.captions.json``."""
        cast = tuple(CastMember(c["id"], c.get("name", c["id"]), c.get("anchor_caption", c.get("description", "")))
                     for c in outline.get("cast", []))
        prompts = [s["caption"] for s in captions["shots"]]
        annotations = [{k: s.get(k) for k in ("title_en", "scene", "camera", "beat", "state") if k in s}
                       for s in outline["shots"]]
        if len(prompts) != len(annotations):
            raise ValueError("outline and captions disagree on the number of shots")
        return Story(
            id=outline.get("case", "story"),
            setup=StorySetup(title=outline.get("title", ""), style=outline.get("style", ""), cast=cast,
                             extras={k: outline[k] for k in EXTRA_KEYS if outline.get(k)}),
            prompts=prompts,
            annotations=annotations,
        )

    def to_unified(self) -> Dict[str, Any]:
        shots = []
        for i, prompt in enumerate(self.prompts):
            shot: Dict[str, Any] = {"prompt": prompt}
            if i < len(self.annotations) and self.annotations[i]:
                shot["annotation"] = self.annotations[i]
            variants = {k: v[i] for k, v in self.variants.items() if i < len(v)}
            if variants:
                shot["variants"] = variants
            shots.append(shot)
        return {
            "id": self.id,
            "setup": {"title": self.setup.title, "style": self.setup.style,
                      "cast": [{"id": c.id, "name": c.name, "description": c.description} for c in self.setup.cast],
                      **({"extras": dict(self.setup.extras)} if self.setup.extras else {})},
            "shots": shots,
        }


def _digest(items: Tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(hashlib.sha256(item.encode()).digest())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class AgentView:
    """Everything the memory agent may read when processing revealed shot ``latest``."""

    setup: StorySetup
    revealed: Tuple[str, ...]  # prompts 0..latest
    # Only populated in the oracle upper-bound setting.
    next_prompt: Optional[str] = None

    @property
    def latest(self) -> int:
        return len(self.revealed) - 1

    @property
    def prompt(self) -> str:
        return self.revealed[-1]

    def fingerprint(self) -> str:
        """Hash of the visible prompts; logged with every decision for boundary audits."""
        visible = self.revealed + ((self.next_prompt,) if self.next_prompt is not None else ())
        return _digest(visible)


class PromptStream:
    """Reveals one prompt per shot. ``reveal`` is the only way prompts leave the story."""

    def __init__(self, story: Story, max_shots: Optional[int] = None, oracle_next_prompt: bool = False):
        self._prompts = list(story.prompts[: max_shots or len(story.prompts)])
        self._setup = story.setup
        self._revealed: List[str] = []
        self._oracle = oracle_next_prompt
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._prompts)

    @property
    def num_revealed(self) -> int:
        return len(self._revealed)

    def reveal(self) -> Tuple[int, str, AgentView]:
        with self._lock:
            index = len(self._revealed)
            if index >= len(self._prompts):
                raise IndexError("all prompts have been revealed")
            self._revealed.append(self._prompts[index])
            return index, self._prompts[index], self._view_locked()

    def view(self) -> AgentView:
        with self._lock:
            return self._view_locked()

    def _view_locked(self) -> AgentView:
        nxt = None
        if self._oracle and len(self._revealed) < len(self._prompts):
            nxt = self._prompts[len(self._revealed)]
        return AgentView(setup=self._setup, revealed=tuple(self._revealed), next_prompt=nxt)
