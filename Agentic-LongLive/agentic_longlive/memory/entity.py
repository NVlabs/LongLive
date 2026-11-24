# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer 1 of the memory: entities (characters, objects, scenes) with text state and images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

KINDS = ("character", "object", "scene")
SOURCES = ("explicit", "inferred")


@dataclass
class Fact:
    """One attribute of an entity, with the shot it came from and whether it is persistent.

    Persistent facts (identity, clothing, stains, damage, assembly stage) are drawn into memory
    images. Transient facts (pose, action, hand-held items, expression) are kept as text only:
    the next prompt decides them, so memory never commits to them.
    """

    attr: str
    value: str
    persistent: bool = True
    source: str = "explicit"
    shot: int = -1


@dataclass
class ImageVersion:
    version: int
    views: Dict[str, str]  # view name -> image path
    shot: int
    signature: str
    prompt: str = ""
    check: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, str] = field(default_factory=dict)  # visual state this image was drawn with

    @property
    def primary(self) -> str:
        return next(iter(self.views.values()))


@dataclass
class Entity:
    id: str
    kind: str
    name: str
    description: str
    protagonist: bool = False
    facts: Dict[str, Fact] = field(default_factory=dict)
    images: List[ImageVersion] = field(default_factory=list)
    created_shot: int = -1
    last_seen_shot: int = -1
    seen_count: int = 0
    status: str = "active"  # active | dormant
    history: List[Dict[str, Any]] = field(default_factory=list)

    def visual_state(self, split: bool = True) -> Dict[str, str]:
        """Facts that images must show. Without the persistent/transient split, everything is drawn."""
        return {a: f.value for a, f in sorted(self.facts.items()) if f.persistent or not split}

    def transient_state(self) -> Dict[str, str]:
        return {a: f.value for a, f in sorted(self.facts.items()) if not f.persistent}

    def signature(self, split: bool = True) -> str:
        payload = json.dumps([self.kind, self.description, self.visual_state(split)], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def needs_image(self, split: bool = True) -> bool:
        return not self.images or self.images[-1].signature != self.signature(split)

    @property
    def anchor(self) -> Optional[ImageVersion]:
        return self.images[0] if self.images else None

    @property
    def latest(self) -> Optional[ImageVersion]:
        return self.images[-1] if self.images else None

    def describe(self, split: bool = True) -> str:
        state = self.visual_state(split)
        text = f"{self.name}: {self.description}"
        if state:
            text += " Current state: " + "; ".join(f"{k}: {v}" for k, v in state.items()) + "."
        return text

    def state_changes(self, state: Dict[str, str], images: List[Optional["ImageVersion"]]) -> List[Tuple[str, List[str], Optional[str]]]:
        """How ``state`` differs from what the given (reference) images show, as (attribute, old values, new value):
        new value None = the attribute no longer applies, no old values = newly added. Image models copy the state of
        their reference images, so a changed state must be spelled out for the generator and the checker. Images
        without a recorded state are skipped."""
        drawn = [im.state for im in images if im is not None and im.state]
        if not drawn:
            return []
        changes: List[Tuple[str, List[str], Optional[str]]] = []
        for attr, value in state.items():
            old = list(dict.fromkeys(s[attr] for s in drawn if attr in s and s[attr] != value))
            if old or not any(attr in s for s in drawn):
                changes.append((attr, old, value))
        gone = list(dict.fromkeys(a for s in drawn for a in s if a not in state))
        changes += [(attr, list(dict.fromkeys(s[attr] for s in drawn if attr in s)), None) for attr in gone]
        return changes

    def table_row(self) -> Dict[str, Any]:
        """Compact row shown to the planner."""
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "protagonist": self.protagonist,
            "status": self.status,
            "last_seen_shot": self.last_seen_shot,
            "seen_count": self.seen_count,
            "facts": [
                {"attr": f.attr, "value": f.value, "persistent": f.persistent, "source": f.source, "shot": f.shot}
                for f in self.facts.values()
            ],
            "has_image": bool(self.images),
        }

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Entity":
        data = dict(data)
        data["facts"] = {k: Fact(**v) for k, v in data.get("facts", {}).items()}
        data["images"] = [ImageVersion(**v) for v in data.get("images", [])]
        return Entity(**data)
