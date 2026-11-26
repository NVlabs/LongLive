# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured state updates produced by the memory policy and applied to the entity store."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .entity import KINDS, SOURCES


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return value or "entity"


@dataclass
class FactUpdate:
    attr: str
    value: str
    persistent: bool = True
    source: str = "explicit"


@dataclass
class EntityUpdate:
    id: str
    kind: str = "object"
    name: str = ""
    description: str = ""
    facts: List[FactUpdate] = field(default_factory=list)
    revoke: List[str] = field(default_factory=list)  # attributes whose earlier value is withdrawn
    present: bool = True  # visible / involved in the revealed shot


@dataclass
class StateUpdate:
    entities: List[EntityUpdate] = field(default_factory=list)
    scene: Optional[str] = None  # scene entity id of the revealed shot
    active: List[str] = field(default_factory=list)  # entities likely to stay relevant
    retire: List[str] = field(default_factory=list)  # entities that left the story for now


@dataclass
class MemoryPlan:
    """Everything the policy writes for one revealed prompt (the RL action)."""

    update: StateUpdate
    image_prompts: Dict[str, str] = field(default_factory=dict)  # entity id -> image description
    g_members: List[str] = field(default_factory=list)
    m_members: List[str] = field(default_factory=list)
    g_prompt: str = ""
    m_prompt: str = ""
    g_caption: str = ""
    m_caption: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return default


def _persistent(value: Any) -> bool:
    """Accepts booleans or the words persistent / transient."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"transient", "false", "no", "0"}
    return True


# Attributes that say where a character is or what it does right now are never drawn into memory (Method 2: pose,
# action and hand-held items stay neutral), even when the planner marks them persistent (e.g. "location:
# standing beside the induction hob" would draw the kitchen into the entity image). Characters only: for objects a
# location is a placement ("left on the counter"), which is persistent.
_TRANSIENT_ATTRS = re.compile(r"^(location|position|where|pose|posture|stance|action|activity|doing|current_action|"
                              r"holding|held_items?|carrying|in_hand|gaze|expression|facial_expression|mood|framing|"
                              r"camera)(_|$)")


def is_transient_attr(attr: str) -> bool:
    return bool(_TRANSIENT_ATTRS.match(attr))


def parse_plan(data: Dict[str, Any]) -> MemoryPlan:
    """Lenient parser for the planner's JSON; unknown fields are ignored, ids are normalised."""
    entities: List[EntityUpdate] = []
    for item in data.get("entities", []) or []:
        if not isinstance(item, dict) or not (item.get("id") or item.get("name")):
            continue
        kind = str(item.get("kind", "object")).lower()
        kind = kind if kind in KINDS else "object"
        facts = []
        for fact in item.get("facts", []) or []:
            if not isinstance(fact, dict) or not fact.get("attr"):
                continue
            persistence = fact.get("persistent", fact.get("persistence", True))
            source = str(fact.get("source", "explicit")).lower()
            facts.append(FactUpdate(
                attr=slug(fact["attr"]),
                value=str(fact.get("value", "")).strip(),
                persistent=_persistent(persistence),
                source=source if source in SOURCES else "explicit",
            ))
        entities.append(EntityUpdate(
            id=slug(item.get("id") or item.get("name")),
            kind=kind,
            name=str(item.get("name", "")).strip(),
            description=str(item.get("description", "")).strip(),
            facts=facts,
            revoke=[slug(a) for a in item.get("revoke", []) or []],
            present=_as_bool(item.get("present", True), True),
        ))
    update = StateUpdate(
        entities=entities,
        scene=slug(data["scene"]) if data.get("scene") else None,
        active=[slug(a) for a in data.get("active", []) or []],
        retire=[slug(a) for a in data.get("retire", []) or []],
    )
    memory = data.get("memory", {}) or {}
    return MemoryPlan(
        update=update,
        image_prompts={slug(k): str(v) for k, v in (data.get("image_prompts", {}) or {}).items()},
        g_members=[slug(a) for a in memory.get("g_members", []) or []],
        m_members=[slug(a) for a in memory.get("m_members", []) or []],
        g_prompt=str(memory.get("g_prompt", "") or ""),
        m_prompt=str(memory.get("m_prompt", "") or ""),
        g_caption=str(memory.get("g_caption", "") or ""),
        m_caption=str(memory.get("m_caption", "") or ""),
        raw=data,
    )
