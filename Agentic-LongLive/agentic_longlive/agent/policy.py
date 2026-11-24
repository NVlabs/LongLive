# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory policies: the LLM planner (the RL-trainable policy) and rule baselines for member selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import AgentConfig, MemoryConfig
from ..memory.store import EntityStore
from ..memory.update import MemoryPlan, parse_plan, slug
from ..models.base import LLMClient
from ..story import AgentView, StorySetup
from . import prompts


@dataclass
class PlanCall:
    """Inputs and outputs of one planner call (logged for audits and RL datasets)."""

    plan: MemoryPlan
    system: str
    user: str
    images: List[str]
    raw_text: str
    model: str
    latency_s: float


class LLMPolicy:
    """Writes the full memory plan with one LLM call per revealed prompt."""

    def __init__(self, llm: LLMClient, cfg: AgentConfig, memory: MemoryConfig):
        self.llm = llm
        self.cfg = cfg
        self.memory = memory

    def plan(self, view: AgentView, store: EntityStore, observations: Optional[List[Path]] = None,
             since: Optional[int] = None) -> PlanCall:
        """One planner call for every prompt revealed after ``since`` (default: only the newest)."""
        cast = []
        for c in view.setup.cast:
            entity = store.get(slug(c.id))
            if entity is not None and any(h.get("event") == "cast_split" for h in entity.history):
                # the card's changeable appearance lives in the entity facts; showing the card's old clothing / beard
                # here would invite copying it back
                cast.append({"id": c.id, "name": c.name, "appearance": "see the ENTITY TABLE (description + facts)"})
            else:
                cast.append({"id": c.id, "name": c.name, "description": c.description})
        setup = {"title": view.setup.title, "style": view.setup.style, "cast": cast}
        first_new = view.latest if since is None else max(0, min(since + 1, view.latest))
        new = [(i, view.revealed[i]) for i in range(first_new, view.latest + 1)]
        k = self.cfg.context_prompts
        recent = list(view.revealed[max(0, first_new - k):first_new]) if k else []
        observation = ""
        images: List[Path] = []
        if observations:
            observation = ("Attached: the last frame of each recently generated shot (oldest first). Generated video "
                           "may contain errors; prompts remain the authority.")
            images = list(observations)
        user = prompts.plan_user(
            setup, store.table(), store.scene, recent, new,
            {"g": self.cfg.max_g_members, "m": self.cfg.max_m_members},
            observation=observation, next_prompt=view.next_prompt or "",
        )
        result = self.llm.chat(prompts.PLAN_SYSTEM, [user, *images], json_mode=True)
        plan = parse_plan(result.data or {})
        return PlanCall(plan=plan, system=prompts.PLAN_SYSTEM, user=user, images=[str(i) for i in images],
                        raw_text=result.text, model=result.model, latency_s=result.latency_s)

    def split_cast(self, setup: StorySetup) -> Dict[str, Dict]:
        """Session setup (memory.cast_split): split each cast card into an unchanging identity description and the
        card's changeable appearance as initial facts. An identity that still names changeable appearance (beard,
        glasses, clothing, ...) is sent back once; what remains is moved into facts clause by clause. Cards that come
        back unusable keep the whole card as identity (no split for that member)."""
        if not setup.cast:
            return {}
        cards = [{"id": c.id, "name": c.name, "card": c.description} for c in setup.cast]
        user = prompts.cast_split_user(cards)
        out: Dict[str, Dict] = {}
        for attempt in range(2):
            try:
                result = self.llm.chat(prompts.CAST_SPLIT_SYSTEM, [user], json_mode=True, temperature=0.0)
                data = result.data or {}
            except Exception:  # the split is an optimisation of the memory text; never block the session on it
                break
            out = self._parse_split(data, setup)
            leaks = {eid: sorted({m.lower() for m in prompts.CHANGEABLE.findall(v["identity"])})
                     for eid, v in out.items() if prompts.CHANGEABLE.search(v["identity"])}
            if not leaks or attempt == 1:
                break
            user = (prompts.cast_split_user(cards) + "\n\n### FIX\nYour identity descriptions still mention changeable "
                    "appearance: " + json.dumps(leaks) + ". Move these into facts and return the corrected JSON.")
        for eid, part in out.items():
            part["identity"] = self._strip_changeable(part["identity"], part["facts"])
        return out

    @staticmethod
    def _parse_split(data: Dict, setup: StorySetup) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        ids = {slug(c.id) for c in setup.cast}
        for item in data.get("cast", []) or []:
            if not isinstance(item, dict):
                continue
            eid = slug(item.get("id", ""))
            identity = str(item.get("identity", "")).strip()
            raw = item.get("facts") if isinstance(item.get("facts"), dict) else {}
            facts = {slug(k): str(v).strip() for k, v in raw.items()
                     if str(v).strip() and str(v).strip().lower() not in {"none", "n/a", "no", "null", "-"}}
            if eid in ids and len(identity.split()) >= 4 and facts:
                out[eid] = {"identity": identity, "facts": facts}
        return out

    @staticmethod
    def _strip_changeable(identity: str, facts: Dict[str, str]) -> str:
        """Last resort: drop identity clauses that name changeable appearance, keeping them as facts."""
        if not prompts.CHANGEABLE.search(identity):
            return identity
        end = "." if identity.rstrip().endswith(".") else ""
        clauses = [c for c in re.split(r",\s*|\s+and\s+", identity.rstrip(". ")) if c]
        kept = []
        for clause in clauses:
            word = prompts.CHANGEABLE.search(clause)
            if not word:
                kept.append(clause)
                continue
            text = re.sub(r"^(with|wearing|and)\s+", "", clause.strip())
            if not any(text.lower() in v.lower() for v in facts.values()):
                w = word.group(0).lower()
                key = ("facial_hair" if re.match(r"beard|moustache|mustache|goatee|stubble|sideburns", w) else
                       "glasses" if "glasses" in w or "spectacles" in w else
                       "hairstyle" if w in {"bun", "braid", "ponytail"} else "appearance_detail")
                facts[key] = f"{facts[key]}; {text}" if key in facts else text
        return ", ".join(kept) + end if kept else identity


def select_members(rule: str, plan: MemoryPlan, store: EntityStore, cfg: AgentConfig) -> Tuple[List[str], List[str]]:
    """G and M members. ``llm`` trusts the plan (falling back to rules); others are ablation baselines."""
    entities = store.entities
    valid = lambda ids: [i for i in dict.fromkeys(ids) if i in entities and entities[i].kind != "scene"]  # noqa: E731
    protagonists = store.protagonists()
    others = [e for e in entities.values() if e.kind != "scene" and not e.protagonist and e.status == "active"]
    if rule == "llm":
        g = valid(plan.g_members) or protagonists
        m = valid(plan.m_members) or valid(store.active) or protagonists
    else:
        if rule == "recent":
            ranked = sorted(others, key=lambda e: e.last_seen_shot, reverse=True)
        elif rule == "frequency":
            ranked = sorted(others, key=lambda e: (e.seen_count, e.last_seen_shot), reverse=True)
        else:  # all_active
            ranked = [entities[i] for i in store.active if i in entities and entities[i] in others]
        ordered = protagonists + [e.id for e in ranked]
        g, m = ordered, ordered
    return g[: cfg.max_g_members], m[: cfg.max_m_members]
