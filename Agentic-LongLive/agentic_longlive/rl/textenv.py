# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-only rollouts of the memory agent: the deployed planner prompt, parser and entity store, without images or
video.

The agent is open loop (it never reads the video), so the planner's inputs (revealed prompts + entity table) do not
depend on the generator. Entity images are replaced by placeholders after every update so that ``has_image`` in the
planner's table matches a live run. Each decision is logged with the store before/after, the raw output, latency and
the G/M members chosen by ``select_members`` (the same rule the live agent applies).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..agent import prompts
from ..agent.policy import LLMPolicy, select_members
from ..config import AgentConfig, MemoryConfig
from ..memory.entity import Entity, ImageVersion
from ..memory.store import EntityStore
from ..memory.update import MemoryPlan, parse_plan
from ..story import AgentView, Story
from .common import LLMAdapter, parse_json

PLAN_SYSTEM = prompts.PLAN_SYSTEM


def fake_images(store: EntityStore, shot: int) -> None:
    """Placeholder image for every entity the live agent would (re)draw now."""
    for eid in store.needs_image():
        e = store.get(eid)
        store.add_image(eid, ImageVersion(version=len(e.images), views={"front": f"placeholder/{eid}_{shot}.png"},
                                          shot=shot, signature=e.signature(store.split),
                                          state=e.visual_state(store.split)))


def memory_view(store: EntityStore, plan: MemoryPlan, g: List[str], m: List[str]) -> Dict[str, Any]:
    """What the memory holds after a decision (input to judges and metrics)."""
    ents = []
    for e in store.entities.values():
        ents.append({"id": e.id, "kind": e.kind, "name": e.name, "description": e.description,
                     "status": e.status, "protagonist": e.protagonist,
                     "persistent": e.visual_state(store.split), "momentary": e.transient_state()})
    return {"entities": ents, "scene": store.scene, "active": list(store.active), "g_members": list(g),
            "m_members": list(m), "g_prompt": plan.g_prompt, "m_prompt": plan.m_prompt,
            "g_caption": plan.g_caption, "m_caption": plan.m_caption, "image_prompts": dict(plan.image_prompts)}


def store_from_json(data: Dict[str, Any]) -> EntityStore:
    """Rebuild an entity store from ``EntityStore.to_json()`` output (as stored in decision states)."""
    store = EntityStore(split=True, revision=True)
    store.scene = data.get("scene")
    store.active = list(data.get("active") or [])
    store.entities = {eid: Entity.from_json(e) for eid, e in (data.get("entities") or {}).items()}
    return store


def apply_output(store_before: Dict[str, Any], output_text: str, shot: int, rule: str = "llm",
                 acfg: Optional[AgentConfig] = None) -> Tuple[EntityStore, MemoryPlan, List[str], List[str], bool]:
    """Apply a candidate planner output to a stored state; returns (store, plan, g, m, parsed_ok). An output is
    malformed when it is not a JSON object, its ``entities`` is not a list, or it updates neither entities nor the
    scene."""
    acfg = acfg or AgentConfig()
    store = store_from_json(store_before)
    data = parse_json(output_text)
    ok = data is not None and isinstance(data.get("entities", []), list)
    plan = parse_plan(data or {})
    if ok and not plan.update.entities and plan.update.scene is None:
        ok = False
    store.apply(plan.update, shot)
    fake_images(store, shot)
    g, m = select_members(rule, plan, store, acfg)
    return store, plan, g, m, ok


def rollout(story: Story, llm: LLMAdapter, *, rule: str = "llm", oracle: bool = False, cast_split: bool = True,
            max_shots: Optional[int] = None, split_llm: Optional[LLMAdapter] = None,
            on_decision: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    """Run the planner through a story online (one decision per revealed prompt); returns one record per decision.
    ``oracle`` shows the next prompt (upper bound only; breaks the online protocol)."""
    acfg, mcfg = AgentConfig(), MemoryConfig()
    policy = LLMPolicy(llm, acfg, mcfg)
    store = EntityStore(split=True, revision=True)
    split = None
    if cast_split:
        split_policy = LLMPolicy(split_llm or llm, acfg, mcfg)
        split = split_policy.split_cast(story.setup)
    store.seed_cast(story.setup, split)
    fake_images(store, -1)
    n = min(len(story.prompts), max_shots or len(story.prompts))
    records = []
    for w in range(n):
        view = AgentView(setup=story.setup, revealed=tuple(story.prompts[:w + 1]),
                         next_prompt=(story.prompts[w + 1] if oracle and w + 1 < len(story.prompts) else None))
        before = store.to_json()
        t0 = time.monotonic()
        call = policy.plan(view, store, None, since=w - 1)
        latency = time.monotonic() - t0
        info = llm.calls[-1] if llm.calls else {}
        store.apply(call.plan.update, w)
        fake_images(store, w)
        g, m = select_members(rule, call.plan, store, acfg)
        rec = {"story": story.id, "shot": w, "system": call.system, "user": call.user, "output": call.raw_text,
               "parsed": bool(info.get("parsed", True)), "latency": latency,
               "completion_tokens": info.get("completion_tokens"), "prompt_tokens": info.get("prompt_tokens"),
               "store_before": before, "store_after": store.to_json(), "g": g, "m": m,
               "memory": memory_view(store, call.plan, g, m), "model": llm.model, "rule": rule, "oracle": oracle}
        records.append(rec)
        if on_decision:
            on_decision(rec)
    return records


def load_story(path: str | Path) -> Story:
    return Story.load(path)


def story_meta(path: str | Path) -> Dict[str, Any]:
    """Raw JSON (hidden annotations + bible) of a story file."""
    return json.loads(Path(path).read_text())


def story_metas(stories: str | Path) -> Dict[str, Dict[str, Any]]:
    """Story id -> raw JSON for every ``*.json`` story in a directory."""
    metas = {}
    for p in sorted(Path(stories).glob("*.json")):
        meta = story_meta(p)
        metas[meta["id"]] = meta
    return metas
