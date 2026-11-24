# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hindsight memory judge: maps memory entities to ground-truth entities and labels each needed fact.

The same prompt is used by the training judge (Qwen3.6-35B-A3B, self-hosted) and the evaluation judge (Gemini 3.8
Flash), so test metrics never come from the model that produced the training reward.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

JUDGE_SYSTEM = """TASK: memory_audit
You audit the memory of a video-generation agent after the fact, using hidden ground truth. The agent wrote its memory
without seeing the future; you only check it. Be literal and consistent. Output one JSON object and nothing else."""

JUDGE_USER = """### GROUND-TRUTH ENTITIES (canonical id: kind, name, unchanging identity)
{canon}

### AGENT MEMORY
Entities (id, kind, name, status, description, persistent facts):
{memory}
G image members: {g}
G composition prompt: {g_prompt}
M image members: {m}
M composition prompt: {m_prompt}

### TRUE CURRENT STATE (persistent properties right now)
{truth}

### NEEDED FACTS (properties the next shots rely on)
{facts}

Tasks:
1. "map": for every ground-truth entity id, the id of the memory entity that represents the same character / object
   (names may differ slightly), or null if the memory has none.
2. "facts": for every needed fact id give "belief": "S" if that entity's memory description or persistent facts state
   this property with the same meaning (wording may differ), "C" if the memory states a different or outdated value
   for this property, "M" if the memory does not say. Judge each fact only for the mapped entity.
3. "gm_conflict": ids of needed facts that the G or M composition prompt describes with a different or outdated value.
4. "false": memory persistent facts, written as "memory_id.attr", that clearly contradict the TRUE CURRENT STATE of
   the same property (an outdated or wrong value), or that only describe something momentary (pose, action,
   expression, gaze, an item being held or used right now, camera framing). A fact that agrees with the true state,
   or that concerns a property the true state does not list and is a lasting property, is NOT false. List every
   such fact (at most 16); [] if none.
Return {{"map": {{}}, "facts": {{"F1": {{"belief": "S"}}}}, "gm_conflict": [], "false": []}}"""

JudgeCall = Callable[[str, str], Optional[Dict[str, Any]]]


def _memory_text(mem: Dict[str, Any], max_chars: int = 9000) -> str:
    rows = []
    for e in mem["entities"]:
        if e["kind"] == "scene":
            continue
        facts = "; ".join(f"{a}={v}" for a, v in e["persistent"].items())
        rows.append(f'- {e["id"]} ({e["kind"]}, "{e["name"]}", {e["status"]}): {e["description"][:300]} | {facts}')
    text = "\n".join(rows) or "(empty)"
    return text[:max_chars]


def build_user(targets: Dict[str, Any], mem: Dict[str, Any]) -> str:
    canon = "\n".join(f'- {e}: {c["kind"]}, "{c["name"]}", {c["identity"][:200]}' for e, c in targets["canon"].items()
                      if c["kind"] != "scene") or "(none)"
    names = {e: c["name"] for e, c in targets["canon"].items()}
    truth = "\n".join(f'- {names.get(e, e)} [{e}]: ' + "; ".join(f"{a}={v}" for a, v in st.items())
                      for e, st in targets["truth"].items() if st) or "(none)"
    facts = "\n".join(f'{f["id"]}: {names.get(f["entity"], f["entity"])} [{f["entity"]}].{f["attr"]} = "{f["value"]}"'
                      for f in targets["F"]) or "(none)"
    return JUDGE_USER.format(canon=canon, memory=_memory_text(mem), g=json.dumps(mem.get("g_members", [])),
                             g_prompt=(mem.get("g_prompt") or "")[:1500], m=json.dumps(mem.get("m_members", [])),
                             m_prompt=(mem.get("m_prompt") or "")[:1500], truth=truth, facts=facts)


def normalise(out: Optional[Dict[str, Any]], targets: Dict[str, Any], mem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate a judge reply: unknown memory ids map to None, missing / invalid labels become "M"."""
    if not isinstance(out, dict):
        return None
    ids = {e["id"] for e in mem["entities"]}
    mp = {}
    for e in targets["canon"]:
        v = (out.get("map") or {}).get(e)
        mp[e] = v if isinstance(v, str) and v in ids else None
    labels = {}
    for f in targets["F"]:
        x = (out.get("facts") or {}).get(f["id"])
        b = (x.get("belief") if isinstance(x, dict) else x) if x is not None else None
        b = str(b or "M").strip().upper()[:1]
        labels[f["id"]] = b if b in ("S", "C", "M") else "M"
    conflict = [str(x) for x in (out.get("gm_conflict") or []) if isinstance(x, (str, int))]
    false = [str(x) for x in (out.get("false") or []) if isinstance(x, str)][:16]
    return {"map": mp, "labels": labels, "gm_conflict": conflict, "false": false}


def judge(call: JudgeCall, targets: Dict[str, Any], mem: Dict[str, Any], retries: int = 2) -> Optional[Dict[str, Any]]:
    """``call(system, user) -> parsed JSON or None``. Empty targets need no call."""
    if not targets["F"] and not targets["E"] and not targets["R"]:
        return {"map": {}, "labels": {}, "gm_conflict": [], "false": []}
    user = build_user(targets, mem)
    for _ in range(retries + 1):
        out = normalise(call(JUDGE_SYSTEM, user), targets, mem)
        if out is not None:
            return out
    return None
