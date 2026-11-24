# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hindsight targets for one memory decision (privileged information: used only for scoring, never as policy input).

Decision w writes memory m_w after prompt w is revealed; m_w conditions shot w+1 (lag 0) and its entity table persists.
From the hidden per-shot annotation (``present``, ``entity_state`` = persistent state at the END of each shot):

* F (needed facts): for shots j in (w, w+H], every persistent attribute value of an entity present in j that already
  held after shot w and still holds at the start of j (unchanged through j-1). Terse prompts do not restate these, so
  only memory can supply them. Weight gamma^(j-w-1); ``visual`` when needed by the very next shot.
* E (slot items): entities present in shot w+1 that were already seen (<= w): they should be in the G or M image.
* R (retention items): entities seen (<= w), absent in w and w+1, that return within (w+1, w+H_ret]: they must stay in
  the memory (not retired). ``absent_for`` is recorded for the revisit analysis.
* truth: the true persistent state after shot w of every entity seen so far (for false-fact checks).
"""

from __future__ import annotations

from typing import Any, Dict, List


def last_seen_before(anns: List[Dict[str, Any]], e: str, j: int) -> int:
    for k in range(j - 1, -1, -1):
        if e in anns[k].get("present", []):
            return k
    return -1


def compute(meta: Dict[str, Any], w: int, H: int = 6, H_ret: int = 12, gamma: float = 0.8) -> Dict[str, Any]:
    """Targets of decision ``w`` from a story file's raw JSON (``shots[*].annotation`` and ``bible.entities``)."""
    anns = [s.get("annotation", {}) for s in meta["shots"]]
    ents = (meta.get("bible") or {}).get("entities", {})
    n = len(anns)
    seen = set()
    for k in range(w + 1):
        seen.update(anns[k].get("present", []))
    state_w = anns[w].get("entity_state", {})
    facts: Dict[tuple, Dict[str, Any]] = {}
    for j in range(w + 1, min(n, w + 1 + H)):
        start_j = anns[j - 1].get("entity_state", {})
        for e in anns[j].get("present", []):
            if e not in seen:
                continue
            for a, v in (state_w.get(e) or {}).items():
                if (start_j.get(e) or {}).get(a) != v:
                    continue
                wt = gamma ** (j - w - 1)
                key = (e, a)
                if key not in facts or facts[key]["weight"] < wt:
                    facts[key] = {"entity": e, "attr": a, "value": v, "first_need": j, "weight": round(wt, 4),
                                  "visual": j == w + 1}
    F = []
    for i, item in enumerate(sorted(facts.values(), key=lambda x: (x["first_need"], x["entity"], x["attr"]))):
        F.append({"id": f"F{i + 1}", **item})
    E = []
    if w + 1 < n:
        for e in anns[w + 1].get("present", []):
            if e in seen:
                E.append({"entity": e, "absent_for": (w + 1) - last_seen_before(anns, e, w + 1) - 1})
    R = []
    for e in sorted(seen):
        if e in anns[w].get("present", []) or (w + 1 < n and e in anns[w + 1].get("present", [])):
            continue
        for j in range(w + 2, min(n, w + 1 + H_ret)):
            if e in anns[j].get("present", []):
                R.append({"entity": e, "return_at": j, "absent_for": j - last_seen_before(anns, e, j) - 1})
                break
    truth = {e: dict(state_w.get(e) or {}) for e in sorted(seen) if e in state_w}
    canon = {e: {"kind": (ents.get(e) or {}).get("kind", "object"), "name": (ents.get(e) or {}).get("name", e),
                 "identity": (ents.get(e) or {}).get("identity", "")} for e in sorted(seen)}
    return {"w": w, "F": F, "E": E, "R": R, "truth": truth, "canon": canon}
