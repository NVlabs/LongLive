# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hindsight reward of one memory decision and the metrics shared by training and evaluation.

Three reward versions are used:

* ``version=1`` (default): text reward that selects the best-of-K teacher outputs for the SFT (RFT) warm start, and
  the ``reward`` column of offline evaluation. Weighted mean of belief (needed facts: S +1, M 0, C -1), visual (needed
  facts of the next shot that are stated correctly *and* whose entity is a G/M member), slots (next-shot entities in
  G or M) and retention, minus a capped false-fact penalty and a length penalty.
* ``version=3``: text-only GRPO reward (the paper's "RL, text reward only" ablation). Belief counts a contradicted
  fact as -1.5; adds an uncapped per-fact precision term; the false-fact count and the number of memory entities that
  match no ground-truth entity (``size_pen``) are penalised so the entity table stays bounded.
* ``version=4``: the paper's reward, r = 0.35 r_text + 0.3 r_img + 0.15 r_prec + 0.05 r_ret - r_pen (weights from the
  GRPO config; in every version the positive weights are renormalised over the terms that have items for the
  decision). The pooled visual / slot terms are replaced by a memory-image utility whose weights were calibrated on
  real videos: a regression of judged entity attributes on memory composition gave +9.0 accuracy points for an
  on-screen entity in G, +8.7 in the newest M and -1.9 per M member absent from the next shot; normalised these are
  ``u_g``, ``u_m`` and ``m_off_cost``. With ``image=True`` membership is measured on rendered G / M images (an entity
  counts only if a VLM finds it recognisably depicted, with up to ``q_wrong`` of its credit lost when its needed
  facts are drawn wrong); with ``image=False`` (the "calibrated, not rendered" ablation) membership alone counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

SCORE = {"S": 1.0, "M": 0.0, "C": -1.0}
VERSIONS = (1, 3, 4)


@dataclass
class RewardConfig:
    version: int = 1
    # weights of the positive terms (normalised over the terms that have items)
    w_belief: float = 0.40
    w_visual: float = 0.20  # versions 1, 3
    w_slots: float = 0.20  # versions 1, 3
    w_retain: float = 0.10
    w_precision: float = 0.25  # versions 3, 4
    w_image: float = 0.30  # version 4
    # penalties
    w_false: float = 0.15  # version 1: false facts, capped at false_cap
    false_cap: int = 4
    w_length: float = 0.05  # completion tokens beyond len_free, saturating after len_span more
    len_free: int = 1600
    len_span: int = 1600
    w_size: float = 0.20  # versions 3, 4: share of memory entities that match no ground-truth entity
    w_false_count: float = 0.10  # versions 3, 4: false facts, saturating at false_count_cap
    false_count_cap: int = 8
    format_fail: float = -1.0  # reward of an unparsable output
    # version 4 memory-image utility
    image: bool = True  # True: membership measured on rendered images; False: on member lists only
    u_g: float = 0.508
    u_m: float = 0.492
    m_off_cost: float = 0.107
    q_wrong: float = 0.5  # share of an entity's image credit lost when all its needed facts are drawn wrong
    c_belief: float = -1.5  # belief score of a contradicted needed fact (version 4)

    def __post_init__(self) -> None:
        if self.version not in VERSIONS:
            raise ValueError(f"reward version {self.version} is not one of {VERSIONS}")


def metrics(targets: Dict[str, Any], mem: Dict[str, Any], j: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Component scores in [0, 1] (belief in [-1, 1]); a component is absent when it has no items."""
    if j is None:
        return {"judge_failed": True}
    gm = set(mem.get("g_members", [])) | set(mem.get("m_members", []))
    status = {e["id"]: e["status"] for e in mem["entities"]}
    mp = j["map"]
    out: Dict[str, Any] = {}
    F = targets["F"]
    if F:
        wsum = sum(f["weight"] for f in F)
        out["belief"] = sum(f["weight"] * SCORE[j["labels"].get(f["id"], "M")] for f in F) / wsum
        out["belief_c15"] = sum(f["weight"] * {"S": 1.0, "M": 0.0, "C": -1.5}[j["labels"].get(f["id"], "M")]
                                for f in F) / wsum
        out["support"] = sum(f["weight"] * (j["labels"].get(f["id"]) == "S") for f in F) / wsum
        out["contradict"] = sum(f["weight"] * (j["labels"].get(f["id"]) == "C") for f in F) / wsum
        vis = [f for f in F if f["visual"]]
        if vis:
            conflict = set(j["gm_conflict"])
            out["visual"] = sum(1.0 for f in vis if j["labels"].get(f["id"]) == "S" and mp.get(f["entity"]) in gm
                                and f["id"] not in conflict) / len(vis)
    if targets["E"]:
        out["slots"] = sum(1.0 for x in targets["E"] if mp.get(x["entity"]) in gm) / len(targets["E"])
    if targets["R"]:
        out["retain"] = sum(1.0 for x in targets["R"] if mp.get(x["entity"]) and
                            status.get(mp[x["entity"]]) == "active") / len(targets["R"])
    out["n_false"] = len(j["false"])
    out["n_facts"] = sum(len(e["persistent"]) for e in mem["entities"] if e["kind"] != "scene")
    out["precision"] = 1.0 - min(1.0, out["n_false"] / max(1, out["n_facts"]))
    mapped = {v for v in mp.values() if v}
    out["n_unmapped"] = sum(1 for e in mem["entities"] if e["kind"] != "scene" and e["id"] not in mapped)
    out["size_pen"] = min(1.0, out["n_unmapped"] / max(2, len(mapped)))
    out["n_entities"] = sum(1 for e in mem["entities"] if e["kind"] != "scene")
    g_ids, m_ids = set(mem.get("g_members", [])), set(mem.get("m_members", []))
    on = {mp.get(x["entity"]) for x in targets["E"]} - {None}
    out["n_g"], out["n_m"] = len(g_ids), len(m_ids)
    out["n_m_off"] = len(m_ids - on)
    if targets["E"]:
        out["g_recall"] = sum(1.0 for x in targets["E"] if mp.get(x["entity"]) in g_ids) / len(targets["E"])
        out["m_recall"] = sum(1.0 for x in targets["E"] if mp.get(x["entity"]) in m_ids) / len(targets["E"])
    return out


def image_utility(targets: Dict[str, Any], mem: Dict[str, Any], j: Dict[str, Any], img: Optional[Dict[str, Any]],
                  cfg: RewardConfig) -> Optional[Dict[str, Any]]:
    """Video-calibrated memory-image utility in [-1, 1]: mean over next-shot entities of u_g*[in G] + u_m*[in M]
    (on rendered images: recognisably depicted, with credit lost for needed facts drawn wrong) minus m_off_cost per
    M member that is not in the next shot. None when the image audit failed."""
    mp = j["map"]
    E = [x["entity"] for x in targets["E"]]
    g_ids, m_ids = set(mem.get("g_members", [])), set(mem.get("m_members", []))
    on = {mp.get(e) for e in E} - {None}
    n_off = len(m_ids - on)
    out: Dict[str, Any] = {"n_m_off": n_off}
    if cfg.image:
        if img is None:
            return None
        dg = {e: float(img["entities"].get(e, {}).get("G", False)) for e in E}
        dm = {e: float(img["entities"].get(e, {}).get("M", False)) for e in E}
        q = {}
        for e in E:
            labs = [img["facts"][f["id"]] for f in targets["F"]
                    if f["visual"] and f["entity"] == e and f["id"] in img["facts"]]
            q[e] = 1.0 - cfg.q_wrong * (sum(1 for b in labs if b == "C") / len(labs)) if labs else 1.0
        labs = list(img["facts"].values())
        if labs:
            out["img_fact_s"] = sum(1 for b in labs if b == "S") / len(labs)
            out["img_fact_c"] = sum(1 for b in labs if b == "C") / len(labs)
        for r in ("G", "M"):
            if img.get("rendered", {}).get(r):
                out[f"img_q{r.lower()}"] = img.get("quality", {}).get(r)
    else:
        dg = {e: float(mp.get(e) in g_ids) for e in E}
        dm = {e: float(mp.get(e) in m_ids) for e in E}
        q = {e: 1.0 for e in E}
    if E:
        out["img_g"] = sum(dg.values()) / len(E)
        out["img_m"] = sum(dm.values()) / len(E)
        score = sum(q[e] * (cfg.u_g * dg[e] + cfg.u_m * dm[e]) for e in E) / len(E)
    else:
        score = 0.0
    out["image"] = max(-1.0, score - cfg.m_off_cost * n_off)
    return out


def reward(targets: Dict[str, Any], mem: Optional[Dict[str, Any]], j: Optional[Dict[str, Any]], n_tokens: int,
           parsed: bool, cfg: RewardConfig, img: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{"total": reward or None (judge / image audit failed), "format": 1/0, components...}."""
    if not parsed or mem is None:
        return {"total": cfg.format_fail, "format": 0.0}
    m = metrics(targets, mem, j)
    if m.get("judge_failed"):
        return {"total": None, "format": 1.0, "judge_failed": True}
    if cfg.version == 4:
        m = dict(m)
        if m.get("belief") is not None:
            F = targets["F"]
            wsum = sum(f["weight"] for f in F)
            val = {"S": 1.0, "M": 0.0, "C": cfg.c_belief}
            m["belief"] = sum(f["weight"] * val[j["labels"].get(f["id"], "M")] for f in F) / wsum
        iu = image_utility(targets, mem, j, img, cfg)
        if iu is None:
            return {"total": None, "format": 1.0, "judge_failed": True, "image_failed": True}
        m.update(iu)
        pos = {"belief": cfg.w_belief, "precision": cfg.w_precision, "image": cfg.w_image, "retain": cfg.w_retain}
    elif cfg.version == 3:
        m = dict(m)
        if m.get("belief") is not None:
            m["belief"] = m["belief_c15"]
        pos = {"belief": cfg.w_belief, "precision": cfg.w_precision, "visual": cfg.w_visual, "slots": cfg.w_slots,
               "retain": cfg.w_retain}
    else:
        pos = {"belief": cfg.w_belief, "visual": cfg.w_visual, "slots": cfg.w_slots, "retain": cfg.w_retain}
    avail = {k: w for k, w in pos.items() if m.get(k) is not None}
    total = sum(m[k] * w for k, w in avail.items()) / sum(avail.values()) if avail else 0.0
    false_pen = min(1.0, m["n_false"] / cfg.false_cap)
    len_pen = min(1.0, max(0.0, (n_tokens - cfg.len_free) / cfg.len_span))
    total -= (cfg.w_false * false_pen if cfg.version == 1 else 0.0) + cfg.w_length * len_pen
    if cfg.version >= 3:
        total -= cfg.w_size * m["size_pen"] + cfg.w_false_count * min(1.0, m["n_false"] / cfg.false_count_cap)
    return {"total": total, "format": 1.0, "false_pen": false_pen, "len_pen": len_pen, **m}
