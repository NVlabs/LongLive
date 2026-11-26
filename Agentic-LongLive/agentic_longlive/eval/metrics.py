# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongStory-Online metrics from the cached judge records (paper Table 1 and supplement).

Per-shot metrics (each shot contributes one value; rates are 0/1):

* ``state``: persistent-state facts judged confirmed / (confirmed + contradicted), per shot with a decided fact;
* ``identity``: on shots where the judge sees the protagonist, whether it matches the identity description;
* ``same_protagonist``: the shots counted are fixed by the plan (``expected_visible``); each scores
  (rating - 1) / 4 of the protagonist pair judge against the run's first shot of the protagonist, and 0 when the shot
  judge does not see the protagonist there (or the shot is missing); the reference shot itself is excluded;
* ``location``: on scene changes (planned place differs from the previous shot's), the shot is at its planned place;
* ``revisit``: on returns to an earlier place, (place rating - 1) / 4 against the last earlier shot there if both
  shots are judged at their planned place, else 0;
* ``imaging``: VBench imaging quality (MUSIQ-SPAQ, 16 frames);
* ``artifacts``: share of shots with severe artifacts (lower is better);
* supplementary: ``framing`` (camera and subject match the plan), ``content`` (planned content shown, 0-1),
  ``location_all`` (all shots), ``place_continuation`` (consecutive shots at one place, when judged).

Story-level metrics: ``imaging_drift`` (imaging quality of shots starting in the last minute minus the first minute,
stories longer than two minutes) and ``mir`` (ArcFace, StoryEngine protocol).

Aggregation as in the paper: per-shot metrics are pooled over all shots of all stories of one seed, story-level
metrics are averaged over stories, and seeds are averaged. Paired differences between two methods use the shots
present in both (same story, shot and seed index) with a story-clustered bootstrap (2000 draws) 95% interval.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..story import Story
from .runs import is_hard

# name -> (column title, higher is better, shown as a percentage)
METRICS: Dict[str, tuple] = {
    "state": ("State", True, True),
    "identity": ("Identity", True, True),
    "same_protagonist": ("Same prot.", True, True),
    "location": ("Location", True, True),
    "revisit": ("Revisit", True, True),
    "imaging": ("Imaging", True, False),
    "artifacts": ("Artifacts", False, True),
    "framing": ("Framing", True, True),
    "content": ("Content", True, True),
    "location_all": ("Location (all)", True, True),
    "place_continuation": ("Place cont.", True, True),
    "imaging_drift": ("Imaging drift", True, False),
    "mir": ("MIR", True, False),
}
TABLE = ("state", "identity", "same_protagonist", "location", "revisit", "imaging", "artifacts")
SUPPLEMENT = ("framing", "imaging_drift", "mir")
SHOT_METRICS = ("state", "identity", "same_protagonist", "location", "revisit", "imaging", "artifacts", "framing",
                "content", "location_all", "place_continuation")
STORY_METRICS = ("imaging_drift", "mir")
SUBSETS = ("all", "hand_written", "hard")

PerShot = Dict[str, Dict[int, float]]


def per_shot_values(story: Story, suite: Dict[str, Any], identity: Optional[Dict[str, Any]] = None,
                    visibility: Optional[Dict[str, bool]] = None) -> PerShot:
    """metric -> {shot: value} for one run."""
    ann = [a or {} for a in story.annotations]
    planned = {i: a.get("scene") for i, a in enumerate(ann)}
    shots = {int(k): v for k, v in (suite.get("shots") or {}).items()}
    out: PerShot = {m: {} for m in SHOT_METRICS}
    for i, r in shots.items():
        j = r.get("judge")
        if r.get("imaging") is not None:
            out["imaging"][i] = float(r["imaging"])
        if not j:
            continue
        p, prev = planned.get(i), planned.get(i - 1)
        out["location_all"][i] = float(j["location"] == p)
        out["framing"][i] = float(j["framing_match"])
        if j.get("content") is not None:
            out["content"][i] = j["content"] / 2.0
        out["artifacts"][i] = float(j["artifacts"])
        if i > 0 and prev and p != prev:
            out["location"][i] = float(j["location"] == p)
        decided = [v for v in (j.get("facts") or {}).values() if v in ("confirmed", "contradicted")]
        if decided:
            out["state"][i] = sum(v == "confirmed" for v in decided) / len(decided)
        if j.get("protagonist_visible"):
            out["identity"][i] = float(bool(j.get("protagonist_match")))
    for key, pair in (suite.get("pairs") or {}).items():
        a, b = (int(x) for x in key.split(":"))
        if pair.get("place") is None:
            continue
        if pair.get("kind") == "revisit":
            ja = (shots.get(a) or {}).get("judge") or {}
            jb = (shots.get(b) or {}).get("judge") or {}
            at_place = ja.get("location") == planned.get(a) and jb.get("location") == planned.get(b)
            out["revisit"][b] = (pair["place"] - 1) / 4.0 if at_place else 0.0
        elif pair.get("kind") == "continuation":
            out["place_continuation"][b] = (pair["place"] - 1) / 4.0
    if visibility is not None:
        out["same_protagonist"] = same_protagonist(shots, identity or {}, visibility)
    return out


def same_protagonist(shots: Dict[int, Dict[str, Any]], identity: Dict[str, Any],
                     visibility: Dict[str, bool]) -> Dict[int, float]:
    ref = identity.get("ref")
    pairs = identity.get("pairs") or {}
    values: Dict[int, float] = {}
    for key, wanted in visibility.items():
        j = int(key)
        if not wanted or j == ref:
            continue
        # a missing or unjudged shot counts as "protagonist not seen" (0), as does a video that breaks down
        seen = bool(((shots.get(j) or {}).get("judge") or {}).get("protagonist_visible"))
        if seen and ref is not None and j > ref:
            record = pairs.get(f"{ref}:{j}")
            if record is None:
                continue  # judge failure: not scored
            values[j] = (record["person"] - 1) / 4.0 if record.get("person") is not None else 0.0
        else:
            values[j] = 0.0
    return values


def imaging_drift(suite: Dict[str, Any]) -> Optional[float]:
    """Imaging quality of shots starting in the last minute minus shots starting in the first (needs durations)."""
    shots = {int(k): v for k, v in (suite.get("shots") or {}).items()}
    ids = sorted(shots)
    if not ids or any(shots[i].get("duration") is None for i in ids):
        return None
    start, t = {}, 0.0
    for i in ids:
        start[i] = t
        t += shots[i]["duration"]
    first = _mean(shots[i].get("imaging") for i in ids if start[i] < 60.0)
    last = _mean(shots[i].get("imaging") for i in ids if start[i] >= max(0.0, t - 60.0))
    return (last - first) if first is not None and last is not None and t > 120.0 else None


def story_values(suite: Dict[str, Any], mir: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[float]]:
    return {"imaging_drift": imaging_drift(suite), "mir": (mir or {}).get("mir")}


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_scores(story: Story, measurements: Dict[str, Any], visibility: Optional[Dict[str, bool]] = None) -> Dict:
    """Scores of one run: per-shot values, their story means and the story-level metrics."""
    per = per_shot_values(story, measurements["suite"], measurements.get("identity"), visibility)
    story_level = story_values(measurements["suite"], measurements.get("mir"))
    means = {m: _mean(v.values()) for m, v in per.items()}
    means.update(story_level)
    return {"per_shot": per, "story_level": story_level, "means": means, "n": {m: len(v) for m, v in per.items()},
            "judged_shots": sum(1 for r in (measurements["suite"].get("shots") or {}).values() if r.get("judge"))}


def subset_ok(key: str, subset: str) -> bool:
    return subset == "all" or (subset == "hard") == is_hard(key)


def summarize(seeds: List[Dict[str, Dict]], subset: str = "all") -> Dict[str, Any]:
    """seeds: [{story key: run_scores(...)}] -> metric -> value (pooled over shots per seed, mean over seeds)."""
    res: Dict[str, List[float]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    for runs in seeds:
        pooled: Dict[str, List[float]] = defaultdict(list)
        stories: Dict[str, List[float]] = defaultdict(list)
        selected = {k: v for k, v in runs.items() if subset_ok(k, subset)}
        for scores in selected.values():
            for m in SHOT_METRICS:
                pooled[m] += list(scores["per_shot"].get(m, {}).values())
            for m in STORY_METRICS:
                if scores["story_level"].get(m) is not None:
                    stories[m].append(scores["story_level"][m])
        for m in SHOT_METRICS:
            if pooled[m]:
                res[m].append(sum(pooled[m]) / len(pooled[m]))
                counts[m] += len(pooled[m])
        for m in STORY_METRICS:
            if stories[m]:
                res[m].append(sum(stories[m]) / len(stories[m]))
                counts[m] += len(stories[m])
        res["stories"].append(len(selected))
        res["shots"].append(sum(s["judged_shots"] for s in selected.values()))
    out = {m: (sum(v) / len(v) if v else None) for m, v in res.items()}
    out["n"] = dict(counts)
    out["seeds"] = len(seeds)
    return out


def _bootstrap(per_story: Dict[str, List[float]], draws: int, seed: int) -> Optional[Dict[str, float]]:
    stories = sorted(s for s in per_story if per_story[s])
    if not stories:
        return None
    values = [x for s in stories for x in per_story[s]]
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        pick = [x for s in (rng.choice(stories) for _ in stories) for x in per_story[s]]
        boots.append(sum(pick) / len(pick))
    boots.sort()
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]
    return {"diff": mean, "lo": lo, "hi": hi, "n": len(values), "stories": len(stories),
            "significant": lo > 0 or hi < 0}


def paired(a_seeds: List[Dict[str, Dict]], b_seeds: List[Dict[str, Dict]], metric: str, subset: str = "all",
           draws: int = 2000, seed: int = 0) -> Optional[Dict[str, float]]:
    """Difference a - b with a story-clustered bootstrap 95% interval; seed k of a is paired with seed k of b.

    Per-shot metrics: mean over shots present in both runs. Story-level metrics: mean over (seed, story)."""
    per_story: Dict[str, List[float]] = defaultdict(list)
    for a, b in zip(a_seeds, b_seeds):
        for key in sorted(set(a) & set(b)):
            if not subset_ok(key, subset):
                continue
            if metric in STORY_METRICS:
                x, y = a[key]["story_level"].get(metric), b[key]["story_level"].get(metric)
                if x is not None and y is not None:
                    per_story[key].append(x - y)
                continue
            va, vb = a[key]["per_shot"].get(metric, {}), b[key]["per_shot"].get(metric, {})
            per_story[key] += [va[i] - vb[i] for i in sorted(set(va) & set(vb))]
    return _bootstrap(per_story, draws, seed)


# ---------------------------------------------------------------------------------------------- formatting
def fmt(metric: str, value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "--"
    _, _, percent = METRICS.get(metric, ("", True, False))
    if percent:
        return f"{100 * value:+.1f}" if signed else f"{100 * value:.1f}"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def markdown_table(rows: Dict[str, Dict[str, Any]], columns=TABLE + SUPPLEMENT) -> str:
    """rows: method -> summarize(...) result. Rates in %, imaging / drift / MIR as raw values."""
    columns = [c for c in columns if any(r.get(c) is not None for r in rows.values())]
    head = "| Method | Seeds | Stories | Shots | " + " | ".join(METRICS[c][0] + ("" if METRICS[c][1] else " (lower)")
                                                         for c in columns) + " |"
    line = "|" + "---|" * (4 + len(columns))
    body = []
    for name, r in rows.items():
        stories = r.get("stories")
        shots = r.get("shots")
        body.append(f"| {name} | {r.get('seeds')} | {'--' if stories is None else round(stories)} | "
                    f"{'--' if shots is None else round(shots)} | " + " | ".join(fmt(c, r.get(c)) for c in columns) + " |")
    return "\n".join([head, line, *body])


def markdown_compare(name_a: str, name_b: str, diffs: Dict[str, Optional[Dict[str, float]]]) -> str:
    lines = [f"| {name_a} - {name_b} | Difference | 95% CI | Shots | Significant |", "|---|---|---|---|---|"]
    for metric, d in diffs.items():
        if d is None:
            continue
        lines.append(f"| {METRICS[metric][0]} | {fmt(metric, d['diff'], True)} | [{fmt(metric, d['lo'], True)}, "
                     f"{fmt(metric, d['hi'], True)}] | {d['n']} | {'yes' if d['significant'] else 'no'} |")
    return "\n".join(lines)
