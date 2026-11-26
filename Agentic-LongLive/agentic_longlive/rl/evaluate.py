# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline (text-layer) evaluation of memory planners on held-out stories with hidden annotations.

Online protocol: each planner rolls through every story on its own states (prompts revealed one at a time).

  rollout  : python -m agentic_longlive.rl.evaluate rollout --name NAME --urls URLS --model MODEL \\
                 --stories STORY_DIR --out EVAL_DIR [--oracle]
  judge    : python -m agentic_longlive.rl.evaluate judge --name NAME --stories STORY_DIR --out EVAL_DIR \\
                 [--judge gemini|openai] [--rule recent --src NAME0]
  summary  : python -m agentic_longlive.rl.evaluate summary --out EVAL_DIR
  from-runs: python -m agentic_longlive.rl.evaluate from-runs --runs RUNS_DIR --name NAME --out EVAL_DIR

Metrics use the same hindsight targets as the training reward but are judged by a different model family by
default (Gemini; the training reward uses Qwen3.6). Rule rows (``--rule recent|frequency|all_active``) reuse another
planner's state trajectory (``--src``) and only replace G/M member selection. ``from-runs`` turns live runs (output
directories of the agent runtime) into records, so text metrics are measured on exactly the memory that drove the
rendered video. Layout: ``EVAL_DIR/rows/<name>/{records,judged_<judge>}.jsonl``, ``EVAL_DIR/summary.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from . import judge as J
from . import targets as T
from .common import Endpoints, LLMAdapter, append_jsonl, log, read_jsonl, write_jsonl
from .reward import RewardConfig, metrics, reward
from .score import GEMINI_MODEL, JUDGE_MODEL, gemini_judge, judge_urls_from_env, openai_judge
from .textenv import apply_output, load_story, memory_view, rollout, story_meta, story_metas

RECORD_KEYS = ("story", "shot", "output", "parsed", "latency", "completion_tokens", "prompt_tokens", "store_before",
               "model", "oracle")


def cmd_rollout(args) -> None:
    out = Path(args.out) / "rows" / args.name
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.stories).glob("*.json"))
    if not files:
        raise SystemExit(f"no story files in {args.stories}")
    eps = Endpoints(args.urls)
    path = out / "records.jsonl"
    done = {r["story"] for r in read_jsonl(path)}

    def run(p: Path):
        meta = story_meta(p)
        if meta["id"] in done:
            return meta["id"], 0
        llm = LLMAdapter(eps, args.model, temperature=args.temperature, max_tokens=4096)
        recs = rollout(load_story(p), llm, oracle=args.oracle)
        for r in recs:
            append_jsonl({k: r[k] for k in RECORD_KEYS}, path)
        return meta["id"], len(recs)

    with ThreadPoolExecutor(args.workers or len(files)) as pool:
        for f in as_completed([pool.submit(run, p) for p in files]):
            sid, n = f.result()
            log(f"eval rollout {args.name} {sid} {n}", Path(args.out) / "eval.log")


def cmd_judge(args) -> None:
    src = args.src or args.name
    recs = read_jsonl(Path(args.out) / "rows" / src / "records.jsonl")
    metas = story_metas(args.stories)
    if args.judge == "gemini":
        call = gemini_judge(args.judge_model or GEMINI_MODEL)
    else:
        call = openai_judge(args.judge_urls or judge_urls_from_env(), args.judge_model or JUDGE_MODEL)
    dest = Path(args.out) / "rows" / args.name / f"judged_{args.judge}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["story"], r["shot"]) for r in read_jsonl(dest)}
    cfg = RewardConfig()

    def run(r):
        if (r["story"], r["shot"]) in done:
            return
        meta = metas[r["story"]]
        tg = T.compute(meta, r["shot"])
        store, plan, g, m, ok = apply_output(r["store_before"], r["output"], r["shot"], rule=args.rule)
        mem = memory_view(store, plan, g, m)
        j = J.judge(call, tg, mem) if ok else None
        met = metrics(tg, mem, j) if ok else {}
        rw = reward(tg, mem, j, r.get("completion_tokens") or 0, ok, cfg)
        gm = set(g) | set(m)
        row = {"story": r["story"], "shot": r["shot"], "parsed": ok, "latency": r.get("latency"),
               "completion_tokens": r.get("completion_tokens"), "reward": rw.get("total"), **met,
               "E": [{"entity": x["entity"], "absent_for": x["absent_for"],
                      "hit": bool(j) and (j["map"].get(x["entity"]) in gm)} for x in tg["E"]],
               "R": [{"entity": x["entity"], "absent_for": x["absent_for"],
                      "hit": bool(j) and bool(j["map"].get(x["entity"]))
                      and any(e["id"] == j["map"][x["entity"]] and e["status"] == "active" for e in mem["entities"])}
                     for x in tg["R"]],
               "n_F": len(tg["F"]), "judge": j, "g": g, "m": m}
        append_jsonl(row, dest)

    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run, r) for r in recs]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 50 == 0:
                log(f"judge {args.name} {i + 1}/{len(recs)}", Path(args.out) / "eval.log")


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def membership(r: Dict[str, Any], cfg: RewardConfig = RewardConfig()) -> Dict[str, Any]:
    """Coverage of next-shot entities by G / M, off-shot M members and the video-calibrated utility (membership
    only, as in reward version 4 with ``image=False``)."""
    mp = (r.get("judge") or {}).get("map") or {}
    E = [x["entity"] for x in r.get("E", [])]
    g, m = set(r.get("g") or []), set(r.get("m") or [])
    on = {mp.get(e) for e in E} - {None}
    off = len(m - on)
    out = {"n_m": len(m), "n_g": len(g), "n_m_off": off, "m_recall": None, "g_recall": None}
    score = 0.0
    if E:
        out["m_recall"] = sum(1.0 for e in E if mp.get(e) in m) / len(E)
        out["g_recall"] = sum(1.0 for e in E if mp.get(e) in g) / len(E)
        score = sum(cfg.u_g * (mp.get(e) in g) + cfg.u_m * (mp.get(e) in m) for e in E) / len(E)
    out["util"] = score - cfg.m_off_cost * off
    return out


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"n": len(rows), "format_ok": _mean([1.0 if r["parsed"] else 0.0 for r in rows])}
    for k in ("reward", "belief", "support", "contradict", "visual", "slots", "retain", "n_false", "n_facts",
              "n_entities", "completion_tokens", "latency"):
        out[k] = _mean([r.get(k) for r in rows if r["parsed"]])
    buckets = {"1": (0, 1), "2-3": (2, 3), "4-7": (4, 7), "8+": (8, 999)}
    for name, key in (("revisit_slot", "E"), ("retain", "R")):
        curve = {}
        for b, (lo, hi) in buckets.items():
            hits = [x["hit"] for r in rows for x in r.get(key, []) if lo <= x["absent_for"] <= hi]
            curve[b] = {"rate": _mean([1.0 if h else 0.0 for h in hits]), "n": len(hits)}
        out[f"{name}_curve"] = curve
    # immediate re-appearance (absent 0) vs returning after absence
    e0 = [x["hit"] for r in rows for x in r.get("E", []) if x["absent_for"] == 0]
    e1 = [x["hit"] for r in rows for x in r.get("E", []) if x["absent_for"] >= 1]
    out["slots_present"] = _mean([1.0 if h else 0.0 for h in e0])
    out["slots_returning"] = _mean([1.0 if h else 0.0 for h in e1])
    mem = [membership(r) for r in rows if r["parsed"] and r.get("judge")]
    for k in ("m_recall", "g_recall", "n_m", "n_g", "n_m_off", "util"):
        out[k] = _mean([x[k] for x in mem])
    return out


def cmd_summary(args) -> None:
    root = Path(args.out) / "rows"
    table = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for jf in sorted(d.glob("judged_*.jsonl")):
            rows = read_jsonl(jf)
            if not rows:
                continue
            key = f"{d.name}|{jf.stem.split('_', 1)[1]}"
            table[key] = summarise(rows)
            by_story = defaultdict(list)
            for r in rows:
                by_story[r["story"]].append(r)
            table[key]["per_story"] = {s: summarise(v)["reward"] for s, v in by_story.items()}
    (Path(args.out) / "summary.json").write_text(json.dumps(table, indent=1))
    for k, v in table.items():
        print(f'{k:40s} n={v["n"]:4d} R={v["reward"] or 0:.3f} belief={v["belief"] or 0:.3f} '
              f'sup={v["support"] or 0:.3f} con={v["contradict"] or 0:.3f} vis={v["visual"] or 0:.3f} '
              f'slots={v["slots"] or 0:.3f} ret={v["retain"] or 0:.3f} false={v["n_false"] or 0:.2f} '
              f'tok={v["completion_tokens"] or 0:.0f} lat={v["latency"] or 0:.1f} fmt={v["format_ok"]:.3f} '
              f'mrec={v.get("m_recall") or 0:.3f} moff={v.get("n_m_off") or 0:.2f} util={v.get("util") or 0:.3f}')


def run_records(run: Path, suffix: str = "_terse") -> List[Dict[str, Any]]:
    """Evaluation records from one live run: each ``plan`` trace event is a decision whose entity store before the
    decision is ``memory/store/w{w-1:03d}.json`` (``setup.json`` for w = 0)."""
    store_dir = run / "memory" / "store"
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").is_file() else {}
    story = config.get("story_id")
    if not story and config.get("story") and Path(config["story"]).is_file():
        story = json.loads(Path(config["story"]).read_text()).get("id")
    if not story:
        story = run.name if run.name.endswith(suffix) else run.name + suffix
    out = []
    for line in (run / "trace.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev.get("kind") != "plan" or ev.get("oracle"):
            continue
        w = int(ev["shot"])
        before = store_dir / ("setup.json" if w == 0 else f"w{w - 1:03d}.json")
        if not before.is_file():
            continue
        out.append({"story": story, "shot": w, "output": ev["output"], "parsed": True, "latency": ev.get("latency_s"),
                    "completion_tokens": max(1, len(ev["output"]) // 4), "store_before": json.loads(before.read_text()),
                    "model": ev.get("model"), "oracle": False, "since": ev.get("since")})
    return out


def cmd_from_runs(args) -> None:
    rows = []
    for run in sorted(Path(args.runs).iterdir()):
        if run.is_dir() and (run / "summary.json").is_file() and (run / "trace.jsonl").is_file():
            rows += run_records(run, args.suffix)
    dest = Path(args.out) / "rows" / args.name / "records.jsonl"
    print(args.name, write_jsonl(rows, dest), "records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rollout", help="roll a planner online through every story")
    a.add_argument("--name", required=True)
    a.add_argument("--urls", required=True, help="comma-separated OpenAI-compatible base URLs (.../v1)")
    a.add_argument("--model", required=True, help="served model name (SGLang LoRA: BASE:ADAPTER)")
    a.add_argument("--stories", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--temperature", type=float, default=0.2)
    a.add_argument("--oracle", action="store_true", help="show the next prompt (upper bound only)")
    a.add_argument("--workers", type=int, default=0, help="parallel stories (0: all)")
    b = sub.add_parser("judge", help="judge the records of a planner against hidden annotations")
    b.add_argument("--name", required=True)
    b.add_argument("--src", default="", help="records of another planner (rule rows)")
    b.add_argument("--rule", default="llm", choices=("llm", "recent", "frequency", "all_active"))
    b.add_argument("--stories", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--judge", default="gemini", choices=("gemini", "openai"))
    b.add_argument("--judge-model", default="")
    b.add_argument("--judge-urls", default="", help="openai judge endpoints (default: $RL_JUDGE_URLS or localhost)")
    b.add_argument("--workers", type=int, default=32)
    c = sub.add_parser("summary", help="summary.json over every judged planner")
    c.add_argument("--out", required=True)
    d = sub.add_parser("from-runs", help="records from live runs of the agent runtime")
    d.add_argument("--runs", required=True, help="directory of run output directories")
    d.add_argument("--name", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--suffix", default="_terse", help="story id suffix when a run does not name its story file")
    args = ap.parse_args()
    {"rollout": cmd_rollout, "judge": cmd_judge, "summary": cmd_summary, "from-runs": cmd_from_runs}[args.cmd](args)


if __name__ == "__main__":
    main()
