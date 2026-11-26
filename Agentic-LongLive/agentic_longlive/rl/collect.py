# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect decision states by rolling a planner through training stories (text only), and build the RFT/SFT data.

  rollouts: python -m agentic_longlive.rl.collect rollouts --stories data/rl/train/terse data/rl/train/full \\
                --planner-urls URLS --planner-model Qwen/Qwen3.6-35B-A3B-FP8 --tag teacher --out data/rl/rollouts
  bestofk : python -m agentic_longlive.rl.collect bestofk --states data/rl/rollouts/states_teacher.jsonl \\
                --planner-urls URLS --judge-urls URLS --k 3 --out data/rl/sft/bestof.jsonl

``rollouts`` writes one state per decision (``<out>/states_<tag>.jsonl``): the planner's system / user prompt, the
entity store before the decision, the behaviour (teacher) output and the hindsight targets. These states form the
GRPO state pool. ``bestofk`` scores the behaviour output plus K fresh teacher samples per state with the text reward
(reward version 1, training judge) and keeps the best as the SFT target (K=3: best of 4). Both commands resume: stories
/ states already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from . import targets as T
from .common import Endpoints, LLMAdapter, append_jsonl, chat, log, read_jsonl
from .score import JUDGE_MODEL, Scorer, judge_urls_from_env, openai_judge
from .textenv import load_story, rollout, story_meta


def state_rows(records: List[Dict[str, Any]], meta: Dict[str, Any], variant: str, tag: str) -> List[Dict[str, Any]]:
    """Decision states (inputs, behaviour output and hindsight targets) from the records of one rollout."""
    rows = []
    for r in records:
        rows.append({"key": f"{tag}/{meta['id']}/{r['shot']}", "story": meta["id"], "variant": variant,
                     "shot": r["shot"], "n_shots": len(meta["shots"]), "system": r["system"], "user": r["user"],
                     "store_before": r["store_before"], "behaviour_output": r["output"],
                     "behaviour_parsed": r["parsed"], "behaviour_model": r["model"], "latency": r["latency"],
                     "completion_tokens": r["completion_tokens"], "targets": T.compute(meta, r["shot"]),
                     "memory": r["memory"], "tag": tag})
    return rows


def cmd_rollouts(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in args.stories:
        files += sorted(Path(d).glob("*.json"))
    if args.limit:
        random.Random(0).shuffle(files)
        files = files[: args.limit]
    eps = Endpoints(args.planner_urls)
    states_path = out / f"states_{args.tag}.jsonl"
    done = {r["story"] for r in read_jsonl(states_path)}

    def run(path: Path):
        meta = story_meta(path)
        if meta["id"] in done:
            return meta["id"], 0
        story = load_story(path)
        llm = LLMAdapter(eps, args.planner_model, temperature=args.temperature, max_tokens=4096)
        recs = rollout(story, llm, oracle=args.oracle)
        variant = "terse" if "terse" in str(path.parent) else "full"
        rows = state_rows(recs, meta, variant, args.tag)
        for row in rows:
            append_jsonl(row, states_path)
        return meta["id"], len(rows)

    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run, f): f for f in files}
        for f in as_completed(futs):
            try:
                sid, n = f.result()
                log(f"rollout {args.tag} {sid}: {n} decisions", out / "collect.log")
            except Exception as exc:  # noqa: BLE001
                log(f"rollout {args.tag} {futs[f].name} FAILED {exc!r}", out / "collect.log")


def cmd_bestofk(args) -> None:
    """Score the behaviour output + K samples per state with the training judge; keep the best as the SFT target."""
    states = read_jsonl(args.states)
    if args.max_states:
        random.Random(1).shuffle(states)
        states = states[: args.max_states]
    eps = Endpoints(args.planner_urls)
    scorer = Scorer(openai_judge(args.judge_urls or judge_urls_from_env(), args.judge_model), workers=args.workers)
    out = Path(args.out)
    done = {r["key"] for r in read_jsonl(out)}

    def run(st):
        if st["key"] in done:
            return None
        cands = [st["behaviour_output"]]
        if args.k:
            info = chat(eps, args.planner_model, st["system"], st["user"], temperature=args.temperature,
                        max_tokens=4096, n=args.k)
            cands += info["texts"]
        scores = scorer.score_many([(st, c) for c in cands])
        totals = [s["total"] if s.get("total") is not None else -9 for s in scores]
        best = max(range(len(cands)), key=lambda i: totals[i])
        row = {"key": st["key"], "story": st["story"], "variant": st["variant"], "shot": st["shot"],
               "system": st["system"], "user": st["user"], "output": cands[best], "best_index": best,
               "rewards": totals, "parts": [{k: v for k, v in s.items() if k != "judge"} for s in scores]}
        append_jsonl(row, out)
        return totals

    n = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(run, s) for s in states]):
            try:
                f.result()
                n += 1
                if n % 50 == 0:
                    log(f"bestofk {n}/{len(states)}", out.with_suffix(".log"))
            except Exception as exc:  # noqa: BLE001
                log(f"bestofk FAILED {exc!r}", out.with_suffix(".log"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rollouts", help="roll a planner through stories and write decision states")
    a.add_argument("--stories", nargs="+", required=True, help="directories of story JSON files")
    a.add_argument("--planner-urls", required=True, help="comma-separated OpenAI-compatible base URLs (.../v1)")
    a.add_argument("--planner-model", required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--temperature", type=float, default=0.2)
    a.add_argument("--workers", type=int, default=32)
    a.add_argument("--limit", type=int, default=0, help="use a random subset of this many story files")
    a.add_argument("--oracle", action="store_true", help="show the next prompt (upper bound only)")
    b = sub.add_parser("bestofk", help="best-of-(K+1) SFT targets from decision states")
    b.add_argument("--states", required=True)
    b.add_argument("--planner-urls", required=True)
    b.add_argument("--planner-model", default=JUDGE_MODEL)
    b.add_argument("--judge-urls", default="", help="training judge endpoints (default: $RL_JUDGE_URLS or localhost)")
    b.add_argument("--judge-model", default=JUDGE_MODEL)
    b.add_argument("--k", type=int, default=3, help="fresh samples besides the behaviour output")
    b.add_argument("--temperature", type=float, default=0.8)
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=48)
    b.add_argument("--max-states", type=int, default=0)
    args = ap.parse_args()
    {"rollouts": cmd_rollouts, "bestofk": cmd_bestofk}[args.cmd](args)


if __name__ == "__main__":
    main()
