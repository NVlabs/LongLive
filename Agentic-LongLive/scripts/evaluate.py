# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score finished runs on LongStory-Online (paper Table 1 metrics) and compare methods.

    # judge every shot of every story run under outputs/ours and outputs/longlive2, then print the table
    python scripts/evaluate.py --runs outputs/ours outputs/longlive2 --stories stories/longstory_online \
        --judge gemini --imaging --out results

    # several seeds of one method: NAME=DIR,DIR,...  (seed k of A is paired with seed k of B in --compare)
    python scripts/evaluate.py --runs ours=outputs/ours_s0,outputs/ours_s1 base=outputs/base_s0,outputs/base_s1 \
        --compare ours base --stories stories/longstory_online --judge gemini --out results

    # an OpenAI-compatible judge, e.g. a Qwen VL model served by vLLM / SGLang (records kept in <run>/eval_qwen)
    python scripts/evaluate.py --runs outputs/ours --judge openai --judge-url http://127.0.0.1:8000/v1 \
        --judge-model Qwen/Qwen3-VL-32B-Instruct --cache eval_qwen --out results_qwen

A method root holds one run directory per story (the ``output_dir`` of ``scripts/run_story.py``). Judge records are
cached in ``<run>/<cache>/`` and only missing ones are requested, so the command can be re-run after interruptions;
``--judge none`` only aggregates existing records. Outputs in ``--out``: ``scores.json`` (every method, subset and
story), ``table.md`` and, with ``--compare``, ``compare.json`` / ``compare.md``. Each run also gets
``<run>/<cache>/scores.json``.

Credentials come from the environment: ``GEMINI_SERVICE_ACCOUNT`` (Vertex AI service-account JSON) or, with
``--judge-backend studio``, ``GEMINI_API_KEY``; ``OPENAI_API_KEY`` (or ``--judge-api-key-env``) for ``openai``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.eval import judge as J  # noqa: E402
from agentic_longlive.eval import metrics as M  # noqa: E402
from agentic_longlive.eval.measure import load_measurements, measure_run, plan_visibility  # noqa: E402
from agentic_longlive.eval.runs import find_runs, load_run  # noqa: E402


def parse_method(spec: str):
    """``DIR`` or ``NAME=DIR[,DIR...]`` -> (name, [roots]); several roots are seeds of one method."""
    if "=" in spec and not Path(spec).exists():
        name, dirs = spec.split("=", 1)
        return name, [Path(d) for d in dirs.split(",") if d]
    roots = [Path(d) for d in spec.split(",") if d]
    return roots[0].resolve().name, roots


def visibility_path(stories: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    for base in (stories, stories.parent):
        path = base / "eval" / "expected_visible.json"
        if path.is_file():
            return path
    return stories / "eval" / "expected_visible.json"


def build_judge(args):
    if args.judge == "none":
        return None
    if args.judge_config:
        spec = J.spec_from_yaml(args.judge_config, args.judge_concurrency)
    else:
        extra = json.loads(args.judge_extra) if args.judge_extra else None
        spec = J.judge_spec(args.judge, model=args.judge_model, base_url=args.judge_url, backend=args.judge_backend,
                            api_key_env=args.judge_api_key_env, concurrency=args.judge_concurrency, extra_body=extra)
    return J.build_judge(spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[], help="method roots: DIR or NAME=DIR[,DIR...] (seeds)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="paired differences A - B (names or roots)")
    ap.add_argument("--stories", default=str(ROOT / "stories" / "longstory_online"),
                    help="benchmark directory (terse prompts; use .../full for the full-prompt control)")
    ap.add_argument("--visibility", default="", help="plan-visibility JSON (default: <stories>/eval/expected_visible.json)")
    ap.add_argument("--judge", default="gemini", choices=["gemini", "openai", "none"])
    ap.add_argument("--judge-config", default="", help="YAML with a ModelSpec (or a config whose agent.checker is used)")
    ap.add_argument("--judge-model", default="", help="default: gemini-3.8-flash (gemini) / $AGL_JUDGE_MODEL")
    ap.add_argument("--judge-url", default="", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-backend", default="vertex", choices=["vertex", "studio"], help="Gemini endpoint")
    ap.add_argument("--judge-api-key-env", default="", help="environment variable holding the judge API key")
    ap.add_argument("--judge-extra", default="", help="JSON merged into OpenAI-compatible requests")
    ap.add_argument("--judge-concurrency", type=int, default=8)
    ap.add_argument("--cache", default="eval", help="folder under each run for the judge records (one per judge)")
    ap.add_argument("--continuation-pairs", action="store_true",
                    help="also judge consecutive shots at one place (supplementary place_continuation)")
    ap.add_argument("--imaging", action="store_true", help="VBench imaging quality (needs torch + pyiqa)")
    ap.add_argument("--mir", action="store_true", help="ArcFace MIR (needs insightface)")
    ap.add_argument("--device", default=None, help="device for --imaging / --mir, e.g. cuda:0")
    ap.add_argument("--vbench-cache", default=None, help="VBench model cache (default $VBENCH_CACHE_DIR or ~/.cache/vbench)")
    ap.add_argument("--face-root", default=None, help="insightface model root")
    ap.add_argument("--draws", type=int, default=2000, help="bootstrap draws")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    methods = OrderedDict(parse_method(s) for s in args.runs)
    for spec in args.compare or []:
        if spec not in methods:
            name, roots = parse_method(spec)
            methods.setdefault(name, roots)
    if not methods:
        ap.error("give --runs and/or --compare")
    compare = [m if m in methods else parse_method(m)[0] for m in args.compare] if args.compare else None

    stories = Path(args.stories)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = build_judge(args)
    table = J.load_visibility(visibility_path(stories, args.visibility))
    cache_path = out / "plan_visibility.json"
    for key, value in J.load_visibility(cache_path).items():  # stories outside the benchmark judged earlier
        table.setdefault(key, value)
    imaging = faces = None
    if args.imaging:
        from agentic_longlive.eval.quality import ImagingQuality

        imaging = ImagingQuality(args.device, args.vbench_cache)
    if args.mir:
        from agentic_longlive.eval.mir import FaceDetector

        device = args.device or "cuda:0"
        gpu = -1 if device == "cpu" else int(device.split(":")[1]) if ":" in device else 0
        faces = FaceDetector(gpu, args.face_root)

    results, seeds_of = OrderedDict(), {}
    for name, roots in methods.items():
        seeds = []
        for root in roots:
            runs = {}
            dirs = find_runs(root)
            if not dirs:
                print(f"warning: no runs under {root}", flush=True)
            for run_dir in dirs:
                run = load_run(run_dir, stories, cache=args.cache)
                if run.key in runs:
                    print(f"warning: {run_dir} repeats story {run.key} in {root}; skipped", flush=True)
                    continue
                vis = plan_visibility(run.key, run, judge, table, cache_path, args.judge_concurrency)
                if judge is not None or imaging is not None or faces is not None:
                    measured = measure_run(run, judge, vis, args.continuation_pairs, imaging, faces,
                                           args.judge_concurrency)
                else:
                    measured = load_measurements(run)
                scores = M.run_scores(run.story, measured, vis)
                if len(run.videos) < len(run.story) or scores["judged_shots"] < len(run.videos):
                    print(f"warning: {run_dir}: {len(run.videos)}/{len(run.story)} shots generated, "
                          f"{scores['judged_shots']} judged", flush=True)
                run.eval_dir.mkdir(parents=True, exist_ok=True)
                (run.eval_dir / "scores.json").write_text(json.dumps(
                    {"story": run.key, "run": str(run_dir), **scores,
                     "per_shot": {m: {str(i): v for i, v in d.items()} for m, d in scores["per_shot"].items()}},
                    indent=1))
                runs[run.key] = scores
                print(f"[{name}] {run.key}: " + "  ".join(f"{m} {M.fmt(m, scores['means'].get(m))}" for m in M.TABLE),
                      flush=True)
            seeds.append(runs)
        seeds_of[name] = seeds
        results[name] = {"roots": [str(r) for r in roots],
                         "subsets": {sub: M.summarize(seeds, sub) for sub in M.SUBSETS},
                         "per_story": [{k: s["means"] for k, s in runs.items()} for runs in seeds]}

    judge_name = getattr(judge, "name", None) if judge is not None else "cached records"
    (out / "scores.json").write_text(json.dumps({"judge": judge_name, "cache": args.cache, "methods": results},
                                                indent=1))
    lines = []
    for sub, title in (("all", "All 25 stories"), ("hand_written", "Hand-written (9)"), ("hard", "LongStory-Hard (16)")):
        lines += [f"### {title}", "", M.markdown_table({n: r["subsets"][sub] for n, r in results.items()}), ""]
    (out / "table.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))

    if compare:
        a, b = compare
        diffs = {sub: {m: M.paired(seeds_of[a], seeds_of[b], m, sub, args.draws) for m in M.TABLE + M.SUPPLEMENT}
                 for sub in M.SUBSETS}
        (out / "compare.json").write_text(json.dumps({"a": a, "b": b, "draws": args.draws, "subsets": diffs},
                                                     indent=1))
        text = [f"### {a} - {b} (story-clustered bootstrap 95% CI, {args.draws} draws)", ""]
        for sub in M.SUBSETS:
            text += [f"{sub}:", "", M.markdown_compare(a, b, diffs[sub]), ""]
        (out / "compare.md").write_text("\n".join(text))
        print("\n".join(text))


if __name__ == "__main__":
    main()
