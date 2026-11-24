# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image-layer evaluation: render the G / M memory images of every held-out decision of a planner (as the deployed
painter composes them: text-only FLUX.2-klein, fixed seed) and audit them with Gemini (vision) against the hindsight
targets of the next shot. The training reward used Qwen3.6 for this audit, so test numbers do not come from the
training judge.

  render : python -m agentic_longlive.rl.imgeval render --eval-dir EVAL_DIR --stories STORY_DIR --out IMG_DIR \\
               --names a,b --image-urls URLS [--src NAME0]
  audit  : python -m agentic_longlive.rl.imgeval audit --eval-dir EVAL_DIR --stories STORY_DIR --out IMG_DIR --names a,b
  summary: python -m agentic_longlive.rl.imgeval summary --eval-dir EVAL_DIR --out IMG_DIR

``EVAL_DIR`` is the output directory of ``evaluate`` (records in ``rows/<name>/records.jsonl``; the summary reads
``rows/<name>/judged_gemini.jsonl`` for the entity mapping). Names ``rule_recent``, ``rule_frequency`` and
``rule_all_active`` reuse the records of ``--src`` with rule-based member selection.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import targets as T
from .common import append_jsonl, log, parse_json, read_jsonl
from .imagelayer import AUDIT_SYSTEM, ROLES, Renderer, audit_prompt, compose_specs, parse_audit
from .reward import RewardConfig
from .score import GEMINI_MODEL, gemini_client
from .textenv import apply_output, story_metas

RULES = ("rule_recent", "rule_frequency", "rule_all_active")


def _records(eval_dir: Path, name: str, src: str) -> List[Dict[str, Any]]:
    if name in RULES and not src:
        raise SystemExit(f"{name} needs --src (the planner whose state trajectory it reuses)")
    return read_jsonl(eval_dir / "rows" / (src if name in RULES else name) / "records.jsonl")


def cmd_render(args) -> None:
    rend = Renderer(args.image_urls, model=args.image_model, per_server=2)
    metas = story_metas(args.stories)
    log_path = Path(args.out) / "imgeval.log"
    for name in args.names.split(","):
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        rule = name.removeprefix("rule_") if name in RULES else "llm"
        recs = _records(Path(args.eval_dir), name, args.src)
        idx_path = out / "index.jsonl"
        done = {(r["story"], r["shot"]) for r in read_jsonl(idx_path)}

        def one(r):
            if (r["story"], r["shot"]) in done or r["story"] not in metas:
                return
            tg = T.compute(metas[r["story"]], r["shot"])
            row = {"story": r["story"], "shot": r["shot"], "E": [x["entity"] for x in tg["E"]]}
            if tg["E"]:
                try:
                    store, plan, g, m, ok = apply_output(r["store_before"], r["output"], r["shot"], rule=rule)
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    row["parsed"] = False
                else:
                    row.update({"parsed": True, "g": g, "m": m})
                    for role, spec in zip(ROLES, compose_specs(store, plan, g, m)):
                        if not spec:
                            continue
                        png = rend.render(spec)
                        if png:
                            f = out / f"{r['story']}_{r['shot']:03d}_{role}.png"
                            f.write_bytes(png)
                            row[role] = f.name
            append_jsonl(row, idx_path)

        with ThreadPoolExecutor(args.workers) as ex:
            for k, f in enumerate(as_completed([ex.submit(one, r) for r in recs])):
                f.result()
                if (k + 1) % 200 == 0:
                    log(f"render {name} {k + 1}/{len(recs)} {rend.stats}", log_path)
        log(f"render {name} done {rend.stats}", log_path)


def gemini_vlm(model: str = GEMINI_MODEL):
    client = gemini_client(model, max_tokens=2048, max_image_side=768)

    def call(system: str, content: list) -> Optional[Dict[str, Any]]:
        for _ in range(3):
            try:
                r = client.chat(system, content, json_mode=True, temperature=0.0)
                d = r.data if isinstance(r.data, dict) else parse_json(r.text)
                if d is not None:
                    return d
            except Exception:  # noqa: BLE001
                continue
        return None
    return call


def cmd_audit(args) -> None:
    call = gemini_vlm(args.vlm_model)
    metas = story_metas(args.stories)
    log_path = Path(args.out) / "imgeval.log"
    for name in args.names.split(","):
        d = Path(args.out) / name
        idx = read_jsonl(d / "index.jsonl")
        dest = d / "audit_gemini.jsonl"
        done = {(r["story"], r["shot"]) for r in read_jsonl(dest)}

        def one(row):
            try:
                _one(row)
            except Exception as exc:  # noqa: BLE001  (one bad reply must not stop the split)
                log(f"audit {name} {row['story']}:{row['shot']} error {exc!r}", log_path)

        def _one(row):
            if (row["story"], row["shot"]) in done or not row.get("E") or not row.get("parsed"):
                return
            tg = T.compute(metas[row["story"]], row["shot"])
            rendered = {r: bool(row.get(r)) and (d / row[r]).is_file() for r in ROLES}
            ents, vis, text = audit_prompt(tg, rendered)
            res: Dict[str, Any] = {"story": row["story"], "shot": row["shot"], "rendered": rendered}
            if not any(rendered.values()):
                res.update({"entities": {e: {"G": False, "M": False} for e in ents}, "facts": {}, "quality": {}})
                append_jsonl(res, dest)
                return
            content: list = [text]
            for r in ROLES:
                if rendered[r]:
                    content += [f"IMAGE {r}:", d / row[r]]
            out = None
            for _ in range(3):  # the reply must give a {"G": bool, "M": bool} object for every entity
                out = call(AUDIT_SYSTEM, content)
                if isinstance(out, dict) and isinstance(out.get("entities"), dict) and all(
                        isinstance(out["entities"].get(e), dict) for e in ents):
                    break
                out = None
            if not isinstance(out, dict):
                res["failed"] = True
                append_jsonl(res, dest)
                return
            parsed = parse_audit(out, ents, vis, rendered)
            res["entities"] = parsed["entities"]
            res["facts"] = parsed["facts"]
            res["fact_entity"] = {f["id"]: f["entity"] for f in vis}
            res["quality"] = {r: parsed["quality"][r] for r in ROLES if rendered[r]}
            append_jsonl(res, dest)

        with ThreadPoolExecutor(args.workers) as ex:
            for k, f in enumerate(as_completed([ex.submit(one, r) for r in idx])):
                f.result()
                if (k + 1) % 200 == 0:
                    log(f"audit {name} {k + 1}/{len(idx)}", log_path)
        log(f"audit {name} done", log_path)


def summarise(rows: List[Dict[str, Any]], judged: Dict[tuple, Dict[str, Any]],
              cfg: RewardConfig = RewardConfig()) -> Dict[str, Any]:
    """Depiction rates, fact accuracy, quality and the image utility of reward version 4 on audited decisions."""
    ok = [r for r in rows if not r.get("failed") and r.get("entities") is not None]
    dg, dm, util, fs, fc, qg, qm, off = [], [], [], [], [], [], [], []
    for r in ok:
        E = list(r["entities"])
        if not E:
            continue
        labs = r.get("facts") or {}
        fe = r.get("fact_entity") or {}
        score = 0.0
        for e in E:
            g_, m_ = float(r["entities"][e]["G"]), float(r["entities"][e]["M"])
            el = [labs[f] for f in labs if fe.get(f) == e]
            q = 1.0 - cfg.q_wrong * (sum(1 for b in el if b == "C") / len(el)) if el else 1.0
            dg.append(g_)
            dm.append(m_)
            score += q * (cfg.u_g * g_ + cfg.u_m * m_)
        n_off = 0
        j = judged.get((r["story"], r["shot"]))
        if j and j.get("judge"):
            mp = j["judge"].get("map") or {}
            on = {mp.get(e) for e in E} - {None}
            n_off = len(set(j.get("m") or []) - on)
        off.append(n_off)
        util.append(score / len(E) - cfg.m_off_cost * n_off)
        fs += [1.0 if b == "S" else 0.0 for b in labs.values()]
        fc += [1.0 if b == "C" else 0.0 for b in labs.values()]
        if "G" in (r.get("quality") or {}):
            qg.append(r["quality"]["G"])
        if "M" in (r.get("quality") or {}):
            qm.append(r["quality"]["M"])

    def mean(x):
        return sum(x) / len(x) if x else None
    return {"n": len(ok), "n_failed": sum(1 for r in rows if r.get("failed")), "depicted_G": mean(dg),
            "depicted_M": mean(dm), "fact_shown": mean(fs), "fact_wrong": mean(fc), "quality_G": mean(qg),
            "quality_M": mean(qm), "m_off": mean(off), "utility": mean(util)}


def cmd_summary(args) -> None:
    base = Path(args.out)
    table = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        rows = read_jsonl(d / "audit_gemini.jsonl")
        if not rows:
            continue
        jf = Path(args.eval_dir) / "rows" / d.name / "judged_gemini.jsonl"
        judged = {(r["story"], r["shot"]): r for r in read_jsonl(jf)}
        table[d.name] = summarise(rows, judged)
    (base / "summary.json").write_text(json.dumps(table, indent=1))
    for k, v in table.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("render")
    a.add_argument("--eval-dir", required=True)
    a.add_argument("--stories", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--names", required=True)
    a.add_argument("--src", default="", help="records reused by rule_* names")
    a.add_argument("--image-urls", required=True, help="FLUX.2-klein servers (OpenAI image API, .../v1)")
    a.add_argument("--image-model", default="black-forest-labs/FLUX.2-klein-4B")
    a.add_argument("--workers", type=int, default=16)
    b = sub.add_parser("audit")
    b.add_argument("--eval-dir", required=True)
    b.add_argument("--stories", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--names", required=True)
    b.add_argument("--vlm-model", default=GEMINI_MODEL)
    b.add_argument("--workers", type=int, default=24)
    c = sub.add_parser("summary")
    c.add_argument("--eval-dir", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"render": cmd_render, "audit": cmd_audit, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
