# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-shot measurements of one run, cached under ``<run>/eval/`` (re-running fills only what is missing).

* ``suite.json``    ``shots[i].judge`` (shot judge), ``shots[i].imaging`` / ``duration`` (VBench imaging quality),
                    ``pairs["j:i"]`` (place pair judge on revisits, optionally consecutive shots);
* ``identity.json`` ``ref`` (the run's first shot in which the shot judge saw the protagonist) and
                    ``pairs["ref:i"]`` (protagonist pair judge against it);
* ``faces_se.json`` / ``mir_se.json`` optional ArcFace detections and MIR.

Judge failures are logged and left missing, so the next call retries them.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..models.base import LLMClient
from . import frames as F
from . import judge as J
from .runs import Run


def _read(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def load_measurements(run: Run) -> Dict[str, Any]:
    """The cached measurements of a run: suite, identity and MIR records (empty when not measured)."""
    return {"suite": _read(run.eval_dir / "suite.json", {"story": run.story.id, "shots": {}, "pairs": {}}),
            "identity": _read(run.eval_dir / "identity.json", {}),
            "mir": _read(run.eval_dir / "mir_se.json", {})}


def measure_run(run: Run, judge: Optional[LLMClient] = None, visibility: Optional[Dict[str, bool]] = None,
                continuation: bool = False, imaging: Optional[Callable] = None, faces: Optional[Any] = None,
                workers: int = 8, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Fill the missing measurements of ``run``.

    ``judge``: VLM client (None: no judge calls). ``visibility``: the plan visibility of the story's shots
    ({"i": bool}); only planned-visible shots need a protagonist pair (None: pair every shot in which the protagonist
    was seen). ``imaging``: frames -> VBench imaging quality (:class:`quality.ImagingQuality`). ``faces``: a
    :class:`mir.FaceDetector` for MIR.
    """
    eval_dir = run.eval_dir
    suite_path, identity_path = eval_dir / "suite.json", eval_dir / "identity.json"
    suite = _read(suite_path, {"story": run.story.id, "shots": {}, "pairs": {}})
    suite.setdefault("shots", {})
    suite.setdefault("pairs", {})
    lock = threading.Lock()

    def rec(i: int) -> Dict[str, Any]:
        return suite["shots"].setdefault(str(i), {})

    if judge is not None:
        name = str(getattr(judge, "name", type(judge).__name__))
        if suite.get("judge") not in (None, name):
            log(f"warning: {suite_path} holds records of judge {suite['judge']}; new records come from {name} "
                f"(use another --cache folder to keep judges apart)")
        suite["judge"] = suite.get("judge") or name
        guide = J.location_guide(run.story)

        def shot_work(i: int) -> None:
            try:
                paths = F.save_judge_frames(run.videos[i], eval_dir, i)
                result = J.judge_shot(judge, run.story, i, paths, guide)
            except Exception as exc:  # noqa: BLE001 - keep going; retried on the next call
                log(f"shot judge failed {run.dir.name} shot {i}: {exc!r}")
                return
            with lock:
                rec(i)["judge"] = result

        todo = [i for i in sorted(run.videos) if "judge" not in rec(i)]
        with ThreadPoolExecutor(max(1, workers)) as pool:
            list(pool.map(shot_work, todo))
        _write(suite_path, suite)

        def pair_work(item) -> None:
            a, b, kind = item
            try:
                paths = F.save_pair_frames(run.videos[a], eval_dir, a) + F.save_pair_frames(run.videos[b], eval_dir, b)
                result = J.judge_pair(judge, run.story, b, paths)
            except Exception as exc:  # noqa: BLE001
                log(f"pair judge failed {run.dir.name} {a}:{b}: {exc!r}")
                return
            with lock:
                suite["pairs"][f"{a}:{b}"] = {**result, "kind": kind}

        todo = [(a, b, kind) for a, b, kind in J.place_pairs(run.story, continuation)
                if a in run.videos and b in run.videos and f"{a}:{b}" not in suite["pairs"]]
        with ThreadPoolExecutor(max(1, workers)) as pool:
            list(pool.map(pair_work, todo))
        _write(suite_path, suite)

        if run.story.setup.cast:
            identity = measure_protagonist(run, judge, suite, visibility, workers, log)
            _write(identity_path, identity)

    if imaging is not None:
        for i in sorted(run.videos):
            r = rec(i)
            if "imaging" in r and "duration" in r:
                continue
            try:
                video = F.Video(run.videos[i])
                r["duration"] = video.duration
                r["imaging"] = float(imaging(video.frames(F.uniform_indices(video.n, F.IMAGING_FRAMES))))
            except Exception as exc:  # noqa: BLE001
                log(f"imaging failed {run.dir.name} shot {i}: {exc!r}")
        _write(suite_path, suite)

    if faces is not None:
        from .mir import measure_mir

        _write(eval_dir / "mir_se.json", measure_mir(run, faces))
    return load_measurements(run)


def measure_protagonist(run: Run, judge: LLMClient, suite: Dict[str, Any], visibility: Optional[Dict[str, bool]],
                        workers: int = 8, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Protagonist pairs: every later shot in which the shot judge saw the protagonist vs. the run's first such shot."""
    path = run.eval_dir / "identity.json"
    out = _read(path, {})
    seen = sorted(int(k) for k, r in suite["shots"].items() if (r.get("judge") or {}).get("protagonist_visible"))
    if not seen:
        return {"ref": None, "visible": [], "pairs": {}}
    ref = seen[0]
    if out.get("ref") != ref:  # a new first appearance invalidates the cached pairs
        out = {"ref": ref, "pairs": {}}
    out["visible"] = seen
    out.setdefault("pairs", {})
    todo = [j for j in seen[1:] if f"{ref}:{j}" not in out["pairs"]
            and (visibility is None or visibility.get(str(j)))]
    lock = threading.Lock()

    def frames_of(i: int):
        paths = F.judge_frame_paths(run.eval_dir, i)
        if not all(p.is_file() for p in paths):
            paths = F.save_judge_frames(run.videos[i], run.eval_dir, i)
        return paths

    def work(j: int) -> None:
        try:
            paths = frames_of(ref) + frames_of(j)
        except Exception as exc:  # noqa: BLE001
            log(f"protagonist frames failed {run.dir.name} {ref}:{j}: {exc!r}")
            return
        result = J.judge_protagonist(judge, run.story, ref, j, paths)
        if result is None:
            log(f"protagonist judge failed {run.dir.name} {ref}:{j}")
            return
        with lock:
            out["pairs"][f"{ref}:{j}"] = result

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(work, todo))
    return out


def plan_visibility(story_key: str, run: Run, judge: Optional[LLMClient], table: Dict[str, Dict[str, bool]],
                    cache: Optional[Path] = None, workers: int = 8,
                    log: Callable[[str], None] = print) -> Optional[Dict[str, bool]]:
    """Plan visibility of a story: from the frozen benchmark table, else judged once (text only) and cached."""
    if story_key in table and len(table[story_key]) >= len(run.story):
        return table[story_key]
    if judge is None or not run.story.setup.cast:
        return table.get(story_key)
    known = dict(table.get(story_key) or {})
    todo = [i for i in range(len(run.story)) if str(i) not in known]

    def work(i: int):
        return i, J.judge_visibility(judge, run.story, i)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for i, value in pool.map(work, todo):
            if value is not None:
                known[str(i)] = value
            else:
                log(f"plan visibility undecided {story_key} shot {i}")
    table[story_key] = known
    if cache is not None:
        _write(cache, table)
    return known
