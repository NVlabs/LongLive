# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a robot episode: a 64 s continuous video rendered segment by segment, with and without the memory agent.

Arms (same seeds per segment):
  base  the generator alone: sink = [latest frame], prompt = the user's instruction.
  ours  the generator + the memory agent: sink = [up to 2 pinned memory pictures, latest frame], prompt = the agent's
        rewrite with a preamble that names each pinned picture. Segment 0 is shared with base (nothing to remember yet).

Every segment continues from the previous segment's hand-off frame (keyframe at frame 0, no cut; segment 0 starts from
the input image). In the sink, picture k sits 0.25*k s after the prompt and the target video starts at 2 s.

Real-time schedule (ours): the agent never blocks the generator. While segment i-1 is generated, the agent observes
segment i-2 and plans segment i (lag 1); the input image is observed while segment 0 is generated. Each segment is
submitted to an out-of-process generator worker through the file queue of ``agentic_longlive.workers.protocol``.

Outputs (``out/``): ``<arm>/seg<k>.mp4``, ``<arm>/seg<k>/last.png`` (hand-off frames), ``<arm>/episode.json``
(prompts, pictures, seeds, timings, plans), ``ours/agent.json`` (agent trace and final memory), ``ours/agent/mem/``
(memory pictures) and the joined videos ``<arm>.mp4``.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..workers import protocol
from . import video
from .agent import RobotAgent, make_agent
from .episode import base_prompt, instructions, ours_prompt

SEED0 = 1000                 # segment i uses seed 1000 + i
REF_STEP_S = 0.25            # sink: picture k at 0.25 * k s
VIDEO_START_S = 2.0          # sink: the target video starts at 2 s (after a cut)
REF_SCALE = 0.5              # pictures are cover-resized to half the video canvas (672x384 for 1344x768)
SEGMENT_S = 8                # seconds per instruction


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


class QueueGenerator:
    """Submits one segment at a time to a generator worker that serves ``queue`` (e.g. ``workers/fasth3.py``)."""

    def __init__(self, queue: Path, tag: str = "robot", timeout_s: float = 1800.0, retries: int = 3,
                 retry_wait_s: float = 20.0, alive: Optional[Callable[[], bool]] = None):
        self.queue = Path(queue).resolve()
        self.tag, self.timeout_s, self.retries = tag, timeout_s, retries
        self.retry_wait_s, self.alive = retry_wait_s, alive

    def job(self, job_id: str, out: Path, prompt: str, refs: Sequence[Path], keyframe: Path,
            seed: int) -> Dict[str, Any]:
        refs = [str(Path(r).resolve()) for r in refs]
        return {"id": job_id, "out": str(Path(out).resolve()), "seed": int(seed), "prompt": prompt, "refs": refs,
                "ref_seconds": [REF_STEP_S * k for k in range(len(refs))], "video_start_seconds": VIDEO_START_S,
                "keyframe": str(Path(keyframe).resolve()), "ref_scale": REF_SCALE, "duration_seconds": SEGMENT_S}

    def render(self, name: str, out: Path, prompt: str, refs: Sequence[Path], keyframe: Path,
               seed: int) -> Dict[str, Any]:
        """Render one segment into ``out``; retried with a fresh job id. Returns the worker's done record and the wall
        time."""
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        err: Optional[Exception] = None
        for attempt in range(self.retries):
            job = self.job(f"{self.tag}-{name}-{uuid.uuid4().hex[:8]}", out, prompt, refs, keyframe, seed)
            t0 = time.monotonic()
            try:
                out.unlink(missing_ok=True)    # never mistake an older file for this segment
                rec = protocol.submit(self.queue, job, timeout_s=self.timeout_s, alive=self.alive)
                if not out.exists():
                    raise RuntimeError(f"worker reported success but {out} does not exist")
                return {"job": job["id"], "wall_s": round(time.monotonic() - t0, 2), "worker_s": rec.get("seconds"),
                        "worker_timing": rec.get("timing")}
            except Exception as e:  # noqa: BLE001 -- retried, then raised
                err = e
                _log(f"  generation retry {attempt} ({name}): {e}")
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_wait_s)
        raise RuntimeError(f"generation failed for {name}: {err}")


@dataclass
class Seeds:
    """Segment 0 (shared by both arms) uses ``first``; segment i >= 1 uses ``base + i``. The paper's runs use 1000 for
    both; its second-seed run used base 2000 with segment 0 unchanged."""

    base: int = SEED0
    first: int = SEED0

    def __call__(self, i: int) -> int:
        return self.first if i == 0 else self.base + i


def run_base(ep: Dict[str, Any], image: Path, gen: QueueGenerator, out_dir: Path, seeds: Seeds) -> Path:
    """The generator alone. Returns the joined video."""
    D = Path(out_dir) / "base"
    D.mkdir(parents=True, exist_ok=True)
    rec: Dict[str, Any] = {"arm": "base", "episode": ep["name"], "first_frame": str(image), "segments": []}
    kf = Path(image)
    instr = instructions(ep)
    for i, text in enumerate(instr):
        out = D / f"seg{i}.mp4"
        p = base_prompt(ep, text)
        t = gen.render(f"{ep['name']}-base-seg{i}", out, p, [kf], kf, seeds(i))
        rec["segments"].append({"seg": i, "prompt": p, "seed": seeds(i), "pictures": [str(kf)], "keyframe": str(kf),
                                "timing": t})
        _log(f"  base seg{i} {t['wall_s']}s")
        kf = video.last_frame(out, D / f"seg{i}")
        _write_json(D / "episode.json", rec)
    return video.concat([D / f"seg{i}.mp4" for i in range(len(instr))], Path(out_dir) / "base.mp4")


def run_ours(ep: Dict[str, Any], image: Path, gen: QueueGenerator, agent: RobotAgent, out_dir: Path, seeds: Seeds,
             base_dir: Optional[Path] = None) -> Path:
    """The generator + the memory agent on the real-time (lag-1) schedule. Returns the joined video.

    base_dir: the base arm's directory of the same episode and seeds; its segment 0 is reused when present."""
    D = Path(out_dir) / "ours"
    D.mkdir(parents=True, exist_ok=True)
    instr = instructions(ep)
    n = len(instr)
    rec: Dict[str, Any] = {"arm": "ours", "episode": ep["name"], "policy": agent.policy, "first_frame": str(image),
                           "segments": [], "agent_windows": []}
    plans: Dict[int, dict] = {}

    def agent_window(i: int, observe_input: bool = False) -> None:
        """Runs while segment i-1 is generated: observe segment i-2 (or the input image), then plan segment i."""
        t0 = time.monotonic()
        w: Dict[str, Any] = {"plans_seg": i, "observe_seg": -1 if observe_input else i - 2}
        if observe_input:
            w["observe_s"] = round(agent.observe_input(image), 2)
        elif i - 2 >= 0:
            w["observe_s"] = round(agent.observe_segment(i - 2, D / f"seg{i - 2}.mp4", instr[i - 2]), 2)
        if i < n:
            plans[i], dt = agent.plan(i, instr[i], instr[:i])
            w["plan_s"] = round(dt, 2)
        w["window_s"] = round(time.monotonic() - t0, 2)
        rec["agent_windows"].append(w)

    # Segment 0: no memory yet, identical to base segment 0. The agent observes the input image and plans segment 1
    # meanwhile.
    seg0 = D / "seg0.mp4"
    if base_dir is not None and (Path(base_dir) / "seg0.mp4").exists():
        shutil.copy2(Path(base_dir) / "seg0.mp4", seg0)
        agent_window(1, observe_input=True)
        rec["segments"].append({"seg": 0, "shared_with_base": True, "seed": seeds(0)})
    else:
        p0 = base_prompt(ep, instr[0])
        res0: Dict[str, Any] = {}

        def g0() -> None:
            try:
                res0["t"] = gen.render(f"{ep['name']}-ours-seg0", seg0, p0, [image], image, seeds(0))
            except Exception as e:  # noqa: BLE001 -- re-raised below
                res0["err"] = e
        th0 = threading.Thread(target=g0, daemon=True)
        th0.start()
        agent_window(1, observe_input=True)
        th0.join()
        if "err" in res0:
            raise res0["err"]
        rec["segments"].append({"seg": 0, "prompt": p0, "seed": seeds(0), "pictures": [str(image)],
                                "keyframe": str(image), "timing": res0["t"]})
    _log(f"  ours seg0 ({agent.policy})")
    kf = video.last_frame(seg0, D / "seg0")
    _write_json(D / "episode.json", rec)

    for i in range(1, n):
        pl = plans[i]
        pinned = agent.pictures(pl)
        pics = [Path(m["path"]) for m in pinned] + [kf]
        p = ours_prompt(ep, pl["prompt"], pinned)
        out = D / f"seg{i}.mp4"
        res: Dict[str, Any] = {}

        def g() -> None:
            try:
                res["t"] = gen.render(f"{ep['name']}-ours-seg{i}", out, p, pics, kf, seeds(i))
            except Exception as e:  # noqa: BLE001 -- re-raised below
                res["err"] = e
            res["done"] = time.monotonic()
        th = threading.Thread(target=g, daemon=True)
        th.start()
        if i + 1 < n:
            agent_window(i + 1)
        window_done = time.monotonic()
        th.join()
        if "err" in res:
            raise res["err"]
        rec["segments"].append({"seg": i, "prompt": p, "seed": seeds(i),
                                "plan": {k: v for k, v in pl.items() if k != "pinned"}, "pinned": pinned,
                                "pictures": [str(x) for x in pics], "keyframe": str(kf), "timing": res["t"],
                                "generator_waited_s": round(max(0.0, window_done - res["done"]), 2)})
        _log(f"  ours seg{i} {res['t']['wall_s']}s pictures={[m['id'] for m in pinned]}")
        kf = video.last_frame(out, D / f"seg{i}")
        _write_json(D / "episode.json", rec)

    # Observe the last segments too, for the record (no generation uses them).
    for j in range(max(0, n - 2), n):
        agent.observe_segment(j, D / f"seg{j}.mp4", instr[j])
    _write_json(D / "agent.json", {"trace": agent.trace, "state": agent.state()})
    _write_json(D / "episode.json", rec)
    return video.concat([D / f"seg{i}.mp4" for i in range(n)], Path(out_dir) / "ours.mp4")


def run_episode(ep: Dict[str, Any], image: Path, queue: Path, out_dir: Path, arms: Sequence[str] = ("base", "ours"),
                policy: str = "v4", llm=None, grounder=None, crop_mode: Optional[str] = None,
                seeds: Optional[Seeds] = None, job_timeout_s: float = 1800.0, gen_retries: int = 3,
                retry_wait_s: float = 20.0) -> List[Path]:
    """Run the requested arms of one episode; returns the joined videos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or Seeds()
    gen = QueueGenerator(queue, tag=f"robot-{uuid.uuid4().hex[:6]}", timeout_s=job_timeout_s, retries=gen_retries,
                         retry_wait_s=retry_wait_s)
    videos = []
    t0 = time.monotonic()
    if "base" in arms:
        _log(f"{ep['name']}: base")
        videos.append(run_base(ep, image, gen, out_dir, seeds))
    if "ours" in arms:
        if llm is None:
            raise ValueError("the ours arm needs an agent LLM (--agent-url)")
        _log(f"{ep['name']}: ours ({policy})")
        agent = make_agent(policy, ep, llm, out_dir / "ours" / "agent", grounder=grounder, crop_mode=crop_mode)
        base_dir = out_dir / "base" if "base" in arms else None
        videos.append(run_ours(ep, image, gen, agent, out_dir, seeds, base_dir=base_dir))
    _log(f"{ep['name']}: done in {time.monotonic() - t0:.0f} s")
    return videos
