# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live session (Method 1): video loop + memory agent + prefill worker, all asynchronous.

Timeline of shot ``i``:
  boundary -> take the newest *prefilled* snapshot built from prompts <= i-1 (never wait in async
  mode) -> reveal prompt ``i`` (the agent starts writing memory for later shots) -> generate shot i.
The agent therefore never sees the prompt of the shot its memory is used for, and the video
loop's throughput does not depend on the agent.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent.agent import MemoryAgent
from ..backends.base import ShotResult, VideoBackend, build_backend
from ..config import RunConfig, to_dict
from ..memory.cards import CardBook
from ..memory.snapshot import MemorySnapshot, SnapshotBoard
from ..models.registry import build_image, build_llm
from ..story import PromptStream, Story
from .trace import Tracer


class PrefillWorker:
    """Prefills committed snapshots on the auxiliary device; always works on the newest one."""

    def __init__(self, backend: VideoBackend, tracer: Tracer, keep: int = 2):
        self.backend = backend
        self.tracer = tracer
        self.keep = keep
        self._cv = threading.Condition()
        self._queue: Optional[MemorySnapshot] = None
        self._ready: Dict[int, Tuple[MemorySnapshot, Any, float]] = {}
        self._stop = False
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="prefill", daemon=True)
        self._thread.start()

    def submit(self, snapshot: MemorySnapshot) -> None:
        with self._cv:
            if self._queue is None or snapshot.version > self._queue.version:
                self._queue = snapshot  # older pending versions are skipped
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._queue is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                snapshot, self._queue = self._queue, None
            start = time.monotonic()
            try:
                handle = self.backend.prefill(snapshot, on_aux=True)
            except BaseException as exc:
                self.error = exc
                self.tracer.log("prefill_error", version=snapshot.version, error=repr(exc))
                continue
            seconds = time.monotonic() - start
            with self._cv:
                self._ready[snapshot.version] = (snapshot, handle, seconds)
                for version in sorted(self._ready)[:-self.keep]:
                    del self._ready[version]
                self._cv.notify_all()
            self.tracer.log("prefill_ready", version=snapshot.version, watermark=snapshot.watermark,
                            seconds=round(seconds, 4), **self.backend.memory_stats())
            hook = getattr(self.backend, "after_prefill", None)
            if hook is not None:  # e.g. LongLive recall pre-prefill: runs after the version is already usable
                try:
                    hook(snapshot, handle)
                except BaseException as exc:
                    self.error = exc
                    self.tracer.log("prefill_error", version=snapshot.version, error=repr(exc), stage="after_prefill")

    def latest(self, max_watermark: int) -> Optional[Tuple[MemorySnapshot, Any, float]]:
        with self._cv:
            for version in sorted(self._ready, reverse=True):
                if self._ready[version][0].watermark <= max_watermark:
                    return self._ready[version]
            return None

    def wait(self, version: int, timeout: Optional[float] = None) -> Optional[Tuple[MemorySnapshot, Any, float]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while version not in self._ready:
                if self.error is not None:
                    raise RuntimeError("prefill failed") from self.error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cv.wait(remaining)
            return self._ready[version]

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=60)


class MemoryReplay:
    """The memory a finished run injected at every shot (``runtime.replay_memory``).

    The agent never reads the generated video, so its memory stream does not depend on how memory is injected;
    replaying one stream makes injection ablations differ only in the backend. ``memory.use_g`` / ``memory.m_slots``
    drop blocks (G only, M only, fewer M slots).
    """

    def __init__(self, run_dir: str, use_g: bool = True, m_slots: int = 2):
        summary = json.loads((Path(run_dir) / "summary.json").read_text())
        self.shots: Dict[int, Optional[dict]] = {}
        for record in summary["per_shot"]:
            if record.get("snapshot") is not None and record.get("memory") is None:
                raise ValueError(f"{run_dir} has no full memory records (run before replay support)")
            self.shots[int(record["shot"])] = record.get("memory")
        self.use_g, self.m_slots = use_g, m_slots

    def at(self, shot: int) -> Optional[MemorySnapshot]:
        data = self.shots.get(shot)
        if data is None:
            return None
        snapshot = MemorySnapshot.from_json(data)
        g = snapshot.g if self.use_g else None
        m = snapshot.m[len(snapshot.m) - self.m_slots:] if self.m_slots > 0 else ()
        if g is None and not m:
            return None
        return dataclasses.replace(snapshot, g=g, m=tuple(m))


class LiveSession:
    def __init__(self, cfg: RunConfig, story: Story, backend: Optional[VideoBackend] = None):
        self.cfg = cfg
        self.story = story
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
        self.tracer = Tracer(self.out_dir / "trace.jsonl" if cfg.trace else None)
        self.backend = backend or build_backend(cfg.backend, cfg, self.out_dir, self.tracer)
        self.board = SnapshotBoard()
        self.agent: Optional[MemoryAgent] = None
        self.replay: Optional[MemoryReplay] = None
        if cfg.runtime.replay_memory:
            if cfg.memory.mode == "none":
                raise ValueError("runtime.replay_memory needs memory (memory.mode != none)")
            self.replay = MemoryReplay(cfg.runtime.replay_memory, cfg.memory.use_g, cfg.memory.m_slots)
        elif cfg.memory.mode != "none":
            planner = build_llm(cfg.agent.planner)
            checker = planner if cfg.agent.checker == cfg.agent.planner else build_llm(cfg.agent.checker)
            self.agent = MemoryAgent(cfg, planner, checker, build_image(cfg.agent.image), self.board,
                                     self.out_dir, self.tracer)
        self.uses_memory = (self.agent is not None or self.replay is not None) and self.backend.injection != "none"
        self.prefill: Optional[PrefillWorker] = None
        # entity cards come from the entity stores of the memory that is injected (this run's, or the replayed run's)
        self.cards = CardBook(cfg.runtime.replay_memory or self.out_dir) if cfg.runtime.entity_cards else None
        self._t_prompt0: Optional[float] = None

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        cfg, tracer = self.cfg, self.tracer
        t_setup = time.monotonic()
        self.backend.start(self.story.setup)
        tracer.log("backend_ready", seconds=round(time.monotonic() - t_setup, 3))
        if self.uses_memory and cfg.runtime.aux_prefill and self.replay is None:
            # A boundary picks its snapshot before the next prompt is revealed, so the newest prefilled version is
            # always usable; older ones are only needed when memory is deliberately delayed (runtime.extra_lag).
            keep = 2 + cfg.runtime.extra_lag * max(1, cfg.agent.refine_rounds)
            self.prefill = PrefillWorker(self.backend, tracer, keep=keep)
            self.board.subscribe(self.prefill.submit)
        t_boot = time.monotonic()
        boot = self.agent.bootstrap(self.story.setup) if self.agent else None
        if boot is not None and self.prefill is not None:
            self.prefill.wait(boot.version, timeout=cfg.runtime.bootstrap_timeout_s)
        setup_seconds = time.monotonic() - t_boot
        tracer.log("session_setup_done", seconds=round(setup_seconds, 3))
        live_agent = self.agent is not None and cfg.memory.mode not in ("anchor",)
        if live_agent:
            self.agent.start()

        stream = PromptStream(self.story, cfg.max_shots, cfg.agent.oracle_next_prompt)
        shots: List[Dict[str, Any]] = []
        progress = self.out_dir / "progress.jsonl"
        resumed = self._resumable(progress)
        try:
            for i in range(len(stream)):
                if i in resumed:
                    shots.append(self._resume_shot(i, stream, resumed[i]))
                    continue
                shots.append(self._shot(i, stream, live_agent))
                with open(progress, "a") as f:
                    f.write(json.dumps(shots[-1], default=str) + "\n")
        finally:
            if self.agent is not None:
                self.agent.stop(wait=True, timeout=600)
            if self.prefill is not None:
                self.prefill.stop()
        final = self.backend.finish()
        summary = self._summary(shots, setup_seconds, final)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        tracer.log("session_done")
        tracer.close()
        self.backend.close()
        return summary

    def _resumable(self, progress: Path) -> Dict[int, Dict[str, Any]]:
        """Finished shots of an interrupted run (runtime.resume_shots); otherwise start a fresh progress file."""
        out: Dict[int, Dict[str, Any]] = {}
        if self.cfg.runtime.resume_shots and self.agent is None and progress.is_file():
            for line in progress.read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    break  # a torn last line
                if record.get("video_path") and Path(record["video_path"]).is_file():
                    out[int(record["shot"])] = record
            keep = []  # only a prefix of shots, so the progress file stays in shot order
            for i in sorted(out):
                if i != len(keep):
                    break
                keep.append(out[i])
            out = {r["shot"]: r for r in keep}
            progress.write_text("".join(json.dumps(r, default=str) + "\n" for r in keep))
            self.tracer.log("resume", shots=len(keep))
        elif progress.exists():
            progress.unlink()
        return out

    def _resume_shot(self, i: int, stream: PromptStream, record: Dict[str, Any]) -> Dict[str, Any]:
        """Keep a shot of an interrupted run: reveal its prompt and replay the backend's per-shot memory state."""
        snapshot = self.replay.at(i) if (self.uses_memory and self.replay is not None) else None
        if snapshot is not None:
            self.backend.prefill(snapshot, on_aux=False)
        stream.reveal()
        fps = float(self.backend.params.get("fps", 24))
        self.backend.shots.append(ShotResult(i, record["video_path"], int(record.get("frames") or 0), fps,
                                             meta=record.get("meta") or {}))
        self.tracer.log("shot_resumed", shot=i, video_path=record["video_path"])
        return record

    def _shot(self, i: int, stream: PromptStream, live_agent: bool) -> Dict[str, Any]:
        cfg, tracer = self.cfg, self.tracer
        if self.agent is not None and self.agent.error is not None and cfg.runtime.stop_on_agent_error:
            raise RuntimeError("memory agent failed") from self.agent.error
        if self.prefill is not None and self.prefill.error is not None:
            raise RuntimeError("memory prefill failed") from self.prefill.error
        pace_s = 0.0
        interval = float(cfg.runtime.prompt_interval_s or 0.0)
        if interval > 0:  # prompt i is not available before its arrival time: the boundary waits for it
            if self._t_prompt0 is None:
                self._t_prompt0 = time.monotonic()
            pace_s = max(0.0, self._t_prompt0 + i * interval - time.monotonic())
            if pace_s > 0:
                time.sleep(pace_s)
        t0 = time.monotonic()
        max_w = i - 1 - cfg.runtime.extra_lag
        snapshot, handle, wait_s, inline_prefill_s = None, None, 0.0, 0.0
        if self.uses_memory and self.replay is not None:
            snapshot = self.replay.at(i)
            if snapshot is not None:
                s = time.monotonic()
                handle = self.backend.prefill(snapshot, on_aux=False)
                inline_prefill_s = time.monotonic() - s
        elif self.uses_memory:
            if cfg.runtime.mode == "sync":
                target = self.board.wait_for(max(max_w, -1), complete=True)
                if target is not None and self.prefill is not None:
                    self.prefill.wait(target.version)
            if self.prefill is not None:
                ready = self.prefill.latest(max_w if i > 0 else -1)
                if ready is None:
                    ready = self.prefill.latest(-1)
                if ready is not None:
                    snapshot, handle, _ = ready
            else:
                snapshot = self.board.latest(max_watermark=max(max_w, -1))
                if snapshot is not None:
                    s = time.monotonic()
                    handle = self.backend.prefill(snapshot, on_aux=False)
                    inline_prefill_s = time.monotonic() - s
            wait_s = time.monotonic() - t0 - inline_prefill_s
        index, prompt, view = stream.reveal()
        if live_agent:
            self.agent.submit(view)
        variant = None
        key = self.backend.params.get("prompt_variant")  # optional per-shot prompt written offline for this backend
        if key and key in self.story.variants and i < len(self.story.variants[key]):
            variant = self.story.variants[key][i]
        t_prompt = time.monotonic()
        cards = 0
        if self.cards is not None and self.uses_memory:
            suffix, cards = self.cards.text(prompt, snapshot.watermark) if snapshot is not None else ("", 0)
            self.backend.text_suffix = suffix
        backend_prompt = self.backend.prepare_prompt(i, prompt, snapshot, variant)
        prompt_s = time.monotonic() - t_prompt
        result = self.backend.generate(i, backend_prompt, handle, self.cfg.seed + i)
        total = time.monotonic() - t0
        if self.agent is not None and cfg.agent.read_generated_video and result.last_frame:
            self.agent.observe(i, Path(result.last_frame))
        record = {
            "shot": i,
            "snapshot": snapshot.summary() if snapshot else None,
            "memory": snapshot.to_json() if snapshot else None,
            "lag": (i - 1 - snapshot.watermark) if snapshot else None,
            "wait_s": round(wait_s, 4),
            "inline_prefill_s": round(inline_prefill_s, 4),
            "prompt_s": round(prompt_s, 4),
            "pace_wait_s": round(pace_s, 4),
            "cards": cards,
            "kv_prepare_s": round(result.timings.get("kv_prepare", 0.0), 4),
            "video_s": round(result.timings.get("video", 0.0), 4),
            "dit_s": round(result.timings["dit"], 4) if "dit" in result.timings else None,
            "text_s": round(result.timings["text"], 4) if "text" in result.timings else None,
            "decode_tail_s": round(result.timings["decode_tail"], 4) if "decode_tail" in result.timings else None,
            "frames": result.num_frames,
            "total_s": round(total, 4),
            "video_path": result.video_path,
            "meta": result.meta,
        }
        tracer.log("shot", **record)
        return record

    def _speed(self, shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Table 2 speed: DiT FPS (denoising only, LongLive-paper protocol) and end-to-end FPS (wall clock of the
        video loop: memory wait, inline prefill, K/V load, text encoding, DiT and the decode tail)."""
        frames = sum(s.get("frames") or 0 for s in shots)
        dit = sum(s.get("dit_s") or 0.0 for s in shots)
        wall = sum(s["total_s"] for s in shots)
        fps = float(getattr(self.backend, "fps", 0) or 0) or None
        out = {"frames": frames, "dit_s": round(dit, 3), "wall_s": round(wall, 3),
               "dit_fps": round(frames / dit, 3) if dit else None,
               "e2e_fps": round(frames / wall, 3) if wall else None}
        if fps and wall:
            out["realtime_ratio"] = round(frames / fps / wall, 4)
        return out

    def _summary(self, shots: List[Dict[str, Any]], setup_seconds: float, final: Dict[str, Any]) -> Dict[str, Any]:
        def mean(key):
            values = [s[key] for s in shots if s.get(key) is not None]
            return round(sum(values) / len(values), 4) if values else None

        lags = [s["lag"] for s in shots if s["lag"] is not None]
        commits = self.tracer.select("commit")
        plans = self.tracer.select("plan")
        return {
            "story": self.story.id,
            "shots": len(shots),
            "backend": self.backend.name,
            "injection": self.backend.injection,
            "memory_mode": self.cfg.memory.mode,
            "runtime_mode": self.cfg.runtime.mode,
            "replay_memory": self.cfg.runtime.replay_memory or None,
            "setup_seconds": round(setup_seconds, 3),
            "mean": {k: mean(k) for k in ("wait_s", "inline_prefill_s", "prompt_s", "kv_prepare_s", "video_s", "total_s",
                                           "dit_s", "text_s", "decode_tail_s")},
            "speed": self._speed(shots),
            "lag_histogram": {str(v): lags.count(v) for v in sorted(set(lags))},
            "agent": {
                "plans": len(plans),
                "mean_plan_latency_s": round(sum(p["latency_s"] for p in plans) / len(plans), 3) if plans else None,
                "commits": len(commits),
                "accepted_commits": sum(1 for c in commits if c["accepted"]),
                "images": len(self.tracer.select("image")),
                "checks": len(self.tracer.select("check")),
                "refused": len(self.tracer.select("image_refused")),
                "error": repr(self.agent.error) if self.agent and self.agent.error else None,
            },
            "final": final,
            "per_shot": shots,
        }
