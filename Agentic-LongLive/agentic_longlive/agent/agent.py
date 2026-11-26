# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The memory agent: turns revealed prompts into new committed memory snapshots.

For the newest revealed prompt ``w`` (while the video model renders shot ``w``):

1. the policy writes one plan (state update + all memory text) from revealed information only; if the
   agent fell behind, the same call integrates every prompt revealed since its last update;
2. the entity store applies it (evidence, persistent/transient split, revision);
3. G and M are composed and refined for up to R rounds ("anytime"): round 0 starts at once from the
   current entity images and the new state text, later rounds use the fresh entity images or the
   checker's feedback. Round 0 is committed at once whatever the check says (``commit_first_round``), so
   memory is never older than the newest finished update; every later round that passes its check and is not
   worse replaces it immediately, so the video loop takes whatever is ready at the next shot boundary;
4. entities whose drawn state changed are redrawn from their identity anchor in the background, queued
   right after the round-0 composition so they never delay it on a shared image server.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait as wait_futures
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

from ..config import RunConfig
from ..memory.entity import Entity
from ..memory.snapshot import MemoryImage, MemorySnapshot, SnapshotBoard
from ..memory.store import EntityStore
from ..memory.update import MemoryPlan
from ..models.base import ImageClient, LLMClient
from ..runtime.trace import Tracer
from ..story import AgentView, StorySetup
from . import prompts
from .painter import Drawn, Painter
from .policy import LLMPolicy, select_members


class MemoryAgent:
    def __init__(self, cfg: RunConfig, planner: LLMClient, checker: LLMClient, image: ImageClient,
                 board: SnapshotBoard, out_dir: Path, tracer: Tracer):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.mcfg = cfg.memory
        self.split = cfg.memory.persistent_transient_split
        self.store = EntityStore(split=self.split, revision=cfg.memory.revision)
        self.policy = LLMPolicy(planner, cfg.agent, cfg.memory)
        self.painter = Painter(image, checker, cfg.agent, cfg.memory, out_dir / "memory", tracer)
        self.board = board
        self.tracer = tracer
        self.out_dir = out_dir
        self.g: Optional[MemoryImage] = None
        self.g_key: Optional[Tuple] = None
        self.m_history: List[MemoryImage] = []
        self.observations: Dict[int, Path] = {}
        self.error: Optional[BaseException] = None
        self._pending: Deque[AgentView] = deque()
        self._cv = threading.Condition()
        self._stop = False
        self._busy = False
        self._latest_revealed = -1
        self._last_planned = -1  # newest prompt already integrated into the entity store
        self._painting: Dict[str, Tuple[str, Future]] = {}  # entity id -> (signature, in-flight redraw)
        self._paint_lock = threading.Lock()
        # Entity redraws are background work: they leave two image slots to G/M composition, and queued redraws
        # yield to composition (image priority). At least two painters keep the image model busy while one checks.
        self._paint_pool = ThreadPoolExecutor(max_workers=max(2, image.spec.max_concurrency - 2),
                                              thread_name_prefix="entity-painter")
        self._compose_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="composer")
        self._thread = threading.Thread(target=self._run, name="memory-agent", daemon=True)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread.start()

    def submit(self, view: AgentView) -> None:
        with self._cv:
            self._pending.append(view)
            self._latest_revealed = max(self._latest_revealed, view.latest)
            self._cv.notify_all()

    def observe(self, shot: int, frame: Path) -> None:
        self.observations[shot] = frame

    def stop(self, wait: bool = True, timeout: Optional[float] = None, drain: bool = False) -> None:
        """Stop the agent. Without ``drain`` pending prompts are dropped (nothing will use their memory)."""
        with self._cv:
            self._stop = True
            if not drain:
                self._pending.clear()
            self._cv.notify_all()
        if wait and self._thread.is_alive():
            self._thread.join(timeout)
        for pool in (self._compose_pool, self._paint_pool):
            pool.shutdown(wait=wait, cancel_futures=not drain)

    def idle(self) -> bool:
        with self._cv:
            return not self._pending and not self._busy

    def _newer_available(self, index: int) -> bool:
        with self._cv:
            return self._latest_revealed > index

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if not self._pending and self._stop:
                    return
                views = list(self._pending)
                self._pending.clear()
                self._busy = True
            try:
                if self.acfg.batch_catch_up:
                    views = views[-1:]  # one plan integrates every prompt since the last update
                for view in views:
                    self.process(view)
            except BaseException as exc:  # surfaced to the session
                self.error = exc
                self.tracer.log("agent_error", error=repr(exc))
                if self.cfg.runtime.stop_on_agent_error:
                    with self._cv:
                        self._busy = False
                    return
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    # ------------------------------------------------------------- bootstrap
    def bootstrap(self, setup: StorySetup) -> Optional[MemorySnapshot]:
        """Session setup: identity anchors of the cast and the first G (memory for shot 0)."""
        mode = self.mcfg.mode
        if mode == "none":
            return None
        cast = self.store.seed_cast(setup, self.policy.split_cast(setup) if self.mcfg.cast_split else None)
        self.tracer.log("bootstrap_start", cast=cast, mode=mode,
                        cast_state={i: {"identity": self.store.get(i).description,
                                        "facts": self.store.get(i).visual_state(self.split)} for i in cast})
        if mode in ("entity", "anchor"):
            self._paint(cast, appearance={}, shot=-1)
        g = None
        if self.mcfg.use_g and cast:
            members = [self.store.get(i) for i in cast]
            if mode != "text" and len(members) == 1 and members[0].latest is not None:
                front = members[0].latest.primary
                g = MemoryImage("G", front, self._caption("G", "", members, None), tuple(cast), -1, 0, None)
            else:
                text = "The main characters of the story: " + "; ".join(m.describe(self.split) for m in members)
                drawn = self._best_of(lambda fb: self.painter.compose("G", text, members, None, "w-01", feedback=fb))
                if drawn.path is not None:
                    g = MemoryImage("G", str(drawn.path), self._caption("G", "", members, None), tuple(cast), -1,
                                    drawn.attempts - 1, drawn.verdict.score)
        self.g = g
        self.g_key = self._g_key(cast)
        snap = MemorySnapshot(version=self.board.next_version(), watermark=-1, g=g, m=(),
                              references=tuple(self.store.references(cast, self.mcfg.max_reference_images,
                                                                     self.mcfg.multi_view)),
                              round=0, score=g.score if g else None)
        self.board.commit(snap)
        self.board.mark_complete(-1)
        self.store.save(self.out_dir / "memory" / "store" / "setup.json")
        self.tracer.log("bootstrap_done", snapshot=snap.summary())
        return snap

    # --------------------------------------------------------------- process
    def process(self, view: AgentView) -> None:
        w = view.latest
        since = self._last_planned
        observations = None
        if self.acfg.read_generated_video:
            observations = [self.observations[s] for s in sorted(self.observations) if s < w][-2:]
        call = self.policy.plan(view, self.store, observations, since=since)
        applied = self.store.apply(call.plan.update, w)
        self._last_planned = w
        self.tracer.log("plan", shot=w, since=since, fingerprint=view.fingerprint(), revealed=len(view.revealed),
                        oracle=view.next_prompt is not None, system=call.system, user=call.user,
                        images=call.images, output=call.raw_text, model=call.model, latency_s=call.latency_s,
                        applied=asdict(applied))
        self.store.save(self.out_dir / "memory" / "store" / f"w{w:03d}.json")
        if w - since > 1:
            self.tracer.log("catch_up", shot=w, prompts=list(range(since + 1, w + 1)))
        start_painting: Optional[Callable[[], Dict[str, Future]]] = None
        if self.mcfg.mode == "entity":
            ids, appearance = self.store.needs_image(), call.plan.image_prompts
            if self.acfg.pipeline_entities:
                start_painting = lambda: self._paint_async(ids, appearance, w)  # noqa: E731
            else:
                wait_futures(list(self._paint_async(ids, appearance, w).values()))
        g_members, m_members = select_members(self.acfg.policy, call.plan, self.store, self.acfg)
        self._refine_and_commit(w, call.plan, g_members, m_members, start_painting, prompt=view.prompt)

    # ---------------------------------------------------------------- helpers
    def _paint(self, ids: List[str], appearance: Dict[str, str], shot: int) -> None:
        wait_futures(list(self._paint_async(ids, appearance, shot).values()))

    def _paint_async(self, ids: List[str], appearance: Dict[str, str], shot: int) -> Dict[str, Future]:
        """Redraw entities in the background; an identical in-flight redraw is reused, not repeated."""
        futures: Dict[str, Future] = {}
        for eid in ids:
            entity = self.store.get(eid)
            if entity is None:
                continue
            signature = entity.signature(self.split)
            with self._paint_lock:
                running = self._painting.get(eid)
                if running is not None and running[0] == signature and not running[1].done():
                    futures[eid] = running[1]
                    continue
            # snapshot the state now; the planner's free-text image prompt is followed by the authoritative state
            state = entity.visual_state(self.split)
            text = appearance.get(eid)
            if text and state:
                text += " Current state (must be shown): " + "; ".join(f"{k}: {v}" for k, v in state.items()) + "."
            text = text or entity.describe(self.split)
            future = self._paint_pool.submit(self._paint_job, eid, text, shot, signature, state)
            with self._paint_lock:
                self._painting[eid] = (signature, future)
            futures[eid] = future
        return futures

    def _paint_job(self, eid: str, text: str, shot: int, signature: str, state: Optional[Dict[str, str]] = None) -> bool:
        entity = self.store.get(eid)
        version = self.painter.paint_entity(entity, text, shot, signature, state)
        stale = version is not None and entity.latest is not None and entity.latest.shot > shot
        if version is not None and not stale:
            self.store.add_image(eid, version)
        self.tracer.log("entity_painted", entity=eid, shot=shot, ok=version is not None and not stale, stale=stale,
                        views=version.views if version else None)
        return version is not None and not stale

    def _wait_redraws(self, futures: List[Future], w: int) -> bool:
        """Wait for entity redraws; give up (False) as soon as a newer prompt is revealed, since later rounds of
        update ``w`` could then only reach the video after the newer update. The redraws keep running."""
        pending = [f for f in futures if not f.done()]
        while pending:
            if self.acfg.stop_refine_on_new_prompt and self._newer_available(w):
                return False
            wait_futures(pending, timeout=0.1)
            pending = [f for f in pending if not f.done()]
        return True

    def _image_versions(self, members: List[Optional[str]]) -> Tuple:
        out = []
        for i in members:
            e = self.store.get(i) if i else None
            out.append((i, e.latest.version if (e is not None and e.latest is not None) else None))
        return tuple(out)

    def _g_key(self, members: List[str]) -> Tuple:
        sigs = []
        for i in members:
            e = self.store.get(i)
            sigs.append(e.latest.signature if (e and e.latest and self.mcfg.mode == "entity") else
                        (e.signature(self.split) if e else ""))
        return tuple(members), tuple(sigs)

    def _caption(self, role: str, written: str, members: List[Entity], scene: Optional[Entity]) -> str:
        if role == "G" and self.acfg.neutral_g:  # member/state text plus the backdrop (prompts.NEUTRAL_G_*)
            return "; ".join(m.describe(self.split) for m in members) + ". " + prompts.NEUTRAL_G_CAPTION_TAIL
        if written:
            return written
        text = "; ".join(m.describe(self.split) for m in members)
        if role == "M" and scene is not None:
            return f"In {scene.name}: {text}" if text else f"The {scene.name}."
        return text

    def _best_of(self, compose, rounds: Optional[int] = None) -> Drawn:
        best: Optional[Drawn] = None
        feedback: List[str] = []
        for _ in range(rounds or self.acfg.refine_rounds):
            drawn = compose(feedback)
            if best is None or drawn.verdict.score > best.verdict.score:
                best = drawn
            if drawn.verdict.passed and not drawn.verdict.issues:
                break
            feedback = drawn.verdict.issues
        return best

    def _refine_and_commit(self, w: int, plan: MemoryPlan, g_members: List[str], m_members: List[str],
                           start_painting: Optional[Callable[[], Dict[str, Future]]] = None, prompt: str = "") -> None:
        mode = self.mcfg.mode
        scene = self.store.get(self.store.scene) if self.store.scene else None
        g_ents = [self.store.get(i) for i in g_members]
        m_ents = [self.store.get(i) for i in m_members]
        g_key = self._g_key(g_members)
        need_g = self.mcfg.use_g and mode != "anchor" and (self.g is None or g_key != self.g_key)
        need_m = self.mcfg.m_slots > 0 and mode in ("entity", "text")
        prev_m = self.m_history[-1] if self.m_history else None
        g_wait: List[Future] = []
        m_wait: List[Future] = []
        used = {"G": self._image_versions(g_members), "M": self._image_versions(m_members + [self.store.scene])}
        best_g: Optional[MemoryImage] = None
        best_m: Optional[MemoryImage] = None
        fb_g: List[str] = []
        fb_m: List[str] = []
        rank: Dict[str, Tuple[bool, float]] = {}  # role -> (passed, score) of the current best
        # Round 0 of this role is committed whatever the check says (agent.commit_first_round): no lag from checks.
        first_role = ("M" if need_m else "G") if self.acfg.commit_first_round else None
        for rnd in range(self.acfg.refine_rounds):
            if self._stop:
                break
            if rnd > 0 and self.acfg.stop_refine_on_new_prompt and self._newer_available(w):
                self.tracer.log("refine_stopped", shot=w, round=rnd, reason="newer prompt revealed")
                break
            fresh = {}
            if rnd > 0:  # wait for the background redraws of this round's members, then use them
                if not self._wait_redraws(g_wait + m_wait, w):
                    self.tracer.log("refine_stopped", shot=w, round=rnd,
                                    reason="newer prompt revealed while waiting for entity redraws")
                    break
                g_key = self._g_key(g_members)
                fresh = {"G": self._image_versions(g_members) != used["G"],
                         "M": self._image_versions(m_members + [self.store.scene]) != used["M"]}
                need_g = need_g or (self.mcfg.use_g and mode != "anchor" and g_key != self.g_key)
            jobs = {}
            if need_g and (rnd == 0 or fb_g or best_g is None or fresh.get("G")):
                extra = [Path(self.g.path)] if (mode == "text" and self.g) else []
                used["G"] = self._image_versions(g_members)
                jobs["G"] = self._compose_pool.submit(self.painter.compose, "G", plan.g_prompt or "The protagonists.",
                                                      g_ents, None, f"w{w:03d}_r{rnd}", extra,
                                                      [] if fresh.get("G") else list(fb_g))
            if need_m and (rnd == 0 or fb_m or best_m is None or fresh.get("M")):
                extra = []
                if mode == "text":
                    extra = [Path(p.path) for p in (self.g, prev_m) if p is not None]
                used["M"] = self._image_versions(m_members + [self.store.scene])
                jobs["M"] = self._compose_pool.submit(self.painter.compose, "M", plan.m_prompt or "The current scene.",
                                                      m_ents, scene, f"w{w:03d}_r{rnd}", extra,
                                                      [] if fresh.get("M") else list(fb_m))
            if rnd == 0 and start_painting is not None:
                # Entity redraws only feed later rounds, so they queue behind the round-0 composition.
                painting, start_painting = start_painting(), None
                g_wait = [painting[i] for i in g_members if i in painting]
                m_wait = [painting[i] for i in m_members + ([self.store.scene] if self.store.scene else [])
                          if i in painting]
            if not jobs:
                break
            self.tracer.log("refine_round", shot=w, round=rnd, roles=sorted(jobs),
                            fresh=sorted(k for k, v in fresh.items() if v))
            improved = False
            verdicts: Dict[str, Optional[Dict]] = {}
            role_of = {future: role for role, future in jobs.items()}
            for future in as_completed(role_of):  # G and M are used as soon as each is ready
                role = role_of[future]
                drawn: Drawn = future.result()
                verdicts[role] = drawn.verdict.to_json() if drawn.path is not None else None
                if drawn.path is None:
                    continue
                members = tuple(g_members if role == "G" else m_members)
                image = MemoryImage(role, str(drawn.path),
                                    self._caption(role, plan.g_caption if role == "G" else plan.m_caption,
                                                  g_ents if role == "G" else m_ents, scene if role == "M" else None),
                                    members, w, rnd, drawn.verdict.score,
                                    scene=scene.name if (role == "M" and scene is not None) else "",
                                    prompt=prompt if role == "M" else "")
                # Fresh entity images make a newer memory: accept it even at a lower score, but never let an
                # image that failed its check replace one that passed.
                newer = bool(fresh.get(role))
                new_rank, cur = (drawn.verdict.passed, drawn.verdict.score), rank.get(role)
                forced = rnd == 0 and role == first_role
                accepted = (forced or self.painter.acceptable(drawn.verdict)) and (
                    cur is None or new_rank >= cur or (newer and (new_rank[0] or not cur[0])))
                if accepted:
                    rank[role] = new_rank
                    if role == "G":
                        best_g = image
                    else:
                        best_m = image
                    improved = True
                issues = drawn.verdict.issues if not drawn.verdict.passed or drawn.verdict.score < 9 else []
                if role == "G":
                    fb_g = issues
                else:
                    fb_m = issues
                # Commit at once: a slow G (e.g. an image API retry) never holds back a ready M, and vice versa.
                if accepted and (not need_m or best_m is not None):
                    ready_g = best_g or (self.g if self.mcfg.use_g else None)
                    self._commit(w, rnd, ready_g, best_m, m_members, passed=all(r[0] for r in rank.values()))
            self.tracer.log("round_done", shot=w, round=rnd, improved=improved, verdicts=verdicts,
                            forced=sorted(r for r in jobs if rnd == 0 and r == first_role))
        if start_painting is not None and not self._stop:  # no composition round ran
            start_painting()
        if best_m is not None:
            self.m_history.append(best_m)
        if best_g is not None:
            self.g, self.g_key = best_g, g_key
        self.board.mark_complete(w)
        self.tracer.log("watermark_done", shot=w, g=best_g.path if best_g else None, m=best_m.path if best_m else None,
                        g_reused=not need_g)

    def _commit(self, w: int, rnd: int, g: Optional[MemoryImage], m: Optional[MemoryImage], m_members: List[str],
                passed: Optional[bool] = None) -> None:
        """Commit a snapshot; the caller has already decided that it is not worse than this watermark's last one."""
        slots = self.mcfg.m_slots
        history = self.m_history[-(slots - 1):] if slots > 1 else []
        m_tuple = tuple(history + ([m] if m is not None else [])) if slots else ()
        scores = [s for s in ((g.score if g and g.watermark == w else None), (m.score if m else None)) if s is not None]
        refs = self.store.protagonists() + [i for i in m_members if i not in self.store.protagonists()]
        if self.store.scene:
            refs.append(self.store.scene)
        snap = MemorySnapshot(version=self.board.next_version(), watermark=w, g=g, m=m_tuple,
                              references=tuple(self.store.references(refs, self.mcfg.max_reference_images,
                                                                     self.mcfg.multi_view)),
                              round=rnd, score=min(scores) if scores else None, passed=passed, replaces=True)
        accepted = self.board.commit(snap)
        self.tracer.log("commit", shot=w, round=rnd, accepted=accepted, snapshot=snap.summary())
