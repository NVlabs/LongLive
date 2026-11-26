# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drawing memory: entity images (layer 1) and composed G/M memory images (layer 2), with checks."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from ..config import AgentConfig, MemoryConfig
from ..memory.entity import Entity, ImageVersion
from ..models.base import ImageClient, ImageRefused, LLMClient
from ..runtime.trace import Tracer
from . import prompts


@dataclass
class Verdict:
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    state_ok: bool = True  # the required persistent state is shown (a state error is never a "minor flaw")

    def to_json(self) -> Dict:
        return {"passed": self.passed, "score": self.score, "issues": self.issues, "state_ok": self.state_ok}


@dataclass
class Drawn:
    path: Optional[Path]
    verdict: Verdict
    prompt: str
    attempts: int


class Painter:
    def __init__(self, image: ImageClient, checker: LLMClient, cfg: AgentConfig, memory: MemoryConfig,
                 out_dir: Path, tracer: Tracer):
        self.image = image
        self.checker = checker
        self.cfg = cfg
        self.memory = memory
        self.out_dir = out_dir
        self.tracer = tracer
        self._counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ primitives
    def _path(self, *parts: str) -> Path:
        path = self.out_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def acceptable(self, verdict: Verdict) -> bool:
        """May this image be committed? Passing, or failing with only minor flaws (agent.commit_min_score). An image
        whose checker reports a wrong persistent state (state_ok false) is never acceptable."""
        if not verdict.state_ok:
            return False
        floor = self.cfg.commit_min_score
        return verdict.passed or (floor is not None and verdict.score >= floor)

    def check(self, candidate: Path, spec: str, refs: Sequence[Path]) -> Verdict:
        if not self.cfg.check_images:
            return Verdict(True, 10.0)
        # every image is preceded by its label: unlabelled, the VLM sometimes judged a reference (e.g. the beard of an
        # older entity image) instead of the candidate
        content: List = [prompts.check_user(spec, len(refs)), "CANDIDATE (judge this image):", candidate]
        for i, ref in enumerate(refs, 1):
            content += [f"REFERENCE {i} (identity only; it may show an older state):", ref]
        result = self.checker.chat(prompts.CHECK_SYSTEM, content, json_mode=True, temperature=0.0)
        data = result.data or {}
        state_ok = data.get("state_ok", True)
        state_ok = state_ok if isinstance(state_ok, bool) else str(state_ok).strip().lower() not in {"false", "no", "0"}
        verdict = Verdict(bool(data.get("pass", False)) and state_ok, float(data.get("score", 0) or 0),
                          [str(i) for i in data.get("issues", []) or []][:6], state_ok)
        self.tracer.log("check", candidate=str(candidate), spec=spec, refs=[str(r) for r in refs],
                        verdict=verdict.to_json(), model=result.model, latency_s=result.latency_s)
        return verdict

    def draw(self, spec: str, refs: Sequence[Path], target: Callable[[int], Path], rounds: int,
             stop: Optional[Callable[[], bool]] = None, feedback: Sequence[str] = (), priority: int = 0,
             gen: Optional[str] = None) -> Drawn:
        """Generate, check and repair up to ``rounds`` extra times; returns the best attempt. G/M composition is on
        the video's critical path (priority 0); entity redraws are background work (priority 1). ``gen``: generation
        prompt when it differs from the checked specification (an edit instruction)."""
        best: Optional[Drawn] = None
        issues = list(feedback)
        for attempt in range(rounds + 1):
            base = gen or spec
            prompt = base if not issues else base + "\nCorrections: " + "; ".join(issues)
            try:
                result = self.image.generate(prompt, refs, self.cfg.image_size, priority=priority)
            except ImageRefused as exc:
                self.tracer.log("image_refused", spec=spec, error=str(exc)[:300])
                break
            path = target(attempt)
            path.write_bytes(result.png)
            self.tracer.log("image", path=str(path), prompt=prompt, refs=[str(r) for r in refs],
                            model=result.model, latency_s=result.latency_s, meta=result.meta)
            verdict = self.check(path, spec, refs)
            if best is None or (verdict.state_ok, verdict.passed, verdict.score) > \
                    (best.verdict.state_ok, best.verdict.passed, best.verdict.score):
                best = Drawn(path, verdict, prompt, attempt + 1)
            if verdict.passed or (stop and stop()):
                break
            issues = verdict.issues or ["follow the specification more closely"]
        return best or Drawn(None, Verdict(False, 0.0, ["refused"]), spec, 0)

    # --------------------------------------------------------------- entities
    def views(self, entity: Entity) -> List[str]:
        if entity.kind == "scene":
            return ["plate"]
        return ["front", "side"] if self.memory.multi_view else ["front"]

    def paint_entity(self, entity: Entity, appearance: str, shot: int, signature: str,
                     state: Optional[Dict[str, str]] = None) -> Optional[ImageVersion]:
        """Draw every view of the entity's current state, conditioned on its identity anchor. ``state`` is the visual
        state being drawn: where it differs from what the reference images show, the change is written out."""
        version = len(entity.images)
        state = dict(state or {})
        base_refs: List[Path] = []
        ref_versions = []
        if entity.anchor is not None:
            base_refs.append(Path(entity.anchor.primary))
            ref_versions.append(entity.anchor)
        if entity.latest is not None and entity.latest is not entity.anchor:
            base_refs.append(Path(entity.latest.primary))
            ref_versions.append(entity.latest)
        changes = entity.state_changes(state, ref_versions)  # vs. everything the references show (checked)
        # what the latest image must change: drawn as an edit of that image (the image model keeps the references'
        # state when given a full description)
        edits = entity.state_changes(state, [entity.latest]) if entity.latest is not None else []
        # The check is against memory only (identity + persistent state): the free-text appearance written by the
        # planner may mention a pose, a place or a held prop, which an entity image deliberately leaves neutral.
        required = f"{entity.name}: {entity.description}"
        if state:
            required += " Current state: " + "; ".join(f"{k}: {v}" for k, v in state.items()) + "."
        views: Dict[str, str] = {}
        checks: Dict[str, Dict] = {}
        prompt0 = ""
        for view in self.views(entity):
            spec = prompts.entity_spec(entity.name, required, view, entity.kind, changes)
            gen = (prompts.entity_edit(entity.name, edits, view, entity.kind) if edits else
                   prompts.entity_spec(entity.name, appearance, view, entity.kind, changes))
            if edits:  # the same view of the latest image only (with the anchor too, edits sometimes reverted other
                # state to the anchor's, e.g. the first T-shirt), then the new first view
                refs = [Path(entity.latest.views.get(view, entity.latest.primary))]
                refs += [Path(p) for p in list(views.values())[:1]]
            else:
                refs = base_refs + [Path(p) for p in list(views.values())[:1]]  # later views follow the first view
            tag = f"v{version:03d}_w{shot:03d}_{view}"  # one file per attempt (a failed version number is reused)
            drawn = self.draw(spec, refs, lambda a, t=tag: self._path("entities", entity.id, f"{t}_a{a}.png"),
                              self.cfg.entity_repair_rounds, priority=1, gen=gen)
            anchor = entity.anchor
            if edits and not self.acceptable(drawn.verdict) and anchor is not None and anchor is not entity.latest:
                # Returning to an earlier look (soaked -> clean, muddy -> washed) often fails as an edit of the latest
                # image but works as an edit of the identity anchor, which may already show it.
                back = entity.state_changes(state, [anchor])
                if back:
                    refs = [Path(anchor.views.get(view, anchor.primary))] + [Path(p) for p in list(views.values())[:1]]
                    retry = self.draw(spec, refs, lambda a, t=tag: self._path("entities", entity.id, f"{t}_b{a}.png"),
                                      self.cfg.entity_repair_rounds, priority=1,
                                      gen=prompts.entity_edit(entity.name, back, view, entity.kind))
                    self.tracer.log("entity_anchor_retry", entity=entity.id, shot=shot, view=view,
                                    verdict=retry.verdict.to_json())
                    if retry.path is not None and (retry.verdict.state_ok, retry.verdict.passed, retry.verdict.score) > \
                            (drawn.verdict.state_ok, drawn.verdict.passed, drawn.verdict.score):
                        drawn = retry
            if drawn.path is None:
                return None
            if not self.acceptable(drawn.verdict) and entity.latest is not None:
                # Keep the previous checked image rather than committing an unchecked change.
                self.tracer.log("entity_kept_previous", entity=entity.id, shot=shot, verdict=drawn.verdict.to_json())
                return None
            views[view] = str(drawn.path)
            checks[view] = drawn.verdict.to_json()
            prompt0 = prompt0 or drawn.prompt
        return ImageVersion(version=version, views=views, shot=shot, signature=signature, prompt=prompt0, check=checks,
                            state=state)

    # ------------------------------------------------------------ composition
    def compose(self, role: str, text: str, members: List[Entity], scene: Optional[Entity], tag: str,
                extra_refs: Sequence[Path] = (), feedback: Sequence[str] = ()) -> Drawn:
        """Compose G or M from entity images (or from text only when no entity images exist)."""
        refs: List[Path] = []
        if role == "M" and scene is not None and scene.latest is not None:
            refs.append(Path(scene.latest.primary))
        refs += [Path(m.latest.primary) for m in members if m.latest is not None]
        refs += list(extra_refs)
        split = self.memory.persistent_transient_split
        names = [m.describe(split) for m in members]
        if role == "M" and scene is not None:
            names.insert(0, "Location: " + scene.describe(split))
        # members whose entity image was drawn before their latest state change (redraw still pending or failed)
        changes = [(f"{m.name}'s {attr}", old, new) for m in members if m.latest is not None
                   for attr, old, new in m.state_changes(m.visual_state(split), [m.latest])]
        spec = prompts.compose_spec(role, text, names, changes, self.cfg.neutral_g)
        return self.draw(spec, refs, lambda a: self._path(role, f"{tag}_a{a}.png"), 0, feedback=feedback)
