# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer 2 of the memory and its versioned hand-off to the video generator.

The agent publishes immutable :class:`MemorySnapshot` versions on a :class:`SnapshotBoard`.
The video loop never waits: at a shot boundary it takes the newest snapshot whose prefill is
ready. A snapshot bundles G and the recent M images so they are always swapped together.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class MemoryImage:
    role: str  # "G" (long-term) or "M" (short-term, rolling)
    path: str
    caption: str  # text condition used when this image is prefilled into the sink
    members: Tuple[str, ...] = ()
    watermark: int = -1  # last revealed prompt index this image was built from
    round: int = 0
    score: Optional[float] = None
    scene: str = ""  # name of the location shown (M only)
    prompt: str = ""  # the revealed shot prompt this image was built from (M only; empty for setup images)


def _rank(snapshot: MemorySnapshot) -> Tuple[bool, float]:
    """Same-watermark order: a checked (passing) version beats a failing one, then the higher score."""
    return snapshot.passed is not False, snapshot.score or 0.0


@dataclass(frozen=True)
class Reference:
    """One entity image for reference-injection backends (e.g. H3 Ref2VA)."""

    entity: str
    name: str
    kind: str  # character | object | scene
    view: str
    path: str
    description: str  # identity + persistent state at commit time

    @property
    def label(self) -> str:
        return f"{self.name} ({self.view})"


@dataclass(frozen=True)
class MemorySnapshot:
    version: int
    watermark: int  # memory was built from prompts 0..watermark (-1 = session setup only)
    g: Optional[MemoryImage] = None
    m: Tuple[MemoryImage, ...] = ()  # oldest -> newest, at most m_slots
    references: Tuple[Reference, ...] = ()  # entity images for reference injection, grouped by entity
    round: int = 0
    score: Optional[float] = None
    created: float = field(default_factory=time.time)
    complete: bool = False  # all refinement rounds for this watermark have finished
    passed: Optional[bool] = None  # every image of this watermark passed its check (None: unchecked / setup)
    replaces: bool = False  # the agent verified this version is not worse than the one it replaces (same watermark)

    @property
    def blocks(self) -> List[MemoryImage]:
        """Sink order: [G, M_{w-1}, M_w]."""
        return ([self.g] if self.g else []) + list(self.m)

    def summary(self) -> dict:
        return {
            "version": self.version, "watermark": self.watermark, "round": self.round,
            "score": self.score, "complete": self.complete, "passed": self.passed,
            "g": self.g.path if self.g else None, "m": [m.path for m in self.m],
            "references": len(self.references),
        }

    def to_json(self) -> dict:
        """Everything a backend reads (paths, captions, source prompts), so a run's memory can be replayed."""
        return json.loads(json.dumps(asdict(self)))  # plain JSON types: the same in memory and in summary.json

    @staticmethod
    def from_json(data: dict) -> "MemorySnapshot":
        def image(d):
            return None if d is None else MemoryImage(**{**d, "members": tuple(d.get("members") or ())})

        return MemorySnapshot(**{**data, "g": image(data.get("g")), "m": tuple(image(m) for m in data.get("m") or ()),
                                 "references": tuple(Reference(**r) for r in data.get("references") or ())})


class SnapshotBoard:
    """Atomic latest-version board between the agent (writer) and the video loop (reader).

    Commit rule ("checked and not worse"): a snapshot replaces the current one if it was built from
    newer prompts, or from the same prompts with a score that is not lower.
    """

    def __init__(self):
        self._lock = threading.Condition()
        self._history: List[MemorySnapshot] = []
        self._current: Optional[MemorySnapshot] = None
        self._next_version = 0
        self._finished: Optional[int] = None  # newest watermark the agent finished, committed or not
        self._subscribers: List[Callable[[MemorySnapshot], None]] = []

    def next_version(self) -> int:
        with self._lock:
            self._next_version += 1
            return self._next_version

    def subscribe(self, callback: Callable[[MemorySnapshot], None]) -> None:
        self._subscribers.append(callback)

    def commit(self, snapshot: MemorySnapshot) -> bool:
        with self._lock:
            cur = self._current
            accept = (
                cur is None
                or snapshot.watermark > cur.watermark
                or (snapshot.watermark == cur.watermark and (snapshot.replaces or _rank(snapshot) >= _rank(cur)))
            )
            if accept:
                self._current = snapshot
                self._history.append(snapshot)
                self._lock.notify_all()
        if accept:
            for callback in self._subscribers:
                callback(snapshot)
        return accept

    def mark_complete(self, watermark: int) -> None:
        """Signal that refinement for ``watermark`` is finished (used by the sync baseline). The agent may
        finish without committing anything new (every image failed its check); waiters then keep the
        current snapshot instead of waiting forever."""
        with self._lock:
            self._finished = watermark if self._finished is None else max(self._finished, watermark)
            cur = self._current
            if cur is not None and cur.watermark == watermark and not cur.complete:
                done = MemorySnapshot(**{**cur.__dict__, "complete": True})
                self._current = done
                self._history.append(done)
            self._lock.notify_all()

    def latest(self, max_watermark: Optional[int] = None) -> Optional[MemorySnapshot]:
        with self._lock:
            if max_watermark is None:
                return self._current
            for snap in reversed(self._history):
                if snap.watermark <= max_watermark:
                    return snap
            return None

    def wait_for(self, watermark: int, complete: bool, timeout: Optional[float] = None) -> Optional[MemorySnapshot]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while True:
                cur = self._current
                if cur is not None and cur.watermark >= watermark and (cur.complete or not complete or cur.watermark > watermark):
                    return cur
                if self._finished is not None and self._finished >= watermark:
                    return cur
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return cur
                self._lock.wait(remaining)

    def history(self) -> List[MemorySnapshot]:
        with self._lock:
            return list(self._history)
