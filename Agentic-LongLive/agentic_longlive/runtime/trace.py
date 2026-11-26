# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append-only JSONL trace of every agent and runtime event (audit, timing, RL datasets)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Tracer:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.events: List[Dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def now(self) -> float:
        return time.monotonic() - self._t0

    def log(self, kind: str, **fields: Any) -> Dict[str, Any]:
        event = {"t": round(self.now(), 4), "wall": time.time(), "kind": kind,
                 "thread": threading.current_thread().name, **fields}
        with self._lock:
            self.events.append(event)
            if self._fh is not None:
                self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                self._fh.flush()
        return event

    def select(self, kind: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e["kind"] == kind]

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
