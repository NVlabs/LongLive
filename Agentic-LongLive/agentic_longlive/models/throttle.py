# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency limit + minimum spacing between calls, shared by all calls of one provider."""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from contextlib import contextmanager


class Throttle:
    """At most ``max_concurrency`` calls at once, at least ``min_interval_s`` apart. Waiting calls are admitted by
    priority (lower first), then in arrival order, so urgent work skips queued background work."""

    def __init__(self, max_concurrency: int = 4, min_interval_s: float = 0.0):
        self._slots = max(1, int(max_concurrency))
        self._busy = 0
        self._waiting: list = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._interval = float(min_interval_s)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self, priority: int = 0) -> None:
        ticket = (priority, next(self._seq))
        with self._cv:
            heapq.heappush(self._waiting, ticket)
            while self._busy >= self._slots or self._waiting[0] != ticket:
                self._cv.wait()
            heapq.heappop(self._waiting)
            self._busy += 1
            self._cv.notify_all()
        if self._interval > 0:
            with self._lock:
                now = time.monotonic()
                wait = self._next - now
                self._next = max(now, self._next) + self._interval
            if wait > 0:
                time.sleep(wait)

    def release(self) -> None:
        with self._cv:
            self._busy -= 1
            self._cv.notify_all()

    @contextmanager
    def slot(self, priority: int = 0):
        self.acquire(priority)
        try:
            yield self
        finally:
            self.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class RetryAfter(RuntimeError):
    """Transient error carrying a server-suggested wait (e.g. HTTP 429)."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after
