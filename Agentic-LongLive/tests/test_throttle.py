# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_longlive.models.throttle import Throttle  # noqa: E402


def test_queued_calls_are_admitted_by_priority_then_arrival():
    throttle, order = Throttle(max_concurrency=1), []
    throttle.acquire()  # the slot is busy

    def call(name, priority):
        with throttle.slot(priority):
            order.append(name)

    threads = []
    for name, priority in (("entity-1", 1), ("entity-2", 1), ("compose", 0)):
        t = threading.Thread(target=call, args=(name, priority))
        t.start()
        threads.append(t)
        time.sleep(0.05)  # fix the arrival order
    throttle.release()
    for t in threads:
        t.join(2)
    assert order == ["compose", "entity-1", "entity-2"]


def test_concurrency_limit_holds():
    throttle, active, peak, lock = Throttle(max_concurrency=2), [0], [0], threading.Lock()

    def call():
        with throttle:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(2)
    assert peak[0] == 2
