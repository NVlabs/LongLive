# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-queue protocol between the agent side and an out-of-process video generator (MiniMax-H3 family).

A queue is a directory with ``jobs/``, ``running/`` and ``done/``. The client writes ``jobs/<id>.json`` atomically and
waits for ``done/<id>.json``; a worker takes the oldest job, renders it and writes the done record. Client and worker
only need a shared directory (same machine, or a shared file system).

Job fields (JSON):
  id                   unique job id (also the file name)
  out                  absolute path of the MP4 to write
  seed                 integer seed
  prompt               the full text prompt, already in the generator's format
  refs                 memory pictures (absolute paths), in the order they enter the generator
  ref_seconds          sink layout: time of each picture on the generator's clock (seconds); empty: no sink
  video_start_seconds  sink layout: time at which the generated video starts (after a cut)
  ref_masks            optional, one foreground mask (PNG path or null) per picture: only foreground tokens are kept
  keyframe             optional, an image the generated video starts from (continuous robot video)
  ref_scale            optional, pictures are cover-resized to this fraction of the video canvas
  duration_seconds     optional, video length (default: the generator's shot length)
  stop                 true: the worker exits

Done record: {"id", "ok", "seconds", "timing" (worker-specific), "error" (when not ok)}.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def ensure(queue: Path) -> Path:
    queue = Path(queue)
    for sub in ("jobs", "running", "done"):
        (queue / sub).mkdir(parents=True, exist_ok=True)
    return queue


def submit(queue: Path, job: Dict[str, Any], timeout_s: float = 3600.0, poll_s: float = 0.05,
           alive: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Enqueue ``job`` and block until its done record exists; raises if the worker reports a failure."""
    queue = ensure(queue)
    job_id = str(job["id"])
    tmp = queue / "jobs" / f".{job_id}.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False))
    os.replace(tmp, queue / "jobs" / f"{job_id}.json")
    done = queue / "done" / f"{job_id}.json"
    start = time.monotonic()
    while not done.exists():
        if alive is not None and not alive():
            raise RuntimeError(f"generator worker for {queue} is not running")
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"job {job_id} timed out after {timeout_s:.0f} s ({queue})")
        time.sleep(poll_s)
    record = json.loads(done.read_text())
    if not record.get("ok"):
        raise RuntimeError(f"job {job_id} failed: {record.get('error')}")
    return record


def serve(queue: Path, handler: Callable[[Dict[str, Any]], Dict[str, Any]], poll_s: float = 0.02,
          ready: Optional[Dict[str, Any]] = None) -> None:
    """Worker loop: ``handler(job) -> timing dict``; runs until a job with ``stop: true`` arrives."""
    queue = ensure(queue)
    (queue / "ready").write_text(json.dumps({**(ready or {}), "time": time.time()}))
    while True:
        jobs = sorted(queue.glob("jobs/*.json"))
        if not jobs:
            time.sleep(poll_s)
            continue
        path = jobs[0]
        try:
            job = json.loads(path.read_text())
            os.replace(path, queue / "running" / path.name)
        except (OSError, ValueError):  # taken by another worker, or still being written
            continue
        if job.get("stop"):
            (queue / "running" / path.name).unlink(missing_ok=True)
            return
        record: Dict[str, Any] = {"id": job["id"], "out": job.get("out")}
        t0 = time.time()
        try:
            record.update(ok=True, timing=handler(job))
        except Exception as exc:  # noqa: BLE001 -- reported to the client
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-3000:])
        record["seconds"] = round(time.time() - t0, 3)
        tmp = queue / "done" / f".{job['id']}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, queue / "done" / f"{job['id']}.json")
        (queue / "running" / path.name).unlink(missing_ok=True)
        print(job["id"], "ok" if record["ok"] else f"FAILED: {record.get('error')}", record["seconds"], flush=True)
