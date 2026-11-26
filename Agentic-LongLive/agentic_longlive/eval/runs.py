# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finished runs and the benchmark stories they belong to.

A run directory is the ``output_dir`` of ``scripts/run_story.py``: ``summary.json`` (``per_shot[].video_path``),
``config.json`` (``story``: the story file) and ``shots/shot_NNN.mp4``. A *method root* is a directory holding one
run directory per story (``outputs/<method>/<story>/summary.json``), or a single run directory.

Runs are matched to benchmark stories by the stem of the story file named in ``config.json`` (``collie_muddy_day``);
the story is then loaded from ``--stories`` so that the hidden annotations always come from the benchmark files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..story import Story

SUFFIXES = ("_terse", "_full")


@dataclass
class Run:
    dir: Path
    key: str  # story key: the benchmark file stem, e.g. "collie_muddy_day" or "hd03_528390"
    story: Story
    videos: Dict[int, str] = field(default_factory=dict)  # shot index -> mp4 path
    cache: str = "eval"  # folder under the run that holds the judge records (one per judge model)

    @property
    def eval_dir(self) -> Path:
        return self.dir / self.cache


def find_runs(root: str | Path) -> List[Path]:
    """Run directories under a method root (or the root itself if it is a run directory)."""
    root = Path(root)
    if (root / "summary.json").is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file()) if root.is_dir() else []


def _config_story(run_dir: Path) -> Optional[str]:
    cfg = run_dir / "config.json"
    if cfg.is_file():
        return json.loads(cfg.read_text()).get("story") or None
    return None


def _strip(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_story(run_dir: str | Path, stories: Optional[str | Path] = None) -> tuple:
    """(story key, story file) of a run; the file is taken from ``stories`` when it holds the story."""
    run_dir = Path(run_dir)
    named = _config_story(run_dir)
    candidates = []
    if named:
        candidates.append(Path(named).stem)
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("story"):
        candidates.append(str(summary["story"]))
    candidates.append(run_dir.name)
    if stories is not None:
        for name in candidates:
            for key in (name, _strip(name)):
                path = Path(stories) / f"{key}.json"
                if path.is_file():
                    return key, path
    if named:  # no benchmark directory given (or the story is not in it): the file the run was generated from
        for path in (Path(named), run_dir / named, Path.cwd() / named):
            if path.is_file():
                return _strip(path.stem), path
    raise FileNotFoundError(f"cannot find the story of run {run_dir} (looked for {candidates} in {stories})")


def shot_videos(run_dir: str | Path) -> Dict[int, str]:
    """shot index -> video file; falls back to ``<run>/shots/<name>`` for runs copied from another machine."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    out: Dict[int, str] = {}
    for record in summary.get("per_shot") or []:
        i = int(record["shot"])
        path = Path(record.get("video_path") or "")
        for candidate in (path, run_dir / "shots" / path.name, run_dir / "shots" / f"shot_{i:03d}.mp4"):
            if candidate.name and candidate.is_file():
                out[i] = str(candidate)
                break
    return out


def load_run(run_dir: str | Path, stories: Optional[str | Path] = None, cache: str = "eval") -> Run:
    run_dir = Path(run_dir)
    key, path = resolve_story(run_dir, stories)
    return Run(dir=run_dir, key=key, story=Story.load(path), videos=shot_videos(run_dir), cache=cache)


def is_hard(key: str) -> bool:
    """LongStory-Hard stories are ``hd00`` .. ``hd15``; the other nine are hand-written."""
    return key.startswith("hd")
