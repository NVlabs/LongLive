# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongStory-Online evaluation (``scripts/evaluate.py``).

* ``frames``  frame sampling of shot videos (decord or imageio-ffmpeg);
* ``judge``   the VLM judge prompts and calls (shot judge, place / protagonist pair judges, plan visibility);
* ``measure`` per-shot measurements of a run, cached in ``<run>/eval/``;
* ``metrics`` Table 1 metrics, pooling over shots and seeds, story-clustered bootstrap for paired differences;
* ``quality`` optional VBench imaging quality (torch + pyiqa);
* ``mir``     optional ArcFace MIR (insightface).
"""

from .judge import build_judge, judge_spec
from .measure import load_measurements, measure_run
from .metrics import TABLE, paired, run_scores, summarize
from .runs import Run, find_runs, load_run

__all__ = ["TABLE", "Run", "build_judge", "find_runs", "judge_spec", "load_measurements", "load_run", "measure_run",
           "paired", "run_scores", "summarize"]
