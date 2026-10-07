# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""Task selection and complete-result aggregation, independent of simulators."""

import json
import math
from pathlib import Path

from longwam.paths import repository_root

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def native_plan(cfg):
    if cfg.benchmark == "libero":
        suites = LIBERO_SUITES if cfg.suite == "all" else (str(cfg.suite),)
        if any(suite not in (*LIBERO_SUITES, "libero_90") for suite in suites):
            raise ValueError(f"Unknown LIBERO suite: {cfg.suite}")
        plan = []
        for suite in suites:
            count = 90 if suite == "libero_90" else 10
            ids = range(count) if cfg.task_id is None else (int(cfg.task_id),)
            for task_id in ids:
                if not 0 <= task_id < count:
                    raise ValueError(f"Invalid {suite} task_id: {task_id}")
                plan.append({"suite": suite, "task_id": task_id})
        return plan
    name = "domino/level1_tasks.txt" if cfg.benchmark == "domino" else "robotwin/tasks.txt"
    available = (repository_root() / "src/longwam/benchmarks" / name).read_text().splitlines()
    tasks = available if cfg.task == "all" else [str(cfg.task)]
    if any(task not in available for task in tasks):
        raise ValueError(f"Unknown {cfg.benchmark} task: {cfg.task}")
    settings = (
        ("demo_clean_dynamic",) if cfg.benchmark == "domino" else ("demo_clean", "demo_randomized")
    )
    selected = settings if cfg.setting == "all" else (str(cfg.setting),)
    if any(setting not in settings for setting in selected):
        raise ValueError(f"Unsupported evaluation setting: {cfg.setting}")
    return [{"task": task, "setting": setting} for setting in selected for task in tasks]


def native_result(cfg, output):
    """Read a completed upstream result; never turn missing results into failures."""
    output = Path(output)
    if cfg.benchmark == "domino":
        metrics = json.loads((output / "_metrics.json").read_text())
        episodes = json.loads((output / "_episodes_detail.json").read_text())
        if int(metrics["total_episodes"]) != int(cfg.episodes) or len(episodes) != int(
            cfg.episodes
        ):
            raise RuntimeError("Incomplete Domino episode inventory")
        successes = int(metrics["success_count"])
        ms = float(metrics["manipulation_score_mean"])
        if not math.isfinite(ms):
            raise ValueError("Non-finite Domino manipulation score")
        result = {"successes": successes, "manipulation_score": ms}
    else:
        filename = "_result_clean.txt" if cfg.setting == "demo_clean" else "_result_random.txt"
        score = float((output / filename).read_text().strip().splitlines()[-1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid success fraction: {score}")
        successes = round(score * int(cfg.episodes))
        if abs(successes / int(cfg.episodes) - score) > 1e-6:
            raise ValueError("Success fraction is inconsistent with episode count")
        result = {"successes": successes}
    if not 0 <= successes <= int(cfg.episodes):
        raise ValueError("Invalid success count")
    return {**result, "total_episodes": int(cfg.episodes), "task": cfg.task, "setting": cfg.setting}


def summarize_cells(cells, expected_cells):
    episodes = sum(cell["total_episodes"] for cell in cells)
    successes = sum(cell["successes"] for cell in cells)
    result = {
        "complete": len(cells) == expected_cells,
        "completed_cells": len(cells),
        "expected_cells": expected_cells,
        "total_episodes": episodes,
        "successes": successes,
        "success_rate_percent": 100 * successes / episodes if episodes else None,
    }
    if cells and all("manipulation_score" in cell for cell in cells):
        result["manipulation_score"] = (
            sum(cell["manipulation_score"] * cell["total_episodes"] for cell in cells) / episodes
        )
    return result
