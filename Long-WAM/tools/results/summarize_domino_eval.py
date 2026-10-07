#!/usr/bin/env python3
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

"""Aggregate DOMINO per-task JSON reports into one reproducible summary."""

import argparse
import csv
import json
import os
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    tasks = [
        line.split("#", 1)[0].strip()
        for line in args.task_file.read_text(encoding="utf-8").splitlines()
    ]
    tasks = [task for task in tasks if task]
    rows = []
    missing = []
    for task in tasks:
        metrics_path = args.run_dir / task / "_metrics.json"
        episodes_path = args.run_dir / task / "_episodes_detail.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            missing.append(task)
            continue
        if (
            int(metrics.get("total_episodes", 0)) != args.expected_episodes
            or not isinstance(episodes, list)
            or len(episodes) != args.expected_episodes
        ):
            missing.append(task)
            continue
        rows.append({"task": task, **metrics})

    args.run_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "total_episodes",
        "success_count",
        "success_rate",
        "manipulation_score_mean",
        "manipulation_score_std",
        "route_completion_mean",
        "route_completion_std",
        "penalty_clutter_collision_total",
        "penalty_out_of_bounds_total",
    ]
    csv_tmp = args.run_dir / "summary.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, args.run_dir / "summary.csv")

    total_episodes = sum(int(row["total_episodes"]) for row in rows)
    successes = sum(int(row["success_count"]) for row in rows)
    aggregate = {
        "tasks_expected": len(tasks),
        "tasks_complete": len(rows),
        "episodes_expected_per_task": args.expected_episodes,
        "total_episodes": total_episodes,
        "success_count": successes,
        "micro_success_rate": 100.0 * successes / total_episodes if total_episodes else None,
        "macro_success_rate": (
            sum(float(row["success_rate"]) for row in rows) / len(rows) if rows else None
        ),
        "macro_manipulation_score": (
            sum(float(row["manipulation_score_mean"]) for row in rows) / len(rows) if rows else None
        ),
        "macro_route_completion": (
            sum(float(row["route_completion_mean"]) for row in rows) / len(rows) if rows else None
        ),
        "missing_tasks": missing,
    }
    atomic_json(args.run_dir / "summary.json", {"aggregate": aggregate, "tasks": rows})
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 2 if args.require_complete and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
