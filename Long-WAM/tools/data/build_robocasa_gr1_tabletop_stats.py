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

"""Build the pinned official GR1 Tabletop action-normalization contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


RAW_ACTION_DIM = 44
FIELD_SLICES = {
    "left_arm": tuple(range(0, 7)),
    "right_arm": tuple(range(22, 29)),
    "left_hand": tuple(range(7, 13)),
    "right_hand": tuple(range(29, 35)),
    "waist": tuple(range(41, 44)),
}
STATISTIC_NAMES = ("min", "max", "mean", "std", "q01", "q99")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _action_statistics(path: Path) -> dict[str, list[float]]:
    action = _read_object(path).get("action")
    if not isinstance(action, dict):
        raise ValueError(f"Missing action statistics: {path}")
    result: dict[str, list[float]] = {}
    for name in STATISTIC_NAMES:
        values = action.get(name)
        if not isinstance(values, list) or len(values) != RAW_ACTION_DIM:
            raise ValueError(
                f"action.{name} in {path} must contain {RAW_ACTION_DIM} values"
            )
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"action.{name} contains non-finite values: {path}")
        result[name] = numeric
    if any(low > high for low, high in zip(result["min"], result["max"])):
        raise ValueError(f"action min exceeds max: {path}")
    return result


def _merge_statistics(
    per_dataset: list[dict[str, list[float]]], frame_weights: list[int]
) -> dict[str, list[float]]:
    total_weight = sum(frame_weights)
    if total_weight <= 0:
        raise ValueError("Dataset frame weights must sum to a positive value")
    weights = [weight / total_weight for weight in frame_weights]

    mean = [
        sum(weight * stats["mean"][index] for weight, stats in zip(weights, per_dataset))
        for index in range(RAW_ACTION_DIM)
    ]
    second_moment = [
        sum(
            weight
            * (stats["std"][index] ** 2 + stats["mean"][index] ** 2)
            for weight, stats in zip(weights, per_dataset)
        )
        for index in range(RAW_ACTION_DIM)
    ]
    return {
        "min": [min(stats["min"][index] for stats in per_dataset) for index in range(RAW_ACTION_DIM)],
        "max": [max(stats["max"][index] for stats in per_dataset) for index in range(RAW_ACTION_DIM)],
        "mean": mean,
        "std": [math.sqrt(max(second - value**2, 0.0)) for second, value in zip(second_moment, mean)],
        # The official finetune entrypoint overrides the mixture default with
        # percentile_mixing_method=weighted_average. These quantiles are retained
        # only as diagnostics; FourierGr1ArmsWaist normalizes actions with the
        # global min/max values computed above.
        "q01": [
            sum(weight * stats["q01"][index] for weight, stats in zip(weights, per_dataset))
            for index in range(RAW_ACTION_DIM)
        ],
        "q99": [
            sum(weight * stats["q99"][index] for weight, stats in zip(weights, per_dataset))
            for index in range(RAW_ACTION_DIM)
        ],
    }


def build_statistics(data_root: Path, inventory_path: Path) -> dict[str, Any]:
    inventory = _read_object(inventory_path)
    if inventory.get("schema") != "longwam.robocasa-gr1-tabletop-inventory/v1":
        raise ValueError(f"Unsupported inventory schema: {inventory.get('schema')!r}")
    datasets = inventory.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Inventory must contain an ordered, non-empty datasets mapping")

    per_dataset: list[dict[str, list[float]]] = []
    frame_weights: list[int] = []
    sources: list[dict[str, Any]] = []
    for dataset_name, expected in datasets.items():
        if not isinstance(expected, dict):
            raise ValueError(f"Invalid inventory entry: {dataset_name}")
        frames = int(expected["frames"])
        path = data_root / dataset_name / "meta" / "stats.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned source statistics: {path}")
        per_dataset.append(_action_statistics(path))
        frame_weights.append(frames)
        sources.append(
            {
                "dataset": dataset_name,
                "frames": frames,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )

    merged = _merge_statistics(per_dataset, frame_weights)
    action_fields = {
        name: {
            statistic: [merged[statistic][index] for index in indices]
            for statistic in ("min", "max")
        }
        for name, indices in FIELD_SLICES.items()
    }
    diagnostic_fields = {
        name: {
            statistic: [merged[statistic][index] for index in indices]
            for statistic in ("mean", "std", "q01", "q99")
        }
        for name, indices in FIELD_SLICES.items()
    }
    selected_raw_indices = [index for indices in FIELD_SLICES.values() for index in indices]
    if len(selected_raw_indices) != 29 or len(set(selected_raw_indices)) != 29:
        raise RuntimeError("GR1 selected action layout must contain 29 unique dimensions")

    return {
        "schema": "longwam.robocasa-gr1-tabletop-action-stats/v1",
        "schema_version": 1,
        "benchmark": "robocasa_gr1_tabletop",
        "repo_id": inventory["repo_id"],
        "dataset_revision": inventory["revision"],
        "dataset_count": len(datasets),
        "total_frames": sum(frame_weights),
        "raw_action_dim": RAW_ACTION_DIM,
        "field_order": list(FIELD_SLICES),
        "raw_field_indices": {name: list(indices) for name, indices in FIELD_SLICES.items()},
        "selected_raw_indices": selected_raw_indices,
        "mixing": {
            "dataset_path_weights": "1.0_each",
            "balance_dataset_weights": True,
            "balance_trajectory_weights": True,
            "effective_frame_sampling": "global_uniform_with_replacement",
            "percentile_mixing_method": "weighted_average",
            "action_normalization": "min_max_to_minus_one_plus_one",
            "constant_dimension_normalization": 0.0,
        },
        "action_fields": action_fields,
        "diagnostics": {"action_fields": diagnostic_fields},
        "sources": sources,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_statistics(args.data_root, args.inventory)
    _write_json_atomic(args.output, result)
    print(
        "ROBOCASA_GR1_TABLETOP_STATS_OK "
        f"datasets={result['dataset_count']} frames={result['total_frames']} "
        f"action_dim={len(result['selected_raw_indices'])}"
    )


if __name__ == "__main__":
    main()
