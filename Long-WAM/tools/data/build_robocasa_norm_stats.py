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

"""Build Human300 normalization stats from RoboCasa365 metadata only.

This intentionally never opens parquet or video files. RoboCasa365 already
ships per-dataset statistics, so aggregating those files on CPU avoids doing a
full-dataset statistics pass after a GPU training allocation has started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


FIELD_SPECS = {
    "action": ("action", 12),
    "state": ("observation.state", 16),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "./data/robocasa365"
        ),
        help="RoboCasa365 root containing v1.0/pretrain.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-datasets",
        type=int,
        default=300,
        help="Fail unless this many Human300 LeRobot datasets are found.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def discover_dataset_dirs(data_root: Path) -> list[Path]:
    pretrain_root = data_root.expanduser().resolve() / "v1.0" / "pretrain"
    dataset_dirs: list[Path] = []
    for family in ("atomic", "composite"):
        family_root = pretrain_root / family
        dataset_dirs.extend(
            path.parent.parent
            for path in family_root.glob("*/*/lerobot/meta/info.json")
        )
    return sorted(set(dataset_dirs), key=lambda path: path.as_posix())


def _numeric_vector(
    value: Any,
    *,
    expected_dim: int,
    label: str,
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_dim:
        raise ValueError(
            f"{label} must be a length-{expected_dim} list, got {value!r}"
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} contains a non-numeric value: {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} contains a non-finite value: {number}")
        result.append(number)
    return result


def _feature_dim(info: dict[str, Any], key: str, path: Path) -> int:
    try:
        shape = info["features"][key]["shape"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} is missing features.{key}.shape") from exc
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"{path}: features.{key}.shape must be one-dimensional")
    return int(shape[0])


def build_stats(
    data_root: Path,
    *,
    expected_datasets: int = 300,
) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    dataset_dirs = discover_dataset_dirs(data_root)
    if len(dataset_dirs) != expected_datasets:
        raise ValueError(
            f"Expected {expected_datasets} Human300 datasets under {data_root}, "
            f"found {len(dataset_dirs)}"
        )

    aggregate: dict[str, dict[str, Any]] = {
        output_key: {
            "min": [math.inf] * dim,
            "max": [-math.inf] * dim,
            "q01": [math.inf] * dim,
            "q99": [-math.inf] * dim,
            "weighted_sum": [0.0] * dim,
            "weighted_second_moment": [0.0] * dim,
        }
        for output_key, (_, dim) in FIELD_SPECS.items()
    }
    manifest_hasher = hashlib.sha256()
    total_episodes = 0
    total_frames = 0

    for dataset_dir in dataset_dirs:
        info_path = dataset_dir / "meta" / "info.json"
        stats_path = dataset_dir / "meta" / "stats.json"
        info_bytes = info_path.read_bytes()
        stats_bytes = stats_path.read_bytes()
        info = _load_json(info_path)
        source_stats = _load_json(stats_path)

        try:
            fps = float(info["fps"])
            num_episodes = int(info["total_episodes"])
            num_frames = int(info["total_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{info_path} has invalid dataset counts or fps") from exc
        if fps != 20.0:
            raise ValueError(f"{info_path}: expected fps=20, got {fps}")
        if num_episodes <= 0 or num_frames <= 0:
            raise ValueError(
                f"{info_path}: total_episodes and total_frames must be positive"
            )
        for _, (source_key, expected_dim) in FIELD_SPECS.items():
            if _feature_dim(info, source_key, info_path) != expected_dim:
                raise ValueError(
                    f"{info_path}: expected {source_key} dim={expected_dim}"
                )

        relative_dir = dataset_dir.relative_to(data_root).as_posix()
        manifest_hasher.update(relative_dir.encode("utf-8"))
        manifest_hasher.update(b"\0")
        manifest_hasher.update(hashlib.sha256(info_bytes).digest())
        manifest_hasher.update(hashlib.sha256(stats_bytes).digest())

        for output_key, (source_key, expected_dim) in FIELD_SPECS.items():
            try:
                field_stats = source_stats[source_key]
            except KeyError as exc:
                raise ValueError(f"{stats_path} is missing {source_key}") from exc
            vectors = {
                name: _numeric_vector(
                    field_stats.get(name),
                    expected_dim=expected_dim,
                    label=f"{stats_path}:{source_key}.{name}",
                )
                for name in ("min", "max", "mean", "std", "q01", "q99")
            }
            current = aggregate[output_key]
            for index in range(expected_dim):
                current["min"][index] = min(
                    current["min"][index], vectors["min"][index]
                )
                current["max"][index] = max(
                    current["max"][index], vectors["max"][index]
                )
                current["q01"][index] = min(
                    current["q01"][index], vectors["q01"][index]
                )
                current["q99"][index] = max(
                    current["q99"][index], vectors["q99"][index]
                )
                mean = vectors["mean"][index]
                std = vectors["std"][index]
                current["weighted_sum"][index] += num_frames * mean
                current["weighted_second_moment"][index] += num_frames * (
                    std * std + mean * mean
                )

        total_episodes += num_episodes
        total_frames += num_frames

    result: dict[str, Any] = {
        "state": {},
        "action": {},
        "num_episodes": total_episodes,
        "num_transition": total_frames,
    }
    for output_key, (_, dim) in FIELD_SPECS.items():
        current = aggregate[output_key]
        means = [value / total_frames for value in current["weighted_sum"]]
        variances = [
            max(
                0.0,
                current["weighted_second_moment"][index] / total_frames
                - means[index] * means[index],
            )
            for index in range(dim)
        ]
        result[output_key]["default"] = {
            "global_min": current["min"],
            "global_max": current["max"],
            "global_mean": means,
            "global_std": [math.sqrt(value) for value in variances],
            # Exact global quantiles cannot be reconstructed from per-dataset
            # quantiles. These conservative envelopes match the existing
            # LongWAMBase stats aggregation and are not used by min/max configs.
            "global_q01": current["q01"],
            "global_q99": current["q99"],
        }

    result["provenance"] = {
        "format": "robocasa365_metadata_aggregate_v1",
        "data_root": data_root.as_posix(),
        "dataset_count": len(dataset_dirs),
        "dataset_manifest_sha256": manifest_hasher.hexdigest(),
        "fps": 20,
        "action_dim": 12,
        "state_dim": 16,
        "quantile_aggregation": "conservative_dataset_envelope",
    }
    return result


def write_json_atomic(payload: dict[str, Any], output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = _parse_args()
    payload = build_stats(
        args.data_root,
        expected_datasets=args.expected_datasets,
    )
    write_json_atomic(payload, args.output)
    print(
        json.dumps(
            {
                "output": args.output.expanduser().resolve().as_posix(),
                "dataset_count": payload["provenance"]["dataset_count"],
                "num_episodes": payload["num_episodes"],
                "num_transition": payload["num_transition"],
                "dataset_manifest_sha256": payload["provenance"][
                    "dataset_manifest_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
