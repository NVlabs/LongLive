# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/datasets/lerobot/episode_selection.py
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

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 2

_MAPPING_EPISODE_FIELDS = (
    "episode_index",
    "length",
    "data_path",
    "canonical_task",
    "phase",
    "source_episode_index",
    "source_archive_path",
    "source_member_path",
    "combined_instruction_set_sha256",
    "official_seen_instruction_set_sha256",
    "official_instruction_json_sha256",
    "source_trajectory_member_path",
    "official_trajectory_pickle_sha256",
    "official_terminal_fingerprint_sha256",
    "combined_terminal_fingerprint_sha256",
    "match_evidence",
)
_MAPPING_ARCHIVE_FIELDS = (
    "task",
    "phase",
    "path",
    "episode_count",
    "size_bytes",
    "lfs_sha256",
    "instruction_registry_sha256",
    "trajectory_registry_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_fields(record: Mapping, fields: Sequence[str], context: str) -> dict:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}.")
    return {field: record[field] for field in fields}


def compute_manifest_mapping_sha256(manifest: Mapping) -> str:
    """Hash the complete episode mapping and its pinned official evidence."""
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Episode manifest is missing its `source` contract.")
    official = source.get("official")
    if not isinstance(official, Mapping):
        raise ValueError("Episode manifest is missing its `source.official` contract.")
    archives = official.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("Episode manifest has no official source archives.")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Episode manifest has no `episodes` mapping.")

    official_identity = _required_fields(
        official,
        ("repository", "revision", "embodiment"),
        "Official source contract",
    )
    official_identity["archives"] = sorted(
        (
            _required_fields(record, _MAPPING_ARCHIVE_FIELDS, "Official archive")
            for record in archives
        ),
        key=lambda record: (record["task"], record["phase"], record["path"]),
    )
    episode_mapping = sorted(
        (
            _required_fields(record, _MAPPING_EPISODE_FIELDS, "Episode mapping")
            for record in episodes
        ),
        key=lambda record: record["episode_index"],
    )
    payload = {
        "official": official_identity,
        "episodes": episode_mapping,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedEpisodeSelection:
    episode_indices: tuple[int, ...]
    group_names: tuple[str, ...]
    group_frame_counts: tuple[int, ...]
    signature: str


@dataclass(frozen=True)
class ResolvedPerDatasetEpisodeSelection:
    """A deterministic episode allowlist for each independent LeRobot root."""

    episode_indices: tuple[tuple[int, ...], ...]
    dataset_frame_counts: tuple[int, ...]
    total_episodes: int
    signature: str


def _load_lerobot_episode_records(dataset_root: Path) -> list[dict]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid LeRobot metadata: {info_path} ({exc}).") from exc

    records = []
    try:
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {episodes_path}:{line_number}.") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read LeRobot episodes: {episodes_path}.") from exc

    try:
        expected_episodes = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"LeRobot metadata has no valid total_episodes: {info_path}.") from exc
    if expected_episodes != len(records):
        raise ValueError(
            f"LeRobot episode inventory mismatch at {dataset_root}: "
            f"info.json={expected_episodes}, episodes.jsonl={len(records)}."
        )
    if not records:
        raise ValueError(f"LeRobot dataset contains no episodes: {dataset_root}.")

    seen_indices: set[int] = set()
    for record in records:
        try:
            episode_index = int(record["episode_index"])
            length = int(record["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid episode metadata in {episodes_path}: {record!r}.") from exc
        if episode_index < 0 or length <= 0:
            raise ValueError(f"Invalid episode {episode_index} length {length} in {episodes_path}.")
        if episode_index in seen_indices:
            raise ValueError(f"Duplicate episode_index {episode_index} in {episodes_path}.")
        seen_indices.add(episode_index)
    return records


def resolve_per_dataset_episode_selection(
    dataset_roots: Sequence[str | Path],
    *,
    episodes_per_dataset: int,
    seed: int,
    expected_dataset_count: int | None = None,
    expected_total_episodes: int | None = None,
) -> ResolvedPerDatasetEpisodeSelection:
    """Match RoboCasa's seeded ``N_demos`` filtering for LeRobot datasets.

    The upstream loader resets Python's RNG to ``seed`` for every independent
    task, shuffles episode ids in ``episodes.jsonl`` order, and keeps the first
    ``episodes_per_dataset`` ids. A dataset containing exactly the requested
    count remains in metadata order, matching the upstream ``subset=None``
    behavior.
    """

    dataset_roots = tuple(Path(root).expanduser().resolve() for root in dataset_roots)
    if not dataset_roots:
        raise ValueError("Per-dataset episode selection requires dataset roots.")
    if len(dataset_roots) != len(set(dataset_roots)):
        raise ValueError("Per-dataset episode selection contains duplicate roots.")

    episodes_per_dataset = int(episodes_per_dataset)
    seed = int(seed)
    if episodes_per_dataset <= 0:
        raise ValueError("`episodes_per_dataset` must be positive.")
    if seed < 0:
        raise ValueError("Per-dataset episode selection seed must be non-negative.")
    if expected_dataset_count is not None and len(dataset_roots) != int(expected_dataset_count):
        raise ValueError(
            "Per-dataset episode selection dataset count mismatch: "
            f"expected={int(expected_dataset_count)}, actual={len(dataset_roots)}."
        )

    selections: list[tuple[int, ...]] = []
    frame_counts: list[int] = []
    signature_datasets = []
    for dataset_root in dataset_roots:
        records = _load_lerobot_episode_records(dataset_root)
        if len(records) < episodes_per_dataset:
            raise ValueError(
                f"Dataset {dataset_root} has only {len(records)} episodes, fewer "
                f"than the requested {episodes_per_dataset}."
            )

        all_episode_indices = [int(record["episode_index"]) for record in records]
        if len(records) == episodes_per_dataset:
            selected_indices = all_episode_indices
        else:
            selected_indices = list(all_episode_indices)
            random.Random(seed).shuffle(selected_indices)
            selected_indices = selected_indices[:episodes_per_dataset]

        lengths = {int(record["episode_index"]): int(record["length"]) for record in records}
        selection = tuple(selected_indices)
        frame_count = sum(lengths[index] for index in selection)
        selections.append(selection)
        frame_counts.append(frame_count)
        signature_datasets.append(
            {
                "root": str(dataset_root),
                "available_episodes": len(records),
                "selected_episode_indices": selection,
                "selected_frame_count": frame_count,
            }
        )

    total_episodes = sum(len(selection) for selection in selections)
    if expected_total_episodes is not None and total_episodes != int(expected_total_episodes):
        raise ValueError(
            "Per-dataset episode selection total mismatch: "
            f"expected={int(expected_total_episodes)}, actual={total_episodes}."
        )

    contract = {
        "algorithm": "python_random_shuffle_per_dataset_v1",
        "episodes_per_dataset": episodes_per_dataset,
        "seed": seed,
        "datasets": signature_datasets,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ResolvedPerDatasetEpisodeSelection(
        episode_indices=tuple(selections),
        dataset_frame_counts=tuple(frame_counts),
        total_episodes=total_episodes,
        signature=signature,
    )


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported episode manifest schema at {path}: {manifest.get('schema_version')!r}."
        )
    if not isinstance(manifest.get("episodes"), list):
        raise ValueError(f"Episode manifest has no `episodes` list: {path}")
    expected_digest = manifest.get("mapping_sha256")
    if not isinstance(expected_digest, str):
        raise ValueError(f"Episode manifest has no `mapping_sha256`: {path}")
    actual_digest = compute_manifest_mapping_sha256(manifest)
    if actual_digest != expected_digest:
        raise ValueError(
            f"Episode manifest mapping changed: {path}; "
            f"expected={expected_digest}, actual={actual_digest}."
        )
    return manifest


def _validate_source_metadata(dataset_root: Path, manifest: Mapping) -> None:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Episode manifest is missing its `source` contract.")

    expected_root_name = source.get("root_name")
    if expected_root_name and dataset_root.name != expected_root_name:
        raise ValueError(
            f"Episode manifest expects dataset root {expected_root_name!r}, "
            f"got {dataset_root.name!r}."
        )

    metadata_sha256 = source.get("metadata_sha256")
    if not isinstance(metadata_sha256, Mapping):
        raise ValueError("Episode manifest is missing metadata SHA-256 values.")
    for relative_path, expected_digest in metadata_sha256.items():
        path = dataset_root / str(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest source metadata is missing: {path}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Manifest source metadata changed: {path}; "
                f"expected={expected_digest}, actual={actual_digest}."
            )


def resolve_episode_selection(
    dataset_root: str | Path,
    selection: Mapping,
) -> ResolvedEpisodeSelection:
    """Resolve a canonical-task selection to an ordered episode allowlist."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_path = Path(str(selection["manifest_path"])).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    if bool(selection.get("verify_source_hashes", True)):
        _validate_source_metadata(dataset_root, manifest)

    canonical_tasks = tuple(str(task) for task in selection["canonical_tasks"])
    if not canonical_tasks or len(set(canonical_tasks)) != len(canonical_tasks):
        raise ValueError("`canonical_tasks` must be non-empty and unique.")

    raw_phases = selection.get("phases")
    phases = None if raw_phases is None else frozenset(str(phase) for phase in raw_phases)
    if phases is not None and not phases:
        raise ValueError("`phases`, when provided, must be non-empty.")

    raw_phase_quotas = selection.get("episodes_per_phase")
    phase_quotas = None
    selection_seed = int(selection.get("seed", 0))
    if raw_phase_quotas is not None:
        if not isinstance(raw_phase_quotas, Mapping) or not raw_phase_quotas:
            raise ValueError("`episodes_per_phase` must be a non-empty mapping.")
        phase_quotas = {str(phase): int(quota) for phase, quota in raw_phase_quotas.items()}
        if any(quota <= 0 for quota in phase_quotas.values()):
            raise ValueError("Every `episodes_per_phase` quota must be positive.")
        if phases is None:
            phases = frozenset(phase_quotas)
        elif phases != frozenset(phase_quotas):
            raise ValueError("`phases` and `episodes_per_phase` must name exactly the same phases.")

    by_task: dict[str, list[tuple[int, int, str]]] = {task: [] for task in canonical_tasks}
    seen_episode_indices: set[int] = set()
    for record in manifest["episodes"]:
        task = str(record["canonical_task"])
        if task not in by_task:
            continue
        phase = record.get("phase")
        if phases is not None and phase not in phases:
            continue

        episode_index = int(record["episode_index"])
        length = int(record["length"])
        if length <= 0:
            raise ValueError(f"Episode {episode_index} has invalid length {length}.")
        if episode_index in seen_episode_indices:
            raise ValueError(f"Episode {episode_index} appears more than once in the selection.")
        seen_episode_indices.add(episode_index)
        by_task[task].append((episode_index, length, str(phase)))

    episode_indices: list[int] = []
    group_frame_counts: list[int] = []
    for task in canonical_tasks:
        episodes = by_task[task]
        if not episodes:
            phase_text = "all phases" if phases is None else f"phases={sorted(phases)}"
            raise ValueError(f"No episodes selected for task {task!r} ({phase_text}).")
        if phase_quotas is not None:
            sampled: list[tuple[int, int, str]] = []
            for phase, quota in phase_quotas.items():
                candidates = [record for record in episodes if record[2] == phase]
                if len(candidates) < quota:
                    raise ValueError(
                        f"Task {task!r} has only {len(candidates)} episodes in phase "
                        f"{phase!r}, fewer than requested quota {quota}."
                    )

                def stable_key(record: tuple[int, int, str]) -> bytes:
                    episode_index = record[0]
                    payload = (f"{selection_seed}\0{task}\0{phase}\0{episode_index}").encode(
                        "utf-8"
                    )
                    return hashlib.sha256(payload).digest()

                sampled.extend(sorted(candidates, key=stable_key)[:quota])
            episodes = sampled
        episodes = sorted(episodes)
        episode_indices.extend(index for index, _, _ in episodes)
        group_frame_counts.append(sum(length for _, length, _ in episodes))

    contract = {
        "manifest_mapping_sha256": manifest.get("mapping_sha256"),
        "canonical_tasks": canonical_tasks,
        "phases": None if phases is None else sorted(phases),
        "episodes_per_phase": phase_quotas,
        "seed": selection_seed,
        "episode_indices": episode_indices,
        "group_frame_counts": group_frame_counts,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ResolvedEpisodeSelection(
        episode_indices=tuple(episode_indices),
        group_names=canonical_tasks,
        group_frame_counts=tuple(group_frame_counts),
        signature=signature,
    )


def build_group_ranges(frame_counts: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ranges = []
    start = 0
    for raw_count in frame_counts:
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"Group frame counts must be positive, got {count}.")
        stop = start + count
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)
