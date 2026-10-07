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

"""Versioned, fail-closed handling of upstream GR1 payload omissions."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


KNOWN_OMISSIONS_SCHEMA = "longwam.robocasa-gr1-tabletop-known-omissions/v1"
_CONTRACT_FIELDS = {
    "schema",
    "repo_id",
    "revision",
    "source_total_episodes",
    "source_total_frames",
    "effective_total_episodes",
    "effective_total_frames",
    "omissions",
}
_OMISSION_FIELDS = {
    "dataset",
    "episode_index",
    "episode_length",
    "payload_kind",
    "relative_path",
    "paired_data_path",
}


@dataclass(frozen=True)
class GR1KnownOmission:
    dataset: str
    episode_index: int
    episode_length: int
    payload_kind: str
    relative_path: str
    paired_data_path: str


@dataclass(frozen=True)
class GR1KnownOmissionContract:
    path: Path
    sha256: str
    repo_id: str
    revision: str
    source_total_episodes: int
    source_total_frames: int
    effective_total_episodes: int
    effective_total_frames: int
    omissions: tuple[GR1KnownOmission, ...]

    def by_dataset(self) -> dict[str, tuple[GR1KnownOmission, ...]]:
        grouped: dict[str, list[GR1KnownOmission]] = {}
        for omission in self.omissions:
            grouped.setdefault(omission.dataset, []).append(omission)
        return {name: tuple(rows) for name, rows in grouped.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} is not a canonical relative POSIX path: {value!r}.")
    return value


def load_gr1_known_omissions(
    path: str | Path,
    *,
    repo_id: str,
    revision: str,
    source_total_episodes: int,
    source_total_frames: int,
    dataset_names: tuple[str, ...] | list[str],
) -> GR1KnownOmissionContract:
    """Load an exact omission allowlist; arbitrary missing payloads stay fatal."""

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError(f"Known-omission contract must be a regular file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Known-omission contract is invalid JSON: {source}") from error
    if not isinstance(payload, Mapping) or set(payload) != _CONTRACT_FIELDS:
        raise ValueError("Known-omission contract fields changed.")
    expected_identity = {
        "schema": KNOWN_OMISSIONS_SCHEMA,
        "repo_id": repo_id,
        "revision": revision,
        "source_total_episodes": int(source_total_episodes),
        "source_total_frames": int(source_total_frames),
    }
    drift = {
        key: (payload.get(key), expected)
        for key, expected in expected_identity.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Known-omission source identity changed: {drift}")

    raw_omissions = payload.get("omissions")
    if not isinstance(raw_omissions, list) or not raw_omissions:
        raise ValueError("Known-omission contract must contain a non-empty omissions list.")
    allowed_datasets = set(dataset_names)
    omissions: list[GR1KnownOmission] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_omissions:
        if not isinstance(raw, Mapping) or set(raw) != _OMISSION_FIELDS:
            raise ValueError("Known-omission entry fields changed.")
        dataset = raw.get("dataset")
        if not isinstance(dataset, str) or dataset not in allowed_datasets:
            raise ValueError(f"Known omission names an unknown dataset: {dataset!r}.")
        episode_index = _nonnegative_int(raw.get("episode_index"), "episode_index")
        episode_length = _positive_int(raw.get("episode_length"), "episode_length")
        payload_kind = raw.get("payload_kind")
        if payload_kind != "video":
            raise ValueError(f"Only an absent video can be allowlisted, got {payload_kind!r}.")
        relative_path = _safe_relative_path(raw.get("relative_path"), "relative_path")
        paired_data_path = _safe_relative_path(raw.get("paired_data_path"), "paired_data_path")
        chunk = episode_index // 1000
        expected_video = (
            f"videos/chunk-{chunk:03d}/observation.images.ego_view/episode_{episode_index:06d}.mp4"
        )
        expected_data = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        if relative_path != expected_video or paired_data_path != expected_data:
            raise ValueError(
                "Known-omission paths do not match their episode index: "
                f"{relative_path!r}, {paired_data_path!r}."
            )
        key = (dataset, episode_index)
        if key in seen:
            raise ValueError(f"Duplicate known omission: {key}.")
        seen.add(key)
        omissions.append(
            GR1KnownOmission(
                dataset=dataset,
                episode_index=episode_index,
                episode_length=episode_length,
                payload_kind=payload_kind,
                relative_path=relative_path,
                paired_data_path=paired_data_path,
            )
        )

    effective_episodes = int(payload.get("effective_total_episodes", -1))
    effective_frames = int(payload.get("effective_total_frames", -1))
    if effective_episodes != int(source_total_episodes) - len(omissions):
        raise ValueError("Known-omission effective episode total is inconsistent.")
    if effective_frames != int(source_total_frames) - sum(row.episode_length for row in omissions):
        raise ValueError("Known-omission effective frame total is inconsistent.")
    return GR1KnownOmissionContract(
        path=source.resolve(),
        sha256=_sha256(source),
        repo_id=repo_id,
        revision=revision,
        source_total_episodes=int(source_total_episodes),
        source_total_frames=int(source_total_frames),
        effective_total_episodes=effective_episodes,
        effective_total_frames=effective_frames,
        omissions=tuple(omissions),
    )


__all__ = [
    "GR1KnownOmission",
    "GR1KnownOmissionContract",
    "KNOWN_OMISSIONS_SCHEMA",
    "load_gr1_known_omissions",
]
