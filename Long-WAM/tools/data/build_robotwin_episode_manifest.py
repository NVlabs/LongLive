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

"""Build a verified RoboTwin episode manifest from official instruction JSONs."""

import argparse
import binascii
import hashlib
import io
import json
import os
import pickle
import re
import struct
import tempfile
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from tqdm import tqdm

from longwam.datasets.lerobot.episode_selection import (
    MANIFEST_SCHEMA_VERSION,
    compute_manifest_mapping_sha256,
    sha256_file,
)


DEFAULT_OFFICIAL_REPOSITORY = "TianxingChen/RoboTwin2.0"
METADATA_FILES = ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl")
PHASE_EPISODE_COUNTS = {"clean": 50, "randomized": 500}
INSTRUCTION_MEMBER_PATTERN = re.compile(r"(?:^|/)instructions/episode(\d+)\.json$")
TRAJECTORY_MEMBER_PATTERN = re.compile(r"(?:^|/)_traj_data/episode(\d+)\.pkl$")
MAX_ZIP_MEMBER_BYTES = 256 * 1024
MAX_TRAJECTORY_DEPTH = 32
MAX_TRAJECTORY_NODES = 4096
EXPECTED_STATE_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)


class AmbiguousInstructionMatchError(ValueError):
    """Official prompts identify a task but not a unique source episode."""


def _canonical_json_sha256(payload) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_canonical_tasks(
    registry_path: Path | None, inline_tasks: Sequence[str] | None
) -> tuple[tuple[str, ...], dict]:
    if registry_path is not None:
        with registry_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("canonical_tasks")
        if not isinstance(payload, list):
            raise ValueError(
                "Canonical registry must be a YAML list or contain `canonical_tasks`."
            )
        tasks = tuple(str(task) for task in payload)
        provenance = {
            "path": str(registry_path.resolve()),
            "file_sha256": sha256_file(registry_path),
        }
    else:
        tasks = tuple(str(task) for task in inline_tasks or ())
        provenance = {"path": None, "file_sha256": None}
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Canonical tasks must be non-empty and unique.")
    provenance["tasks_sha256"] = _canonical_json_sha256(list(tasks))
    return tasks, provenance


def _resolve_official_revision(repository: str, requested_revision: str) -> str:
    if requested_revision == "main":
        url = f"https://huggingface.co/api/datasets/{repository}"
    else:
        revision = quote(requested_revision, safe="")
        url = f"https://huggingface.co/api/datasets/{repository}/revision/{revision}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    resolved = str(response.json().get("sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"Could not resolve an immutable revision for {repository}.")
    return resolved


def _official_archive_paths(
    tasks: Sequence[str], embodiment: str
) -> dict[tuple[str, str], str]:
    return {
        (task, phase): f"dataset/{task}/{embodiment}_{phase}_{episode_count}.zip"
        for task in tasks
        for phase, episode_count in PHASE_EPISODE_COUNTS.items()
    }


def _fetch_archive_metadata(
    repository: str, revision: str, paths: Sequence[str]
) -> dict[str, dict]:
    url = (
        f"https://huggingface.co/api/datasets/{repository}/paths-info/{revision}"
    )
    response = requests.post(url, json={"paths": list(paths)}, timeout=120)
    response.raise_for_status()
    records = {str(record["path"]): record for record in response.json()}
    if set(records) != set(paths):
        raise ValueError(
            "Official archive inventory mismatch: "
            f"missing={sorted(set(paths) - set(records))}, "
            f"extra={sorted(set(records) - set(paths))}."
        )
    normalized = {}
    for path, record in records.items():
        lfs = record.get("lfs")
        if record.get("type") != "file" or not isinstance(lfs, Mapping):
            raise ValueError(f"Official archive is not an LFS file: {path}")
        lfs_sha256 = str(lfs.get("oid", ""))
        size_bytes = int(lfs.get("size", -1))
        if not re.fullmatch(r"[0-9a-f]{64}", lfs_sha256) or size_bytes <= 0:
            raise ValueError(f"Invalid official LFS evidence for {path}.")
        normalized[path] = {
            "size_bytes": size_bytes,
            "lfs_sha256": lfs_sha256,
        }
    return normalized


class _HttpRangeReader(io.RawIOBase):
    """Seekable HTTP reader that rejects ignored or oversized Range responses."""

    def __init__(self, url: str, size: int, retries: int = 3):
        self.url = url
        self.size = int(size)
        self.position = 0
        self.retries = int(retries)
        self.transferred_bytes = 0
        self._session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError(f"Negative seek position: {position}")
        self.position = position
        return position

    def _fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise ValueError(f"Invalid byte range {start}-{end} for size {self.size}.")
        expected_size = end - start + 1
        expected_range = f"bytes {start}-{end}/{self.size}"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            response = None
            try:
                response = self._session.get(
                    self.url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    allow_redirects=True,
                    stream=True,
                    timeout=(20, 120),
                )
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range request returned HTTP {response.status_code}."
                    )
                if response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(
                        "Unexpected Content-Range: "
                        f"{response.headers.get('Content-Range')!r}; "
                        f"expected {expected_range!r}."
                    )
                content_length = int(response.headers.get("Content-Length", -1))
                if content_length != expected_size:
                    raise RuntimeError(
                        f"Unexpected Range response size {content_length}; "
                        f"expected {expected_size}."
                    )
                chunks = []
                received = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    received += len(chunk)
                    if received > expected_size:
                        raise RuntimeError("Range response exceeded its declared size.")
                    chunks.append(chunk)
                if received != expected_size:
                    raise RuntimeError(
                        f"Truncated Range response: received={received}, "
                        f"expected={expected_size}."
                    )
                payload = b"".join(chunks)
                self.transferred_bytes += received
                return payload
            except (requests.RequestException, RuntimeError, ValueError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError(
            f"Failed HTTP Range request {start}-{end} for {self.url}: {last_error}"
        ) from last_error

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        payload = self._fetch(self.position, self.position + size - 1)
        self.position += len(payload)
        return payload

    def fetch(self, start: int, end: int) -> bytes:
        return self._fetch(start, end)

    def close(self) -> None:
        self._session.close()
        super().close()


def _decode_zip_member(
    blob: bytes,
    span_start: int,
    info: zipfile.ZipInfo,
    maximum_compressed_bytes: int = MAX_ZIP_MEMBER_BYTES,
    maximum_uncompressed_bytes: int = MAX_ZIP_MEMBER_BYTES,
) -> bytes:
    if not 0 <= info.compress_size <= maximum_compressed_bytes:
        raise ValueError(
            f"ZIP member compressed size exceeds limit: {info.filename}; "
            f"size={info.compress_size}, limit={maximum_compressed_bytes}."
        )
    if not 0 <= info.file_size <= maximum_uncompressed_bytes:
        raise ValueError(
            f"ZIP member uncompressed size exceeds limit: {info.filename}; "
            f"size={info.file_size}, limit={maximum_uncompressed_bytes}."
        )
    offset = info.header_offset - span_start
    if offset < 0 or offset + 30 > len(blob):
        raise ValueError(f"Local ZIP header is outside fetched span: {info.filename}")
    header = blob[offset : offset + 30]
    (
        signature,
        _,
        flags,
        _,
        _,
        _,
        _,
        _,
        _,
        filename_length,
        extra_length,
    ) = struct.unpack("<4s5H3L2H", header)
    if signature != b"PK\x03\x04" or flags & 1:
        raise ValueError(f"Unsupported local ZIP header: {info.filename}")
    data_start = offset + 30 + filename_length + extra_length
    data_end = data_start + info.compress_size
    if data_end > len(blob):
        raise ValueError(f"ZIP member is outside fetched span: {info.filename}")
    compressed = blob[data_start:data_end]
    if info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-15)
        payload = decompressor.decompress(
            compressed, maximum_uncompressed_bytes + 1
        )
        if len(payload) > maximum_uncompressed_bytes or decompressor.unconsumed_tail:
            raise ValueError(
                f"ZIP member decompressed data exceeds limit: {info.filename}."
            )
        payload += decompressor.flush(maximum_uncompressed_bytes + 1 - len(payload))
        if len(payload) > maximum_uncompressed_bytes or not decompressor.eof:
            raise ValueError(
                f"ZIP member decompressed data exceeds limit or is truncated: "
                f"{info.filename}."
            )
        if decompressor.unused_data:
            raise ValueError(f"ZIP member has trailing compressed data: {info.filename}")
    else:
        raise ValueError(
            f"Unsupported ZIP compression {info.compress_type}: {info.filename}"
        )
    if len(payload) != info.file_size:
        raise ValueError(f"ZIP member size mismatch: {info.filename}")
    if binascii.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"ZIP member CRC mismatch: {info.filename}")
    return payload


def _bounded_member_span(
    all_members: Sequence[zipfile.ZipInfo],
    selected_members: Sequence[zipfile.ZipInfo],
    central_directory_offset: int,
    maximum_span_bytes: int,
    context: str,
) -> tuple[int, int]:
    if not selected_members:
        raise ValueError(f"No ZIP members selected for {context}.")
    span_start = min(info.header_offset for info in selected_members)
    last_header = max(info.header_offset for info in selected_members)
    following_headers = [
        info.header_offset for info in all_members if info.header_offset > last_header
    ]
    span_end = min(following_headers, default=central_directory_offset)
    span_bytes = span_end - span_start
    if span_bytes <= 0 or span_bytes > maximum_span_bytes:
        raise ValueError(
            f"ZIP member span for {context} is {span_bytes} bytes; "
            f"limit={maximum_span_bytes}. Refusing a potentially large transfer."
        )
    return span_start, span_end


try:
    from numpy._core.multiarray import _reconstruct as _NUMPY_RECONSTRUCT
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import _reconstruct as _NUMPY_RECONSTRUCT


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    _ALLOWED_GLOBALS = {
        ("numpy.core.multiarray", "_reconstruct"): _NUMPY_RECONSTRUCT,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str):
        allowed = self._ALLOWED_GLOBALS.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(
                f"Forbidden global in official trajectory pickle: {module}.{name}"
            )
        return allowed


    def persistent_load(self, persistent_id):
        raise pickle.UnpicklingError(
            "Persistent IDs are forbidden in official trajectory pickle: "
            f"{persistent_id!r}"
        )


def _validate_trajectory_tree(
    value,
    maximum_depth: int = MAX_TRAJECTORY_DEPTH,
    maximum_nodes: int = MAX_TRAJECTORY_NODES,
) -> None:
    if maximum_depth < 0 or maximum_nodes <= 0:
        raise ValueError("Trajectory tree limits must be positive.")
    active_containers: set[int] = set()
    visited_nodes = 0

    def visit(item, depth: int) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > maximum_nodes:
            raise ValueError(
                f"Official trajectory pickle exceeds node limit {maximum_nodes}."
            )
        if depth > maximum_depth:
            raise ValueError(
                f"Official trajectory pickle exceeds depth limit {maximum_depth}."
            )
        if isinstance(item, str):
            return
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Official trajectory pickle contains an object array.")
            return
        if not isinstance(item, (list, dict)):
            raise ValueError(
                "Official trajectory pickle contains forbidden value type "
                f"{type(item).__name__}."
            )
        identity = id(item)
        if identity in active_containers:
            raise ValueError("Official trajectory pickle contains a container cycle.")
        active_containers.add(identity)
        try:
            if isinstance(item, list):
                for child in item:
                    visit(child, depth + 1)
            else:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise ValueError(
                            "Official trajectory pickle has a non-string key."
                        )
                    visit(child, depth + 1)
        finally:
            active_containers.remove(identity)

    visit(value, 0)


def _terminal_trajectory_fingerprint(raw_pickle: bytes) -> str:
    if len(raw_pickle) > MAX_ZIP_MEMBER_BYTES:
        raise ValueError(
            f"Official trajectory pickle exceeds {MAX_ZIP_MEMBER_BYTES} bytes."
        )
    stream = io.BytesIO(raw_pickle)
    trajectory = _RestrictedNumpyUnpickler(stream).load()
    if stream.read(1):
        raise pickle.UnpicklingError(
            "Official trajectory pickle contains trailing data."
        )
    _validate_trajectory_tree(trajectory)
    if not isinstance(trajectory, dict):
        raise ValueError("Official trajectory pickle is not a dictionary.")
    terminal_positions = []
    for arm in ("left_joint_path", "right_joint_path"):
        segments = trajectory.get(arm)
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"Official trajectory pickle has invalid {arm}.")
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError(
                    f"Official trajectory pickle has invalid {arm} segment "
                    f"{segment_index}."
                )
            positions = segment.get("position")
            if not isinstance(positions, np.ndarray):
                raise ValueError(
                    f"Official trajectory pickle has non-array {arm} positions "
                    f"at segment {segment_index}."
                )
            if (
                positions.dtype != np.dtype(np.float32)
                or positions.ndim != 2
                or positions.shape[0] <= 0
                or positions.shape[1] != 6
            ):
                raise ValueError(
                    f"Official trajectory pickle has invalid {arm} positions at "
                    f"segment {segment_index}: shape={positions.shape}, "
                    f"dtype={positions.dtype}; expected non-empty [N, 6] float32."
                )
            if not np.isfinite(positions).all():
                raise ValueError(
                    f"Official trajectory pickle has non-finite {arm} positions "
                    f"at segment {segment_index}."
                )
        terminal_positions.append(
            segments[-1]["position"][-1].astype("<f4", copy=False)
        )
    terminal = np.concatenate(terminal_positions).astype("<f4", copy=False)
    return hashlib.sha256(terminal.tobytes()).hexdigest()


def _fetch_official_instructions(
    repository: str,
    revision: str,
    archive_path: str,
    archive_metadata: Mapping,
    expected_episode_count: int,
    maximum_span_bytes: int,
) -> tuple[dict[int, dict], dict]:
    url = (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
        f"{archive_path}"
    )
    reader = _HttpRangeReader(url, int(archive_metadata["size_bytes"]))
    try:
        with zipfile.ZipFile(reader) as archive:
            members = []
            for info in archive.infolist():
                match = INSTRUCTION_MEMBER_PATTERN.search(info.filename)
                if match:
                    members.append((int(match.group(1)), info))
            source_indices = {source_index for source_index, _ in members}
            expected_indices = set(range(expected_episode_count))
            if len(members) != expected_episode_count or source_indices != expected_indices:
                raise ValueError(
                    f"Official instruction member coverage mismatch for {archive_path}: "
                    f"members={len(members)}, "
                    f"missing={sorted(expected_indices - source_indices)[:10]}, "
                    f"extra={sorted(source_indices - expected_indices)[:10]}."
                )
            span_start, span_end = _bounded_member_span(
                archive.infolist(),
                [info for _, info in members],
                archive.start_dir,
                maximum_span_bytes,
                f"official instructions in {archive_path}",
            )
            span_bytes = span_end - span_start
            blob = reader.fetch(span_start, span_end - 1)

        sources = {}
        member_evidence = []
        for source_index, info in members:
            raw_json = _decode_zip_member(blob, span_start, info)
            payload = json.loads(raw_json)
            seen = payload.get("seen")
            if not isinstance(seen, list) or not seen or not all(
                isinstance(prompt, str) for prompt in seen
            ):
                raise ValueError(
                    f"Official instruction JSON has invalid `seen`: {info.filename}"
                )
            seen_prompts = frozenset(seen)
            raw_sha256 = hashlib.sha256(raw_json).hexdigest()
            seen_sha256 = _canonical_json_sha256(sorted(seen_prompts))
            sources[source_index] = {
                "seen_prompts": seen_prompts,
                "member_path": info.filename,
                "json_sha256": raw_sha256,
                "seen_instruction_set_sha256": seen_sha256,
            }
            member_evidence.append(
                {
                    "source_episode_index": source_index,
                    "member_path": info.filename,
                    "json_sha256": raw_sha256,
                    "seen_instruction_set_sha256": seen_sha256,
                }
            )
        archive_evidence = {
            "instruction_registry_sha256": _canonical_json_sha256(
                sorted(member_evidence, key=lambda record: record["source_episode_index"])
            ),
            "instruction_span_bytes": span_bytes,
            "metadata_bytes_read": reader.transferred_bytes,
        }
        return sources, archive_evidence
    finally:
        reader.close()


def _fetch_official_trajectory_fingerprints(
    repository: str,
    revision: str,
    archive_path: str,
    archive_metadata: Mapping,
    expected_episode_count: int,
    maximum_span_bytes: int,
) -> tuple[dict[int, dict], dict]:
    url = (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
        f"{archive_path}"
    )
    reader = _HttpRangeReader(url, int(archive_metadata["size_bytes"]))
    try:
        with zipfile.ZipFile(reader) as archive:
            members = []
            for info in archive.infolist():
                match = TRAJECTORY_MEMBER_PATTERN.search(info.filename)
                if match:
                    members.append((int(match.group(1)), info))
            source_indices = {source_index for source_index, _ in members}
            expected_indices = set(range(expected_episode_count))
            if len(members) != expected_episode_count or source_indices != expected_indices:
                raise ValueError(
                    f"Official trajectory member coverage mismatch for {archive_path}: "
                    f"members={len(members)}, "
                    f"missing={sorted(expected_indices - source_indices)[:10]}, "
                    f"extra={sorted(source_indices - expected_indices)[:10]}."
                )
            span_start, span_end = _bounded_member_span(
                archive.infolist(),
                [info for _, info in members],
                archive.start_dir,
                maximum_span_bytes,
                f"official trajectory pickles in {archive_path}",
            )
            span_bytes = span_end - span_start
            blob = reader.fetch(span_start, span_end - 1)

        sources = {}
        member_evidence = []
        for source_index, info in members:
            raw_pickle = _decode_zip_member(blob, span_start, info)
            pickle_sha256 = hashlib.sha256(raw_pickle).hexdigest()
            terminal_sha256 = _terminal_trajectory_fingerprint(raw_pickle)
            sources[source_index] = {
                "trajectory_member_path": info.filename,
                "trajectory_pickle_sha256": pickle_sha256,
                "terminal_fingerprint_sha256": terminal_sha256,
            }
            member_evidence.append(
                {
                    "source_episode_index": source_index,
                    "member_path": info.filename,
                    "pickle_sha256": pickle_sha256,
                    "terminal_fingerprint_sha256": terminal_sha256,
                }
            )
        fingerprints = [
            evidence["terminal_fingerprint_sha256"] for evidence in sources.values()
        ]
        if len(fingerprints) != len(set(fingerprints)):
            duplicates = [
                fingerprint
                for fingerprint, count in Counter(fingerprints).items()
                if count > 1
            ]
            raise ValueError(
                f"Official terminal trajectory fingerprints are not unique in "
                f"{archive_path}: {duplicates[:5]}"
            )
        archive_evidence = {
            "trajectory_registry_sha256": _canonical_json_sha256(
                sorted(member_evidence, key=lambda record: record["source_episode_index"])
            ),
            "trajectory_span_bytes": span_bytes,
            "trajectory_metadata_bytes_read": reader.transferred_bytes,
        }
        return sources, archive_evidence
    finally:
        reader.close()


def _validate_state_feature_contract(info: Mapping) -> None:
    features = info.get("features")
    feature = features.get("observation.state") if isinstance(features, Mapping) else None
    if not isinstance(feature, Mapping):
        raise ValueError("Dataset info has no `observation.state` feature contract.")
    if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
        raise ValueError(
            "Dataset info has an incompatible `observation.state` dtype/shape: "
            f"dtype={feature.get('dtype')!r}, shape={feature.get('shape')!r}."
        )
    names = feature.get("names")
    if (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], list)
    ):
        names = names[0]
    if not isinstance(names, list) or tuple(names) != EXPECTED_STATE_NAMES:
        raise ValueError(
            "Dataset info has incompatible `observation.state` names; "
            f"expected={list(EXPECTED_STATE_NAMES)}, actual={names!r}."
        )


def _combined_terminal_fingerprint(
    parquet_path: Path,
    info: Mapping,
    expected_episode_index: int,
    expected_length: int,
) -> str:
    _validate_state_feature_contract(info)
    if expected_length <= 0:
        raise ValueError(
            f"Combined episode {expected_episode_index} has invalid length "
            f"{expected_length}."
        )
    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows != expected_length:
        raise ValueError(
            f"Combined episode {expected_episode_index} length mismatch: "
            f"expected={expected_length}, parquet={parquet.metadata.num_rows}."
        )
    schema = parquet.schema_arrow
    expected_arrow_types = {
        "episode_index": pa.int64(),
        "frame_index": pa.int64(),
    }
    for column, expected_type in expected_arrow_types.items():
        if column not in schema.names or schema.field(column).type != expected_type:
            actual_type = schema.field(column).type if column in schema.names else None
            raise ValueError(
                f"Combined episode has invalid Arrow type for {column}: "
                f"{actual_type}; expected={expected_type}."
            )
    if "observation.state" not in schema.names:
        raise ValueError("Combined episode has no Arrow `observation.state` column.")
    state_type = schema.field("observation.state").type
    if not (
        pa.types.is_list(state_type)
        or pa.types.is_large_list(state_type)
        or pa.types.is_fixed_size_list(state_type)
    ) or state_type.value_type != pa.float32():
        raise ValueError(
            "Combined episode `observation.state` must be an Arrow list of "
            f"float32, got {state_type}."
        )
    table = parquet.read(
        columns=["observation.state", "episode_index", "frame_index"]
    )
    if table.num_rows != expected_length:
        raise ValueError(
            f"Combined episode changed length while reading: {parquet_path}."
        )
    if table["episode_index"].null_count or table["frame_index"].null_count:
        raise ValueError(f"Combined episode has null identity fields: {parquet_path}")
    episode_indices = table["episode_index"].combine_chunks().to_numpy()
    frame_indices = table["frame_index"].combine_chunks().to_numpy()
    if not np.array_equal(
        episode_indices, np.full(expected_length, expected_episode_index, dtype=np.int64)
    ):
        raise ValueError(
            f"Combined episode has invalid episode_index sequence: {parquet_path}."
        )
    expected_frames = np.arange(expected_length, dtype=np.int64)
    if not np.array_equal(frame_indices, expected_frames):
        raise ValueError(
            f"Combined episode has invalid frame_index sequence: {parquet_path}."
        )
    terminal_rows = np.flatnonzero(frame_indices == expected_length - 1)
    if len(terminal_rows) != 1 or int(terminal_rows[0]) != table.num_rows - 1:
        raise ValueError(
            f"Combined episode terminal frame is not the final row: {parquet_path}."
        )
    states = table["observation.state"].combine_chunks()
    if states.null_count or states.values.null_count:
        raise ValueError(f"Combined episode has null state values: {parquet_path}")
    state_rows = states.to_pylist()
    if any(not isinstance(row, list) or len(row) != 14 for row in state_rows):
        raise ValueError(
            f"Combined episode has a non-14D state row: {parquet_path}."
        )
    terminal_state = np.asarray(state_rows[-1], dtype="<f4")
    if terminal_state.shape != (14,) or not np.isfinite(terminal_state).all():
        raise ValueError(
            f"Combined episode has invalid terminal state: {parquet_path}, "
            f"shape={terminal_state.shape}."
        )
    terminal_joints = np.concatenate(
        (terminal_state[:6], terminal_state[7:13])
    ).astype("<f4", copy=False)
    return hashlib.sha256(terminal_joints.tobytes()).hexdigest()


def _match_task_episodes(
    episodes_path: Path,
    info: Mapping,
    canonical_task: str,
    official_sources: Mapping[tuple[str, int], Mapping],
    dataset_root: Path | None = None,
    use_trajectory_fingerprints: bool = False,
) -> tuple[list[dict], dict]:
    prompt_to_sources: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for source, evidence in official_sources.items():
        for prompt in evidence["seen_prompts"]:
            prompt_to_sources[prompt].add(source)

    matched = []
    source_to_combined: dict[tuple[str, int], int] = {}
    ambiguous = []
    instruction_unique_matches = 0
    trajectory_fingerprint_matches = 0
    if use_trajectory_fingerprints:
        if dataset_root is None:
            raise ValueError("Trajectory disambiguation requires `dataset_root`.")
        fingerprints = [
            evidence.get("terminal_fingerprint_sha256")
            for evidence in official_sources.values()
        ]
        if any(fingerprint is None for fingerprint in fingerprints):
            raise ValueError("Official trajectory fingerprints are incomplete.")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError(
                f"{canonical_task} official trajectory fingerprints are not unique."
            )
    scanned = 0
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            if episode_index != scanned:
                raise ValueError(
                    f"episodes.jsonl is not contiguous at line {line_number}: "
                    f"expected={scanned}, actual={episode_index}."
                )
            scanned += 1
            prompts = episode.get("tasks")
            if not isinstance(prompts, list) or not prompts or not all(
                isinstance(prompt, str) for prompt in prompts
            ):
                raise ValueError(f"Episode {episode_index} has invalid task prompts.")
            prompt_set = frozenset(prompts)
            candidates: set[tuple[str, int]] | None = None
            for prompt in prompt_set:
                prompt_sources = prompt_to_sources.get(prompt)
                if not prompt_sources:
                    candidates = set()
                    break
                candidates = (
                    set(prompt_sources)
                    if candidates is None
                    else candidates.intersection(prompt_sources)
                )
                if not candidates:
                    break
            if not candidates:
                continue
            chunk = episode_index // int(info["chunks_size"])
            data_path = str(info["data_path"]).format(
                episode_chunk=chunk,
                episode_index=episode_index,
            )
            combined_terminal_sha256 = None
            if len(candidates) != 1 and use_trajectory_fingerprints:
                combined_terminal_sha256 = _combined_terminal_fingerprint(
                    dataset_root / data_path,
                    info,
                    episode_index,
                    int(episode["length"]),
                )
                candidates = {
                    source
                    for source in candidates
                    if official_sources[source]["terminal_fingerprint_sha256"]
                    == combined_terminal_sha256
                }
                if len(candidates) == 1:
                    trajectory_fingerprint_matches += 1
            if len(candidates) != 1:
                ambiguous.append(
                    {
                        "episode_index": episode_index,
                        "candidate_count": len(candidates),
                        "candidate_head": sorted(candidates)[:10],
                        "trajectory_fallback_used": use_trajectory_fingerprints,
                    }
                )
                continue
            if combined_terminal_sha256 is None:
                instruction_unique_matches += 1
            phase, source_episode_index = next(iter(candidates))
            source = (phase, source_episode_index)
            if source in source_to_combined:
                raise ValueError(
                    f"Official source {canonical_task}/{phase}/{source_episode_index} "
                    f"matches combined episodes {source_to_combined[source]} and "
                    f"{episode_index}."
                )
            source_to_combined[source] = episode_index
            evidence = official_sources[source]
            matched.append(
                {
                    "episode_index": episode_index,
                    "length": int(episode["length"]),
                    "data_path": data_path,
                    "canonical_task": canonical_task,
                    "phase": phase,
                    "source_episode_index": source_episode_index,
                    "source_archive_path": evidence["archive_path"],
                    "source_member_path": evidence["member_path"],
                    "combined_instruction_count": len(prompt_set),
                    "combined_instruction_set_sha256": _canonical_json_sha256(
                        sorted(prompt_set)
                    ),
                    "official_seen_instruction_set_sha256": evidence[
                        "seen_instruction_set_sha256"
                    ],
                    "official_instruction_json_sha256": evidence["json_sha256"],
                    "source_trajectory_member_path": evidence.get(
                        "trajectory_member_path"
                    ),
                    "official_trajectory_pickle_sha256": evidence.get(
                        "trajectory_pickle_sha256"
                    ),
                    "official_terminal_fingerprint_sha256": evidence.get(
                        "terminal_fingerprint_sha256"
                    ),
                    "combined_terminal_fingerprint_sha256": (
                        combined_terminal_sha256
                    ),
                    "match_evidence": (
                        "canonical_task_by_instruction_exact_subset;"
                        "source_by_exact_terminal_trajectory_fingerprint"
                        if combined_terminal_sha256 is not None
                        else "source_by_instruction_exact_subset"
                    ),
                }
            )

    if scanned != int(info["total_episodes"]):
        raise ValueError(
            f"episodes.jsonl count mismatch: scanned={scanned}, "
            f"declared={info['total_episodes']}."
        )
    if ambiguous:
        error_type = (
            ValueError
            if use_trajectory_fingerprints
            else AmbiguousInstructionMatchError
        )
        raise error_type(
            f"{canonical_task} has unresolved exact-subset matches: {ambiguous[:5]}"
        )
    missing_sources = set(official_sources) - set(source_to_combined)
    if missing_sources or len(matched) != len(official_sources):
        raise ValueError(
            f"{canonical_task} official-source coverage mismatch: "
            f"matched={len(matched)}/{len(official_sources)}, "
            f"missing={sorted(missing_sources)[:10]}."
        )
    phase_counts = Counter(record["phase"] for record in matched)
    expected_phase_counts = Counter(PHASE_EPISODE_COUNTS)
    if phase_counts != expected_phase_counts:
        raise ValueError(
            f"{canonical_task} phase counts mismatch: {dict(phase_counts)}."
        )
    numeric_order_by_phase = {}
    for phase in PHASE_EPISODE_COUNTS:
        source_order = [
            record["source_episode_index"]
            for record in sorted(matched, key=lambda record: record["episode_index"])
            if record["phase"] == phase
        ]
        numeric_order_by_phase[phase] = source_order == sorted(source_order)
    return matched, {
        "matched": len(matched),
        "ambiguous": 0,
        "missing_official_sources": 0,
        "instruction_unique_matches": instruction_unique_matches,
        "trajectory_fingerprint_matches": trajectory_fingerprint_matches,
        "phase_counts": dict(sorted(phase_counts.items())),
        "combined_order_matches_numeric_source_order": numeric_order_by_phase,
    }


def _constant_ints(table, column: str) -> set[int]:
    values = table[column].combine_chunks().to_pylist()
    return {int(value) for value in values}


def _validate_selected_parquets(
    dataset_root: Path, info: Mapping, episodes: list[dict]
) -> None:
    needed_task_indices: set[int] = set()
    episode_task_indices: dict[int, tuple[int, ...]] = {}
    for episode in tqdm(episodes, desc="Validating selected RoboTwin parquets"):
        episode_index = int(episode["episode_index"])
        parquet_path = dataset_root / episode["data_path"]
        parquet = pq.ParquetFile(parquet_path)
        if parquet.metadata.num_rows != int(episode["length"]):
            raise ValueError(
                f"Episode {episode_index} length mismatch: "
                f"episodes.jsonl={episode['length']}, "
                f"parquet={parquet.metadata.num_rows}."
            )
        table = parquet.read(columns=["episode_index", "frame_index", "task_index"])
        parquet_episode_indices = _constant_ints(table, "episode_index")
        if parquet_episode_indices != {episode_index}:
            raise ValueError(
                f"Episode identity mismatch for {parquet_path}: "
                f"{sorted(parquet_episode_indices)}."
            )
        frame_indices = table["frame_index"].combine_chunks().to_numpy()
        if not np.array_equal(
            frame_indices, np.arange(int(episode["length"]), dtype=np.int64)
        ):
            raise ValueError(
                f"Episode {episode_index} has an invalid frame_index sequence."
            )
        task_indices = tuple(sorted(_constant_ints(table, "task_index")))
        if not task_indices:
            raise ValueError(f"Episode {episode_index} contains no task indices.")
        episode_task_indices[episode_index] = task_indices
        needed_task_indices.update(task_indices)

    task_labels = {}
    tasks_path = dataset_root / "meta/tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_index = int(record["task_index"])
            if task_index != line_number - 1:
                raise ValueError(
                    f"tasks.jsonl is not contiguous at line {line_number}: "
                    f"task_index={task_index}."
                )
            if task_index in needed_task_indices:
                task_labels[task_index] = str(record["task"])
    missing_task_indices = needed_task_indices - set(task_labels)
    if missing_task_indices:
        raise ValueError(
            f"Selected parquets reference unknown task indices: "
            f"{sorted(missing_task_indices)[:10]}."
        )
    for episode in episodes:
        labels = sorted(
            {task_labels[index] for index in episode_task_indices[episode["episode_index"]]}
        )
        parquet_instruction_sha256 = _canonical_json_sha256(labels)
        if parquet_instruction_sha256 != episode["combined_instruction_set_sha256"]:
            raise ValueError(
                f"Episode {episode['episode_index']} task labels disagree between "
                "Parquet/task registry and episodes.jsonl."
            )
        episode["parquet_task_registry_validated"] = True


def build_manifest(
    dataset_root: Path,
    canonical_tasks: Sequence[str],
    registry_provenance: Mapping,
    official_repository: str,
    requested_revision: str,
    embodiment: str,
    maximum_instruction_span_bytes: int,
    maximum_trajectory_span_bytes: int,
    validate_parquet_tasks: bool,
) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    info = _read_json(dataset_root / "meta/info.json")
    _validate_state_feature_contract(info)
    expected_indices = set(range(int(info["total_episodes"])))
    if int(info["total_episodes"]) <= 0 or int(info["chunks_size"]) <= 0:
        raise ValueError("Dataset info declares invalid episode metadata.")

    resolved_revision = _resolve_official_revision(
        official_repository, requested_revision
    )
    archive_paths = _official_archive_paths(canonical_tasks, embodiment)
    archive_metadata = _fetch_archive_metadata(
        official_repository, resolved_revision, list(archive_paths.values())
    )

    all_episodes = []
    official_archives = []
    task_audit = {}
    claimed_combined_indices: dict[int, str] = {}
    episodes_path = dataset_root / "meta/episodes.jsonl"
    for task_number, canonical_task in enumerate(canonical_tasks, start=1):
        task_sources = {}
        task_archive_records = {}
        for phase, expected_count in PHASE_EPISODE_COUNTS.items():
            archive_path = archive_paths[(canonical_task, phase)]
            sources, range_evidence = _fetch_official_instructions(
                official_repository,
                resolved_revision,
                archive_path,
                archive_metadata[archive_path],
                expected_count,
                maximum_instruction_span_bytes,
            )
            for source_episode_index, evidence in sources.items():
                task_sources[(phase, source_episode_index)] = {
                    **evidence,
                    "archive_path": archive_path,
                }
            archive_record = {
                "task": canonical_task,
                "phase": phase,
                "path": archive_path,
                "episode_count": expected_count,
                "size_bytes": archive_metadata[archive_path]["size_bytes"],
                "lfs_sha256": archive_metadata[archive_path]["lfs_sha256"],
                "trajectory_registry_sha256": None,
                **range_evidence,
            }
            official_archives.append(archive_record)
            task_archive_records[phase] = archive_record
        try:
            matched, audit = _match_task_episodes(
                episodes_path, info, canonical_task, task_sources
            )
        except AmbiguousInstructionMatchError:
            print(
                f"[{task_number}/{len(canonical_tasks)}] {canonical_task}: "
                "instruction source is ambiguous; validating official trajectory "
                "fingerprints",
                flush=True,
            )
            for phase, expected_count in PHASE_EPISODE_COUNTS.items():
                archive_path = archive_paths[(canonical_task, phase)]
                trajectories, trajectory_evidence = (
                    _fetch_official_trajectory_fingerprints(
                        official_repository,
                        resolved_revision,
                        archive_path,
                        archive_metadata[archive_path],
                        expected_count,
                        maximum_trajectory_span_bytes,
                    )
                )
                for source_episode_index, evidence in trajectories.items():
                    task_sources[(phase, source_episode_index)].update(evidence)
                task_archive_records[phase].update(trajectory_evidence)
            matched, audit = _match_task_episodes(
                episodes_path,
                info,
                canonical_task,
                task_sources,
                dataset_root=dataset_root,
                use_trajectory_fingerprints=True,
            )
        for episode in matched:
            episode_index = episode["episode_index"]
            previous_task = claimed_combined_indices.get(episode_index)
            if previous_task is not None:
                raise ValueError(
                    f"Combined episode {episode_index} matches canonical tasks "
                    f"{previous_task!r} and {canonical_task!r}."
                )
            claimed_combined_indices[episode_index] = canonical_task
        all_episodes.extend(matched)
        task_audit[canonical_task] = audit
        print(
            f"[{task_number}/{len(canonical_tasks)}] {canonical_task}: "
            f"matched={audit['matched']}, phases={audit['phase_counts']}",
            flush=True,
        )

    if not set(claimed_combined_indices).issubset(expected_indices):
        raise ValueError("Mapped episode index is outside the declared dataset range.")
    all_episodes.sort(key=lambda record: record["episode_index"])
    if validate_parquet_tasks:
        _validate_selected_parquets(dataset_root, info, all_episodes)
    else:
        for episode in all_episodes:
            episode["parquet_task_registry_validated"] = False

    task_summary = {}
    for task in canonical_tasks:
        task_episodes = [
            episode for episode in all_episodes if episode["canonical_task"] == task
        ]
        task_summary[task] = {
            "episodes": len(task_episodes),
            "frames": sum(episode["length"] for episode in task_episodes),
            "phases": dict(
                sorted(Counter(episode["phase"] for episode in task_episodes).items())
            ),
        }

    official_archives.sort(
        key=lambda record: (record["task"], record["phase"], record["path"])
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "root_name": dataset_root.name,
            "total_episodes": int(info["total_episodes"]),
            "total_frames": int(info["total_frames"]),
            "fps": int(info["fps"]),
            "metadata_sha256": {
                relative_path: sha256_file(dataset_root / relative_path)
                for relative_path in METADATA_FILES
            },
            "canonical_registry": {
                **registry_provenance,
                "tasks": list(canonical_tasks),
            },
            "official": {
                "repository": official_repository,
                "requested_revision": requested_revision,
                "revision": resolved_revision,
                "embodiment": embodiment,
                "archives": official_archives,
            },
        },
        "audit": {
            "match_method": (
                "instruction exact-subset; pinned official terminal-trajectory "
                "fingerprint only when source episode is ambiguous"
            ),
            "uses_combined_episode_block_arithmetic": False,
            "mapped_episodes": len(all_episodes),
            "official_sources": sum(PHASE_EPISODE_COUNTS.values())
            * len(canonical_tasks),
            "unique_combined_episodes": len(claimed_combined_indices),
            "unique_official_sources": len(all_episodes),
            "ambiguous_matches": 0,
            "unmatched_official_sources": 0,
            "instruction_unique_matches": sum(
                audit["instruction_unique_matches"] for audit in task_audit.values()
            ),
            "trajectory_fingerprint_matches": sum(
                audit["trajectory_fingerprint_matches"]
                for audit in task_audit.values()
            ),
            "parquet_task_registry_validated": validate_parquet_tasks,
            "tasks": task_audit,
        },
        "task_summary": task_summary,
        "episodes": all_episodes,
    }
    manifest["mapping_sha256"] = compute_manifest_mapping_sha256(manifest)
    return manifest


def _atomic_write_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--canonical-registry", type=Path)
    task_source.add_argument(
        "--canonical-task",
        action="append",
        dest="canonical_tasks",
        help="Repeat once per canonical task.",
    )
    parser.add_argument(
        "--official-repository", default=DEFAULT_OFFICIAL_REPOSITORY
    )
    parser.add_argument(
        "--official-revision",
        default="main",
        help="HF revision to resolve and pin in the manifest.",
    )
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument("--maximum-instruction-span-mib", type=int, default=16)
    parser.add_argument("--maximum-trajectory-span-mib", type=int, default=128)
    parser.add_argument("--skip-parquet-task-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_tasks, registry_provenance = _load_canonical_tasks(
        args.canonical_registry, args.canonical_tasks
    )
    manifest = build_manifest(
        dataset_root=args.dataset_root,
        canonical_tasks=canonical_tasks,
        registry_provenance=registry_provenance,
        official_repository=args.official_repository,
        requested_revision=args.official_revision,
        embodiment=args.embodiment,
        maximum_instruction_span_bytes=args.maximum_instruction_span_mib
        * 1024
        * 1024,
        maximum_trajectory_span_bytes=args.maximum_trajectory_span_mib
        * 1024
        * 1024,
        validate_parquet_tasks=not args.skip_parquet_task_validation,
    )
    _atomic_write_json(manifest, args.output)
    print(
        f"Wrote {len(manifest['episodes'])} episodes for "
        f"{len(manifest['task_summary'])} tasks to {args.output}"
    )
    print(
        f"official_revision={manifest['source']['official']['revision']}\n"
        f"mapping_sha256={manifest['mapping_sha256']}"
    )


if __name__ == "__main__":
    main()
