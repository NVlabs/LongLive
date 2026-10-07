# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/utils/index_manifest_sampler.py
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

"""Immutable index-manifest sampling for a single, resumable data pass.

The manifest stores an ordered, unique set of dataset indices plus the bitmap of
indices consumed before the pass.  Loading is intentionally strict: payload
hashes, membership proofs, dataset identity, source-state identity, and the
distributed batch contract must all agree before an iterator is exposed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Sized

import numpy as np
from torch.utils.data import Sampler


MANIFEST_SCHEMA = "longwam.index-complement-manifest/v1"
INDEX_DTYPE = np.dtype("<u4")
BIT_ORDER = "little"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field} keys changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )
    return value


def _ranges_sha256(ranges: Sequence[Sequence[int]]) -> str:
    normalized = [[int(start), int(stop)] for start, stop in ranges]
    return canonical_sha256(normalized)


def _packed_membership(mask: np.ndarray) -> bytes:
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise TypeError("Membership masks must be one-dimensional numpy bool arrays.")
    return np.packbits(mask, bitorder=BIT_ORDER).tobytes()


def membership_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(_packed_membership(mask)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_array(path: Path, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values, dtype=INDEX_DTYPE)
    _atomic_write_bytes(path, contiguous.tobytes(order="C"))


def _validate_ranges(ranges: Any, dataset_length: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)) or not ranges:
        raise ValueError("dataset.ranges must be a non-empty sequence.")
    normalized = tuple((int(item[0]), int(item[1])) for item in ranges)
    expected_start = 0
    for start, stop in normalized:
        if start != expected_start or stop <= start:
            raise ValueError("dataset.ranges must be contiguous, non-empty half-open ranges.")
        expected_start = stop
    if expected_start != dataset_length:
        raise ValueError(f"dataset.ranges end at {expected_start}, expected {dataset_length}.")
    return normalized


def _validate_family_ranges(raw_ranges: Any, dataset_length: int) -> dict[str, tuple[int, int]]:
    if not isinstance(raw_ranges, Mapping) or not raw_ranges:
        raise ValueError("dataset.family_ranges must be a non-empty mapping.")
    result: dict[str, tuple[int, int]] = {}
    expected_start = 0
    for raw_name, raw_range in raw_ranges.items():
        name = str(raw_name)
        if not name or not isinstance(raw_range, Sequence) or len(raw_range) != 2:
            raise ValueError("dataset.family_ranges contains an invalid entry.")
        start, stop = (int(raw_range[0]), int(raw_range[1]))
        if start != expected_start or stop <= start:
            raise ValueError(
                "dataset.family_ranges must be contiguous, non-empty half-open ranges."
            )
        result[name] = (start, stop)
        expected_start = stop
    if expected_start != dataset_length:
        raise ValueError("dataset.family_ranges does not cover the full dataset.")
    return result


def write_complement_manifest(
    output_path: str | Path,
    *,
    ordered_indices: np.ndarray,
    seen_mask: np.ndarray,
    dataset_identity: Mapping[str, Any],
    history_identity: Mapping[str, Any],
    shuffle_seed: int,
    world_size: int,
    per_rank_batch_size: int,
) -> dict[str, Any]:
    """Atomically write a complement manifest and its two binary payloads.

    ``ordered_indices`` must be a permutation of exactly ``~seen_mask``.  The
    index payload is uint32 little-endian; the proof payload is a little-bit-
    ordered packed bitmap.  Existing outputs are never overwritten.
    """

    output_path = Path(output_path).expanduser().resolve(strict=False)
    index_path = output_path.with_suffix(".indices.u32le")
    seen_path = output_path.with_suffix(".seen.bitset")
    for path in (output_path, index_path, seen_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")

    seen_mask = np.asarray(seen_mask, dtype=np.bool_)
    ordered_indices = np.asarray(ordered_indices, dtype=INDEX_DTYPE)
    if seen_mask.ndim != 1 or not 0 < seen_mask.size <= np.iinfo(np.uint32).max:
        raise ValueError("seen_mask must describe 1..2^32-1 dataset indices.")
    if ordered_indices.ndim != 1 or ordered_indices.size == 0:
        raise ValueError("ordered_indices must be a non-empty one-dimensional array.")

    dataset_length = int(seen_mask.size)
    if int(ordered_indices.max()) >= dataset_length:
        raise ValueError("ordered_indices contains an out-of-range dataset index.")
    complement_mask = np.zeros(dataset_length, dtype=np.bool_)
    complement_mask[ordered_indices] = True
    unique_count = int(np.count_nonzero(complement_mask))
    if unique_count != ordered_indices.size:
        raise ValueError("ordered_indices contains duplicates.")

    disjoint_count = int(np.count_nonzero(seen_mask & complement_mask))
    union_mask = seen_mask | complement_mask
    union_count = int(np.count_nonzero(union_mask))
    if disjoint_count != 0 or union_count != dataset_length:
        raise ValueError(
            "ordered_indices is not the exact complement of seen_mask: "
            f"disjoint={disjoint_count}, union={union_count}, total={dataset_length}."
        )

    dataset = _require_exact_keys(
        dataset_identity,
        {
            "selection_signature",
            "sampling_signature",
            "historical_group_signature",
            "ranges",
            "group_names",
            "family_ranges",
        },
        "dataset_identity",
    )
    ranges = _validate_ranges(dataset["ranges"], dataset_length)
    group_names = tuple(str(name) for name in dataset["group_names"])
    if len(group_names) != len(ranges) or any(not name for name in group_names):
        raise ValueError("dataset_identity requires one non-empty name per range.")
    family_ranges = _validate_family_ranges(dataset["family_ranges"], dataset_length)
    for field in (
        "selection_signature",
        "sampling_signature",
        "historical_group_signature",
    ):
        _require_sha256(dataset[field], f"dataset_identity.{field}")

    history = dict(history_identity)
    _require_exact_keys(
        history,
        {
            "source_plan",
            "continuation_plan",
            "source_state",
            "historical_sampler_sha256",
            "draw_count",
        },
        "history_identity",
    )
    _require_sha256(
        history["historical_sampler_sha256"],
        "history_identity.historical_sampler_sha256",
    )
    for name in ("source_plan", "continuation_plan", "source_state"):
        item = _require_exact_keys(
            history[name],
            {"path", "file_sha256", "contract_sha256"},
            f"history_identity.{name}",
        )
        _require_sha256(item["file_sha256"], f"history_identity.{name}.file_sha256")
        _require_sha256(
            item["contract_sha256"],
            f"history_identity.{name}.contract_sha256",
        )
    draw_count = int(history["draw_count"])
    if draw_count < int(np.count_nonzero(seen_mask)):
        raise ValueError("history draw_count cannot be smaller than its unique count.")

    world_size = int(world_size)
    per_rank_batch_size = int(per_rank_batch_size)
    shuffle_seed = int(shuffle_seed)
    if world_size <= 0 or per_rank_batch_size <= 0 or shuffle_seed < 0:
        raise ValueError("world size, batch size, and shuffle seed are invalid.")
    global_batch_size = world_size * per_rank_batch_size
    padded_count = math.ceil(unique_count / global_batch_size) * global_batch_size
    padding_count = padded_count - unique_count
    if padding_count >= unique_count:
        raise ValueError("The deterministic final-batch padding contract is invalid.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_array(index_path, ordered_indices)
    try:
        _atomic_write_bytes(seen_path, _packed_membership(seen_mask))
        padding = ordered_indices[:padding_count]
        task_counts: dict[str, dict[str, int]] = {}
        for name, (start, stop) in zip(group_names, ranges, strict=True):
            task_counts[name] = {
                "total": stop - start,
                "seen": int(np.count_nonzero(seen_mask[start:stop])),
                "complement": int(np.count_nonzero(complement_mask[start:stop])),
            }
        family_counts = {
            name: {
                "total": stop - start,
                "seen": int(np.count_nonzero(seen_mask[start:stop])),
                "complement": int(np.count_nonzero(complement_mask[start:stop])),
            }
            for name, (start, stop) in family_ranges.items()
        }

        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "kind": "exact_unseen_index_complement",
            "dataset": {
                "length": dataset_length,
                "index_range": [0, dataset_length],
                "selection_signature": dataset["selection_signature"],
                "sampling_signature": dataset["sampling_signature"],
                "historical_group_signature": dataset["historical_group_signature"],
                "ranges": [list(item) for item in ranges],
                "ranges_sha256": _ranges_sha256(ranges),
                "group_names": list(group_names),
                "family_ranges": {
                    name: [start, stop] for name, (start, stop) in family_ranges.items()
                },
                "family_ranges_sha256": canonical_sha256(family_ranges),
            },
            "history": history,
            "payloads": {
                "ordered_indices": {
                    "path": index_path.name,
                    "format": "uint32_little_endian_v1",
                    "count": unique_count,
                    "nbytes": int(index_path.stat().st_size),
                    "sha256": sha256_file(index_path),
                    "ordering": "numpy_pcg64_in_place_shuffle_v1",
                    "shuffle_seed": shuffle_seed,
                },
                "seen_bitmap": {
                    "path": seen_path.name,
                    "format": "numpy_packbits_little_v1",
                    "count": dataset_length,
                    "nbytes": int(seen_path.stat().st_size),
                    "sha256": sha256_file(seen_path),
                },
            },
            "proof": {
                "draw_count": draw_count,
                "seen_unique_count": int(np.count_nonzero(seen_mask)),
                "complement_count": unique_count,
                "complement_unique_count": unique_count,
                "complement_min": int(ordered_indices.min()),
                "complement_max": int(ordered_indices.max()),
                "intersection_count": disjoint_count,
                "union_count": union_count,
                "seen_membership_sha256": membership_sha256(seen_mask),
                "complement_membership_sha256": membership_sha256(complement_mask),
                "union_membership_sha256": membership_sha256(union_mask),
                "task_counts": task_counts,
                "family_counts": family_counts,
            },
            "distribution": {
                "world_size": world_size,
                "per_rank_batch_size": per_rank_batch_size,
                "global_batch_size": global_batch_size,
                "unique_count": unique_count,
                "padded_count": padded_count,
                "padding_count": padding_count,
                "padding_policy": "repeat_order_prefix_in_final_global_batch_v1",
                "padding_indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(padding, dtype=INDEX_DTYPE).tobytes()
                ).hexdigest(),
                "global_steps": padded_count // global_batch_size,
            },
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _atomic_write_bytes(
            output_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return manifest
    except BaseException:
        index_path.unlink(missing_ok=True)
        seen_path.unlink(missing_ok=True)
        raise


def _safe_payload_path(manifest_path: Path, raw_name: Any, field: str) -> Path:
    if not isinstance(raw_name, str) or not raw_name or Path(raw_name).name != raw_name:
        raise ValueError(f"{field} must be a plain payload filename.")
    path = manifest_path.parent / raw_name
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest payload: {path}")
    return path


def load_index_manifest(
    manifest_path: str | Path,
    *,
    verify_payload_membership: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Load and cryptographically/procedurally verify an index manifest."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid index manifest {manifest_path}: {exc}") from exc
    top = _require_exact_keys(
        manifest,
        {
            "schema",
            "kind",
            "dataset",
            "history",
            "payloads",
            "proof",
            "distribution",
            "manifest_sha256",
        },
        "manifest",
    )
    if top["schema"] != MANIFEST_SCHEMA or top["kind"] != "exact_unseen_index_complement":
        raise ValueError("Unsupported index manifest schema or kind.")
    expected_manifest_sha256 = _require_sha256(top["manifest_sha256"], "manifest.manifest_sha256")
    unsigned = dict(top)
    unsigned.pop("manifest_sha256")
    if canonical_sha256(unsigned) != expected_manifest_sha256:
        raise ValueError("Index manifest self-hash mismatch.")

    dataset = _require_exact_keys(
        top["dataset"],
        {
            "length",
            "index_range",
            "selection_signature",
            "sampling_signature",
            "historical_group_signature",
            "ranges",
            "ranges_sha256",
            "group_names",
            "family_ranges",
            "family_ranges_sha256",
        },
        "manifest.dataset",
    )
    dataset_length = int(dataset["length"])
    if dataset["index_range"] != [0, dataset_length]:
        raise ValueError("Manifest dataset index range changed.")
    ranges = _validate_ranges(dataset["ranges"], dataset_length)
    if dataset["ranges_sha256"] != _ranges_sha256(ranges):
        raise ValueError("Manifest dataset range hash mismatch.")
    if len(dataset["group_names"]) != len(ranges):
        raise ValueError("Manifest group names/ranges length mismatch.")
    family_ranges = _validate_family_ranges(dataset["family_ranges"], dataset_length)
    if dataset["family_ranges_sha256"] != canonical_sha256(family_ranges):
        raise ValueError("Manifest family-range hash mismatch.")
    for field in (
        "selection_signature",
        "sampling_signature",
        "historical_group_signature",
    ):
        _require_sha256(dataset[field], f"manifest.dataset.{field}")

    history = _require_exact_keys(
        top["history"],
        {
            "source_plan",
            "continuation_plan",
            "source_state",
            "historical_sampler_sha256",
            "draw_count",
        },
        "manifest.history",
    )
    _require_sha256(
        history["historical_sampler_sha256"],
        "manifest.history.historical_sampler_sha256",
    )
    for name in ("source_plan", "continuation_plan", "source_state"):
        artifact = _require_exact_keys(
            history[name],
            {"path", "file_sha256", "contract_sha256"},
            f"manifest.history.{name}",
        )
        _require_sha256(artifact["file_sha256"], f"manifest.history.{name}.file_sha256")
        _require_sha256(
            artifact["contract_sha256"],
            f"manifest.history.{name}.contract_sha256",
        )

    proof = _require_exact_keys(
        top["proof"],
        {
            "draw_count",
            "seen_unique_count",
            "complement_count",
            "complement_unique_count",
            "complement_min",
            "complement_max",
            "intersection_count",
            "union_count",
            "seen_membership_sha256",
            "complement_membership_sha256",
            "union_membership_sha256",
            "task_counts",
            "family_counts",
        },
        "manifest.proof",
    )
    if int(proof["draw_count"]) != int(history["draw_count"]):
        raise ValueError("Manifest history/proof draw counts disagree.")

    payloads = _require_exact_keys(
        top["payloads"], {"ordered_indices", "seen_bitmap"}, "manifest.payloads"
    )
    index_info = _require_exact_keys(
        payloads["ordered_indices"],
        {
            "path",
            "format",
            "count",
            "nbytes",
            "sha256",
            "ordering",
            "shuffle_seed",
        },
        "manifest.payloads.ordered_indices",
    )
    seen_info = _require_exact_keys(
        payloads["seen_bitmap"],
        {"path", "format", "count", "nbytes", "sha256"},
        "manifest.payloads.seen_bitmap",
    )
    if (
        index_info["format"] != "uint32_little_endian_v1"
        or index_info["ordering"] != "numpy_pcg64_in_place_shuffle_v1"
        or seen_info["format"] != "numpy_packbits_little_v1"
    ):
        raise ValueError("Unsupported index manifest payload format.")
    index_path = _safe_payload_path(manifest_path, index_info["path"], "ordered_indices.path")
    seen_path = _safe_payload_path(manifest_path, seen_info["path"], "seen_bitmap.path")
    index_count = int(index_info["count"])
    expected_index_bytes = index_count * INDEX_DTYPE.itemsize
    expected_seen_bytes = math.ceil(dataset_length / 8)
    if (
        int(index_info["nbytes"]) != expected_index_bytes
        or index_path.stat().st_size != expected_index_bytes
        or int(seen_info["count"]) != dataset_length
        or int(seen_info["nbytes"]) != expected_seen_bytes
        or seen_path.stat().st_size != expected_seen_bytes
    ):
        raise ValueError("Manifest payload size/count contract mismatch.")
    if sha256_file(index_path) != _require_sha256(index_info["sha256"], "ordered_indices.sha256"):
        raise ValueError("Ordered-index payload SHA-256 mismatch.")
    if sha256_file(seen_path) != _require_sha256(seen_info["sha256"], "seen_bitmap.sha256"):
        raise ValueError("Seen-bitmap payload SHA-256 mismatch.")

    distribution = _require_exact_keys(
        top["distribution"],
        {
            "world_size",
            "per_rank_batch_size",
            "global_batch_size",
            "unique_count",
            "padded_count",
            "padding_count",
            "padding_policy",
            "padding_indices_sha256",
            "global_steps",
        },
        "manifest.distribution",
    )
    world_size = int(distribution["world_size"])
    per_rank_batch = int(distribution["per_rank_batch_size"])
    if world_size <= 0 or per_rank_batch <= 0:
        raise ValueError("Manifest world size and per-rank batch must be positive.")
    global_batch = world_size * per_rank_batch
    padded_count = math.ceil(index_count / global_batch) * global_batch
    padding_count = padded_count - index_count
    if (
        int(distribution["global_batch_size"]) != global_batch
        or int(distribution["unique_count"]) != index_count
        or int(distribution["padded_count"]) != padded_count
        or int(distribution["padding_count"]) != padding_count
        or distribution["padding_policy"] != "repeat_order_prefix_in_final_global_batch_v1"
        or int(distribution["global_steps"]) != padded_count // global_batch
    ):
        raise ValueError("Manifest distributed padding contract mismatch.")
    _require_sha256(
        distribution["padding_indices_sha256"],
        "manifest.distribution.padding_indices_sha256",
    )

    if verify_payload_membership:
        indices = np.memmap(index_path, mode="r", dtype=INDEX_DTYPE, shape=(index_count,))
        seen_packed = np.fromfile(seen_path, dtype=np.uint8)
        seen_mask = np.unpackbits(seen_packed, bitorder=BIT_ORDER)[:dataset_length].astype(
            np.bool_, copy=False
        )
        complement_mask = np.zeros(dataset_length, dtype=np.bool_)
        if index_count:
            if int(indices.max()) >= dataset_length:
                raise ValueError("Ordered-index payload contains an out-of-range index.")
            complement_mask[indices] = True
        unique_count = int(np.count_nonzero(complement_mask))
        union_mask = seen_mask | complement_mask
        expected_proof = {
            "seen_unique_count": int(np.count_nonzero(seen_mask)),
            "complement_count": index_count,
            "complement_unique_count": unique_count,
            "complement_min": int(indices.min()) if index_count else None,
            "complement_max": int(indices.max()) if index_count else None,
            "intersection_count": int(np.count_nonzero(seen_mask & complement_mask)),
            "union_count": int(np.count_nonzero(union_mask)),
            "seen_membership_sha256": membership_sha256(seen_mask),
            "complement_membership_sha256": membership_sha256(complement_mask),
            "union_membership_sha256": membership_sha256(union_mask),
        }
        for field, expected in expected_proof.items():
            if proof.get(field) != expected:
                raise ValueError(
                    f"Manifest membership proof mismatch for {field}: "
                    f"saved={proof.get(field)!r}, computed={expected!r}."
                )
        if (
            unique_count != index_count
            or expected_proof["intersection_count"] != 0
            or expected_proof["union_count"] != dataset_length
        ):
            raise ValueError("Manifest payloads do not prove an exact disjoint complement.")
        expected_task_counts = {
            str(name): {
                "total": stop - start,
                "seen": int(np.count_nonzero(seen_mask[start:stop])),
                "complement": int(np.count_nonzero(complement_mask[start:stop])),
            }
            for name, (start, stop) in zip(dataset["group_names"], ranges, strict=True)
        }
        expected_family_counts = {
            name: {
                "total": stop - start,
                "seen": int(np.count_nonzero(seen_mask[start:stop])),
                "complement": int(np.count_nonzero(complement_mask[start:stop])),
            }
            for name, (start, stop) in family_ranges.items()
        }
        if proof["task_counts"] != expected_task_counts:
            raise ValueError("Manifest per-task membership proof mismatch.")
        if proof["family_counts"] != expected_family_counts:
            raise ValueError("Manifest family membership proof mismatch.")
        padding = np.ascontiguousarray(indices[:padding_count], dtype=INDEX_DTYPE)
        if hashlib.sha256(padding.tobytes()).hexdigest() != distribution["padding_indices_sha256"]:
            raise ValueError("Manifest deterministic padding proof mismatch.")

    return dict(top), index_path


class IndexManifestSampler(Sampler[int]):
    """One-pass, globally batched sampler backed by an immutable index manifest."""

    strategy = "index_manifest"

    def __init__(
        self,
        dataset: Sized,
        manifest_path: str | Path,
        *,
        seed: int,
        batch_size: int,
        num_processes: int,
        expected_manifest_sha256: str,
        expected_selection_signature: str,
        expected_sampling_signature: str,
        expected_historical_group_signature: str,
        expected_source_state_sha256: str,
    ) -> None:
        self.dataset = dataset
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest, self.index_path = load_index_manifest(self.manifest_path)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("batch_size and num_processes must be positive.")
        self.global_micro_batch_size = self.batch_size * self.num_processes

        expected = {
            "manifest.manifest_sha256": _require_sha256(
                expected_manifest_sha256, "expected_manifest_sha256"
            ),
            "dataset.selection_signature": _require_sha256(
                expected_selection_signature, "expected_selection_signature"
            ),
            "dataset.sampling_signature": _require_sha256(
                expected_sampling_signature, "expected_sampling_signature"
            ),
            "dataset.historical_group_signature": _require_sha256(
                expected_historical_group_signature,
                "expected_historical_group_signature",
            ),
            "history.source_state.file_sha256": _require_sha256(
                expected_source_state_sha256, "expected_source_state_sha256"
            ),
        }
        actual = {
            "manifest.manifest_sha256": self.manifest["manifest_sha256"],
            "dataset.selection_signature": self.manifest["dataset"]["selection_signature"],
            "dataset.sampling_signature": self.manifest["dataset"]["sampling_signature"],
            "dataset.historical_group_signature": self.manifest["dataset"][
                "historical_group_signature"
            ],
            "history.source_state.file_sha256": self.manifest["history"]["source_state"][
                "file_sha256"
            ],
        }
        mismatches = [
            f"{field}: manifest={actual[field]!r}, expected={value!r}"
            for field, value in expected.items()
            if actual[field] != value
        ]
        if mismatches:
            raise ValueError("Index manifest identity mismatch: " + "; ".join(mismatches))

        dataset_contract = self.manifest["dataset"]
        if len(dataset) != int(dataset_contract["length"]):
            raise ValueError(
                f"Dataset length changed: manifest={dataset_contract['length']}, "
                f"runtime={len(dataset)}."
            )
        runtime_ranges = getattr(dataset, "dataset_frame_ranges", None)
        if (
            runtime_ranges is None
            or _ranges_sha256(runtime_ranges) != dataset_contract["ranges_sha256"]
        ):
            raise ValueError("Runtime dataset frame ranges do not match the manifest.")
        runtime_sampling_signature = getattr(dataset, "sampling_signature", None)
        if runtime_sampling_signature != expected_sampling_signature:
            raise ValueError("Runtime dataset sampling signature changed.")
        resolved_selection = getattr(dataset, "human300_episode_selection", None)
        runtime_selection_signature = getattr(resolved_selection, "signature", None)
        if runtime_selection_signature != expected_selection_signature:
            raise ValueError("Runtime official episode selection signature changed.")

        distribution = self.manifest["distribution"]
        if (
            int(distribution["world_size"]) != self.num_processes
            or int(distribution["per_rank_batch_size"]) != self.batch_size
            or int(distribution["global_batch_size"]) != self.global_micro_batch_size
            or int(self.manifest["payloads"]["ordered_indices"]["shuffle_seed"]) != self.seed
        ):
            raise ValueError("Runtime seed or distributed batch contract changed.")
        if distribution["padding_policy"] != "repeat_order_prefix_in_final_global_batch_v1":
            raise ValueError("Unsupported final-batch padding policy.")

        self.unique_samples_per_epoch = int(distribution["unique_count"])
        self.samples_per_epoch = int(distribution["padded_count"])
        self.padding_count = int(distribution["padding_count"])
        if self.samples_per_epoch % self.global_micro_batch_size:
            raise ValueError("Manifest length is not globally batch aligned.")
        self.group_signature = self.manifest["manifest_sha256"]
        self.dataset_frame_ranges = tuple(tuple(item) for item in dataset_contract["ranges"])
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != 0:
            raise RuntimeError(
                "IndexManifestSampler is a sealed one-pass sampler; epoch must remain 0."
            )
        self.epoch = epoch

    def set_epoch_offset(self, epoch_offset: int) -> None:
        epoch_offset = int(epoch_offset)
        if epoch_offset != 0:
            raise RuntimeError("IndexManifestSampler does not permit an epoch offset.")
        self.epoch_offset = epoch_offset

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        batch_in_epoch = int(batch_in_epoch)
        total_batches = self.samples_per_epoch // self.global_micro_batch_size
        if batch_in_epoch < 0 or batch_in_epoch > total_batches:
            raise ValueError(f"Resume batch {batch_in_epoch} is outside [0, {total_batches}].")
        self.resume_batch_offset = batch_in_epoch

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        if self.epoch != 0 or self.epoch_offset != 0:
            raise RuntimeError("Index manifest iteration escaped its sealed epoch.")
        start = self.resume_batch_offset * self.global_micro_batch_size
        unique_count = self.unique_samples_per_epoch
        indices = np.memmap(
            self.index_path,
            mode="r",
            dtype=INDEX_DTYPE,
            shape=(unique_count,),
        )
        try:
            for position in range(start, self.samples_per_epoch):
                payload_position = position if position < unique_count else position - unique_count
                yield int(indices[payload_position])
        finally:
            del indices

    def __len__(self) -> int:
        return self.samples_per_epoch
