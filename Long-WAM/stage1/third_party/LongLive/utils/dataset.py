# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: The Self-Forcing contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: utils/dataset.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/utils/dataset.py
# Source: https://github.com/guandeh17/Self-Forcing
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
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

# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
from collections.abc import Sequence
import fcntl
import hashlib
import mmap
import numpy as np
import torch
import random
import json
from pathlib import Path
from PIL import Image
import os
import subprocess
import struct
import tempfile
import time
import warnings
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

try:
    import decord
except ModuleNotFoundError:
    decord = None

DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "

ROBOT_VIDEO_ROW_CONTRACT_VERSION = "robot-ti2v-length-mixed-index-v1"
ROBOT_VIDEO_RECEIPT_CONTRACT_VERSION = "robot-ti2v-length-mixed-receipt-v1"
ROBOT_VIDEO_ACCEPTED_RECEIPT_STATUS = "accepted_metadata_index_no_media_copy"
ROBOT_VIDEO_CONDITIONING = {"frame_index": 0, "mode": "i2v"}
ROBOT_VIDEO_STAGE_BUCKETS = {
    "S": tuple(range(24, 65, 8)),
    "M": tuple(range(72, 121, 8)),
    "L": tuple(range(128, 193, 8)),
}
# M rows stop at F120, while SP8 requires a fixed carrier divisible by eight.
# F128 is padding capacity, not an admitted M source bucket.
ROBOT_VIDEO_STAGE_CARRIERS = {"S": 64, "M": 128, "L": 192}
ROBOT_VIDEO_INDEX_MAGIC = b"LLIDX001"
ROBOT_VIDEO_INDEX_VERSION = "longlive-robot-video-jsonl-offsets-v1"
_UINT64 = struct.Struct("<Q")
_MAX_JSONL_ROW_BYTES = 4 * 1024 * 1024


class RobotVideoManifestError(ValueError):
    """Raised when a frozen robot_video manifest or one of its rows is invalid."""


def _json_object_without_duplicate_keys(raw: bytes, *, context: str):
    """Decode one JSON object while rejecting duplicate keys."""

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RobotVideoManifestError(
                    f"{context}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs_hook)
    except RobotVideoManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RobotVideoManifestError(f"{context}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RobotVideoManifestError(f"{context}: row must be a JSON object")
    return value


def _canonical_json_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_int(row, field, *, context):
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RobotVideoManifestError(
            f"{context}: {field} must be an integer, got {value!r}"
        )
    return value


def _infer_robot_video_manifest_coordinates(manifest_path: Path):
    """Infer and strictly validate ``.../manifests/<stage>/<split>.jsonl``."""
    if manifest_path.suffix != ".jsonl":
        raise RobotVideoManifestError(
            f"robot_video manifest must end in .jsonl: {manifest_path}"
        )
    stage = manifest_path.parent.name
    split = manifest_path.stem
    if manifest_path.parent.parent.name != "manifests":
        raise RobotVideoManifestError(
            "robot_video manifest path must have the exact form "
            f".../manifests/<S|M|L>/<train|val|test>.jsonl: {manifest_path}"
        )
    if stage not in ROBOT_VIDEO_STAGE_BUCKETS:
        raise RobotVideoManifestError(f"invalid robot_video stage in path: {stage!r}")
    if split not in {"train", "val", "test"}:
        raise RobotVideoManifestError(f"invalid robot_video split in path: {split!r}")
    return stage, split


def _validate_robot_video_row(
    row,
    *,
    context,
    expected_stage,
    expected_split,
    temporal_compression_ratio,
    carrier_latent_frames,
):
    """Validate the complete training-critical contract for one frozen row."""
    required = {
        "bucket",
        "conditioning",
        "contract_version",
        "first_frame_source",
        "id",
        "instruction",
        "source_dataset",
        "source_id",
        "source_manifest_id",
        "source_name",
        "split",
        "stage",
        "temporal_padding",
        "temporal_stretching",
        "tier",
        "valid_duration_seconds",
        "valid_latent_frames",
        "valid_raw_frames",
        "video",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RobotVideoManifestError(f"{context}: missing required fields {missing}")
    if row["contract_version"] != ROBOT_VIDEO_ROW_CONTRACT_VERSION:
        raise RobotVideoManifestError(
            f"{context}: contract_version must be "
            f"{ROBOT_VIDEO_ROW_CONTRACT_VERSION!r}, got {row['contract_version']!r}"
        )
    if row["conditioning"] != ROBOT_VIDEO_CONDITIONING:
        raise RobotVideoManifestError(
            f"{context}: conditioning must be exactly {ROBOT_VIDEO_CONDITIONING!r}"
        )
    if row["first_frame_source"] != "decoded frame 0 of video":
        raise RobotVideoManifestError(
            f"{context}: first_frame_source is not decoded video frame 0"
        )
    if row["stage"] != expected_stage:
        raise RobotVideoManifestError(
            f"{context}: stage {row['stage']!r} != {expected_stage!r}"
        )
    if row["split"] != expected_split:
        raise RobotVideoManifestError(
            f"{context}: split {row['split']!r} != {expected_split!r}"
        )
    if row["temporal_padding"] is not False:
        raise RobotVideoManifestError(
            f"{context}: frozen source row already has temporal_padding"
        )
    if row["temporal_stretching"] is not False:
        raise RobotVideoManifestError(
            f"{context}: frozen source row already has temporal_stretching"
        )

    sample_id = row["id"]
    if (
        not isinstance(sample_id, str)
        or len(sample_id) != 32
        or any(character not in "0123456789abcdef" for character in sample_id)
    ):
        raise RobotVideoManifestError(
            f"{context}: id must be exactly 32 lowercase hexadecimal characters"
        )
    instruction = row["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise RobotVideoManifestError(f"{context}: instruction must be non-empty")

    valid_latent_frames = _require_exact_int(
        row, "valid_latent_frames", context=context
    )
    valid_raw_frames = _require_exact_int(row, "valid_raw_frames", context=context)
    allowed_buckets = ROBOT_VIDEO_STAGE_BUCKETS[expected_stage]
    if valid_latent_frames not in allowed_buckets:
        raise RobotVideoManifestError(
            f"{context}: F{valid_latent_frames} is outside stage {expected_stage}"
        )
    if row["bucket"] != f"F{valid_latent_frames}":
        raise RobotVideoManifestError(
            f"{context}: bucket {row['bucket']!r} does not match "
            f"valid_latent_frames={valid_latent_frames}"
        )
    expected_raw_frames = 1 + (
        valid_latent_frames - 1
    ) * temporal_compression_ratio
    if valid_raw_frames != expected_raw_frames:
        raise RobotVideoManifestError(
            f"{context}: valid_raw_frames={valid_raw_frames} must equal "
            f"1 + ({valid_latent_frames} - 1) * "
            f"{temporal_compression_ratio} = {expected_raw_frames}"
        )
    if valid_latent_frames > carrier_latent_frames:
        raise RobotVideoManifestError(
            f"{context}: F{valid_latent_frames} exceeds fixed "
            f"F{carrier_latent_frames} stage carrier"
        )
    duration = row["valid_duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise RobotVideoManifestError(
            f"{context}: valid_duration_seconds must be numeric"
        )
    if duration <= 0:
        raise RobotVideoManifestError(
            f"{context}: valid_duration_seconds must be positive"
        )

    video = row["video"]
    if not isinstance(video, str) or not video:
        raise RobotVideoManifestError(f"{context}: video must be a non-empty path")
    video_path = Path(video)
    if not video_path.is_absolute() or ".." in video_path.parts:
        raise RobotVideoManifestError(
            f"{context}: video must be a normalized absolute path: {video!r}"
        )
    if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise RobotVideoManifestError(
            f"{context}: unsupported video extension: {video_path.suffix!r}"
        )
    return row


class _ManifestSampleIdSequence(Sequence):
    """Lazy ``folders`` compatibility for fixed-ID evaluation selection."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return Path(self.dataset._read_row(index)["id"])


class RobotVideoManifestDataset(Dataset):
    """Direct, bounded-memory reader for the frozen robot_video TI2V manifests.

    The JSONL itself remains the sample-order authority.  A compact binary
    offset sidecar is built once under a filesystem lock, then both files are
    memory-mapped by every worker.  No rank materializes the 1.5 GB S manifest
    (or its rows) as a Python list.

    Every row is one single-shot video.  Training decoding always begins at
    source frame zero and stops at ``valid_raw_frames``.  If the stage carrier
    is longer, only an in-memory repeat of the final decoded frame is appended;
    the original ``valid_latent_frames`` is returned for loss masking.

    ``evaluation_mode`` is deliberately narrower.  It is valid only for a
    non-training manifest and one native bucket in that manifest's stage.  It
    decodes frame zero only, repeats the exact instruction for the requested
    rollout blocks, and validates selected rows against the requested bucket.
    Manifest/receipt scanning still uses the stage's fixed maximum carrier, so
    all source rows remain strictly validated and one offset sidecar is safely
    shared by every evaluation horizon in the same manifest.
    """

    def __init__(
        self,
        manifest_path,
        *,
        video_size,
        total_frames,
        target_fps=24,
        num_frame_per_block=8,
        temporal_compression_ratio=4,
        expected_stage=None,
        expected_split=None,
        receipt_path=None,
        expected_receipt_sha256=None,
        index_path=None,
        index_cache_dir=None,
        return_image=True,
        evaluation_mode=False,
        requested_eval_latent_frames=None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise RobotVideoManifestError(
                f"robot_video manifest is not a regular file: {self.manifest_path}"
            )
        path_stage, path_split = _infer_robot_video_manifest_coordinates(
            self.manifest_path
        )
        self.stage = path_stage if expected_stage is None else str(expected_stage)
        self.split = path_split if expected_split is None else str(expected_split)
        if self.stage != path_stage:
            raise RobotVideoManifestError(
                f"configured stage {self.stage!r} disagrees with manifest path "
                f"stage {path_stage!r}"
            )
        if self.split != path_split:
            raise RobotVideoManifestError(
                f"configured split {self.split!r} disagrees with manifest path "
                f"split {path_split!r}"
            )
        if self.stage not in ROBOT_VIDEO_STAGE_BUCKETS:
            raise RobotVideoManifestError(f"invalid expected stage: {self.stage!r}")
        if self.split not in {"train", "val", "test"}:
            raise RobotVideoManifestError(f"invalid expected split: {self.split!r}")

        if not isinstance(evaluation_mode, bool):
            raise RobotVideoManifestError(
                f"evaluation_mode must be a boolean, got {evaluation_mode!r}"
            )
        self.evaluation_mode = evaluation_mode
        if self.evaluation_mode and self.split == "train":
            raise RobotVideoManifestError(
                "robot_video frame0 evaluation mode is forbidden for train manifests"
            )
        if not self.evaluation_mode and requested_eval_latent_frames is not None:
            raise RobotVideoManifestError(
                "requested_eval_latent_frames is valid only in evaluation_mode"
            )
        if self.evaluation_mode and not return_image:
            raise RobotVideoManifestError(
                "robot_video frame0 evaluation mode requires return_image=true"
            )

        self.video_size = tuple(int(value) for value in video_size)
        if len(self.video_size) != 2 or min(self.video_size) <= 0:
            raise RobotVideoManifestError(
                f"video_size must contain two positive integers: {video_size!r}"
            )
        self.total_frames = int(total_frames)
        self.target_fps = int(target_fps)
        self.num_frame_per_block = int(num_frame_per_block)
        self.temporal_compression_ratio = int(temporal_compression_ratio)
        self.return_image = bool(return_image)
        if self.target_fps != 24:
            raise RobotVideoManifestError(
                f"frozen robot_video target_fps is 24, got {self.target_fps}"
            )
        if self.temporal_compression_ratio != 4:
            raise RobotVideoManifestError(
                "frozen robot_video temporal_compression_ratio is 4, got "
                f"{self.temporal_compression_ratio}"
            )
        if self.total_frames <= 0 or (
            self.total_frames - 1
        ) % self.temporal_compression_ratio:
            raise RobotVideoManifestError(
                f"invalid raw carrier length: {self.total_frames}"
            )
        configured_latent_frames = 1 + (
            self.total_frames - 1
        ) // self.temporal_compression_ratio
        expected_carrier = ROBOT_VIDEO_STAGE_CARRIERS[self.stage]
        # ``carrier_latent_frames`` is always the source-validation carrier.
        # It is intentionally independent of the requested eval horizon and is
        # therefore safe to bind into a manifest offset sidecar.
        self.carrier_latent_frames = expected_carrier
        self.requested_eval_latent_frames = None
        if self.evaluation_mode:
            if (
                isinstance(requested_eval_latent_frames, bool)
                or not isinstance(requested_eval_latent_frames, int)
            ):
                raise RobotVideoManifestError(
                    "evaluation_mode requires integer "
                    "requested_eval_latent_frames"
                )
            requested_eval_latent_frames = int(requested_eval_latent_frames)
            if requested_eval_latent_frames not in ROBOT_VIDEO_STAGE_BUCKETS[self.stage]:
                raise RobotVideoManifestError(
                    f"F{requested_eval_latent_frames} is not an allowed "
                    f"stage {self.stage} evaluation bucket"
                )
            if configured_latent_frames != requested_eval_latent_frames:
                raise RobotVideoManifestError(
                    "evaluation raw frame count disagrees with requested "
                    f"horizon: total_frames={self.total_frames} gives "
                    f"F{configured_latent_frames}, requested "
                    f"F{requested_eval_latent_frames}"
                )
            self.requested_eval_latent_frames = requested_eval_latent_frames
            prompt_latent_frames = requested_eval_latent_frames
        else:
            if configured_latent_frames != expected_carrier:
                raise RobotVideoManifestError(
                    f"stage {self.stage} requires fixed F{expected_carrier} "
                    f"carrier, got F{configured_latent_frames}"
                )
            prompt_latent_frames = self.carrier_latent_frames

        if prompt_latent_frames % self.num_frame_per_block:
            raise RobotVideoManifestError(
                f"F{prompt_latent_frames} horizon must be divisible by "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
        self.num_prompt_blocks = (
            prompt_latent_frames // self.num_frame_per_block
        )

        inferred_receipt = self.manifest_path.parents[2] / "receipt.json"
        self.receipt_path = Path(receipt_path or inferred_receipt).expanduser().resolve()
        self.expected_receipt_sha256 = expected_receipt_sha256
        if self.expected_receipt_sha256 is None:
            raise RobotVideoManifestError(
                "expected_receipt_sha256 is required for frozen robot_video manifests; "
                "bind the accepted receipt file SHA-256 in the training config"
            )
        self._receipt_output = self._load_and_validate_receipt()

        if index_path is not None and index_cache_dir is not None:
            raise RobotVideoManifestError(
                "configure at most one of index_path and index_cache_dir"
            )
        if index_path is not None:
            self.index_path = Path(index_path).expanduser().resolve()
        elif index_cache_dir is not None:
            cache_dir = Path(index_cache_dir).expanduser().resolve()
            cache_key = hashlib.sha256(
                str(self.manifest_path).encode("utf-8")
            ).hexdigest()[:20]
            self.index_path = cache_dir / (
                f"{self.stage}-{self.split}-{cache_key}.llidx"
            )
        else:
            raise RobotVideoManifestError(
                "robot_video offset sidecar location must be explicit: configure "
                "manifest_index_path or manifest_index_cache_dir on Lustre; "
                "the frozen manifest directory is read-only"
            )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_offset_index()
        self._manifest_file = None
        self._manifest_mmap = None
        self._index_file = None
        self._index_mmap = None
        self._open_mmaps()
        self.folders = _ManifestSampleIdSequence(self)
        self.resize_transform = transforms.Resize(self.video_size, antialias=True)
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        if decord is not None:
            decord.bridge.set_bridge("torch")

    def _load_and_validate_receipt(self):
        if not self.receipt_path.is_file():
            raise RobotVideoManifestError(
                f"frozen robot_video receipt is missing: {self.receipt_path}"
            )
        try:
            receipt_bytes = self.receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RobotVideoManifestError(
                f"cannot read frozen robot_video receipt {self.receipt_path}: {exc}"
            ) from exc
        if not isinstance(receipt, dict):
            raise RobotVideoManifestError("frozen robot_video receipt must be a JSON object")
        self.receipt_file_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        expected = self.expected_receipt_sha256
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise RobotVideoManifestError(
                "expected_receipt_sha256 must be 64 lowercase hexadecimal characters"
            )
        if self.receipt_file_sha256 != expected:
            raise RobotVideoManifestError(
                "frozen robot_video receipt file SHA-256 changed: "
                f"{self.receipt_file_sha256} != {expected}"
            )
        receipt_body = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _canonical_json_sha256(receipt_body):
            raise RobotVideoManifestError(
                "frozen robot_video receipt embedded canonical SHA-256 is invalid"
            )
        if receipt.get("contract_version") != ROBOT_VIDEO_RECEIPT_CONTRACT_VERSION:
            raise RobotVideoManifestError("frozen robot_video receipt contract_version changed")
        if receipt.get("status") != ROBOT_VIDEO_ACCEPTED_RECEIPT_STATUS:
            raise RobotVideoManifestError("frozen robot_video receipt is not accepted")
        if receipt.get("conditioning") != ROBOT_VIDEO_CONDITIONING:
            raise RobotVideoManifestError("frozen robot_video receipt conditioning changed")
        media_policy = receipt.get("media_policy")
        if not isinstance(media_policy, dict):
            raise RobotVideoManifestError("frozen robot_video receipt media_policy is missing")
        if media_policy.get("target_fps") != self.target_fps:
            raise RobotVideoManifestError("frozen robot_video receipt target_fps changed")
        if (
            media_policy.get("temporal_compression_ratio")
            != self.temporal_compression_ratio
        ):
            raise RobotVideoManifestError(
                "frozen robot_video receipt temporal_compression_ratio changed"
            )
        expected_stage_policy = {
            key: list(value) for key, value in ROBOT_VIDEO_STAGE_BUCKETS.items()
        }
        if receipt.get("stage_policy") != expected_stage_policy:
            raise RobotVideoManifestError("frozen robot_video receipt stage_policy changed")

        outputs = receipt.get("outputs")
        if not isinstance(outputs, list):
            raise RobotVideoManifestError("frozen robot_video receipt outputs are missing")
        matches = []
        for output in outputs:
            if not isinstance(output, dict) or not isinstance(output.get("path"), str):
                continue
            if Path(output["path"]).expanduser().resolve() == self.manifest_path:
                matches.append(output)
        if len(matches) != 1:
            raise RobotVideoManifestError(
                "frozen robot_video receipt must bind the manifest exactly once"
            )
        output = matches[0]
        if output.get("stage") != self.stage or output.get("split") != self.split:
            raise RobotVideoManifestError(
                "frozen robot_video receipt output stage/split disagrees with path"
            )
        stat = self.manifest_path.stat()
        if output.get("size_bytes") != stat.st_size:
            raise RobotVideoManifestError(
                "frozen robot_video manifest size disagrees with receipt"
            )
        count = output.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RobotVideoManifestError(
                "frozen robot_video receipt record_count must be a positive integer"
            )
        digest = output.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RobotVideoManifestError(
                "frozen robot_video receipt manifest SHA-256 is invalid"
            )
        return dict(output)

    def _manifest_identity(self):
        stat = self.manifest_path.stat()
        return {
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _read_index_header(self):
        try:
            with self.index_path.open("rb") as handle:
                if handle.read(len(ROBOT_VIDEO_INDEX_MAGIC)) != ROBOT_VIDEO_INDEX_MAGIC:
                    return None
                raw_length = handle.read(_UINT64.size)
                if len(raw_length) != _UINT64.size:
                    return None
                header_length = _UINT64.unpack(raw_length)[0]
                if header_length <= 0 or header_length > 1024 * 1024:
                    return None
                raw_header = handle.read(header_length)
                if len(raw_header) != header_length:
                    return None
                header = json.loads(raw_header)
                if not isinstance(header, dict):
                    return None
                header["offsets_start"] = (
                    len(ROBOT_VIDEO_INDEX_MAGIC) + _UINT64.size + header_length
                )
                return header
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _index_header_matches(self, header):
        if header is None:
            return False
        count = self._receipt_output["record_count"]
        expected = {
            "index_version": ROBOT_VIDEO_INDEX_VERSION,
            "manifest_path": str(self.manifest_path),
            "manifest_identity": self._manifest_identity(),
            "manifest_sha256": self._receipt_output["sha256"],
            "receipt_file_sha256": self.receipt_file_sha256,
            "record_count": count,
            "stage": self.stage,
            "split": self.split,
            "carrier_latent_frames": self.carrier_latent_frames,
            "temporal_compression_ratio": self.temporal_compression_ratio,
        }
        if any(header.get(key) != value for key, value in expected.items()):
            return False
        expected_size = header["offsets_start"] + (count + 1) * _UINT64.size
        try:
            return self.index_path.stat().st_size == expected_size
        except OSError:
            return False

    def _ensure_offset_index(self):
        header = self._read_index_header()
        if self._index_header_matches(header):
            self._index_header = header
            return
        lock_path = self.index_path.with_name(self.index_path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            header = self._read_index_header()
            if not self._index_header_matches(header):
                self._build_offset_index()
                header = self._read_index_header()
            if not self._index_header_matches(header):
                raise RobotVideoManifestError(
                    f"failed to publish valid offset index: {self.index_path}"
                )
            self._index_header = header

    def _build_offset_index(self):
        manifest_hasher = hashlib.sha256()
        ids_hasher = hashlib.sha256()
        record_count = 0
        offset = 0
        raw_offsets_path = None
        published_temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{self.index_path.name}.offsets.",
                dir=self.index_path.parent,
                delete=False,
            ) as offsets_handle:
                raw_offsets_path = Path(offsets_handle.name)
                offsets_handle.write(_UINT64.pack(0))
                with self.manifest_path.open("rb") as manifest_handle:
                    for line_number, raw_line in enumerate(manifest_handle, start=1):
                        manifest_hasher.update(raw_line)
                        if len(raw_line) > _MAX_JSONL_ROW_BYTES:
                            raise RobotVideoManifestError(
                                f"{self.manifest_path}:{line_number}: row exceeds "
                                f"{_MAX_JSONL_ROW_BYTES} bytes"
                            )
                        row_bytes = raw_line.rstrip(b"\r\n")
                        if not row_bytes:
                            raise RobotVideoManifestError(
                                f"{self.manifest_path}:{line_number}: blank rows are forbidden"
                            )
                        context = f"{self.manifest_path}:{line_number}"
                        row = _json_object_without_duplicate_keys(
                            row_bytes, context=context
                        )
                        _validate_robot_video_row(
                            row,
                            context=context,
                            expected_stage=self.stage,
                            expected_split=self.split,
                            temporal_compression_ratio=self.temporal_compression_ratio,
                            carrier_latent_frames=self.carrier_latent_frames,
                        )
                        ids_hasher.update(row["id"].encode("ascii") + b"\n")
                        record_count += 1
                        offset += len(raw_line)
                        offsets_handle.write(_UINT64.pack(offset))
                offsets_handle.flush()
                os.fsync(offsets_handle.fileno())

            expected_count = self._receipt_output["record_count"]
            if record_count != expected_count:
                raise RobotVideoManifestError(
                    f"manifest row count {record_count} != receipt {expected_count}"
                )
            manifest_digest = manifest_hasher.hexdigest()
            if manifest_digest != self._receipt_output["sha256"]:
                raise RobotVideoManifestError(
                    "manifest SHA-256 does not match frozen robot_video receipt: "
                    f"{manifest_digest} != {self._receipt_output['sha256']}"
                )
            if offset != self._receipt_output["size_bytes"]:
                raise RobotVideoManifestError(
                    f"manifest scanned size {offset} disagrees with receipt"
                )
            header = {
                "carrier_latent_frames": self.carrier_latent_frames,
                "ids_sha256": ids_hasher.hexdigest(),
                "index_version": ROBOT_VIDEO_INDEX_VERSION,
                "manifest_identity": self._manifest_identity(),
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": manifest_digest,
                "record_count": record_count,
                "receipt_file_sha256": self.receipt_file_sha256,
                "split": self.split,
                "stage": self.stage,
                "temporal_compression_ratio": self.temporal_compression_ratio,
            }
            raw_header = json.dumps(
                header, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{self.index_path.name}.publish.",
                dir=self.index_path.parent,
                delete=False,
            ) as output_handle:
                published_temp_path = Path(output_handle.name)
                output_handle.write(ROBOT_VIDEO_INDEX_MAGIC)
                output_handle.write(_UINT64.pack(len(raw_header)))
                output_handle.write(raw_header)
                with raw_offsets_path.open("rb") as offsets_handle:
                    while True:
                        chunk = offsets_handle.read(1024 * 1024)
                        if not chunk:
                            break
                        output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(published_temp_path, self.index_path)
            published_temp_path = None
            directory_fd = os.open(self.index_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            for temporary_path in (raw_offsets_path, published_temp_path):
                if temporary_path is not None:
                    try:
                        temporary_path.unlink()
                    except FileNotFoundError:
                        pass

    def _open_mmaps(self):
        if self._manifest_mmap is not None:
            return
        self._manifest_file = self.manifest_path.open("rb")
        self._manifest_mmap = mmap.mmap(
            self._manifest_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._index_file = self.index_path.open("rb")
        self._index_mmap = mmap.mmap(
            self._index_file.fileno(), 0, access=mmap.ACCESS_READ
        )

    def _close_mmaps(self):
        for attribute in ("_manifest_mmap", "_index_mmap"):
            value = getattr(self, attribute, None)
            if value is not None:
                value.close()
                setattr(self, attribute, None)
        for attribute in ("_manifest_file", "_index_file"):
            value = getattr(self, attribute, None)
            if value is not None:
                value.close()
                setattr(self, attribute, None)

    def __del__(self):
        try:
            self._close_mmaps()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in (
            "_manifest_file",
            "_manifest_mmap",
            "_index_file",
            "_index_mmap",
        ):
            state[key] = None
        return state

    def __len__(self):
        return int(self._index_header["record_count"])

    def _row_bounds(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"manifest index must be an integer, got {index!r}")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._open_mmaps()
        base = int(self._index_header["offsets_start"])
        start = _UINT64.unpack_from(
            self._index_mmap, base + index * _UINT64.size
        )[0]
        end = _UINT64.unpack_from(
            self._index_mmap, base + (index + 1) * _UINT64.size
        )[0]
        if start >= end or end > len(self._manifest_mmap):
            raise RobotVideoManifestError(
                f"offset index is corrupt at row {index}: [{start}, {end})"
            )
        return start, end

    def _read_row(self, index):
        start, end = self._row_bounds(index)
        raw = self._manifest_mmap[start:end].rstrip(b"\r\n")
        context = f"{self.manifest_path}:row_index={index}"
        row = _json_object_without_duplicate_keys(raw, context=context)
        return _validate_robot_video_row(
            row,
            context=context,
            expected_stage=self.stage,
            expected_split=self.split,
            temporal_compression_ratio=self.temporal_compression_ratio,
            carrier_latent_frames=self.carrier_latent_frames,
        )

    def distributed_index_contract(self):
        """Small rank-comparison contract without reading every JSON row."""
        return {
            "count": len(self),
            "ids_sha256": self._index_header["ids_sha256"],
            "manifest_sha256": self._index_header["manifest_sha256"],
            "stage": self.stage,
            "split": self.split,
        }

    def validate_fixed_evaluation_indices(self, indices):
        """Fail before decoding if selected IDs are not native to the horizon.

        Selection helpers call this only for the explicitly configured fixed
        IDs.  Other rows in the stage manifest may belong to any admitted
        bucket and must not make construction of an F32/F96/etc. panel fail.
        """
        if not self.evaluation_mode:
            raise RobotVideoManifestError(
                "fixed evaluation horizon validation requires evaluation_mode"
            )
        validated = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    f"evaluation manifest index must be an integer, got {index!r}"
                )
            row = self._read_row(index)
            self._validate_evaluation_row(row, index=index)
            validated.append(row["id"])
        return tuple(validated)

    def _validate_evaluation_row(self, row, *, index):
        if not self.evaluation_mode:
            raise RobotVideoManifestError(
                "evaluation row validation requires evaluation_mode"
            )
        actual = row["valid_latent_frames"]
        requested = self.requested_eval_latent_frames
        if actual != requested:
            raise RobotVideoManifestError(
                f"fixed evaluation sample id={row['id']} index={index} is "
                f"F{actual}, but this group requests native F{requested}"
            )

    def _decode_video_prefix(self, video_path, valid_raw_frames):
        """Decode an exact frame-zero prefix; never pad or choose another file."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"manifest video is missing: {video_path}")
        if decord is None:
            return self._decode_video_prefix_ffmpeg(video_path, valid_raw_frames)
        try:
            reader = decord.VideoReader(
                str(video_path), width=self.video_size[1], height=self.video_size[0]
            )
        except Exception:
            reader = decord.VideoReader(str(video_path))
        source_frames = len(reader)
        source_fps = float(reader.get_avg_fps())
        if source_frames <= 0 or source_fps <= 0:
            raise RuntimeError(f"video has invalid frame/fps metadata: {video_path}")
        sampling_interval = source_fps / self.target_fps
        indices = np.rint(
            np.arange(valid_raw_frames, dtype=np.float64) * sampling_interval
        ).astype(np.int64)
        if indices[-1] >= source_frames:
            raise RuntimeError(
                f"video {video_path} cannot provide {valid_raw_frames} frames "
                f"from frame 0 at {self.target_fps} fps"
            )
        frames = reader.get_batch(indices.astype(np.int32)).numpy()
        if len(frames) != valid_raw_frames:
            raise RuntimeError(
                f"video {video_path} decoded {len(frames)} != "
                f"{valid_raw_frames} requested frames"
            )
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float() / 255.0
        if tuple(tensor.shape[2:]) != self.video_size:
            tensor = torch.stack(
                [self.resize_transform(frame) for frame in tensor], dim=0
            )
        tensor = self.normalize(tensor).to(torch.float16)
        return tensor

    def _decode_video_prefix_ffmpeg(self, video_path, valid_raw_frames):
        height, width = self.video_size
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={self.target_fps},scale={width}:{height}",
            "-frames:v",
            str(valid_raw_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        try:
            process = subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise RuntimeError(
                f"ffmpeg failed to decode frame-zero prefix from {video_path}: {stderr}"
            ) from exc
        frame_bytes = height * width * 3
        if len(process.stdout) % frame_bytes:
            raise RuntimeError(f"ffmpeg returned a partial frame for {video_path}")
        decoded_frames = len(process.stdout) // frame_bytes
        if decoded_frames != valid_raw_frames:
            raise RuntimeError(
                f"video {video_path} decoded {decoded_frames} != "
                f"{valid_raw_frames} requested frames"
            )
        frames = np.frombuffer(process.stdout, dtype=np.uint8).reshape(
            decoded_frames, height, width, 3
        )
        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float() / 255.0
        return self.normalize(tensor).to(torch.float16)

    def __getitem__(self, index):
        row = self._read_row(index)
        if self.evaluation_mode:
            self._validate_evaluation_row(row, index=index)
            try:
                decoded = self._decode_video_prefix(row["video"], 1)
            except Exception as exc:
                raise RuntimeError(
                    f"robot_video evaluation sample index={index} id={row['id']} "
                    f"video={row['video']} frame0 decode failed; no sample "
                    f"substitution is permitted: {exc}"
                ) from exc
            if decoded.ndim != 4 or decoded.shape[0] != 1:
                raise RuntimeError(
                    f"robot_video evaluation sample {row['id']} decoder returned "
                    f"invalid frame0 shape {tuple(decoded.shape)}"
                )
            image = decoded[0]
            return {
                # The shared video collator requires a frames tensor, although
                # inference consumes only ``image``.  Keep exactly one frame;
                # never decode or synthesize the target rollout here.
                "frames": image.unsqueeze(1),
                "image": image,
                "prompts": [row["instruction"]] * self.num_prompt_blocks,
                "idx": int(index),
                "sample_id": row["id"],
                "video": row["video"],
            }

        valid_raw_frames = row["valid_raw_frames"]
        valid_latent_frames = row["valid_latent_frames"]
        try:
            decoded = self._decode_video_prefix(row["video"], valid_raw_frames)
        except Exception as exc:
            raise RuntimeError(
                f"robot_video sample index={index} id={row['id']} video={row['video']} "
                f"failed; no sample substitution is permitted: {exc}"
            ) from exc
        if decoded.ndim != 4 or decoded.shape[0] != valid_raw_frames:
            raise RuntimeError(
                f"robot_video sample {row['id']} decoder returned invalid shape "
                f"{tuple(decoded.shape)}"
            )
        if valid_raw_frames < self.total_frames:
            tail = decoded[-1:].repeat(
                self.total_frames - valid_raw_frames, 1, 1, 1
            )
            carrier = torch.cat((decoded, tail), dim=0)
        elif valid_raw_frames == self.total_frames:
            carrier = decoded
        else:
            raise RobotVideoManifestError(
                f"robot_video sample {row['id']} exceeds its fixed stage carrier"
            )
        result = {
            "frames": carrier.permute(1, 0, 2, 3),
            "prompts": [row["instruction"]] * self.num_prompt_blocks,
            "idx": int(index),
            "sample_id": row["id"],
            "video": row["video"],
            "num_valid_raw_frames": valid_raw_frames,
            "num_valid_latent_frames": valid_latent_frames,
        }
        if self.return_image:
            result["image"] = decoded[0]
        return result


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        try:
            import datasets
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'datasets' package is required for MultiTextDataset. "
                "Use MultiTextConcatDataset for plain txt/json-caption directories "
                "or install datasets."
            ) from exc
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


class MultiTextConcatDataset(Dataset):
    """Text-only dataset for multi-shot training and inference.

    Supports two input modes:

    **txt file** — each line is one caption. Each sample uses the caption at
    index ``idx``, repeated ``num_blocks`` times (single-shot, no scene cut
    prefix).

    **directory** — reads ``caption/<subfolder>/*.json`` files (no video dir
    needed). Shot durations are resolved with a three-level fallback:

    1. ``shot_durations.txt`` in the caption subfolder (per-sample override)
    2. ``chunks_per_shot`` from config (global fixed repeat)
    3. Even distribution across all available captions

    Scene cut prefix is prepended at shot boundaries (first block of each
    shot except shot 0). Output is always exactly ``num_blocks`` prompts:
    truncated if too many, padded with the last caption if too few.
    """

    def __init__(
        self,
        data_path: str,
        num_blocks: int,
        chunks_per_shot: int = 0,
        scene_cut_prefix: str = DEFAULT_SCENE_CUT_PREFIX,
        caption_field: str = "caption",
        deterministic: bool = False,
    ):
        self.num_blocks = num_blocks
        self.chunks_per_shot = chunks_per_shot
        self.scene_cut_prefix = scene_cut_prefix
        self.caption_field = caption_field
        self.deterministic = deterministic

        path = Path(data_path)
        if data_path.endswith(".txt") or path.is_file():
            self._mode = "txt"
            with open(data_path, encoding="utf-8") as f:
                self._prompts = [line.rstrip() for line in f if line.strip()]
            assert len(self._prompts) > 0, f"No prompts found in {data_path}"
        else:
            self._mode = "dir"
            self._caption_dir = path / "caption" if (path / "caption").is_dir() else path
            self._folders = sorted([d for d in self._caption_dir.iterdir() if d.is_dir()])
            assert len(self._folders) > 0, (
                f"No caption subfolders found in {self._caption_dir}"
            )

    def __len__(self):
        if self._mode == "txt":
            return len(self._prompts)
        return len(self._folders)

    def __getitem__(self, idx):
        if self._mode == "txt":
            return self._get_txt_item(idx)
        return self._get_dir_item(idx)

    # ------------------------------------------------------------------
    # txt mode
    # ------------------------------------------------------------------

    def _get_txt_item(self, idx):
        caption = self._prompts[idx % len(self._prompts)]
        return {
            "prompts": [caption] * self.num_blocks,
            "idx": idx,
        }

    # ------------------------------------------------------------------
    # directory mode
    # ------------------------------------------------------------------

    def _get_dir_item(self, idx):
        folder = self._folders[idx % len(self._folders)]
        raw_captions = self._load_captions_from_folder(folder)
        if not raw_captions:
            raw_captions = [""]

        shot_durations = self._resolve_shot_durations(folder, len(raw_captions))
        prompts = self._apply_shot_durations(raw_captions, shot_durations)

        # Ensure exactly num_blocks prompts
        if len(prompts) > self.num_blocks:
            prompts = prompts[: self.num_blocks]
        elif len(prompts) < self.num_blocks:
            last = prompts[-1] if prompts else ""
            prompts.extend([last] * (self.num_blocks - len(prompts)))

        return {
            "prompts": prompts,
            "idx": idx,
        }

    def _load_captions_from_folder(self, folder: Path):
        json_files = sorted(
            [f for f in folder.glob("*.json") if f.name != "global.json"],
            key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem),
        )
        captions = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    captions.append(data.get(self.caption_field, ""))
            except Exception:
                captions.append("")
        return captions

    # ------------------------------------------------------------------
    # shot duration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_shot_durations(folder: Path):
        txt_path = folder / "shot_durations.txt"
        if not txt_path.exists():
            return None
        try:
            with open(txt_path, "r") as f:
                content = f.read().strip()
            parts = content.replace(",", " ").split()
            durations = [int(x) for x in parts if x.strip()]
            return durations if durations else None
        except Exception:
            return None

    def _resolve_shot_durations(self, folder: Path, num_captions: int):
        durations = self._load_shot_durations(folder)
        if durations is not None:
            return durations[:num_captions]
        if self.chunks_per_shot > 0:
            return [self.chunks_per_shot] * num_captions
        return self._even_durations(num_captions)

    def _even_durations(self, num_shots: int):
        total = self.num_blocks
        base, extra = divmod(total, num_shots)
        return [base + (1 if i < extra else 0) for i in range(num_shots)]

    def _apply_shot_durations(self, raw_captions, shot_durations):
        target = self.num_blocks
        clamped: list[int] = []
        remaining = target
        for d in shot_durations:
            if remaining <= 0:
                break
            take = min(d, remaining)
            clamped.append(take)
            remaining -= take
        if remaining > 0 and clamped:
            clamped[-1] += remaining

        prompts: list[str] = []
        for shot_idx, (caption, duration) in enumerate(zip(raw_captions, clamped)):
            for block_in_shot in range(duration):
                if shot_idx > 0 and block_in_shot == 0 and self.scene_cut_prefix:
                    prompts.append(self.scene_cut_prefix + caption)
                else:
                    prompts.append(caption)
        return prompts


class MultiVideoConcatDataset(Dataset):
    """Dataset that concatenates multiple videos from a folder into a fixed-length video.
    
    Each item consists of multiple video segments concatenated together:
    - First segment: first_chunk_frames frames (chunk)
    - Subsequent segments: subsequent_chunk_frames frames each (chunk)
    - Total: total_frames frames (first_chunk_frames + subsequent_chunk_frames*num_subsequent_segments = total_frames)
    
    Videos are sampled preserving original duration, and if a video doesn't have enough frames,
    it moves to the next video. If a video has enough frames, it can be sampled repeatedly.
    """
    def __init__(
        self,
        data_dir,
        video_size,
        total_frames,
        target_fps=16,
        video_extensions=('.mp4', '.avi', '.mov', '.mkv', '.webm'),
        caption_field='caption',
        filter_invalid_folders=False,
        deterministic: bool = False,
        num_frame_per_block=8,
        temporal_compression_ratio=4,
        allow_padding: bool = False,
        min_latent_frames: int = 0,
        single_video_only: bool = False,
        independent_first_frame: bool = False,
        return_image: bool = False,
        max_chunks_per_shot: int = 0,
        scene_cut_prefix: str = DEFAULT_SCENE_CUT_PREFIX,
        sample_warning_seconds: float = 60.0,
        sample_warning_interval_seconds: float = 60.0,
    ):
        self.root_dir = Path(data_dir)
        self.data_dir = self.root_dir / "video"
        self.caption_dir = self.root_dir / "caption"
        self.video_size = video_size
        self.total_frames = total_frames

        total_latent_frames = 1 + (total_frames - 1) // temporal_compression_ratio
        separate_first_latent = (
            independent_first_frame
            and total_latent_frames % num_frame_per_block != 0
        )
        if separate_first_latent:
            assert (total_latent_frames - 1) % num_frame_per_block == 0, (
                f"total latent frames ({total_latent_frames}) must be divisible by "
                f"num_frame_per_block ({num_frame_per_block}) or equal to "
                f"1 + N * num_frame_per_block when independent_first_frame=True"
            )
        first_chunk_latent_frames = (
            num_frame_per_block + 1 if separate_first_latent else num_frame_per_block
        )
        first_chunk_frames = 1 + (first_chunk_latent_frames - 1) * temporal_compression_ratio
        subsequent_chunk_frames = num_frame_per_block * temporal_compression_ratio

        self.first_chunk_frames = first_chunk_frames
        self.subsequent_chunk_frames = subsequent_chunk_frames
        self.target_fps = target_fps
        self.caption_field = caption_field
        self.video_extensions = video_extensions
        self.filter_invalid_folders = filter_invalid_folders
        self.deterministic = deterministic
        self.allow_padding = allow_padding
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame
        self.first_chunk_latent_frames = first_chunk_latent_frames
        self.return_image = return_image
        if min_latent_frames > 0:
            assert min_latent_frames % num_frame_per_block == 0, (
                f"min_latent_frames ({min_latent_frames}) must be a multiple of "
                f"num_frame_per_block ({num_frame_per_block})"
            )
        self.min_latent_frames = min_latent_frames
        self.single_video_only = single_video_only
        self.max_chunks_per_shot = max_chunks_per_shot
        self.scene_cut_prefix = scene_cut_prefix
        self.sample_warning_seconds = float(sample_warning_seconds or 0.0)
        self.sample_warning_interval_seconds = float(sample_warning_interval_seconds or 0.0)
        
        remaining_frames = total_frames - first_chunk_frames
        self.num_subsequent_segments = remaining_frames // subsequent_chunk_frames
        self.total_segments = 1 + self.num_subsequent_segments
        
        assert total_frames == first_chunk_frames + self.num_subsequent_segments * subsequent_chunk_frames, \
            f"Total frames ({total_frames}) must equal first_chunk_frames ({first_chunk_frames}) + " \
            f"num_subsequent_segments ({self.num_subsequent_segments}) * subsequent_chunk_frames ({subsequent_chunk_frames})"
        
        if not self.data_dir.exists():
            raise ValueError(f"Video directory not found: {self.data_dir}")
        if not self.caption_dir.exists():
            raise ValueError(f"Caption directory not found: {self.caption_dir}")
        
        # DistributedSampler assumes that every rank maps a given integer index
        # to the same sample.  Filesystem enumeration order is not a contract and
        # may differ across clients/nodes, so freeze the mapping by sample ID.
        self.folders = sorted(
            (d for d in self.data_dir.iterdir() if d.is_dir()),
            key=lambda path: path.name,
        )
        if len(self.folders) == 0:
            raise ValueError(f"No subdirectories found in {self.data_dir}")

        # Optionally pre-filter folders with insufficient frames
        # Note: This can be slow for large datasets due to IO operations
        if self.filter_invalid_folders:
            print(f"[MultiVideoConcatDataset] Pre-filtering {len(self.folders)} folders for sufficient frames...")
            valid_folders = []
            skipped_folders = []
            for folder in self.folders:
                if self._check_folder_has_enough_frames(folder):
                    valid_folders.append(folder)
                else:
                    skipped_folders.append(folder.name)
            
            if len(skipped_folders) > 0:
                print(f"[MultiVideoConcatDataset] Skipped {len(skipped_folders)} folders due to insufficient frames: {skipped_folders}")
            
            self.folders = valid_folders
            
            if len(self.folders) == 0:
                raise ValueError(f"No folders with sufficient frames found in {self.data_dir}")
        
        # Setup transforms
        self.resize_transform = transforms.Resize(
            self.video_size, 
            antialias=True
        )
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        # Lazy caches: avoid repeated decord.VideoReader / filesystem IO
        self._video_info_cache = {}   # video_path -> (total_frames, fps)
        self._folder_files_cache = {} # folder_path -> list of video paths
        
        if decord is not None:
            decord.bridge.set_bridge('torch')
    
    def _get_caption_folder(self, folder_name):
        """Return the caption directory path for a given folder (sample)."""
        return self.caption_dir / folder_name

    def _load_caption(self, video_path, folder_name):
        """Load caption for a video file."""
        video_stem = video_path.stem
        caption_folder = self._get_caption_folder(folder_name)
        caption_path = caption_folder / f"{video_stem}.json"
        
        if caption_path.exists():
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                    return caption_data.get(self.caption_field, "")
            except Exception:
                return ""
        return ""

    def _get_video_files_in_folder(self, folder_path):
        """Get sorted video files in a folder, keeping only those with a per-video caption (cached)."""
        key = str(folder_path)
        if key in self._folder_files_cache:
            return self._folder_files_cache[key]

        video_files = []
        for ext in self.video_extensions:
            video_files.extend(list(folder_path.glob(f'*{ext}')))
            video_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
        
        def get_numeric_key(path):
            try:
                return int(path.stem)
            except ValueError:
                return float('inf')
        
        video_files.sort(key=get_numeric_key)

        folder_name = folder_path.name
        filtered_videos = []
        for video_path in video_files:
            caption = self._load_caption(video_path, folder_name)
            if caption is not None and caption != "":
                filtered_videos.append(video_path)

        self._folder_files_cache[key] = filtered_videos
        return filtered_videos
    
    def _check_folder_has_enough_frames(self, folder_path):
        """Check if a folder has enough total frames across all videos to complete all segments.
        
        This is a lenient check: we verify that the total available frames across all videos
        is sufficient for the required segments, assuming ideal sampling.
        """
        video_files = self._get_video_files_in_folder(folder_path)
        if len(video_files) == 0:
            return False
        
        # Calculate total available frames (in target_fps timebase)
        total_available_frames = 0
        for video_path in video_files:
            try:
                total_frames, original_fps = self._get_video_info(video_path)
                # Convert to target_fps timebase
                available_in_target_fps = total_frames * self.target_fps / original_fps
                total_available_frames += available_in_target_fps
            except Exception:
                # If we can't read a video, be conservative and skip this folder
                return False
        
        # Check if we have enough frames for all segments
        required_frames = self.total_frames
        return total_available_frames >= required_frames
    
    def _get_video_info(self, video_path):
        """Get video information without loading frames (cached)."""
        key = str(video_path)
        if key in self._video_info_cache:
            return self._video_info_cache[key]
        if decord is None:
            info = self._get_video_info_ffprobe(video_path)
            self._video_info_cache[key] = info
            return info
        try:
            vr = decord.VideoReader(key, width=self.video_size[1], height=self.video_size[0])
        except:
            vr = decord.VideoReader(key)
        info = (len(vr), vr.get_avg_fps())
        self._video_info_cache[key] = info
        return info

    @staticmethod
    def _parse_fps(value):
        if not value or value == "0/0":
            return 0.0
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(value)

    def _get_video_info_ffprobe(self, video_path):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate,duration",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stream = json.loads(proc.stdout)["streams"][0]
            fps = self._parse_fps(stream.get("avg_frame_rate")) or self._parse_fps(stream.get("r_frame_rate"))
            frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
            if frames_raw and str(frames_raw).isdigit():
                total_frames = int(frames_raw)
            else:
                total_frames = int(round(float(stream.get("duration", 0.0)) * fps))
            if total_frames <= 0 or fps <= 0:
                raise ValueError(f"Could not infer frames/fps from ffprobe output for {video_path}")
            return total_frames, fps
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ffprobe failed to read video metadata for {video_path}: {exc}") from exc
    
    def _can_sample_from_position(self, total_frames, original_fps, num_frames, start_frame):
        """Check if we can sample num_frames starting from start_frame.
        
        Returns True if we can sample num_frames without exceeding video bounds.
        """
        if start_frame >= total_frames:
            return False
        
        # Calculate sampling interval: original_fps / target_fps
        sampling_interval = original_fps / self.target_fps
        
        # Calculate the last frame index we need (with rounding)
        # We need to check if we can get num_frames frames
        last_frame_needed = start_frame + (num_frames - 1) * sampling_interval
        
        # Account for rounding: the actual last frame index will be rounded
        # So we need some margin to ensure we don't exceed bounds
        return int(np.round(last_frame_needed)) < total_frames

    def _can_complete_all_segments_without_wrap(self, video_files, start_video_idx, start_frame):
        """Check if from (start_video_idx, start_frame) we can sample all segments
        without ever wrapping to the beginning (i.e. only use this video and later ones).
        When single_video_only is True, all segments must come from the same video.
        """
        vidx = start_video_idx
        start = start_frame

        # First segment
        total_frames, original_fps = self._get_video_info(video_files[vidx])
        if not self._can_sample_from_position(
            total_frames, original_fps, self.first_chunk_frames, start
        ):
            return False
        sampling_interval = original_fps / self.target_fps
        next_start = start + self.first_chunk_frames * sampling_interval
        if int(np.round(next_start)) >= total_frames:
            if self.single_video_only:
                return self.num_subsequent_segments == 0
            vidx += 1
            start = 0
        else:
            start = int(np.round(next_start))
        if vidx >= len(video_files):
            return False

        # Subsequent segments
        for seg_i in range(self.num_subsequent_segments):
            while vidx < len(video_files):
                total_frames, original_fps = self._get_video_info(video_files[vidx])
                if self._can_sample_from_position(
                    total_frames, original_fps, self.subsequent_chunk_frames, start
                ):
                    break
                if self.single_video_only:
                    return False
                vidx += 1
                start = 0
            if vidx >= len(video_files):
                return False
            sampling_interval = original_fps / self.target_fps
            next_start = start + self.subsequent_chunk_frames * sampling_interval
            if int(np.round(next_start)) >= total_frames:
                if self.single_video_only and seg_i < self.num_subsequent_segments - 1:
                    return False
                vidx += 1
                start = 0
            else:
                start = int(np.round(next_start))
        return True

    def _sample_random_start(self, video_files):
        """Sample a random (video_idx, start_frame) valid for the first segment,
        and from which we can complete ALL segments without wrapping to the start.
        Returns (video_idx, start_frame); falls back to (0, 0) if no valid start found.
        """
        candidates = []
        for video_idx, video_path in enumerate(video_files):
            total_frames, original_fps = self._get_video_info(video_path)
            sampling_interval = original_fps / self.target_fps
            last_needed = (self.first_chunk_frames - 1) * sampling_interval
            max_start = int(np.floor(total_frames - 1 - last_needed))
            if max_start < 0:
                continue
            step = max(1, max_start // 50)
            for start in range(0, max_start + 1, step):
                if not self._can_sample_from_position(
                    total_frames, original_fps, self.first_chunk_frames, start
                ):
                    continue
                if self._can_complete_all_segments_without_wrap(video_files, video_idx, start):
                    candidates.append((video_idx, start))
        if candidates:
            chosen = random.choice(candidates)
            return chosen
        return (0, 0)
    
    def _sample_frames_from_video(self, video_path, num_frames, start_frame=0):
        """Sample frames from a video preserving original duration.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            start_frame: Starting frame index (for repeated sampling)
        
        Returns:
            tuple: (frames_tensor, total_frames_in_video, original_fps)
        """
        if decord is None:
            return self._sample_frames_from_video_ffmpeg(video_path, num_frames, start_frame)

        try:
            vr = decord.VideoReader(str(video_path), width=self.video_size[1], height=self.video_size[0])
        except:
            vr = decord.VideoReader(str(video_path))
        
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        if total_frames == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        # Calculate frame sampling based on fps to preserve duration
        # Calculate sampling interval: original_fps / target_fps
        sampling_interval = original_fps / self.target_fps
        
        # Generate frame indices starting from start_frame
        indices = []
        current_frame = float(start_frame)
        for _ in range(num_frames):
            frame_idx = int(np.round(current_frame))
            frame_idx = min(frame_idx, total_frames - 1)
            indices.append(frame_idx)
            current_frame += sampling_interval
            if current_frame >= total_frames:
                # If we run out of frames, pad with the last frame
                remaining = num_frames - len(indices)
                indices.extend([total_frames - 1] * remaining)
                break
        
        indices = np.array(indices[:num_frames], dtype=np.int32)
        
        # Get video frames: shape (num_frames, height, width, 3)
        video_frames = vr.get_batch(indices).numpy()
        
        # Convert to tensor and permute to (num_frames, 3, height, width)
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous()
        
        # Convert to float and normalize pixel values from [0, 255] to [0, 1]
        video_tensor = video_tensor.float() / 255.0
        
        # Resize if needed
        if video_tensor.shape[2] != self.video_size[0] or video_tensor.shape[3] != self.video_size[1]:
            resized_frames = []
            for i in range(video_tensor.shape[0]):
                resized_frames.append(self.resize_transform(video_tensor[i]))
            video_tensor = torch.stack(resized_frames, dim=0)
        
        # Apply normalization: (x - 0.5) / 0.5 -> range [-1, 1]
        video_tensor = self.normalize(video_tensor)
        video_tensor = video_tensor.to(torch.float16)
        
        return video_tensor, total_frames, original_fps

    def _sample_frames_from_video_ffmpeg(self, video_path, num_frames, start_frame=0):
        total_frames, original_fps = self._get_video_info(video_path)
        start_time = max(0.0, float(start_frame) / max(original_fps, 1e-6))
        height, width = self.video_size
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            str(video_path),
            "-vf",
            f"fps={self.target_fps},scale={width}:{height}",
            "-frames:v",
            str(num_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise RuntimeError(f"ffmpeg failed to sample {video_path}: {stderr}") from exc

        frame_bytes = height * width * 3
        decoded_frames = len(proc.stdout) // frame_bytes
        if decoded_frames <= 0:
            raise RuntimeError(f"ffmpeg produced no frames for {video_path}")

        video_frames = np.frombuffer(proc.stdout[: decoded_frames * frame_bytes], dtype=np.uint8)
        video_frames = video_frames.reshape(decoded_frames, height, width, 3)
        video_tensor = torch.from_numpy(video_frames.copy()).permute(0, 3, 1, 2).contiguous()
        video_tensor = video_tensor.float() / 255.0
        if decoded_frames < num_frames:
            pad = video_tensor[-1:].repeat(num_frames - decoded_frames, 1, 1, 1)
            video_tensor = torch.cat([video_tensor, pad], dim=0)
        elif decoded_frames > num_frames:
            video_tensor = video_tensor[:num_frames]

        video_tensor = self.normalize(video_tensor)
        video_tensor = video_tensor.to(torch.float16)
        return video_tensor, total_frames, original_fps
    
    def __len__(self):
        return len(self.folders)
    
    @staticmethod
    def _sample_failure(reason):
        return False, reason

    def _try_get_item_from_folder(self, folder_idx, deterministic: bool = False):
        """Try to get item from a specific folder.

        Returns (True, result) on success and (False, reason) on failure.
        The reason is used by __getitem__ warnings so a long scan does not
        look like a silent hang when most folders are too short.
        """
        folder_path = self.folders[folder_idx]
        folder_name = folder_path.name
        
        video_files = self._get_video_files_in_folder(folder_path)
        
        if len(video_files) == 0:
            return self._sample_failure(f"{folder_name}: no videos with captions")
        
        # Collect all segments
        all_segments = []
        prompts_list = []
        
        try:
            # Start position
            if deterministic:
                current_video_idx, current_start_frame = 0, 0
            else:
                # Random start: don't always start from the first video, so we get more diversity
                current_video_idx, current_start_frame = self._sample_random_start(video_files)
            
            # Sample first segment (9 frames)
            video_path = video_files[current_video_idx]
            prev_seg_video_idx = current_video_idx
            total_frames, original_fps = self._get_video_info(video_path)

            # Ensure the current video is long enough for the first segment
            while not self._can_sample_from_position(
                total_frames, original_fps, self.first_chunk_frames, current_start_frame
            ):
                if self.single_video_only:
                    return self._sample_failure(
                        f"{folder_name}: first video is too short for the first chunk"
                    )
                current_video_idx += 1
                current_start_frame = 0
                if current_video_idx >= len(video_files):
                    return self._sample_failure(
                        f"{folder_name}: no video can provide the first chunk"
                    )
                video_path = video_files[current_video_idx]
                prev_seg_video_idx = current_video_idx
                total_frames, original_fps = self._get_video_info(video_path)

            # Sample first segment
            segment_frames, total_frames, original_fps = self._sample_frames_from_video(
                video_path, self.first_chunk_frames, current_start_frame
            )
            all_segments.append(segment_frames)
            
            prompt = self._load_caption(video_path, folder_name)
            prompts_list.append(prompt)
            
            # Update position for next sampling
            sampling_interval = original_fps / self.target_fps
            next_start_frame = current_start_frame + self.first_chunk_frames * sampling_interval
            
            # If we've exhausted this video, move to next
            source_exhausted = False
            if int(np.round(next_start_frame)) >= total_frames:
                if self.single_video_only:
                    if self.num_subsequent_segments > 0:
                        if self.allow_padding:
                            source_exhausted = True
                        else:
                            return self._sample_failure(
                                f"{folder_name}: single-video sample ends after first chunk"
                            )
                else:
                    current_video_idx += 1
                    current_start_frame = 0
            else:
                current_start_frame = int(np.round(next_start_frame))

            chunks_from_current_video = 1 if current_video_idx == prev_seg_video_idx else 0

            # Sample subsequent segments (12 frames each). No wrap: we only use videos from start onward.
            for seg_idx in range(self.num_subsequent_segments):
                if source_exhausted:
                    # In single-video variable-length mode, the remainder is
                    # padded below from this video's last decoded frame.  Do
                    # not advance to another file in the folder.
                    if self.allow_padding:
                        break
                    return self._sample_failure(
                        f"{folder_name}: single-video source is exhausted"
                    )
                if current_video_idx >= len(video_files):
                    if self.allow_padding:
                        break
                    return self._sample_failure(
                        f"{folder_name}: ran out of videos before all chunks were filled"
                    )

                # Force virtual scene cut if max_shot_chunks reached:
                # skip 1 second of video and treat the remainder as a new shot.
                forced_scene_cut = False
                if (self.max_chunks_per_shot > 0
                        and chunks_from_current_video >= self.max_chunks_per_shot):
                    vp = video_files[current_video_idx]
                    _, ofps = self._get_video_info(vp)
                    current_start_frame += int(np.round(ofps))
                    chunks_from_current_video = 0
                    forced_scene_cut = True

                video_path = video_files[current_video_idx]
                total_frames, original_fps = self._get_video_info(video_path)

                can_sample = True
                while not self._can_sample_from_position(
                    total_frames, original_fps, self.subsequent_chunk_frames, current_start_frame
                ):
                    if self.single_video_only:
                        can_sample = False
                        break
                    current_video_idx += 1
                    current_start_frame = 0
                    chunks_from_current_video = 0
                    if current_video_idx >= len(video_files):
                        can_sample = False
                        break
                    video_path = video_files[current_video_idx]
                    total_frames, original_fps = self._get_video_info(video_path)

                if not can_sample:
                    if self.allow_padding:
                        break
                    return self._sample_failure(
                        f"{folder_name}: remaining videos are too short for the next chunk"
                    )

                is_scene_cut = (current_video_idx != prev_seg_video_idx) or forced_scene_cut

                # Sample segment
                segment_frames, total_frames, original_fps = self._sample_frames_from_video(
                    video_path, self.subsequent_chunk_frames, current_start_frame
                )
                all_segments.append(segment_frames)
                
                prompt = self._load_caption(video_path, folder_name)
                if is_scene_cut and self.scene_cut_prefix:
                    prompt = self.scene_cut_prefix + prompt
                prompts_list.append(prompt)

                prev_seg_video_idx = current_video_idx
                chunks_from_current_video += 1

                # Update position for next sampling
                sampling_interval = original_fps / self.target_fps
                next_start_frame = current_start_frame + self.subsequent_chunk_frames * sampling_interval
                
                # If we've exhausted this video, move to next
                if int(np.round(next_start_frame)) >= total_frames:
                    if self.single_video_only:
                        if seg_idx < self.num_subsequent_segments - 1:
                            if self.allow_padding:
                                # Tail-pad after this complete block; never
                                # continue with the next video in the folder.
                                break
                            return self._sample_failure(
                                f"{folder_name}: single-video sample is too short"
                            )
                    else:
                        current_video_idx += 1
                        current_start_frame = 0
                        chunks_from_current_video = 0
                else:
                    current_start_frame = int(np.round(next_start_frame))

            num_filled_segments = len(all_segments)
            if num_filled_segments == 0:
                num_valid_latent_frames = 0
            else:
                num_valid_latent_frames = (
                    self.first_chunk_latent_frames
                    + (num_filled_segments - 1) * self.num_frame_per_block
                )

            # Reject if below minimum latent frame threshold
            if self.allow_padding and self.min_latent_frames > 0:
                if num_valid_latent_frames < self.min_latent_frames:
                    return self._sample_failure(
                        f"{folder_name}: only {num_valid_latent_frames} valid latent frames, "
                        f"below min_latent_frames={self.min_latent_frames}"
                    )

            if num_filled_segments < self.total_segments:
                last_prompt = prompts_list[-1] if prompts_list else ""
                prompts_list.extend([last_prompt] * (self.total_segments - num_filled_segments))
            
            # Concatenate all segments: (total_frames, 3, height, width)
            concatenated_video = torch.cat(all_segments, dim=0)
            
            # Ensure we have exactly total_frames
            if concatenated_video.shape[0] != self.total_frames:
                # Pad or trim if necessary
                if concatenated_video.shape[0] < self.total_frames:
                    # Pad with last frame
                    last_frame = concatenated_video[-1:].repeat(self.total_frames - concatenated_video.shape[0], 1, 1, 1)
                    concatenated_video = torch.cat([concatenated_video, last_frame], dim=0)
                else:
                    # Trim
                    concatenated_video = concatenated_video[:self.total_frames]
            
            result = {
                'frames': concatenated_video.permute(1, 0, 2, 3),
                'prompts': prompts_list,
                'idx': folder_idx
            }
            if self.return_image:
                result['image'] = concatenated_video[0]
            if self.allow_padding:
                result['num_valid_latent_frames'] = num_valid_latent_frames
            return True, result
        except Exception as exc:
            return self._sample_failure(f"{folder_name}: {type(exc).__name__}: {exc}")

    def __getitem__(self, idx):
        start_time = time.monotonic()
        last_warning_time = start_time
        attempts = 0
        last_failure = None

        def maybe_warn(folder_idx, failure_reason):
            nonlocal last_warning_time
            if self.sample_warning_seconds <= 0:
                return
            elapsed = time.monotonic() - start_time
            if elapsed < self.sample_warning_seconds:
                return
            if (
                self.sample_warning_interval_seconds > 0
                and time.monotonic() - last_warning_time < self.sample_warning_interval_seconds
            ):
                return
            last_warning_time = time.monotonic()
            folder_name = self.folders[folder_idx % len(self.folders)].name
            warnings.warn(
                "[MultiVideoConcatDataset] Still searching for a valid sample "
                f"after {elapsed:.1f}s and {attempts} folder attempts "
                f"(requested_idx={idx}, current_folder={folder_name}, "
                f"last_failure={failure_reason}). This usually means the dataset "
                f"does not contain enough video duration for total_frames={self.total_frames} "
                f"at target_fps={self.target_fps}. Consider reducing the training "
                "window, enabling allow_padding, lowering min_latent_frames, or "
                "pre-filtering invalid folders.",
                RuntimeWarning,
                stacklevel=2,
            )

        # First try the requested folder
        attempts += 1
        success, result = self._try_get_item_from_folder(idx, deterministic=self.deterministic)
        if success:
            return result
        last_failure = result
        maybe_warn(idx, last_failure)
        
        # If the requested folder fails, try other folders.
        # If any valid folder exists in the dataset, try to return data from it:
        # scan every other folder starting from idx + 1 and return the first
        # successful sample.
        num_folders = len(self.folders)
        for i in range(1, num_folders):
            alt_idx = (idx + i) % num_folders
            attempts += 1
            success, result = self._try_get_item_from_folder(
                alt_idx,
                deterministic=self.deterministic,
            )
            if success:
                return result
            last_failure = result
            maybe_warn(alt_idx, last_failure)
        
        # If all attempts fail, raise an error
        elapsed = time.monotonic() - start_time
        if self.sample_warning_seconds > 0 and elapsed >= self.sample_warning_seconds:
            warnings.warn(
                "[MultiVideoConcatDataset] No valid sample was found after "
                f"{elapsed:.1f}s and {attempts} folder attempts "
                f"(requested_idx={idx}, last_failure={last_failure}).",
                RuntimeWarning,
                stacklevel=2,
            )
        raise ValueError(
            f"Failed to sample valid data from folder {idx} and nearby folders. "
            f"This may indicate insufficient valid videos in the dataset. "
            f"Tried {attempts} folders in {elapsed:.1f}s. Last failure: {last_failure}"
        )


class ResumableDistributedSampler(
    torch.utils.data.distributed.DistributedSampler
):
    """DistributedSampler with a one-shot rank-local resume offset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_index = 0

    def set_start_index(self, start_index: int):
        start_index = int(start_index)
        if start_index < 0 or start_index > self.num_samples:
            raise ValueError(
                f"start_index must be in [0, {self.num_samples}], got {start_index}"
            )
        self._start_index = start_index

    def __iter__(self):
        start_index = self._start_index
        # The cursor applies only to the first iterator created after resume.
        # Later epochs must expose the complete rank-local shard.
        self._start_index = 0
        if not self.shuffle:
            # Match torch DistributedSampler exactly without materializing
            # range(N), a padded copy, or a million-element randperm per rank.
            # For drop_last=False, padded global positions wrap to the head in
            # the same order as DistributedSampler's repeated prefix.
            dataset_size = len(self.dataset)

            def sequential_shard():
                for local_position in range(start_index, self.num_samples):
                    global_position = (
                        self.rank + local_position * self.num_replicas
                    )
                    if global_position >= dataset_size:
                        global_position %= dataset_size
                    yield global_position

            return sequential_shard()
        indices = list(super().__iter__())
        return iter(indices[start_index:])


def build_distributed_sampler(
    dataset,
    *,
    rank: int,
    num_replicas: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
):
    """Build a DistributedSampler whose replicas share one permutation.

    ``rank`` chooses a disjoint slice after the shared permutation is built;
    it must never be mixed into ``seed``. Under sequence parallelism callers
    pass the DP rank and size so every SP rank in a group receives the same
    sample while distinct DP replicas remain disjoint.
    """
    rank = int(rank)
    num_replicas = int(num_replicas)
    if num_replicas <= 0:
        raise ValueError(f"num_replicas must be positive, got {num_replicas}")
    if rank < 0 or rank >= num_replicas:
        raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
    return ResumableDistributedSampler(
        dataset,
        rank=rank,
        num_replicas=num_replicas,
        shuffle=bool(shuffle),
        seed=int(seed),
        drop_last=bool(drop_last),
    )


def resolve_resume_data_cursor(
    *,
    optimizer_step: int,
    gradient_accumulation_steps: int,
    batches_per_epoch: int,
):
    """Map a saved optimizer step to dataloader epoch and batch offset."""
    optimizer_step = int(optimizer_step)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    batches_per_epoch = int(batches_per_epoch)
    if optimizer_step < 0:
        raise ValueError(f"optimizer_step must be non-negative, got {optimizer_step}")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, got "
            f"{gradient_accumulation_steps}"
        )
    if batches_per_epoch <= 0:
        raise ValueError(
            f"batches_per_epoch must be positive, got {batches_per_epoch}"
        )
    consumed_microbatches = optimizer_step * gradient_accumulation_steps
    return divmod(consumed_microbatches, batches_per_epoch)


def cycle(dl, start_epoch: int = 0):
    """Yield forever and advance epoch-aware samplers between passes."""
    epoch = int(start_epoch)
    if epoch < 0:
        raise ValueError(f"start_epoch must be non-negative, got {epoch}")
    sampler = getattr(dl, "sampler", None)
    while True:
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        for data in dl:
            yield data
        epoch += 1

def multi_video_collate_fn(batch):
    # batch is a length-B list of dictionaries returned by __getitem__.
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B, T, C, H, W)

    # Keep prompts as one list per sample:
    # [[p0_seg0, p0_seg1, ...], [p1_seg0, ...], ...].
    prompts_list = [b["prompts"] for b in batch]          # List[List[str]]

    idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)

    result = {
        "frames": frames,
        "prompts": prompts_list,
        "idx": idx,
    }

    if "image" in batch[0]:
        result["image"] = torch.stack([b["image"] for b in batch], dim=0)

    if "num_valid_latent_frames" in batch[0]:
        result["num_valid_latent_frames"] = torch.tensor(
            [b["num_valid_latent_frames"] for b in batch], dtype=torch.long
        )
    if "num_valid_raw_frames" in batch[0]:
        result["num_valid_raw_frames"] = torch.tensor(
            [b["num_valid_raw_frames"] for b in batch], dtype=torch.long
        )
    if "sample_id" in batch[0]:
        result["sample_id"] = [b["sample_id"] for b in batch]
    if "video" in batch[0]:
        result["video"] = [b["video"] for b in batch]

    return result


def eval_collate_fn(batch):
    """Collate for text-only datasets (no frames)."""
    prompts_list = [b["prompts"] for b in batch]
    idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    result = {
        "prompts": prompts_list,
        "idx": idx,
    }
    if "shot_durations" in batch[0]:
        result["shot_durations"] = [b["shot_durations"] for b in batch]
    return result
