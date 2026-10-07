#!/usr/bin/env python3
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot-video integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 (robot-video index prebuild utility)
# Changes: Public robot-video names and environment variables; manifest validation unchanged.
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

"""Prebuild and attest mmap offset indexes for frozen robot_video manifests.

This command reads the immutable JSONL manifests and receipt, writes only
compact ``.llidx`` sidecars under this repository, and publishes a receipt for
the resulting indexes.  It never opens or modifies the referenced videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LONGLIVE_ROOT = ROOT / "third_party" / "LongLive"
if str(LONGLIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(LONGLIVE_ROOT))

from utils.dataset import RobotVideoManifestDataset  # noqa: E402


RELEASE_ROOT = Path(os.environ.get("LONGWAM_VIDEO_DATA_ROOT", "/path/to/robot-video-data"))
RECEIPT_SHA256 = (
    "0a68f0c07a05d7bab571ea205d108c4ad0f0bca0509cc639cb796e6aaee21976"
)
DEFAULT_OUTPUT_DIR = Path(os.environ.get("LONGWAM_VIDEO_INDEX_CACHE", str(ROOT / "cache/robot_video_indexes")))
STAGE_CARRIERS = {"S": 64, "M": 128, "L": 192}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def build(output_dir: Path) -> dict:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = RELEASE_ROOT / "receipt.json"
    if sha256_file(receipt_path) != RECEIPT_SHA256:
        raise RuntimeError("frozen robot_video receipt SHA-256 changed")

    outputs = []
    for stage, carrier in STAGE_CARRIERS.items():
        manifest_path = RELEASE_ROOT / "manifests" / stage / "train.jsonl"
        dataset = RobotVideoManifestDataset(
            manifest_path=manifest_path,
            video_size=(384, 640),
            total_frames=1 + (carrier - 1) * 4,
            target_fps=24,
            num_frame_per_block=8,
            temporal_compression_ratio=4,
            expected_stage=stage,
            expected_split="train",
            receipt_path=receipt_path,
            expected_receipt_sha256=RECEIPT_SHA256,
            index_cache_dir=output_dir,
            return_image=True,
        )
        contract = dataset.distributed_index_contract()
        index_path = dataset.index_path
        outputs.append(
            {
                "stage": stage,
                "carrier_latent_frames": carrier,
                "manifest_path": str(manifest_path),
                "manifest_sha256": contract["manifest_sha256"],
                "record_count": contract["count"],
                "ids_sha256": contract["ids_sha256"],
                "index_path": str(index_path),
                "index_size_bytes": index_path.stat().st_size,
                "index_sha256": sha256_file(index_path),
            }
        )
        dataset._close_mmaps()
        print(
            f"[{stage}] rows={contract['count']} index={index_path} "
            f"sha256={outputs[-1]['index_sha256']}",
            flush=True,
        )

    body = {
        "contract_version": "longlive-robot-video-offset-index-receipt-v1",
        "dataset_media_read": False,
        "frozen_receipt_path": str(receipt_path),
        "frozen_receipt_sha256": RECEIPT_SHA256,
        "index_root": str(output_dir),
        "outputs": outputs,
        "status": "accepted_offset_indexes",
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    atomic_write_json(output_dir / "receipt.json", receipt)
    print(f"receipt_sha256={receipt['receipt_sha256']}", flush=True)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cache directory for .llidx files and their receipt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
