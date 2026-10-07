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

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from longwam.utils.index_manifest_sampler import (
    IndexManifestSampler,
    load_index_manifest,
    write_complement_manifest,
)


SELECTION = "1" * 64
SAMPLING = "2" * 64
HISTORICAL_GROUP = "3" * 64
SOURCE_STATE = "4" * 64


class _Dataset:
    dataset_frame_ranges = ((0, 4), (4, 10))
    sampling_signature = SAMPLING
    human300_episode_selection = SimpleNamespace(signature=SELECTION)

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> int:
        return index


def _history() -> dict:
    artifact = {"path": "/sealed/artifact", "file_sha256": "5" * 64}
    return {
        "source_plan": {**artifact, "contract_sha256": "6" * 64},
        "continuation_plan": {**artifact, "contract_sha256": "7" * 64},
        "source_state": {
            "path": "/sealed/state",
            "file_sha256": SOURCE_STATE,
            "contract_sha256": "8" * 64,
        },
        "historical_sampler_sha256": "9" * 64,
        "draw_count": 17,
    }


def _write(tmp_path: Path) -> tuple[Path, dict]:
    seen = np.zeros(10, dtype=np.bool_)
    seen[[0, 2, 5]] = True
    ordered = np.asarray([7, 1, 8, 3, 9, 4, 6], dtype="<u4")
    path = tmp_path / "complement.json"
    manifest = write_complement_manifest(
        path,
        ordered_indices=ordered,
        seen_mask=seen,
        dataset_identity={
            "selection_signature": SELECTION,
            "sampling_signature": SAMPLING,
            "historical_group_signature": HISTORICAL_GROUP,
            "ranges": ((0, 4), (4, 10)),
            "group_names": ("atomic", "composite"),
            "family_ranges": {"atomic": (0, 4), "composite": (4, 10)},
        },
        history_identity=_history(),
        shuffle_seed=123,
        world_size=2,
        per_rank_batch_size=2,
    )
    return path, manifest


def _sampler(path: Path, manifest: dict) -> IndexManifestSampler:
    return IndexManifestSampler(
        _Dataset(),
        path,
        seed=123,
        batch_size=2,
        num_processes=2,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_selection_signature=SELECTION,
        expected_sampling_signature=SAMPLING,
        expected_historical_group_signature=HISTORICAL_GROUP,
        expected_source_state_sha256=SOURCE_STATE,
    )


def _accelerate_worker(
    rank: int,
    world_size: int,
    master_port: int,
    manifest_path: str,
    manifest_sha256: str,
    resume_batch_offset: int,
    output_dir: str,
) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "OMP_NUM_THREADS": "1",
            "ACCELERATE_TORCH_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(cpu=True)
    sampler = IndexManifestSampler(
        _Dataset(),
        manifest_path,
        seed=123,
        batch_size=2,
        num_processes=world_size,
        expected_manifest_sha256=manifest_sha256,
        expected_selection_signature=SELECTION,
        expected_sampling_signature=SAMPLING,
        expected_historical_group_signature=HISTORICAL_GROUP,
        expected_source_state_sha256=SOURCE_STATE,
    )
    sampler.set_resume_batch_offset(resume_batch_offset)
    loader = DataLoader(
        _Dataset(),
        batch_size=2,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
    )
    prepared = accelerator.prepare(loader)
    batches = [[int(value) for value in batch.tolist()] for batch in prepared]
    result = {
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "device": accelerator.device.type,
        "loader_type": type(prepared).__name__,
        "batches": batches,
    }
    Path(output_dir, f"offset_{resume_batch_offset}_rank_{rank}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _run_accelerate_workers(
    *, manifest_path: Path, manifest_sha256: str, offset: int, output_dir: Path
) -> list[dict]:
    context = mp.spawn(
        _accelerate_worker,
        args=(
            2,
            _unused_local_port(),
            str(manifest_path),
            manifest_sha256,
            offset,
            str(output_dir),
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if context.join(timeout=1):
            break
    else:
        for process in context.processes:
            process.terminate()
        for process in context.processes:
            process.join(timeout=5)
        pytest.fail("2-rank Accelerate sampler integration timed out.")
    return [
        json.loads((output_dir / f"offset_{offset}_rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]


def _interleave_rank_batches(results: list[dict]) -> list[int]:
    batches_by_rank = [result["batches"] for result in results]
    assert len({len(batches) for batches in batches_by_rank}) == 1
    return [
        value
        for step in range(len(batches_by_rank[0]))
        for rank in range(len(batches_by_rank))
        for value in batches_by_rank[rank][step]
    ]


def test_manifest_proves_exact_complement_and_sampler_pads_once(tmp_path: Path):
    path, written = _write(tmp_path)
    loaded, index_path = load_index_manifest(path)

    assert loaded == written
    assert index_path.stat().st_size == 7 * 4
    assert loaded["proof"]["seen_unique_count"] == 3
    assert loaded["proof"]["complement_unique_count"] == 7
    assert loaded["proof"]["intersection_count"] == 0
    assert loaded["proof"]["union_count"] == 10
    assert loaded["proof"]["family_counts"] == {
        "atomic": {"total": 4, "seen": 2, "complement": 2},
        "composite": {"total": 6, "seen": 1, "complement": 5},
    }
    assert loaded["distribution"] == {
        "world_size": 2,
        "per_rank_batch_size": 2,
        "global_batch_size": 4,
        "unique_count": 7,
        "padded_count": 8,
        "padding_count": 1,
        "padding_policy": "repeat_order_prefix_in_final_global_batch_v1",
        "padding_indices_sha256": loaded["distribution"]["padding_indices_sha256"],
        "global_steps": 2,
    }

    sampler = _sampler(path, loaded)
    assert len(sampler) == 8
    assert list(sampler) == [7, 1, 8, 3, 9, 4, 6, 7]
    sampler.set_resume_batch_offset(1)
    assert list(sampler) == [9, 4, 6, 7]
    sampler.clear_resume_batch_offset()
    assert list(sampler) == [7, 1, 8, 3, 9, 4, 6, 7]


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="PyTorch gloo is unavailable")
def test_accelerate_two_rank_sharding_and_resume_are_exact(tmp_path: Path):
    """Exercise the same prepare/DataLoaderShard path used by training."""

    path, manifest = _write(tmp_path / "manifest")
    full_results = _run_accelerate_workers(
        manifest_path=path,
        manifest_sha256=manifest["manifest_sha256"],
        offset=0,
        output_dir=tmp_path,
    )
    resumed_results = _run_accelerate_workers(
        manifest_path=path,
        manifest_sha256=manifest["manifest_sha256"],
        offset=1,
        output_dir=tmp_path,
    )

    for results in (full_results, resumed_results):
        assert [result["rank"] for result in results] == [0, 1]
        assert all(result["world_size"] == 2 for result in results)
        assert all(result["device"] == "cpu" for result in results)
        assert all(result["loader_type"] == "DataLoaderShard" for result in results)

    full = _interleave_rank_batches(full_results)
    resumed = _interleave_rank_batches(resumed_results)
    assert full == [7, 1, 8, 3, 9, 4, 6, 7]
    assert len(set(full[:-1])) == 7
    assert full[-1] == full[0]
    assert resumed == full[4:]


def test_sampler_is_sealed_to_one_epoch_and_exact_identity(tmp_path: Path):
    path, manifest = _write(tmp_path)
    sampler = _sampler(path, manifest)

    with pytest.raises(RuntimeError, match="one-pass"):
        sampler.set_epoch(1)
    with pytest.raises(RuntimeError, match="epoch offset"):
        sampler.set_epoch_offset(1)
    with pytest.raises(ValueError, match="outside"):
        sampler.set_resume_batch_offset(3)
    with pytest.raises(ValueError, match="identity mismatch"):
        IndexManifestSampler(
            _Dataset(),
            path,
            seed=123,
            batch_size=2,
            num_processes=2,
            expected_manifest_sha256="a" * 64,
            expected_selection_signature=SELECTION,
            expected_sampling_signature=SAMPLING,
            expected_historical_group_signature=HISTORICAL_GROUP,
            expected_source_state_sha256=SOURCE_STATE,
        )


def test_manifest_rejects_duplicates_and_non_complement(tmp_path: Path):
    seen = np.zeros(5, dtype=np.bool_)
    seen[0] = True
    common = {
        "output_path": tmp_path / "bad.json",
        "seen_mask": seen,
        "dataset_identity": {
            "selection_signature": SELECTION,
            "sampling_signature": SAMPLING,
            "historical_group_signature": HISTORICAL_GROUP,
            "ranges": ((0, 5),),
            "group_names": ("all",),
            "family_ranges": {"all": (0, 5)},
        },
        "history_identity": _history(),
        "shuffle_seed": 123,
        "world_size": 1,
        "per_rank_batch_size": 4,
    }
    with pytest.raises(ValueError, match="duplicates"):
        write_complement_manifest(
            ordered_indices=np.asarray([1, 1, 2, 3, 4], dtype="<u4"),
            **common,
        )
    with pytest.raises(ValueError, match="exact complement"):
        write_complement_manifest(
            ordered_indices=np.asarray([1, 2, 3], dtype="<u4"),
            **common,
        )


def test_loader_detects_payload_and_manifest_tampering(tmp_path: Path):
    path, _ = _write(tmp_path)
    document = json.loads(path.read_text())
    index_path = path.parent / document["payloads"]["ordered_indices"]["path"]
    with index_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="payload SHA-256"):
        load_index_manifest(path)

    # Restore a fresh artifact in another directory, then alter signed JSON.
    clean_path, _ = _write(tmp_path / "clean")
    clean = json.loads(clean_path.read_text())
    clean["distribution"]["global_steps"] += 1
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        load_index_manifest(clean_path)


def test_writer_refuses_to_replace_immutable_artifacts(tmp_path: Path):
    path, _ = _write(tmp_path)
    assert path.exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write(tmp_path)
