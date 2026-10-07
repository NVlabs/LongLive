# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot integration imported from the author video-training branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/tests/test_fixed_evaluation.py
# Changes: Robot-training/validation additions; existing upstream notices are retained where present.
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

import json
import tempfile
import unittest
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from utils.evaluation import (
    deterministic_evaluation_seed,
    evaluation_artifact_run_modes,
    evaluation_group_specs,
    evaluation_run_modes,
    fixed_evaluation_log_key,
    fixed_evaluation_writer,
    require_equal_evaluation_value,
    select_fixed_evaluation_subset,
    synchronize_evaluation_failure,
)


class _FolderDataset(Dataset):
    def __init__(self, root, sample_ids):
        self.folders = [Path(root) / sample_id for sample_id in sample_ids]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        return self.folders[index].name


def _distributed_failure_worker(rank, world_size, init_file, output_dir):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    messages = []
    try:
        for owner, phase in ((0, "rank0 manifest"), (1, "rank1 generation")):
            try:
                local_error = (
                    ValueError(f"failure-from-rank-{rank}")
                    if rank == owner
                    else None
                )
                synchronize_evaluation_failure(local_error, phase=phase)
            except RuntimeError as exc:
                messages.append(str(exc))

        try:
            require_equal_evaluation_value(
                rank + 1, phase="batch sample count"
            )
        except RuntimeError as exc:
            messages.append(str(exc))
        Path(output_dir, f"rank_{rank}.json").write_text(
            json.dumps(messages), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


class FixedEvaluationTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is required")
    def test_distributed_failures_reach_every_rank_with_same_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = Path(tmp) / "gloo_init"
            mp.spawn(
                _distributed_failure_worker,
                args=(2, str(init_file), tmp),
                nprocs=2,
                join=True,
            )
            rank_messages = [
                json.loads(Path(tmp, f"rank_{rank}.json").read_text())
                for rank in range(2)
            ]

        self.assertEqual(rank_messages[0], rank_messages[1])
        self.assertEqual(len(rank_messages[0]), 3)
        self.assertIn("rank=0 phase='rank0 manifest'", rank_messages[0][0])
        self.assertIn("rank=1 phase='rank1 generation'", rank_messages[0][1])
        self.assertIn("rank=0 value=1, rank=1 value=2", rank_messages[0][2])

    def test_legacy_config_resolves_to_one_unnamed_group(self):
        groups = evaluation_group_specs(
            {},
            default_num_frames=64,
            default_sample_ids=("a", "b"),
            default_seed=17,
            default_data_path="/validation",
            num_frame_per_block=8,
        )
        self.assertEqual(len(groups), 1)
        self.assertIsNone(groups[0].name)
        self.assertEqual(groups[0].num_frames, 64)
        self.assertEqual(groups[0].sample_ids, ("a", "b"))
        self.assertEqual(groups[0].seed, 17)

    def test_named_groups_preserve_short_then_long_order(self):
        groups = evaluation_group_specs(
            {
                "groups": [
                    {
                        "name": "short_f32",
                        "num_frames": 32,
                        "sample_ids": ["a"],
                    },
                    {
                        "name": "long_f64",
                        "num_frames": 64,
                        "sample_ids": ["a"],
                    },
                ]
            },
            default_num_frames=64,
            default_sample_ids=("fallback",),
            default_seed=20260630,
            default_data_path="/validation",
            num_frame_per_block=8,
        )
        self.assertEqual(
            [group.name for group in groups], ["short_f32", "long_f64"]
        )
        self.assertEqual([group.num_frames for group in groups], [32, 64])
        self.assertEqual([group.sample_ids for group in groups], [("a",), ("a",)])

    def test_invalid_named_group_contract_fails_early(self):
        common = {
            "default_num_frames": 64,
            "default_sample_ids": ("a",),
            "default_seed": 1,
            "default_data_path": "/validation",
            "num_frame_per_block": 8,
        }
        with self.assertRaisesRegex(ValueError, "duplicated"):
            evaluation_group_specs(
                {
                    "groups": [
                        {"name": "same", "num_frames": 32},
                        {"name": "same", "num_frames": 64},
                    ]
                },
                **common,
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            evaluation_group_specs(
                {"groups": [{"name": "bad_f30", "num_frames": 30}]},
                **common,
            )

    def test_selection_uses_ids_not_folder_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _FolderDataset(tmp, ["c", "a", "b"])
            subset, samples = select_fixed_evaluation_subset(dataset, ["a", "c"])

        self.assertEqual(list(subset), ["a", "c"])
        self.assertEqual([sample.dataset_index for sample in samples], [1, 0])
        self.assertEqual([sample.slot for sample in samples], [0, 1])

    def test_selection_validates_only_selected_manifest_indices(self):
        class HorizonDataset(_FolderDataset):
            def __init__(self, root, sample_ids):
                super().__init__(root, sample_ids)
                self.validated = None

            def validate_fixed_evaluation_indices(self, indices):
                self.validated = tuple(indices)
                return tuple(self.folders[index].name for index in indices)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = HorizonDataset(tmp, ["f32", "f64", "f96"])
            subset, samples = select_fixed_evaluation_subset(
                dataset, ["f96", "f32"]
            )

        self.assertEqual(dataset.validated, (2, 0))
        self.assertEqual([sample.dataset_index for sample in samples], [2, 0])
        self.assertEqual(list(subset), ["f96", "f32"])

    def test_dataset_cannot_relabel_selected_evaluation_ids(self):
        class RelabelingDataset(_FolderDataset):
            def validate_fixed_evaluation_indices(self, indices):
                return tuple("wrong" for _ in indices)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = RelabelingDataset(tmp, ["a"])
            with self.assertRaisesRegex(ValueError, "different IDs"):
                select_fixed_evaluation_subset(dataset, ["a"])

    def test_missing_and_duplicate_ids_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _FolderDataset(tmp, ["a", "b"])
            with self.assertRaisesRegex(ValueError, "must be unique"):
                select_fixed_evaluation_subset(dataset, ["a", "a"])
            with self.assertRaisesRegex(ValueError, "absent"):
                select_fixed_evaluation_subset(dataset, ["missing"])

    def test_padded_replicas_have_one_canonical_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _FolderDataset(tmp, ["a", "b", "c", "d", "e", "f"])
            _, samples = select_fixed_evaluation_subset(
                dataset, ["a", "b", "c", "d", "e", "f"]
            )

        writers = []
        for sample in samples:
            for data_rank in range(16):
                writer = fixed_evaluation_writer(
                    sample.dataset_index,
                    samples,
                    data_rank=data_rank,
                    data_replicas=16,
                    sequence_parallel_rank=0,
                )
                if writer is not None:
                    writers.append((writer.slot, data_rank))
        self.assertEqual(writers, [(slot, slot) for slot in range(6)])
        self.assertIsNone(
            fixed_evaluation_writer(
                samples[0].dataset_index,
                samples,
                data_rank=0,
                data_replicas=16,
                sequence_parallel_rank=1,
            )
        )

    def test_seed_and_weight_mode_are_stable(self):
        self.assertEqual(deterministic_evaluation_seed(20260630, 3), 20260633)
        self.assertEqual(
            evaluation_run_modes("model", ema_available=False),
            (("_model", False, "model"),),
        )
        self.assertEqual(
            evaluation_run_modes("ema_if_available", ema_available=True),
            (("_ema", True, "ema"),),
        )
        with self.assertRaisesRegex(RuntimeError, "before EMA"):
            evaluation_run_modes("ema", ema_available=False)

    def test_legacy_nonfixed_model_artifact_keeps_empty_suffix(self):
        model_modes = evaluation_run_modes("model", ema_available=False)
        self.assertEqual(
            evaluation_artifact_run_modes(
                model_modes,
                group_name=None,
                fixed_evaluation=False,
            ),
            (("", False, "model"),),
        )
        self.assertEqual(
            evaluation_artifact_run_modes(
                model_modes,
                group_name=None,
                fixed_evaluation=True,
            ),
            (("_model", False, "model"),),
        )
        both_modes = evaluation_run_modes("both", ema_available=True)
        self.assertEqual(
            evaluation_artifact_run_modes(
                both_modes,
                group_name=None,
                fixed_evaluation=False,
            ),
            (("", False, "model"), ("_ema", True, "ema")),
        )
        self.assertEqual(
            evaluation_artifact_run_modes(
                model_modes,
                group_name="short_f32",
                fixed_evaluation=True,
            ),
            (("_model", False, "model"),),
        )

    def test_single_process_failure_uses_distributed_error_contract(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "rank=0 phase='write manifest' ValueError: disk full",
        ):
            synchronize_evaluation_failure(
                ValueError("disk full"), phase="write manifest"
            )

    def test_six_single_weight_panels_have_stable_wandb_keys(self):
        keys = [
            fixed_evaluation_log_key(
                slot,
                weight_label="model",
                multiple_weight_modes=False,
            )
            for slot in range(6)
        ]
        self.assertEqual(
            keys,
            [f"validation/video_{slot:02d}" for slot in range(6)],
        )
        short_keys = [
            fixed_evaluation_log_key(
                slot,
                weight_label="model",
                multiple_weight_modes=False,
                group_name="short_f32",
            )
            for slot in range(6)
        ]
        self.assertEqual(
            short_keys,
            [f"validation/short_f32/video_{slot:02d}" for slot in range(6)],
        )


if __name__ == "__main__":
    unittest.main()
