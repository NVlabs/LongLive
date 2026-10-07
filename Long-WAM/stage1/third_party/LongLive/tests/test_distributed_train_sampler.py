# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot integration imported from the author video-training branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/tests/test_distributed_train_sampler.py
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

import unittest

from torch.utils.data import DataLoader, Dataset

from utils.dataset import (
    build_distributed_sampler,
    cycle,
    resolve_resume_data_cursor,
)


class _RangeDataset(Dataset):
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class _CountingDataset(_RangeDataset):
    def __init__(self, size):
        super().__init__(size)
        self.seen = []

    def __getitem__(self, index):
        self.seen.append(index)
        return index


class _RecordingSampler:
    def __init__(self):
        self.epoch = None
        self.epochs = []

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.epochs.append(self.epoch)


class _RecordingLoader:
    def __init__(self):
        self.sampler = _RecordingSampler()

    def __iter__(self):
        yield self.sampler.epoch, 0
        yield self.sampler.epoch, 1


class DistributedTrainSamplerTest(unittest.TestCase):
    @staticmethod
    def _sp_samplers(dataset, *, world_size=8, sp_size=4, **kwargs):
        dp_size = world_size // sp_size
        return [
            build_distributed_sampler(
                dataset,
                rank=global_rank // sp_size,
                num_replicas=dp_size,
                seed=1234,
                **kwargs,
            )
            for global_rank in range(world_size)
        ]

    def test_train_sampler_is_identical_within_sp_and_disjoint_across_dp(self):
        dataset = _RangeDataset(17)
        samplers = self._sp_samplers(
            dataset, shuffle=True, drop_last=True
        )
        shards = [list(sampler) for sampler in samplers]

        self.assertEqual(shards[0], shards[1])
        self.assertEqual(shards[0], shards[2])
        self.assertEqual(shards[0], shards[3])
        self.assertEqual(shards[4], shards[5])
        self.assertEqual(shards[4], shards[6])
        self.assertEqual(shards[4], shards[7])
        self.assertTrue(set(shards[0]).isdisjoint(shards[4]))
        self.assertEqual(len(shards[0]), 8)
        self.assertEqual(len(shards[4]), 8)
        self.assertEqual(len(set(shards[0] + shards[4])), 16)

    def test_eval_sampler_is_identical_within_sp_and_covers_dataset(self):
        dataset = _RangeDataset(9)
        samplers = self._sp_samplers(
            dataset, shuffle=False, drop_last=False
        )
        shards = [list(sampler) for sampler in samplers]

        self.assertEqual(shards[0], shards[1])
        self.assertEqual(shards[0], shards[2])
        self.assertEqual(shards[0], shards[3])
        self.assertEqual(shards[4], shards[5])
        self.assertEqual(shards[4], shards[6])
        self.assertEqual(shards[4], shards[7])
        self.assertEqual(set(shards[0] + shards[4]), set(range(9)))

    def test_epoch_changes_shared_train_order_deterministically(self):
        dataset = _RangeDataset(32)
        samplers_a = self._sp_samplers(
            dataset, shuffle=True, drop_last=True
        )
        samplers_b = self._sp_samplers(
            dataset, shuffle=True, drop_last=True
        )
        epoch_zero = list(samplers_a[0])
        for sampler in samplers_a + samplers_b:
            sampler.set_epoch(1)

        self.assertNotEqual(epoch_zero, list(samplers_a[0]))
        self.assertEqual(list(samplers_a[0]), list(samplers_a[3]))
        self.assertEqual(list(samplers_a[0]), list(samplers_b[0]))

    def test_cycle_calls_set_epoch_once_per_completed_pass(self):
        loader = _RecordingLoader()
        iterator = cycle(loader, start_epoch=3)
        batches = [next(iterator) for _ in range(5)]

        self.assertEqual(
            batches,
            [(3, 0), (3, 1), (4, 0), (4, 1), (5, 0)],
        )
        self.assertEqual(loader.sampler.epochs, [3, 4, 5])

    def test_resume_cursor_cold_mid_epoch_and_epoch_boundary(self):
        self.assertEqual(
            resolve_resume_data_cursor(
                optimizer_step=0,
                gradient_accumulation_steps=1,
                batches_per_epoch=1269,
            ),
            (0, 0),
        )
        self.assertEqual(
            resolve_resume_data_cursor(
                optimizer_step=300,
                gradient_accumulation_steps=1,
                batches_per_epoch=1269,
            ),
            (0, 300),
        )
        self.assertEqual(
            resolve_resume_data_cursor(
                optimizer_step=1269,
                gradient_accumulation_steps=1,
                batches_per_epoch=1269,
            ),
            (1, 0),
        )
        self.assertEqual(
            resolve_resume_data_cursor(
                optimizer_step=635,
                gradient_accumulation_steps=2,
                batches_per_epoch=1269,
            ),
            (1, 1),
        )

    def test_mid_epoch_offset_skips_indices_without_decoding_them(self):
        dataset = _CountingDataset(17)
        baseline = build_distributed_sampler(
            dataset,
            rank=0,
            num_replicas=2,
            seed=1234,
            shuffle=True,
            drop_last=True,
        )
        baseline_indices = list(baseline)

        resumed = build_distributed_sampler(
            dataset,
            rank=0,
            num_replicas=2,
            seed=1234,
            shuffle=True,
            drop_last=True,
        )
        resumed.set_start_index(3)
        loader = DataLoader(dataset, batch_size=1, sampler=resumed, num_workers=0)
        iterator = cycle(loader, start_epoch=0)

        first = int(next(iterator).item())
        self.assertEqual(first, baseline_indices[3])
        self.assertEqual(dataset.seen, [baseline_indices[3]])

        # Exhaust the resumed suffix; the next item starts a complete epoch 1.
        for _ in range(len(baseline_indices) - 4):
            next(iterator)
        epoch_one = build_distributed_sampler(
            dataset,
            rank=0,
            num_replicas=2,
            seed=1234,
            shuffle=True,
            drop_last=True,
        )
        epoch_one.set_epoch(1)
        self.assertEqual(int(next(iterator).item()), list(epoch_one)[0])

    def test_resume_offset_remains_sp_consistent_and_dp_disjoint(self):
        dataset = _RangeDataset(32)
        samplers = self._sp_samplers(
            dataset, shuffle=True, drop_last=True
        )
        for sampler in samplers:
            sampler.set_epoch(2)
            sampler.set_start_index(5)
        shards = [list(sampler) for sampler in samplers]

        self.assertEqual(shards[0], shards[3])
        self.assertEqual(shards[4], shards[7])
        self.assertTrue(set(shards[0]).isdisjoint(shards[4]))

    def test_cycle_keeps_plain_iterable_compatibility(self):
        iterator = cycle(["a", "b"])
        self.assertEqual(
            [next(iterator) for _ in range(5)],
            ["a", "b", "a", "b", "a"],
        )

    def test_invalid_coordinates_and_epoch_fail_early(self):
        dataset = _RangeDataset(8)
        with self.assertRaisesRegex(ValueError, "num_replicas"):
            build_distributed_sampler(
                dataset,
                rank=0,
                num_replicas=0,
                seed=1,
                shuffle=True,
                drop_last=True,
            )
        with self.assertRaisesRegex(ValueError, "rank"):
            build_distributed_sampler(
                dataset,
                rank=2,
                num_replicas=2,
                seed=1,
                shuffle=True,
                drop_last=True,
            )
        iterator = cycle([1], start_epoch=-1)
        with self.assertRaisesRegex(ValueError, "start_epoch"):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
