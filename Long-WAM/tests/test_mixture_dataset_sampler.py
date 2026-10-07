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

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from longwam.datasets.mixture import DomainConcatDataset
from longwam.utils.samplers import ResumableEpochSampler


class ToyDataset:
    def __init__(self, name: str, ranges: tuple[tuple[int, int], ...]):
        self.name = name
        self.dataset_frame_ranges = ranges
        self.samples = [{"sample": f"{name}-{index}"} for index in range(ranges[-1][1])]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _mixed_dataset() -> DomainConcatDataset:
    return DomainConcatDataset(
        OrderedDict(
            robotwin=ToyDataset("robotwin", ((0, 3), (3, 5))),
            domino=ToyDataset("domino", ((0, 2), (2, 6))),
        )
    )


def _identified_mixed_dataset(
    *,
    robotwin_group_names=("pick", "place"),
    robotwin_signature="robotwin-manifest-v1",
) -> DomainConcatDataset:
    robotwin = ToyDataset("robotwin", ((0, 3), (3, 5)))
    robotwin.dataset_group_names = robotwin_group_names
    robotwin.sampling_signature = robotwin_signature
    domino = ToyDataset("domino", ((0, 2), (2, 6)))
    domino.dataset_group_names = ("dynamic-a", "dynamic-b")
    domino.sampling_signature = "domino-manifest-v1"
    return DomainConcatDataset(OrderedDict(robotwin=robotwin, domino=domino))


def _hierarchical_sampler(
    dataset=None,
    *,
    weights=None,
    strategies=None,
    samples_per_epoch=32,
    seed=19,
    batch_size=2,
    world_size=4,
):
    dataset = _mixed_dataset() if dataset is None else dataset
    return ResumableEpochSampler(
        dataset,
        seed=seed,
        batch_size=batch_size,
        num_processes=world_size,
        strategy="hierarchical",
        domain_weights=weights or {"robotwin": 0.5, "domino": 0.5},
        domain_strategies=strategies or {"robotwin": "frame_uniform", "domino": "group_uniform"},
        samples_per_epoch=samples_per_epoch,
    )


def _domain_for_index(dataset: DomainConcatDataset, index: int) -> str:
    return next(
        name for name, (start, stop) in dataset.domain_ranges.items() if start <= index < stop
    )


def test_domain_concat_forwards_samples_and_exposes_global_ranges():
    dataset = _mixed_dataset()

    assert dataset.domain_names == ("robotwin", "domino")
    assert dataset.domain_ranges == {
        "robotwin": (0, 5),
        "domino": (5, 11),
    }
    assert dataset.domain_group_ranges == {
        "robotwin": ((0, 3), (3, 5)),
        "domino": ((5, 7), (7, 11)),
    }
    assert dataset.domain_frame_ranges == ((0, 5), (5, 11))
    assert dataset.dataset_frame_ranges == ((0, 3), (3, 5), (5, 7), (7, 11))
    assert len(dataset) == 11
    assert dataset[0] == {"sample": "robotwin-0", "domain_index": 0}
    assert dataset[5] == {"sample": "domino-0", "domain_index": 1}
    assert dataset[-1] == {"sample": "domino-5", "domain_index": 1}
    assert "domain_index" not in dataset.domains["robotwin"].samples[0]

    with pytest.raises(IndexError):
        dataset[11]


def test_domain_concat_exposes_group_names_and_child_fingerprints(tmp_path):
    identified = _identified_mixed_dataset()
    assert identified.domain_group_names == {
        "robotwin": ("pick", "place"),
        "domino": ("dynamic-a", "dynamic-b"),
    }
    assert identified.domain_signature_sources == {
        "robotwin": "child.sampling_signature",
        "domino": "child.sampling_signature",
    }
    assert identified.domain_data_fingerprints == identified.domain_sampling_signatures

    fallback_child = ToyDataset("fallback", ((0, 2), (2, 4)))
    fallback_child.dataset_dirs = [tmp_path / "task-b", tmp_path / "task-a"]
    fallback = DomainConcatDataset(OrderedDict(fallback=fallback_child))
    expected_dirs = tuple(str(path.resolve()) for path in fallback_child.dataset_dirs)
    assert fallback.domain_group_names["fallback"] == expected_dirs
    assert fallback.domain_dataset_dirs["fallback"] == expected_dirs
    assert fallback.domain_signature_sources["fallback"] == "resolved_dataset_dirs"

    nested_child = ToyDataset("nested", ((0, 2), (2, 4)))
    nested_child.lerobot_dataset = SimpleNamespace(
        dataset_dirs=[tmp_path / "ignored-a", tmp_path / "ignored-b"],
        dataset_group_names=("manifest-a", "manifest-b"),
        sampling_signature="selected-episodes-v2",
    )
    nested = DomainConcatDataset(OrderedDict(nested=nested_child))
    assert nested.domain_group_names["nested"] == ("manifest-a", "manifest-b")
    assert nested.domain_signature_sources["nested"] == "child.sampling_signature"


def test_hierarchical_sampler_enforces_50_50_in_every_global_micro_batch():
    dataset = _mixed_dataset()
    sampler = _hierarchical_sampler(dataset)
    indices = list(sampler)

    assert len(indices) == len(sampler) == 32
    assert sampler.global_micro_batch_size == 8
    assert sampler.domain_quotas == {"robotwin": 4, "domino": 4}
    for batch_start in range(0, len(indices), sampler.global_micro_batch_size):
        domains = [
            _domain_for_index(dataset, index)
            for index in indices[batch_start : batch_start + sampler.global_micro_batch_size]
        ]
        assert domains.count("robotwin") == 4
        assert domains.count("domino") == 4


def test_hierarchical_domain_cycles_are_deterministic_and_with_replacement():
    dataset = _mixed_dataset()
    sampler = _hierarchical_sampler(
        dataset,
        strategies={"robotwin": "frame_uniform", "domino": "frame_uniform"},
    )
    indices = list(sampler)
    same = list(
        _hierarchical_sampler(
            dataset,
            strategies={"robotwin": "frame_uniform", "domino": "frame_uniform"},
        )
    )
    robotwin = [index for index in indices if index < 5]
    domino = [index for index in indices if index >= 5]

    assert indices == same
    assert len(robotwin) == len(domino) == 16
    assert len(set(robotwin[:5])) == 5
    assert len(set(robotwin[5:10])) == 5
    assert len(set(domino[:6])) == 6
    assert len(set(domino[6:12])) == 6

    sampler.set_epoch(1)
    assert list(sampler) != indices


def test_group_uniform_balances_child_groups_and_cycles_samples():
    dataset = _mixed_dataset()
    sampler = _hierarchical_sampler(
        dataset,
        strategies={"robotwin": "group_uniform", "domino": "group_uniform"},
    )
    indices = list(sampler)

    robotwin = [index for index in indices if index < 5]
    domino = [index for index in indices if index >= 5]
    assert sum(index < 3 for index in robotwin) == 8
    assert sum(index >= 3 for index in robotwin) == 8
    assert sum(index < 7 for index in domino) == 8
    assert sum(index >= 7 for index in domino) == 8
    assert set(robotwin) == set(range(5))
    assert set(domino) == set(range(5, 11))


def test_hierarchical_resume_is_exact_global_batch_suffix():
    full = _hierarchical_sampler(seed=71)
    full.set_epoch_offset(3)
    full_indices = list(full)

    resumed = _hierarchical_sampler(seed=71)
    resumed.set_epoch_offset(3)
    resumed.set_resume_batch_offset(2)

    assert list(resumed) == full_indices[16:]


def test_hierarchical_signature_covers_the_full_sampling_contract():
    baseline = _hierarchical_sampler()
    same = _hierarchical_sampler()
    variants = [
        _hierarchical_sampler(weights={"robotwin": 0.25, "domino": 0.75}),
        _hierarchical_sampler(strategies={"robotwin": "group_uniform", "domino": "group_uniform"}),
        _hierarchical_sampler(samples_per_epoch=40),
        _hierarchical_sampler(batch_size=4, world_size=2),
        _hierarchical_sampler(batch_size=2, world_size=8, samples_per_epoch=32),
        _hierarchical_sampler(seed=20),
        _hierarchical_sampler(
            DomainConcatDataset(
                OrderedDict(
                    domino=ToyDataset("domino", ((0, 2), (2, 6))),
                    robotwin=ToyDataset("robotwin", ((0, 3), (3, 5))),
                )
            ),
            weights={"domino": 0.5, "robotwin": 0.5},
            strategies={"domino": "group_uniform", "robotwin": "frame_uniform"},
        ),
    ]

    assert baseline.group_signature == same.group_signature
    assert all(variant.group_signature != baseline.group_signature for variant in variants)


def test_hierarchical_signature_rejects_same_ranges_with_changed_data_identity():
    baseline = _hierarchical_sampler(_identified_mixed_dataset())
    reordered_tasks = _hierarchical_sampler(
        _identified_mixed_dataset(robotwin_group_names=("place", "pick"))
    )
    changed_manifest = _hierarchical_sampler(
        _identified_mixed_dataset(robotwin_signature="robotwin-manifest-v2")
    )

    assert reordered_tasks.dataset_frame_ranges == baseline.dataset_frame_ranges
    assert changed_manifest.dataset_frame_ranges == baseline.dataset_frame_ranges
    assert reordered_tasks.group_signature != baseline.group_signature
    assert changed_manifest.group_signature != baseline.group_signature


def test_hierarchical_default_epoch_length_is_full_global_batches():
    sampler = _hierarchical_sampler(samples_per_epoch=None)

    assert len(sampler) == 16
    assert len(list(sampler)) == 16


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weights": {"robotwin": 1.0}}, "exactly"),
        (
            {"weights": {"robotwin": 0.01, "domino": 0.99}},
            "at least one sample",
        ),
        (
            {"strategies": {"robotwin": "frame_uniform", "domino": "bad"}},
            "Unsupported domain strategies",
        ),
        ({"samples_per_epoch": 30}, "divisible"),
    ],
)
def test_hierarchical_sampler_rejects_ambiguous_contracts(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _hierarchical_sampler(**kwargs)
