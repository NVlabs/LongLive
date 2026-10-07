# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/datasets/mixture.py
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
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from operator import index as to_index
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def child_sampling_signature(source: str, value: Any) -> str:
    """Return the canonical digest used to bind a child dataset to a domain."""
    serialized = json.dumps(
        {"source": source, "value": value},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DomainConcatDataset(Dataset):
    """Concatenate named domains while preserving their sampling boundaries."""

    def __init__(self, domains: Mapping[str, Dataset]):
        if not isinstance(domains, Mapping) or not domains:
            raise ValueError("`domains` must be a non-empty ordered mapping.")

        self.domains = OrderedDict(domains.items())
        self.domain_names = tuple(self.domains)
        if any(not isinstance(name, str) or not name for name in self.domain_names):
            raise ValueError("Every domain name must be a non-empty string.")

        self.domain_ranges: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self.domain_group_ranges: OrderedDict[str, tuple[tuple[int, int], ...]] = OrderedDict()
        self.domain_group_names: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self.domain_dataset_dirs: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self.domain_sampling_signatures: OrderedDict[str, str] = OrderedDict()
        self.domain_signature_sources: OrderedDict[str, str] = OrderedDict()
        offset = 0
        for name, dataset in self.domains.items():
            domain_length = len(dataset)
            if domain_length <= 0:
                raise ValueError(f"Domain {name!r} must contain at least one sample.")

            start, stop = offset, offset + domain_length
            self.domain_ranges[name] = (start, stop)
            local_ranges = self._validated_local_group_ranges(dataset, name)
            self.domain_group_ranges[name] = tuple(
                (start + group_start, start + group_stop)
                for group_start, group_stop in local_ranges
            )
            group_names, dataset_dirs, signature, signature_source = self._child_sampling_identity(
                dataset, name, len(local_ranges)
            )
            self.domain_group_names[name] = group_names
            self.domain_dataset_dirs[name] = dataset_dirs
            self.domain_sampling_signatures[name] = signature
            self.domain_signature_sources[name] = signature_source
            offset = stop

        self._length = offset
        self._domain_stops = tuple(stop for _, stop in self.domain_ranges.values())
        self.domain_frame_ranges = tuple(self.domain_ranges.values())
        self.dataset_frame_ranges = tuple(
            group_range
            for name in self.domain_names
            for group_range in self.domain_group_ranges[name]
        )

        # A descriptive alias for callers that log data provenance rather than
        # sampler implementation details.
        self.domain_data_fingerprints = self.domain_sampling_signatures

    @staticmethod
    def _validated_local_group_ranges(
        dataset: Dataset,
        domain_name: str,
    ) -> tuple[tuple[int, int], ...]:
        dataset_length = len(dataset)
        raw_ranges = getattr(dataset, "dataset_frame_ranges", ((0, dataset_length),))
        ranges = tuple((int(start), int(stop)) for start, stop in raw_ranges)
        if not ranges:
            raise ValueError(f"Domain {domain_name!r} has an empty `dataset_frame_ranges`.")

        expected_start = 0
        for start, stop in ranges:
            if start != expected_start or stop <= start:
                raise ValueError(
                    f"Domain {domain_name!r} has invalid `dataset_frame_ranges`: {ranges}. "
                    "Ranges must be non-empty, contiguous, and half-open."
                )
            expected_start = stop
        if expected_start != dataset_length:
            raise ValueError(
                f"Domain {domain_name!r} has `dataset_frame_ranges` ending at "
                f"{expected_start}, but its length is {dataset_length}."
            )
        return ranges

    @classmethod
    def _child_sampling_identity(
        cls,
        dataset: Dataset,
        domain_name: str,
        num_groups: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
        lerobot_dataset = getattr(dataset, "lerobot_dataset", None)
        raw_dirs = getattr(dataset, "dataset_dirs", None)
        if raw_dirs is None:
            raw_dirs = getattr(lerobot_dataset, "dataset_dirs", None)
        if isinstance(raw_dirs, (str, Path)):
            raw_dirs = (raw_dirs,)
        if raw_dirs is None:
            raw_dirs = ()
        dataset_dirs = tuple(
            str(Path(path).expanduser().resolve(strict=False)) for path in raw_dirs
        )

        raw_group_names = getattr(dataset, "dataset_group_names", None)
        if raw_group_names is None:
            raw_group_names = getattr(lerobot_dataset, "dataset_group_names", None)
        if raw_group_names is not None:
            if isinstance(raw_group_names, str):
                raw_group_names = (raw_group_names,)
            group_names = tuple(str(group_name) for group_name in raw_group_names)
        elif dataset_dirs:
            group_names = dataset_dirs
        else:
            group_names = tuple(
                f"{type(dataset).__module__}.{type(dataset).__qualname__}:{group_index}"
                for group_index in range(num_groups)
            )
        if len(group_names) != num_groups:
            raise ValueError(
                f"Domain {domain_name!r} exposes {len(group_names)} group names for "
                f"{num_groups} group ranges."
            )

        child_signature = getattr(dataset, "sampling_signature", None)
        if child_signature is None:
            child_signature = getattr(lerobot_dataset, "sampling_signature", None)
        if child_signature is not None:
            signature_source = "child.sampling_signature"
            signature_payload = child_signature
        elif dataset_dirs:
            signature_source = "resolved_dataset_dirs"
            signature_payload = dataset_dirs
        else:
            signature_source = "dataset_group_names"
            signature_payload = group_names
        signature = child_sampling_signature(signature_source, signature_payload)
        return group_names, dataset_dirs, signature, signature_source

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = to_index(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"DomainConcatDataset index out of range: {index}")

        domain_index = bisect_right(self._domain_stops, index)
        domain_name = self.domain_names[domain_index]
        domain_start = self.domain_ranges[domain_name][0]
        sample = self.domains[domain_name][index - domain_start]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"Domain {domain_name!r} returned {type(sample).__name__}; "
                "expected a mapping so `domain_index` can be attached."
            )

        result = dict(sample)
        result["domain_index"] = domain_index
        return result
