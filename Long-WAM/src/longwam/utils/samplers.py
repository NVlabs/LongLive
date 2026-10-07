# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/utils/samplers.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    SUPPORTED_STRATEGIES = {"frame_uniform", "dataset_uniform", "hierarchical"}
    SUPPORTED_DOMAIN_STRATEGIES = {"frame_uniform", "group_uniform"}

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        strategy: str = "frame_uniform",
        domain_weights: Mapping[str, float] | None = None,
        domain_strategies: Mapping[str, str] | None = None,
        samples_per_epoch: int | None = None,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("`batch_size` and `num_processes` must be positive.")
        self.global_micro_batch_size = self.batch_size * self.num_processes
        self.strategy = str(strategy).strip().lower()
        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported sampler strategy: {strategy!r}. "
                f"Expected one of: {sorted(self.SUPPORTED_STRATEGIES)}."
            )

        self.dataset_frame_ranges = self._get_dataset_frame_ranges(dataset)
        self.group_signature = self._make_group_signature(self.dataset_frame_ranges)
        self.samples_per_epoch = len(dataset)
        self.domain_names: tuple[str, ...] = ()
        self.domain_ranges: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self.domain_group_ranges: OrderedDict[str, tuple[tuple[int, int], ...]] = OrderedDict()
        self.domain_group_names: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self.domain_sampling_signatures: OrderedDict[str, str] = OrderedDict()
        self.domain_weights: OrderedDict[str, float] = OrderedDict()
        self.domain_strategies: OrderedDict[str, str] = OrderedDict()
        self.domain_quotas: OrderedDict[str, int] | None = OrderedDict()
        self.domain_epoch_quotas: OrderedDict[str, int] = OrderedDict()
        self.domain_quota_ranges: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._domain_quota_schedule: tuple[tuple[int, ...], ...] = ()
        if self.strategy == "hierarchical":
            self._configure_hierarchical(
                domain_weights=domain_weights,
                domain_strategies=domain_strategies,
                samples_per_epoch=samples_per_epoch,
            )
        elif (
            domain_weights is not None
            or domain_strategies is not None
            or samples_per_epoch is not None
        ):
            raise ValueError(
                "`domain_weights`, `domain_strategies`, and `samples_per_epoch` "
                "are only valid with strategy='hierarchical'."
            )
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    @staticmethod
    def _get_dataset_frame_ranges(dataset: Sized) -> tuple[tuple[int, int], ...]:
        dataset_length = len(dataset)
        raw_ranges = getattr(dataset, "dataset_frame_ranges", ((0, dataset_length),))
        ranges = tuple((int(start), int(stop)) for start, stop in raw_ranges)

        if not ranges:
            raise ValueError("`dataset_frame_ranges` must contain at least one range.")
        if dataset_length == 0 and ranges == ((0, 0),):
            return ranges

        expected_start = 0
        for start, stop in ranges:
            if start != expected_start or stop <= start:
                raise ValueError(
                    "`dataset_frame_ranges` must be non-empty, contiguous [start, stop) ranges "
                    f"covering the dataset; got {ranges}."
                )
            expected_start = stop
        if expected_start != dataset_length:
            raise ValueError(
                "`dataset_frame_ranges` must cover the full dataset; "
                f"last stop is {expected_start}, dataset length is {dataset_length}."
            )
        return ranges

    @staticmethod
    def _make_group_signature(ranges: tuple[tuple[int, int], ...]) -> str:
        serialized = json.dumps(ranges, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _configure_hierarchical(
        self,
        *,
        domain_weights: Mapping[str, float] | None,
        domain_strategies: Mapping[str, str] | None,
        samples_per_epoch: int | None,
    ) -> None:
        raw_names = getattr(self.dataset, "domain_names", None)
        raw_ranges = getattr(self.dataset, "domain_ranges", None)
        raw_group_ranges = getattr(self.dataset, "domain_group_ranges", None)
        raw_group_names = getattr(self.dataset, "domain_group_names", None)
        raw_sampling_signatures = getattr(self.dataset, "domain_sampling_signatures", None)
        if (
            raw_names is None
            or not isinstance(raw_ranges, Mapping)
            or not isinstance(raw_group_ranges, Mapping)
            or not isinstance(raw_group_names, Mapping)
            or not isinstance(raw_sampling_signatures, Mapping)
        ):
            raise ValueError(
                "strategy='hierarchical' requires a dataset exposing `domain_names`, "
                "`domain_ranges`, `domain_group_ranges`, `domain_group_names`, and "
                "`domain_sampling_signatures` (for example, DomainConcatDataset)."
            )

        self.domain_names = tuple(str(name) for name in raw_names)
        if not self.domain_names or len(set(self.domain_names)) != len(self.domain_names):
            raise ValueError("`domain_names` must be non-empty and unique.")
        named_mappings = (
            raw_ranges,
            raw_group_ranges,
            raw_group_names,
            raw_sampling_signatures,
        )
        if any(tuple(mapping) != self.domain_names for mapping in named_mappings):
            raise ValueError(
                "Domain metadata mappings must use exactly the order declared by `domain_names`."
            )

        self.domain_ranges = OrderedDict(
            (name, tuple(int(value) for value in raw_ranges[name])) for name in self.domain_names
        )
        self.domain_group_ranges = OrderedDict(
            (
                name,
                tuple((int(start), int(stop)) for start, stop in raw_group_ranges[name]),
            )
            for name in self.domain_names
        )
        self.domain_group_names = OrderedDict(
            (name, tuple(str(group_name) for group_name in raw_group_names[name]))
            for name in self.domain_names
        )
        self.domain_sampling_signatures = OrderedDict(
            (name, str(raw_sampling_signatures[name])) for name in self.domain_names
        )
        self._validate_domain_ranges()

        if domain_weights is None:
            domain_weights = {name: 1.0 for name in self.domain_names}
        self._validate_named_mapping(domain_weights, "domain_weights")
        weights = [float(domain_weights[name]) for name in self.domain_names]
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("Every domain weight must be finite and positive.")
        weight_sum = sum(weights)
        normalized_weights = [weight / weight_sum for weight in weights]
        self.domain_weights = OrderedDict(zip(self.domain_names, normalized_weights))

        if domain_strategies is None:
            domain_strategies = {name: "frame_uniform" for name in self.domain_names}
        self._validate_named_mapping(domain_strategies, "domain_strategies")
        strategies = [str(domain_strategies[name]).strip().lower() for name in self.domain_names]
        invalid_strategies = sorted(set(strategies) - self.SUPPORTED_DOMAIN_STRATEGIES)
        if invalid_strategies:
            raise ValueError(
                f"Unsupported domain strategies: {invalid_strategies}. Expected only "
                f"{sorted(self.SUPPORTED_DOMAIN_STRATEGIES)}."
            )
        self.domain_strategies = OrderedDict(zip(self.domain_names, strategies))

        if samples_per_epoch is None:
            samples_per_epoch = (
                math.ceil(len(self.dataset) / self.global_micro_batch_size)
                * self.global_micro_batch_size
            )
        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError("`samples_per_epoch` must be positive.")
        if self.samples_per_epoch % self.global_micro_batch_size != 0:
            raise ValueError(
                "`samples_per_epoch` must be divisible by batch_size * num_processes "
                "so every global micro-batch has the configured domain quota."
            )

        exact_quotas = [weight * self.global_micro_batch_size for weight in normalized_weights]
        quotas = [round(quota) for quota in exact_quotas]
        fixed_quotas = (
            all(
                math.isclose(exact, quota, rel_tol=0.0, abs_tol=1e-9)
                for exact, quota in zip(exact_quotas, quotas)
            )
            and sum(quotas) == self.global_micro_batch_size
        )
        num_global_micro_batches = self.samples_per_epoch // self.global_micro_batch_size
        if fixed_quotas:
            if any(quota <= 0 for quota in quotas):
                raise ValueError(
                    "Every domain must receive at least one sample per global micro-batch."
                )
            self.domain_quotas = OrderedDict(zip(self.domain_names, quotas))
            self.domain_epoch_quotas = OrderedDict(
                (name, self.domain_quotas[name] * num_global_micro_batches)
                for name in self.domain_names
            )
            self.domain_quota_ranges = OrderedDict(
                (name, (quota, quota)) for name, quota in self.domain_quotas.items()
            )
        else:
            self.domain_quotas = None
            self._configure_balanced_domain_quotas(normalized_weights, num_global_micro_batches)

        contract = {
            "strategy": self.strategy,
            "domain_names": self.domain_names,
            "domain_ranges": self.domain_ranges,
            "domain_group_ranges": self.domain_group_ranges,
            "domain_group_names": self.domain_group_names,
            "domain_sampling_signatures": self.domain_sampling_signatures,
            "domain_weights": {name: self.domain_weights[name].hex() for name in self.domain_names},
            "domain_strategies": self.domain_strategies,
            "samples_per_epoch": self.samples_per_epoch,
            "batch_size": self.batch_size,
            "num_processes": self.num_processes,
            "seed": self.seed,
        }
        if self._domain_quota_schedule:
            # Fixed-quota configurations keep their historical signature so an
            # existing exact-ratio run remains resumable.
            contract["quota_policy"] = "residual_balanced_v1"
        serialized = json.dumps(contract, separators=(",", ":"))
        self.group_signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _configure_balanced_domain_quotas(
        self,
        normalized_weights: list[float],
        num_global_micro_batches: int,
    ) -> None:
        residuals = [0.0] * len(self.domain_names)
        totals = [0] * len(self.domain_names)
        quota_min = [self.global_micro_batch_size] * len(self.domain_names)
        quota_max = [0] * len(self.domain_names)
        schedule: list[tuple[int, ...]] = []

        for _ in range(num_global_micro_batches):
            targets = [
                weight * self.global_micro_batch_size + residual
                for weight, residual in zip(normalized_weights, residuals)
            ]
            quotas = [math.floor(target) for target in targets]
            remainder = self.global_micro_batch_size - sum(quotas)
            if remainder < 0 or remainder > len(quotas):
                raise RuntimeError("Residual-balanced domain quota apportionment became unstable.")
            priority = sorted(
                range(len(quotas)),
                key=lambda index: (-(targets[index] - quotas[index]), index),
            )
            for index in priority[:remainder]:
                quotas[index] += 1
            if any(quota <= 0 for quota in quotas):
                raise ValueError(
                    "Every domain must receive at least one sample per global "
                    "micro-batch; increase its weight or the global micro-batch size."
                )

            residuals = [target - quota for target, quota in zip(targets, quotas)]
            schedule.append(tuple(quotas))
            for index, quota in enumerate(quotas):
                totals[index] += quota
                quota_min[index] = min(quota_min[index], quota)
                quota_max[index] = max(quota_max[index], quota)

        self._domain_quota_schedule = tuple(schedule)
        self.domain_epoch_quotas = OrderedDict(zip(self.domain_names, totals))
        self.domain_quota_ranges = OrderedDict(
            (
                name,
                (quota_min[index], quota_max[index]),
            )
            for index, name in enumerate(self.domain_names)
        )

    def _validate_named_mapping(self, mapping: Mapping, field_name: str) -> None:
        if not isinstance(mapping, Mapping):
            raise TypeError(f"`{field_name}` must be a mapping keyed by domain name.")
        missing = [name for name in self.domain_names if name not in mapping]
        extra = [name for name in mapping if name not in self.domain_names]
        if missing or extra:
            raise ValueError(
                f"`{field_name}` must contain exactly {self.domain_names}; "
                f"missing={missing}, extra={extra}."
            )

    def _validate_domain_ranges(self) -> None:
        expected_start = 0
        for name in self.domain_names:
            start, stop = self.domain_ranges[name]
            if start != expected_start or stop <= start:
                raise ValueError(
                    "`domain_ranges` must be non-empty, contiguous [start, stop) "
                    f"ranges; got {self.domain_ranges}."
                )

            expected_group_start = start
            groups = self.domain_group_ranges[name]
            if not groups:
                raise ValueError(f"Domain {name!r} must expose at least one group range.")
            for group_start, group_stop in groups:
                if group_start != expected_group_start or group_stop <= group_start:
                    raise ValueError(f"Domain {name!r} has invalid group ranges: {groups}.")
                expected_group_start = group_stop
            if expected_group_start != stop:
                raise ValueError(
                    f"Group ranges for domain {name!r} must cover its full domain range."
                )
            if len(self.domain_group_names[name]) != len(groups):
                raise ValueError(f"Domain {name!r} must expose one group name per group range.")
            expected_start = stop
        if expected_start != len(self.dataset):
            raise ValueError("`domain_ranges` must cover the full dataset.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        dataset_set_epoch = getattr(self.dataset, "set_epoch", None)
        if dataset_set_epoch is not None:
            dataset_set_epoch(self.epoch + self.epoch_offset)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)
        dataset_set_epoch = getattr(self.dataset, "set_epoch", None)
        if dataset_set_epoch is not None:
            dataset_set_epoch(self.epoch + self.epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _frame_uniform_indices(self, generator: torch.Generator) -> list[int]:
        # Keep this expression identical to the historical sampler so the default
        # strategy preserves every epoch's index sequence bit-for-bit.
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def _dataset_uniform_indices(self, generator: torch.Generator) -> list[int]:
        dataset_length = len(self.dataset)
        num_groups = len(self.dataset_frame_ranges)
        quota, remainder = divmod(dataset_length, num_groups)
        indices: list[int] = []

        for group_index, (start, stop) in enumerate(self.dataset_frame_ranges):
            group_quota = quota + int(group_index < remainder)
            group_size = stop - start
            while group_quota > 0:
                cycle = torch.randperm(group_size, generator=generator)
                take = min(group_quota, group_size)
                indices.extend((cycle[:take] + start).tolist())
                group_quota -= take

        shuffle = torch.randperm(dataset_length, generator=generator).tolist()
        return [indices[index] for index in shuffle]

    @staticmethod
    def _range_cycle(
        start: int,
        stop: int,
        generator: torch.Generator,
    ) -> Iterator[int]:
        size = stop - start
        while True:
            permutation = torch.randperm(size, generator=generator).tolist()
            yield from (start + index for index in permutation)

    @classmethod
    def _group_uniform_cycle(
        cls,
        ranges: tuple[tuple[int, int], ...],
        generator: torch.Generator,
    ) -> Iterator[int]:
        group_cycles = [cls._range_cycle(start, stop, generator) for start, stop in ranges]
        while True:
            group_order = torch.randperm(len(group_cycles), generator=generator).tolist()
            for group_index in group_order:
                yield next(group_cycles[group_index])

    def _hierarchical_indices(self, generator: torch.Generator) -> list[int]:
        domain_cycles: list[Iterator[int]] = []
        for name in self.domain_names:
            if self.domain_strategies[name] == "frame_uniform":
                start, stop = self.domain_ranges[name]
                domain_cycles.append(self._range_cycle(start, stop, generator))
            else:
                domain_cycles.append(
                    self._group_uniform_cycle(self.domain_group_ranges[name], generator)
                )

        num_global_micro_batches = self.samples_per_epoch // self.global_micro_batch_size
        if self.domain_quotas is not None:
            domain_slots = [
                domain_index
                for domain_index, name in enumerate(self.domain_names)
                for _ in range(self.domain_quotas[name])
            ]
            quota_schedule = None
        else:
            quota_schedule = self._domain_quota_schedule

        indices: list[int] = []
        for batch_index in range(num_global_micro_batches):
            if quota_schedule is not None:
                domain_slots = [
                    domain_index
                    for domain_index, quota in enumerate(quota_schedule[batch_index])
                    for _ in range(quota)
                ]
            slot_order = torch.randperm(self.global_micro_batch_size, generator=generator).tolist()
            for slot_index in slot_order:
                domain_index = domain_slots[slot_index]
                indices.append(next(domain_cycles[domain_index]))
        return indices

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        if self.strategy == "frame_uniform":
            indices = self._frame_uniform_indices(g)
        elif self.strategy == "dataset_uniform":
            indices = self._dataset_uniform_indices(g)
        else:
            indices = self._hierarchical_indices(g)
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return self.samples_per_epoch
