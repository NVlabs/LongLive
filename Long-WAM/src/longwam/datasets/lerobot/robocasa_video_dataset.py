# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/datasets/lerobot/robocasa_video_dataset.py
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

"""RoboCasa365-specific validation around the shared LeRobot video dataset."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from operator import index as to_index
from pathlib import Path

from .episode_selection import resolve_per_dataset_episode_selection
from .robocasa_contract import (
    ROBOCASA_FPS,
    ROBOCASA_HUMAN300_GROUP_COUNTS,
    validate_human300_dataset_inventory,
    validate_robocasa_lerobot_contract,
)
from .robot_video_dataset import RobotVideoDataset


class RoboCasaHuman300VideoDataset(RobotVideoDataset):
    """Human300 dataset with fail-fast count, schema, and frequency checks."""

    def __init__(
        self,
        dataset_dirs,
        *,
        expected_dataset_count: int = 300,
        expected_fps: int = ROBOCASA_FPS,
        validate_dataset_contract: bool = True,
        human300_episode_selection=None,
        expose_task_family_domains: bool = False,
        **kwargs,
    ):
        dataset_dirs = tuple(str(path) for path in dataset_dirs)
        expected_group_total = sum(ROBOCASA_HUMAN300_GROUP_COUNTS.values())
        if expected_dataset_count != expected_group_total:
            raise ValueError(
                "RoboCasa Human300 expected_dataset_count must remain "
                f"{expected_group_total}, got {expected_dataset_count}."
            )
        if len(dataset_dirs) != expected_dataset_count:
            raise ValueError(
                f"RoboCasa Human300 requires {expected_dataset_count} dataset "
                f"paths, got {len(dataset_dirs)}."
            )
        if validate_dataset_contract:
            validated_dataset_dirs = validate_human300_dataset_inventory(dataset_dirs)
            validate_robocasa_lerobot_contract(validated_dataset_dirs, expected_fps=expected_fps)

        resolved_episode_selection = None
        if human300_episode_selection is not None:
            if kwargs.get("episode_selection") is not None:
                raise ValueError(
                    "Human300 official selection cannot be combined with a "
                    "manifest episode selection."
                )
            selection = dict(human300_episode_selection)
            required_fields = {
                "strategy",
                "demos_per_task",
                "seed",
                "expected_task_count",
                "expected_episode_count",
            }
            missing = sorted(required_fields.difference(selection))
            unknown = sorted(set(selection).difference(required_fields))
            if missing or unknown:
                raise ValueError(
                    "Invalid Human300 episode selection config: "
                    f"missing={missing}, unknown={unknown}."
                )
            if selection["strategy"] != "official_random_subset":
                raise ValueError(
                    "Human300 episode selection strategy must be 'official_random_subset'."
                )

            expected_task_count = int(selection["expected_task_count"])
            demos_per_task = int(selection["demos_per_task"])
            selection_seed = int(selection["seed"])
            expected_episode_count = int(selection["expected_episode_count"])
            if expected_task_count != expected_dataset_count:
                raise ValueError(
                    "Human300 episode selection task count changed: "
                    f"expected={expected_dataset_count}, "
                    f"configured={expected_task_count}."
                )
            if (demos_per_task, selection_seed) != (100, 0):
                raise ValueError(
                    "Official Human300 training requires exactly 100 demos per "
                    "task selected independently with seed 0."
                )
            if expected_episode_count != expected_task_count * demos_per_task:
                raise ValueError(
                    "Human300 selected episode count must equal "
                    "expected_task_count * demos_per_task."
                )
            resolved_episode_selection = resolve_per_dataset_episode_selection(
                dataset_dirs,
                episodes_per_dataset=demos_per_task,
                seed=selection_seed,
                expected_dataset_count=expected_task_count,
                expected_total_episodes=expected_episode_count,
            )
            kwargs["per_dataset_episode_selection"] = resolved_episode_selection

        self.expected_dataset_count = expected_dataset_count
        self.expected_fps = expected_fps
        self.human300_episode_selection = resolved_episode_selection
        # Keep the configured spelling (rather than resolving mount aliases)
        # because Hugging Face includes every parquet
        # path string in the Arrow-cache fingerprint. Validation may resolve
        # paths, but silently replacing them here invalidates the CPU prewarm.
        super().__init__(dataset_dirs=dataset_dirs, **kwargs)
        self.expose_task_family_domains = bool(expose_task_family_domains)
        if self.expose_task_family_domains:
            self._configure_task_family_domains(dataset_dirs)

    def _configure_task_family_domains(self, dataset_dirs: tuple[str, ...]) -> None:
        ranges = self.dataset_frame_ranges
        if len(ranges) != len(dataset_dirs):
            raise ValueError(
                "RoboCasa task-family metadata requires one frame range per "
                f"dataset path, got ranges={len(ranges)}, paths={len(dataset_dirs)}."
            )

        family_counts = OrderedDict(ROBOCASA_HUMAN300_GROUP_COUNTS)
        expected_families = tuple(
            family for family, count in family_counts.items() for _ in range(count)
        )
        actual_families = tuple(Path(path).parents[2].name for path in dataset_dirs)
        if actual_families != expected_families:
            raise ValueError(
                "RoboCasa task-family sampling requires the canonical contiguous "
                "Human300 order: 65 atomic paths followed by 235 composite paths."
            )

        task_names = tuple(Path(path).parents[1].name for path in dataset_dirs)
        self.domain_names = tuple(family_counts)
        self.domain_ranges: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self.domain_group_ranges: OrderedDict[str, tuple[tuple[int, int], ...]] = OrderedDict()
        self.domain_group_names: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self.domain_sampling_signatures: OrderedDict[str, str] = OrderedDict()

        group_offset = 0
        for family, count in family_counts.items():
            group_stop = group_offset + count
            family_ranges = ranges[group_offset:group_stop]
            family_tasks = task_names[group_offset:group_stop]
            self.domain_ranges[family] = (
                family_ranges[0][0],
                family_ranges[-1][1],
            )
            self.domain_group_ranges[family] = family_ranges
            self.domain_group_names[family] = family_tasks
            payload = {
                "dataset_sampling_signature": self.sampling_signature,
                "family": family,
                "task_names": family_tasks,
                "frame_ranges": family_ranges,
            }
            serialized = json.dumps(payload, separators=(",", ":"))
            self.domain_sampling_signatures[family] = hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()
            group_offset = group_stop

        self._domain_stops = tuple(stop for _, stop in self.domain_ranges.values())

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self.expose_task_family_domains:
            return sample

        normalized_index = to_index(index)
        if normalized_index < 0:
            normalized_index += len(self)
        result = dict(sample)
        result["domain_index"] = bisect_right(self._domain_stops, normalized_index)
        return result
