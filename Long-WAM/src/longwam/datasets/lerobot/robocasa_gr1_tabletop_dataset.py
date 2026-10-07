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

"""Strict Long-WAM adapter for the official RoboCasa GR1 Tabletop data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from operator import index as to_index
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from .base_lerobot_dataset import BaseLerobotDataset
from .episode_selection import ResolvedPerDatasetEpisodeSelection
from .gr1_tabletop_omissions import (
    GR1KnownOmissionContract,
    load_gr1_known_omissions,
)
from .gr1_tabletop_recipe import (
    GR1TabletopSourceSpec,
    OLD_GR1_SOURCE,
    TELEOP_GR1_SOURCE,
    get_gr1_source_spec,
)
from .processors.longwam_processor import LongWAMProcessor
from .robot_video_dataset import RobotVideoDataset
from .transforms.gr1_tabletop import (
    GR1_ACTION_DIM,
    GR1_DATASET_REVISION,
    GR1_RAW_DIM,
    GR1_STATE_DIM,
    GR1ActionNormalizer,
    NoClampIdentityNormalizer,
    validate_coarse_prompt,
)


# Backward-compatible public alias used by the original 30K recipe and tests.
GR1_DATASET_NAMES = OLD_GR1_SOURCE.dataset_names

_RAW_MODALITY_FIELDS = {
    "left_arm": (0, 7),
    "left_hand": (7, 13),
    "left_leg": (13, 19),
    "neck": (19, 22),
    "right_arm": (22, 29),
    "right_hand": (29, 35),
    "right_leg": (35, 41),
    "waist": (41, 44),
}

_EMPTY_TASK_SENTINEL_INDEX = 0
_GR1_REPO_ID = OLD_GR1_SOURCE.repo_id
_GR1_SOURCE_EPISODES = OLD_GR1_SOURCE.total_episodes
_GR1_SOURCE_FRAMES = OLD_GR1_SOURCE.total_frames


class GR1DatasetContractError(ValueError):
    """Raised when local source metadata differs from the pinned official data."""


def gr1_safe_hash(input_tuple: tuple[Any, ...]) -> int:
    """Bit-exact copy of Isaac-GR00T N1.5's 128-bit tuple hash."""

    digest = hashlib.sha256(repr(input_tuple).encode("utf-8")).hexdigest()
    return int(digest, 16) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class GR1VirtualSample:
    dataset_index: int
    trajectory_index: int
    step_index: int
    physical_index: int


class GR1OfficialIndexMapper:
    """Official dataset -> trajectory -> step sampler over virtual indices.

    With all 24 path weights equal to one and both N1.5 balancing flags true,
    the hierarchy is globally frame-uniform with replacement. Keeping the
    hierarchy (instead of one flat draw) preserves reference RNG parity.
    """

    algorithm = "isaac_groot_n15_balanced_mixture_v1"

    def __init__(
        self,
        trajectory_lengths: Sequence[Sequence[int]],
        *,
        seed: int = 42,
    ):
        if not trajectory_lengths:
            raise ValueError("GR1 sampling requires at least one dataset.")
        self.seed = int(seed)
        self.epoch = 0
        self.trajectory_lengths = tuple(
            np.asarray(lengths, dtype=np.int64) for lengths in trajectory_lengths
        )
        if any(lengths.ndim != 1 or len(lengths) == 0 for lengths in self.trajectory_lengths):
            raise ValueError("Every GR1 dataset must contain at least one trajectory.")
        if any(bool((lengths <= 0).any()) for lengths in self.trajectory_lengths):
            raise ValueError("Every GR1 trajectory length must be positive.")

        self.dataset_lengths = np.asarray(
            [int(lengths.sum()) for lengths in self.trajectory_lengths],
            dtype=np.int64,
        )
        self.dataset_sampling_weights = self.dataset_lengths.astype(np.float64)
        self.dataset_sampling_weights /= self.dataset_sampling_weights.sum()
        self.trajectory_sampling_weights = tuple(
            lengths.astype(np.float64) / float(lengths.sum()) for lengths in self.trajectory_lengths
        )
        self.dataset_starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self.dataset_lengths[:-1])]
        )
        self.trajectory_starts = tuple(
            np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1])])
            for lengths in self.trajectory_lengths
        )
        self.num_virtual_samples = int(self.dataset_lengths.sum())

        contract = {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "dataset_lengths": self.dataset_lengths.tolist(),
            "trajectory_lengths_sha256": [
                hashlib.sha256(lengths.tobytes()).hexdigest() for lengths in self.trajectory_lengths
            ],
            "balance_dataset_weights": True,
            "balance_trajectory_weights": True,
        }
        self.sampling_signature = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return self.num_virtual_samples

    def set_epoch(self, epoch: int) -> None:
        epoch = to_index(epoch)
        if epoch < 0:
            raise ValueError(f"GR1 sampling epoch must be non-negative, got {epoch}.")
        self.epoch = epoch

    def sample_step(self, index: int) -> GR1VirtualSample:
        index = to_index(index)
        if not 0 <= index < len(self):
            raise IndexError(f"GR1 virtual index {index} is outside [0, {len(self)}).")
        rng = np.random.default_rng(gr1_safe_hash((self.epoch, index, self.seed)))
        dataset_index = int(rng.choice(len(self.dataset_lengths), p=self.dataset_sampling_weights))
        trajectory_index = int(
            rng.choice(
                len(self.trajectory_lengths[dataset_index]),
                p=self.trajectory_sampling_weights[dataset_index],
            )
        )
        step_index = int(rng.choice(self.trajectory_lengths[dataset_index][trajectory_index]))
        physical_index = int(
            self.dataset_starts[dataset_index]
            + self.trajectory_starts[dataset_index][trajectory_index]
            + step_index
        )
        return GR1VirtualSample(
            dataset_index=dataset_index,
            trajectory_index=trajectory_index,
            step_index=step_index,
            physical_index=physical_index,
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise GR1DatasetContractError(f"Cannot read valid JSON from {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise GR1DatasetContractError(f"Expected a JSON object in {path}.")
    return payload


def _validate_info(
    dataset_dir: Path,
    *,
    expected_total_tasks: int,
    source_spec: GR1TabletopSourceSpec = OLD_GR1_SOURCE,
) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    info = _load_json(info_path)
    expected_scalars = {
        "codebase_version": "v2.0",
        "robot_type": "GR1ArmsAndWaistFourierHands",
        "fps": 20.0,
    }
    for key, expected in expected_scalars.items():
        if info.get(key) != expected:
            raise GR1DatasetContractError(
                f"{info_path}: {key} must be {expected!r}, got {info.get(key)!r}."
            )
    if info.get("total_tasks") != expected_total_tasks:
        raise GR1DatasetContractError(
            f"{info_path}: total_tasks must match tasks.jsonl "
            f"({expected_total_tasks}), got {info.get('total_tasks')!r}."
        )
    if not isinstance(info.get("total_episodes"), int) or info["total_episodes"] <= 0:
        raise GR1DatasetContractError(f"{info_path}: total_episodes must be positive.")

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise GR1DatasetContractError(f"{info_path}: missing features mapping.")
    expected_shapes = {
        "observation.state": [GR1_RAW_DIM],
        "action": [GR1_RAW_DIM],
        "observation.images.ego_view": [256, 256, 3],
    }
    expected_shapes.update({f"annotation.{key}": [1] for key in source_spec.annotation_keys})
    for key, shape in expected_shapes.items():
        feature = features.get(key)
        if not isinstance(feature, Mapping) or feature.get("shape") != shape:
            raise GR1DatasetContractError(
                f"{info_path}: feature {key!r} must have shape {shape}, got {feature!r}."
            )


def _validate_modality(
    dataset_dir: Path,
    *,
    source_spec: GR1TabletopSourceSpec = OLD_GR1_SOURCE,
) -> None:
    modality_path = dataset_dir / "meta" / "modality.json"
    modality = _load_json(modality_path)
    for category, original_key in (
        ("state", "observation.state"),
        ("action", "action"),
    ):
        actual = modality.get(category)
        if not isinstance(actual, Mapping) or tuple(actual) != tuple(_RAW_MODALITY_FIELDS):
            raise GR1DatasetContractError(
                f"{modality_path}: {category} field inventory/order changed."
            )
        for field, (start, end) in _RAW_MODALITY_FIELDS.items():
            expected = {"original_key": original_key, "start": start, "end": end}
            if actual[field] != expected:
                raise GR1DatasetContractError(
                    f"{modality_path}: {category}.{field} must be {expected}, "
                    f"got {actual[field]!r}."
                )
    if modality.get("video") != {"ego_view": {"original_key": "observation.images.ego_view"}}:
        raise GR1DatasetContractError(f"{modality_path}: ego-view mapping changed.")
    expected_annotations = {
        key: {"original_key": f"annotation.{key}"} for key in source_spec.annotation_keys
    }
    if modality.get("annotation") != expected_annotations:
        raise GR1DatasetContractError(
            f"{modality_path}: annotation mapping must be {expected_annotations}."
        )


def _validate_tasks(
    dataset_dir: Path,
    *,
    source_spec: GR1TabletopSourceSpec = OLD_GR1_SOURCE,
) -> tuple[str, ...]:
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    try:
        lines = [
            line for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        tasks = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise GR1DatasetContractError(
            f"Cannot read valid tasks from {tasks_path}: {error}"
        ) from error
    if len(tasks) < 2:
        raise GR1DatasetContractError(
            f"{tasks_path}: expected an empty sentinel and at least one task."
        )
    if any(
        not isinstance(task, Mapping)
        or set(task) != {"task_index", "task"}
        or type(task["task_index"]) is not int
        for task in tasks
    ):
        raise GR1DatasetContractError(f"{tasks_path}: invalid task entry schema.")
    indices = [task["task_index"] for task in tasks]
    if indices != list(range(len(tasks))):
        raise GR1DatasetContractError(
            f"{tasks_path}: task indices must be continuous from 0 to {len(tasks) - 1}."
        )
    if tasks[0] != {"task_index": _EMPTY_TASK_SENTINEL_INDEX, "task": ""}:
        raise GR1DatasetContractError(f"{tasks_path}: task index zero must remain empty.")
    prompts = []
    for task in tasks[1:]:
        try:
            prompt = task["task"]
            if source_spec.prompt_source == "coarse_annotation":
                prompt = validate_coarse_prompt(prompt)
            elif not isinstance(prompt, str) or not prompt or prompt != prompt.strip():
                raise ValueError("Teleop task label must be a non-empty stripped string.")
            prompts.append(prompt)
        except (TypeError, ValueError) as error:
            raise GR1DatasetContractError(
                f"{tasks_path}: invalid task index {task['task_index']}: {error}"
            ) from error
    if len(set(prompts)) != len(prompts):
        raise GR1DatasetContractError(f"{tasks_path}: non-empty task prompts must be unique.")
    return ("", *prompts)


def coarse_prompt_from_episode_remark(remark: Any) -> str:
    """Convert one raw Teleop episode remark to the canonical GR1 prompt."""

    if not isinstance(remark, str):
        raise TypeError(f"GR1 Teleop episode remark must be a string, got {type(remark).__name__}.")
    if not remark or remark != remark.strip():
        raise ValueError("GR1 Teleop episode remark must be non-empty and stripped.")
    if remark.startswith("unlocked_waist: "):
        raise ValueError("GR1 Teleop episode remark must not contain the waist prefix.")
    return validate_coarse_prompt(f"unlocked_waist: {remark}")


def _validate_episode_remarks(dataset_dir: Path) -> None:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    try:
        records = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise GR1DatasetContractError(
            f"Cannot read valid episodes from {episodes_path}: {error}"
        ) from error
    indices = [record.get("episode_index") for record in records]
    if indices != list(range(len(records))):
        raise GR1DatasetContractError(
            f"{episodes_path}: episode indices must be continuous from zero."
        )
    for record in records:
        try:
            coarse_prompt_from_episode_remark(record.get("remarks"))
        except (TypeError, ValueError) as error:
            raise GR1DatasetContractError(
                f"{episodes_path}: invalid remarks for episode "
                f"{record.get('episode_index')!r}: {error}"
            ) from error


def validate_gr1_dataset_inventory(
    dataset_dirs: Sequence[str | Path],
    *,
    source_recipe: str = "old",
) -> tuple[str, ...]:
    """Read-only validation of the exact 24-directory official inventory."""

    source_spec = get_gr1_source_spec(source_recipe)
    configured = tuple(str(path) for path in dataset_dirs)
    actual_names = tuple(Path(path).name for path in configured)
    if actual_names != source_spec.dataset_names:
        raise GR1DatasetContractError(
            f"GR1 Tabletop source {source_recipe!r} requires all 24 official "
            "directories in the pinned order; "
            f"got {actual_names}."
        )
    for configured_path in configured:
        dataset_dir = Path(configured_path)
        if not dataset_dir.is_dir():
            raise GR1DatasetContractError(f"Missing GR1 dataset directory: {dataset_dir}")
        for filename in ("stats.json", "episodes.jsonl"):
            path = dataset_dir / "meta" / filename
            if not path.is_file():
                raise GR1DatasetContractError(f"Missing official metadata file: {path}")
        tasks = _validate_tasks(dataset_dir, source_spec=source_spec)
        _validate_info(
            dataset_dir,
            expected_total_tasks=len(tasks),
            source_spec=source_spec,
        )
        _validate_modality(dataset_dir, source_spec=source_spec)
        if source_spec.prompt_source == "episode_remarks":
            _validate_episode_remarks(dataset_dir)
    # Keep the configured spelling: changing fsw/fs1 paths changes HF fingerprints.
    return configured


def resolve_gr1_known_omission_selection(
    dataset_dirs: Sequence[str | Path],
    known_omissions_path: str | Path,
) -> tuple[ResolvedPerDatasetEpisodeSelection, GR1KnownOmissionContract]:
    """Select every official episode except the exact unavailable video pair."""

    roots = tuple(Path(path).expanduser().resolve() for path in dataset_dirs)
    contract = load_gr1_known_omissions(
        known_omissions_path,
        repo_id=_GR1_REPO_ID,
        revision=GR1_DATASET_REVISION,
        source_total_episodes=_GR1_SOURCE_EPISODES,
        source_total_frames=_GR1_SOURCE_FRAMES,
        dataset_names=list(GR1_DATASET_NAMES),
    )
    grouped = contract.by_dataset()
    selections: list[tuple[int, ...]] = []
    frame_counts: list[int] = []
    source_episode_count = 0
    source_frame_count = 0
    signature_rows = []
    for root, expected_name in zip(roots, GR1_DATASET_NAMES, strict=True):
        if root.name != expected_name:
            raise GR1DatasetContractError(
                f"Known-omission selection expected {expected_name}, got {root.name}."
            )
        episodes_path = root / "meta/episodes.jsonl"
        try:
            records = [
                json.loads(line)
                for line in episodes_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise GR1DatasetContractError(
                f"Cannot read official episode metadata: {episodes_path}"
            ) from error
        indices = [int(row.get("episode_index", -1)) for row in records]
        if indices != list(range(len(records))):
            raise GR1DatasetContractError(f"Official episode order changed: {episodes_path}")
        lengths = [int(row.get("length", -1)) for row in records]
        if any(length <= 0 for length in lengths):
            raise GR1DatasetContractError(f"Official episode lengths are invalid: {episodes_path}")
        omitted = grouped.get(expected_name, ())
        omitted_indices = {row.episode_index for row in omitted}
        for row in omitted:
            if (
                row.episode_index >= len(records)
                or lengths[row.episode_index] != row.episode_length
            ):
                raise GR1DatasetContractError(
                    f"Known omission no longer matches {episodes_path}: {row}."
                )
            video = root / row.relative_path
            parquet = root / row.paired_data_path
            if video.exists() or video.is_symlink():
                raise GR1DatasetContractError(
                    f"Allowlisted upstream video now exists; remove the exception: {video}"
                )
            if parquet.is_symlink() or not parquet.is_file() or parquet.stat().st_size == 0:
                raise GR1DatasetContractError(
                    f"Known omission lost its paired parquet payload: {parquet}"
                )
        selected = tuple(index for index in indices if index not in omitted_indices)
        selected_frames = sum(lengths[index] for index in selected)
        selections.append(selected)
        frame_counts.append(selected_frames)
        source_episode_count += len(records)
        source_frame_count += sum(lengths)
        signature_rows.append(
            {
                "dataset": expected_name,
                "source_episodes": len(records),
                "selected_episodes": len(selected),
                "selected_frames": selected_frames,
                "omitted_indices": sorted(omitted_indices),
            }
        )
    if (
        source_episode_count != contract.source_total_episodes
        or source_frame_count != contract.source_total_frames
        or sum(len(row) for row in selections) != contract.effective_total_episodes
        or sum(frame_counts) != contract.effective_total_frames
    ):
        raise GR1DatasetContractError(
            "Known-omission selection totals differ from official metadata."
        )
    signature_payload = {
        "algorithm": "all_official_complete_pairs_v1",
        "known_omissions_sha256": contract.sha256,
        "datasets": signature_rows,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        ResolvedPerDatasetEpisodeSelection(
            episode_indices=tuple(selections),
            dataset_frame_counts=tuple(frame_counts),
            total_episodes=contract.effective_total_episodes,
            signature=signature,
        ),
        contract,
    )


class _GR1BaseLerobotDataset(BaseLerobotDataset):
    """Resolve source-specific prompts and retain replicated action targets."""

    annotation_key = "annotation.human.coarse_action"
    source_spec = OLD_GR1_SOURCE

    def _get_additional_data(self, sample, lerobot_sample):
        dataset_index_value = lerobot_sample.get("dataset_index")
        if dataset_index_value is None:
            raise GR1DatasetContractError("Sample is missing dataset_index provenance.")
        dataset_index = int(torch.as_tensor(dataset_index_value).item())
        if not 0 <= dataset_index < len(self.multi_dataset._datasets):
            raise GR1DatasetContractError(f"Invalid dataset_index {dataset_index}.")
        metadata = self.multi_dataset._datasets[dataset_index].meta

        if self.source_spec.prompt_source == "episode_remarks":
            episode_index_value = lerobot_sample.get("episode_index")
            if episode_index_value is None:
                raise GR1DatasetContractError(
                    "GR1 Teleop sample is missing episode_index provenance."
                )
            episode_index = int(torch.as_tensor(episode_index_value).item())
            episodes = metadata.episodes
            if not isinstance(episodes, Mapping) or episode_index not in episodes:
                raise GR1DatasetContractError(
                    f"Episode index {episode_index} is absent from dataset "
                    f"{dataset_index} metadata."
                )
            try:
                sample["task"] = coarse_prompt_from_episode_remark(
                    episodes[episode_index].get("remarks")
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise GR1DatasetContractError(
                    f"Invalid Teleop remarks for dataset {dataset_index}, "
                    f"episode {episode_index}: {error}"
                ) from error
        else:
            self._set_coarse_annotation_prompt(sample, lerobot_sample, metadata, dataset_index)

        # LeRobot has already replicated the first/last absolute action. The
        # official N1.5 objective trains on it, so do not expose it as padding
        # to Long-WAM's action timestep loss mask.
        action_is_pad = torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
        sample["action_boundary_is_replicated"] = action_is_pad.clone()
        sample["action_is_pad"] = torch.zeros_like(action_is_pad)
        return sample

    def _set_coarse_annotation_prompt(
        self, sample, lerobot_sample, metadata, dataset_index: int
    ) -> None:
        if self.annotation_key not in lerobot_sample:
            raise GR1DatasetContractError(f"Sample is missing required {self.annotation_key!r}.")
        annotation = torch.as_tensor(lerobot_sample[self.annotation_key])
        if annotation.numel() != 1:
            raise GR1DatasetContractError(
                f"{self.annotation_key} must resolve to one current index, "
                f"got shape {tuple(annotation.shape)}."
            )
        annotation_value = annotation.item()
        if isinstance(annotation_value, bool):
            raise GR1DatasetContractError(f"{self.annotation_key} must be an integer task index.")
        try:
            task_index = to_index(annotation_value)
        except TypeError as error:
            raise GR1DatasetContractError(
                f"{self.annotation_key} must be an integer task index."
            ) from error
        if task_index == _EMPTY_TASK_SENTINEL_INDEX:
            raise GR1DatasetContractError(
                "task index 0 is the official empty sentinel and is never a training prompt."
            )
        tasks = metadata.tasks
        if not isinstance(tasks, Mapping):
            raise GR1DatasetContractError(f"Dataset {dataset_index} has an invalid task mapping.")
        if task_index < 1 or task_index not in tasks:
            raise GR1DatasetContractError(
                f"Coarse task index {task_index} is outside the valid task range "
                f"for dataset {dataset_index}."
            )
        sample["task"] = validate_coarse_prompt(tasks[task_index])


class _GR1TeleopBaseLerobotDataset(_GR1BaseLerobotDataset):
    source_spec = TELEOP_GR1_SOURCE


class RoboCasaGR1TabletopProcessor(LongWAMProcessor):
    """LongWAMBase processor whose only normalization is the official GR1 transform."""

    def __init__(
        self,
        *args,
        expected_dataset_revision: str = GR1_DATASET_REVISION,
        **kwargs,
    ):
        self.expected_dataset_revision = expected_dataset_revision
        super().__init__(*args, **kwargs)

    def augment_instruction(self, data: Mapping[str, Any]) -> str:
        if "task" not in data:
            raise GR1DatasetContractError("GR1 sample is missing its coarse prompt.")
        return validate_coarse_prompt(data["task"])

    def set_normalizer_from_stats(self, dataset_stats: Mapping[str, Any] | None = None):
        if dataset_stats is None:
            raise ValueError("GR1 training requires the pinned global action statistics.")
        # Validate the exact same schema consumed by GR1StateActionTransform.
        GR1ActionNormalizer.from_payload(
            dataset_stats,
            expected_dataset_revision=self.expected_dataset_revision,
        )
        self._normalizer = NoClampIdentityNormalizer()


class RoboCasaGR1TabletopDataset(RobotVideoDataset):
    """All complete pairs from the official 24-task GR1 Tabletop dataset."""

    def __init__(
        self,
        dataset_dirs,
        *,
        source_recipe: str = "old",
        expected_dataset_count: int = 24,
        expected_dataset_revision: str | None = None,
        validate_dataset_contract: bool = True,
        known_omissions_path: str | Path | None = None,
        pretrained_norm_stats: str | Path | None = None,
        sampling_seed: int = 42,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        **kwargs,
    ):
        source_spec = get_gr1_source_spec(source_recipe)
        if expected_dataset_revision is None:
            expected_dataset_revision = source_spec.revision
        if expected_dataset_count != len(source_spec.dataset_names):
            raise GR1DatasetContractError(
                f"GR1 expected_dataset_count must remain 24, got {expected_dataset_count}."
            )
        if expected_dataset_revision != source_spec.revision:
            raise GR1DatasetContractError(
                f"GR1 source {source_recipe!r} dataset revision must remain pinned "
                f"to {source_spec.revision}, got {expected_dataset_revision}."
            )
        dataset_dirs = tuple(str(path) for path in dataset_dirs)
        if len(dataset_dirs) != expected_dataset_count:
            raise GR1DatasetContractError(
                f"GR1 Tabletop requires 24 dataset directories, got {len(dataset_dirs)}."
            )
        if validate_dataset_contract:
            validate_gr1_dataset_inventory(dataset_dirs, source_recipe=source_recipe)
        if pretrained_norm_stats is None:
            raise GR1DatasetContractError(
                "GR1 training requires a derived global action-statistics file; "
                "statistics must not be recomputed from sampled draws."
            )
        GR1ActionNormalizer.from_json(
            pretrained_norm_stats,
            expected_dataset_revision=source_spec.revision,
        )
        shape_meta = kwargs.get("shape_meta")
        if shape_meta is None:
            raise GR1DatasetContractError("GR1 training requires shape_meta.")
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        validate_gr1_shape_meta(shape_meta)
        if "lerobot_dataset_cls" in kwargs:
            raise TypeError("RoboCasaGR1TabletopDataset owns its low-level dataset adapter.")
        if "per_dataset_episode_selection" in kwargs:
            raise TypeError("GR1 owns its exact complete-pair episode selection.")
        if not balance_dataset_weights or not balance_trajectory_weights:
            raise GR1DatasetContractError(
                "The first GR1 run requires both official N1.5 balancing flags."
            )
        omission_contract = None
        selection = None
        if source_spec.requires_known_omissions:
            if known_omissions_path is None:
                raise GR1DatasetContractError(
                    "The old GR1 source requires the pinned upstream known-omission contract."
                )
            selection, omission_contract = resolve_gr1_known_omission_selection(
                dataset_dirs, known_omissions_path
            )
        elif known_omissions_path is not None:
            raise GR1DatasetContractError(
                "The Teleop GR1 source has no known omissions; do not apply the "
                "old source's omission contract."
            )
        adapter_cls = (
            _GR1TeleopBaseLerobotDataset
            if source_spec.prompt_source == "episode_remarks"
            else _GR1BaseLerobotDataset
        )
        super().__init__(
            dataset_dirs=dataset_dirs,
            pretrained_norm_stats=str(pretrained_norm_stats),
            lerobot_dataset_cls=adapter_cls,
            per_dataset_episode_selection=selection,
            **kwargs,
        )
        trajectory_lengths = []
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            starts = dataset.episode_data_index["from"].to(torch.int64)
            stops = dataset.episode_data_index["to"].to(torch.int64)
            trajectory_lengths.append((stops - starts).cpu().numpy())
        self._official_index_mapper = GR1OfficialIndexMapper(
            trajectory_lengths,
            seed=sampling_seed,
        )
        if len(self._official_index_mapper) != super().__len__():
            raise GR1DatasetContractError(
                "Official trajectory lengths do not cover the complete local dataset."
            )
        signature_contract = {
            "index_mapper_signature": self._official_index_mapper.sampling_signature,
            "physical_dataset_signature": self.lerobot_dataset.sampling_signature,
            "source_recipe": source_recipe,
            "repo_id": source_spec.repo_id,
            "dataset_revision": source_spec.revision,
            "dataset_names": source_spec.dataset_names,
            "known_omissions_sha256": (
                omission_contract.sha256 if omission_contract is not None else None
            ),
            "effective_total_episodes": (
                omission_contract.effective_total_episodes
                if omission_contract is not None
                else source_spec.total_episodes
            ),
            "effective_total_frames": (
                omission_contract.effective_total_frames
                if omission_contract is not None
                else source_spec.total_frames
            ),
        }
        self._gr1_sampling_signature = hashlib.sha256(
            json.dumps(signature_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def sampling_signature(self) -> str:
        return self._gr1_sampling_signature

    @property
    def sampling_epoch(self) -> int:
        return self._official_index_mapper.epoch

    def set_epoch(self, epoch: int) -> None:
        self._official_index_mapper.set_epoch(epoch)

    def sample_step(self, index: int) -> GR1VirtualSample:
        return self._official_index_mapper.sample_step(index)

    def __getitem__(self, index):
        sample = self.sample_step(index)
        return super().__getitem__(sample.physical_index)


def validate_gr1_shape_meta(shape_meta: Mapping[str, Any]) -> None:
    """Focused preflight helper for the frozen raw/transformed dimensions."""

    expected = {
        "images": [{"key": "ego_view", "raw_shape": [3, 256, 256], "shape": [3, 224, 224]}],
        "action": [{"key": "default", "raw_shape": GR1_RAW_DIM, "shape": GR1_ACTION_DIM}],
        "state": [{"key": "default", "raw_shape": GR1_RAW_DIM, "shape": GR1_STATE_DIM}],
    }
    if dict(shape_meta) != expected:
        raise GR1DatasetContractError(
            f"GR1 shape_meta must equal the official Long-WAM contract {expected}, got {shape_meta}."
        )


__all__ = [
    "GR1_DATASET_NAMES",
    "TELEOP_GR1_SOURCE",
    "GR1DatasetContractError",
    "GR1OfficialIndexMapper",
    "GR1VirtualSample",
    "RoboCasaGR1TabletopDataset",
    "RoboCasaGR1TabletopProcessor",
    "validate_gr1_dataset_inventory",
    "coarse_prompt_from_episode_remark",
    "resolve_gr1_known_omission_selection",
    "validate_gr1_shape_meta",
    "gr1_safe_hash",
]
