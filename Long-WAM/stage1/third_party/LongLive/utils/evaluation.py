# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot integration imported from the author video-training branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/utils/evaluation.py
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

"""Deterministic, distributed helpers for fixed validation samples."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch.distributed as dist
from torch.utils.data import Subset


@dataclass(frozen=True)
class FixedEvaluationSample:
    """Stable identity and ownership metadata for one validation sample."""

    slot: int
    sample_id: str
    dataset_index: int


@dataclass(frozen=True)
class FixedEvaluationGroup:
    """One fixed-shape validation pass executed by every distributed rank.

    ``name=None`` is the legacy single-group contract.  It deliberately keeps
    the historical flat output directory and ``validation/video_*`` W&B keys.
    Named groups use their name as both an output subdirectory and W&B
    namespace.
    """

    name: Optional[str]
    num_frames: int
    sample_ids: Tuple[str, ...]
    seed: int
    data_path: str


_EVALUATION_GROUP_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _all_gather_evaluation_object(value):
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size()
    # launch_distributed_job sets the current CUDA device before evaluation;
    # all_gather_object therefore also works with the NCCL default group.
    dist.all_gather_object(gathered, value)
    return gathered


def synchronize_evaluation_failure(local_error, *, phase: str) -> None:
    """Propagate a rank-local validation failure to every distributed rank.

    Call this at boundaries where every rank is known to arrive.  A serialized
    status from every rank makes rank-zero filesystem/W&B failures visible to
    peers before they enter another world barrier, and gives every rank the
    same terminal exception message.

    This cannot recover a rank that is already stuck or terminated inside an
    NCCL/model collective.  It does cover Python, dataloader, generation-return,
    media-write, manifest, and logging failures once ranks reach the boundary.
    """

    phase = str(phase)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    payload = None
    if local_error is not None:
        payload = {
            "rank": int(rank),
            "phase": phase,
            "type": type(local_error).__name__,
            "message": str(local_error)[:4096],
        }
    failures = [item for item in _all_gather_evaluation_object(payload) if item]
    if not failures:
        return

    details = "; ".join(
        "rank={rank} phase={phase!r} {type}: {message}".format(**failure)
        for failure in failures
    )
    error = RuntimeError(f"Distributed evaluation failed: {details}")
    if local_error is not None:
        raise error from local_error
    raise error


def require_equal_evaluation_value(value, *, phase: str):
    """Fail collectively when ranks would execute different inference counts."""

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    records = _all_gather_evaluation_object(
        {"rank": int(rank), "phase": str(phase), "value": value}
    )
    reference = records[0]["value"]
    if all(record["value"] == reference for record in records[1:]):
        return value
    details = ", ".join(
        f"rank={record['rank']} value={record['value']!r}" for record in records
    )
    raise RuntimeError(
        f"Distributed evaluation shape/count mismatch during {phase}: {details}"
    )


def _config_value(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _one_num_frames(value, *, field_name: str, allow_zero: bool = False) -> int:
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{field_name} cannot be empty")
        value = value[0]
    frames = int(value)
    if frames < 0 or (frames == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier}")
    return frames


def evaluation_group_specs(
    evaluation_config,
    *,
    default_num_frames: int,
    default_sample_ids: Iterable[str],
    default_seed: int,
    default_data_path: str,
    num_frame_per_block: int,
) -> Tuple[FixedEvaluationGroup, ...]:
    """Resolve named validation passes while preserving old single-group configs.

    Every named group is required to be fixed-sample validation.  This keeps
    the number and ordering of inference calls identical on all ranks after
    ``DistributedSampler`` padding.  Group order is the configuration order
    and is therefore also the distributed collective order.
    """

    default_ids = tuple(str(value).strip() for value in default_sample_ids)
    raw_groups = _config_value(evaluation_config, "groups", None)
    if not raw_groups:
        frames = _one_num_frames(
            default_num_frames,
            field_name="evaluation.num_frames",
            allow_zero=True,
        )
        return (
            FixedEvaluationGroup(
                name=None,
                num_frames=frames,
                sample_ids=default_ids,
                seed=int(default_seed),
                data_path=str(default_data_path),
            ),
        )

    groups = []
    names = set()
    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            raise TypeError(f"evaluation.groups[{index}] must be a mapping")
        name = str(_config_value(raw_group, "name", "")).strip()
        if not _EVALUATION_GROUP_NAME.fullmatch(name):
            raise ValueError(
                "evaluation group names must match "
                f"{_EVALUATION_GROUP_NAME.pattern!r}; got {name!r}"
            )
        if name in names:
            raise ValueError(f"evaluation group name is duplicated: {name}")
        names.add(name)

        frames = _one_num_frames(
            _config_value(raw_group, "num_frames", default_num_frames),
            field_name=f"evaluation.groups[{index}].num_frames",
        )
        if frames % int(num_frame_per_block) != 0:
            raise ValueError(
                f"evaluation group {name!r} num_frames={frames} must be divisible "
                f"by num_frame_per_block={num_frame_per_block}"
            )
        sample_ids = tuple(
            str(value).strip()
            for value in _config_value(raw_group, "sample_ids", default_ids)
        )
        if not sample_ids:
            raise ValueError(
                f"evaluation group {name!r} must configure fixed sample_ids"
            )
        if any(not sample_id for sample_id in sample_ids):
            raise ValueError(
                f"evaluation group {name!r} sample_ids cannot contain empty IDs"
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(
                f"evaluation group {name!r} sample_ids must be unique"
            )
        groups.append(
            FixedEvaluationGroup(
                name=name,
                num_frames=frames,
                sample_ids=sample_ids,
                seed=int(_config_value(raw_group, "seed", default_seed)),
                data_path=str(
                    _config_value(raw_group, "eval_data_path", default_data_path)
                ),
            )
        )
    return tuple(groups)


def select_fixed_evaluation_subset(dataset, sample_ids: Iterable[str]):
    """Select dataset folders by stable ID instead of filesystem order.

    ``MultiVideoConcatDataset`` exposes one folder per sample.  The folder
    order is an implementation detail, so configs name fixed validation
    examples by folder ID and this helper resolves them on every rank.
    """

    normalized_ids = tuple(str(sample_id).strip() for sample_id in sample_ids)
    if not normalized_ids:
        raise ValueError("evaluation.sample_ids must contain at least one ID")
    if any(not sample_id for sample_id in normalized_ids):
        raise ValueError("evaluation.sample_ids cannot contain empty IDs")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("evaluation.sample_ids must be unique")

    folders = getattr(dataset, "folders", None)
    if folders is None:
        raise TypeError(
            "Fixed video evaluation requires a dataset with a 'folders' attribute"
        )

    index_by_id = {}
    for dataset_index, folder in enumerate(folders):
        sample_id = Path(folder).name
        if sample_id in index_by_id:
            raise ValueError(f"Duplicate evaluation folder ID: {sample_id}")
        index_by_id[sample_id] = dataset_index

    missing = [sample_id for sample_id in normalized_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(
            "Configured evaluation sample IDs are absent from the dataset: "
            f"{missing}"
        )

    samples = tuple(
        FixedEvaluationSample(
            slot=slot,
            sample_id=sample_id,
            dataset_index=index_by_id[sample_id],
        )
        for slot, sample_id in enumerate(normalized_ids)
    )
    validate_indices = getattr(
        dataset, "validate_fixed_evaluation_indices", None
    )
    if callable(validate_indices):
        validated_ids = tuple(
            validate_indices(
                tuple(sample.dataset_index for sample in samples)
            )
        )
        expected_ids = tuple(sample.sample_id for sample in samples)
        if validated_ids != expected_ids:
            raise ValueError(
                "Dataset fixed-evaluation validation returned different IDs: "
                f"{validated_ids!r} != {expected_ids!r}"
            )
    subset = Subset(dataset, [sample.dataset_index for sample in samples])
    return subset, samples


def fixed_evaluation_writer(
    sample_index: int,
    samples: Sequence[FixedEvaluationSample],
    *,
    data_rank: int,
    data_replicas: int,
    sequence_parallel_rank: int,
) -> Optional[FixedEvaluationSample]:
    """Return the sample metadata only on its one canonical writer rank.

    PyTorch's ``DistributedSampler(drop_last=False)`` pads a small validation
    subset so every data-parallel replica executes the same number of model
    calls.  Equal call counts are required by FSDP, but padded replicas must
    not write or upload duplicate videos.  The canonical owner of slot ``s``
    is DP rank ``s % data_replicas`` and only SP rank zero writes it.
    """

    data_rank = int(data_rank)
    data_replicas = int(data_replicas)
    sequence_parallel_rank = int(sequence_parallel_rank)
    if data_replicas < 1:
        raise ValueError("data_replicas must be positive")
    if data_rank < 0 or data_rank >= data_replicas:
        raise ValueError("data_rank is outside the data-parallel world")
    if sequence_parallel_rank != 0:
        return None

    for sample in samples:
        if sample.dataset_index == int(sample_index):
            owner = sample.slot % data_replicas
            return sample if owner == data_rank else None
    raise ValueError(f"Evaluation sample index {sample_index} is not in the fixed set")


def deterministic_evaluation_seed(base_seed: int, slot: int) -> int:
    """Return a stable per-slot seed without touching the training RNG."""

    base_seed = int(base_seed)
    slot = int(slot)
    if slot < 0:
        raise ValueError("slot must be non-negative")
    return (base_seed + slot) % (2**63 - 1)


def fixed_evaluation_log_key(
    slot: int,
    *,
    weight_label: str,
    multiple_weight_modes: bool,
    group_name: Optional[str] = None,
) -> str:
    """Return the stable W&B history key for one validation panel."""

    slot = int(slot)
    if slot < 0:
        raise ValueError("slot must be non-negative")
    prefix = "validation"
    if group_name is not None:
        group_name = str(group_name).strip()
        if not _EVALUATION_GROUP_NAME.fullmatch(group_name):
            raise ValueError(f"invalid evaluation group name: {group_name!r}")
        prefix = f"{prefix}/{group_name}"
    if multiple_weight_modes:
        return f"{prefix}/{weight_label}/video_{slot:02d}"
    return f"{prefix}/video_{slot:02d}"


def evaluation_run_modes(
    weight_mode: str,
    *,
    ema_available: bool,
) -> Tuple[Tuple[str, bool, str], ...]:
    """Resolve which weights to visualize.

    Each tuple contains ``(filename_suffix, use_ema, display_label)``.
    ``model`` is recommended for progress comparisons because it is available
    from step zero and keeps the weight definition identical at every tenth.
    """

    mode = str(weight_mode).strip().lower()
    if mode == "model":
        return (("_model", False, "model"),)
    if mode == "ema":
        if not ema_available:
            raise RuntimeError("EMA evaluation requested before EMA is available")
        return (("_ema", True, "ema"),)
    if mode == "ema_if_available":
        if ema_available:
            return (("_ema", True, "ema"),)
        return (("_model", False, "model"),)
    if mode == "both":
        modes = [("_model", False, "model")]
        if ema_available:
            modes.append(("_ema", True, "ema"))
        return tuple(modes)
    raise ValueError(
        "evaluation.weights must be one of: model, ema, ema_if_available, both"
    )


def evaluation_artifact_run_modes(
    run_modes: Sequence[Tuple[str, bool, str]],
    *,
    group_name: Optional[str],
    fixed_evaluation: bool,
) -> Tuple[Tuple[str, bool, str], ...]:
    """Restore historical artifact names only for legacy free-form eval.

    Before fixed validation existed, normal model outputs used an empty
    suffix (``video_rank...`` and ``generated_video``), while EMA used
    ``_ema``.  Named fixed groups intentionally retain explicit ``_model`` /
    ``_ema`` filenames and namespaced W&B keys.
    """

    modes = tuple(run_modes)
    if group_name is not None or fixed_evaluation:
        return modes
    return tuple(
        ("" if weight_label == "model" else suffix, use_ema, weight_label)
        for suffix, use_ema, weight_label in modes
    )
