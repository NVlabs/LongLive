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

"""Fail-closed initialization provenance for the GR1 Tabletop recipe.

The GR1 run starts from two strict expert payloads and exactly six tensors that
are initialized after ``seed=42`` is installed.  This module records that
boundary and proves that the optimizer contains every intended parameter once,
without putting process-local parameter IDs in the persistent manifest.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from longwam.models.wan22.action_dit import ActionDiT


MANIFEST_SCHEMA = "longwam.robocasa-gr1-initialization-provenance/v1"
INITIALIZATION_SEED = 42
MANIFEST_FILENAME = "initialization_manifest.json"
LEGACY_VIDEO_STATE_CONTAINER = "root (optional model. prefix stripped)"

ACTION_BOUNDARY_NAMES = (
    "action_expert.action_encoder.weight",
    "action_expert.action_encoder.bias",
    "action_expert.head.weight",
    "action_expert.head.bias",
)
PROPRIO_BOUNDARY_NAMES = (
    "proprio_encoder.weight",
    "proprio_encoder.bias",
)
BOUNDARY_NAMES = ACTION_BOUNDARY_NAMES + PROPRIO_BOUNDARY_NAMES
OPTIMIZER_CATEGORIES = (
    "video_strict",
    "action_backbone_strict",
    "seed42_action_boundary",
    "seed42_proprio",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _regular_file_identity(path: str | Path, sealed_sha256: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError(f"Expert source must be a regular non-symlink file: {source}")
    metadata = source.stat()
    return {
        "path": str(source.resolve()),
        "sha256": _require_sha256(sealed_sha256, "sealed expert identity"),
        "bytes": int(metadata.st_size),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(dimension) for dimension in tensor.shape]


def _target_tensor_specs(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": _shape(tensor),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in sorted(state.items())
    ]


def _validated_actual_load_report(
    *,
    label: str,
    actual: Any,
    expected_path: Path,
    expected_container: str,
    target: Mapping[str, torch.Tensor],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{label} has no real loader report")
    if (
        Path(str(actual.get("source_path", ""))).expanduser().resolve() != expected_path
        or actual.get("state_container") != expected_container
        or actual.get("strict") is not True
        or actual.get("missing_keys") != []
        or actual.get("unexpected_keys") != []
    ):
        raise ValueError(f"{label} real strict loader report is invalid: {actual}")
    source_tensors = actual.get("source_tensors")
    if not isinstance(source_tensors, list) or any(
        not isinstance(item, Mapping) for item in source_tensors
    ):
        raise ValueError(f"{label} loader report has no tensor inventory")
    source_by_name = {str(item.get("name")): dict(item) for item in source_tensors}
    if len(source_by_name) != len(source_tensors) or set(source_by_name) != set(target):
        raise ValueError(f"{label} strict loader key inventory drifted")
    target_tensors = _target_tensor_specs(target)
    for item in target_tensors:
        source = source_by_name[item["name"]]
        if (
            source.get("shape") != item["shape"]
            or source.get("numel") != item["numel"]
            or not isinstance(source.get("dtype"), str)
        ):
            raise ValueError(f"{label} strict loader tensor schema drifted: {item['name']}")
    ordered_source = [source_by_name[item["name"]] for item in target_tensors]
    return {
        "source": dict(source_identity),
        "state_container": expected_container,
        "strict": True,
        "load_result": {"missing_keys": [], "unexpected_keys": []},
        "loaded_keys": len(target_tensors),
        "source_tensors": ordered_source,
        "target_tensors": target_tensors,
        "source_tensors_sha256": canonical_sha256(ordered_source),
        "target_tensors_sha256": canonical_sha256(target_tensors),
    }


def _resolved_model_source(model: Any, key: str, expected: Path) -> None:
    model_paths = getattr(model, "model_paths", None)
    if not isinstance(model_paths, Mapping):
        raise ValueError("GR1 model has no model_paths provenance mapping")
    observed = model_paths.get(key)
    if observed is None or Path(str(observed)).expanduser().resolve() != expected:
        raise ValueError(f"GR1 model path {key!r} is not the sealed expert: {observed!r}")


def build_strict_expert_load_reports(
    model: Any,
    *,
    video_path: str | Path,
    video_sha256: str,
    action_path: str | Path,
    action_sha256: str,
    video_state_container: str = LEGACY_VIDEO_STATE_CONTAINER,
) -> dict[str, Any]:
    """Validate real ``load_state_dict(strict=True)`` reports from both loaders."""

    if not isinstance(video_state_container, str) or not video_state_container.strip():
        raise ValueError("video_state_container must be a non-empty string")

    video_source = Path(video_path).expanduser().resolve()
    action_source = Path(action_path).expanduser().resolve()
    _resolved_model_source(model, "longlive_video_expert", video_source)
    _resolved_model_source(model, "action_dit_backbone", action_source)

    video_identity = _regular_file_identity(video_source, video_sha256)
    action_identity = _regular_file_identity(action_source, action_sha256)

    video_target = getattr(model, "video_expert", None)
    if not isinstance(video_target, nn.Module):
        raise ValueError("GR1 model has no video_expert module")
    video_report = _validated_actual_load_report(
        label="video expert",
        actual=getattr(video_target, "_longwam_strict_load_report", None),
        expected_path=video_source,
        expected_container=video_state_container,
        target=video_target.state_dict(),
        source_identity=video_identity,
    )

    action_target = getattr(model, "action_expert", None)
    if not isinstance(action_target, ActionDiT):
        raise ValueError("GR1 model action_expert must be ActionDiT")
    if ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES != ("action_encoder.", "head."):
        raise ValueError("ActionDiT random-boundary prefix policy drifted")
    full_action_state = action_target.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(full_action_state)
    excluded = sorted(set(full_action_state) - backbone_keys)
    expected_excluded = sorted(
        name.removeprefix("action_expert.") for name in ACTION_BOUNDARY_NAMES
    )
    if excluded != expected_excluded:
        raise ValueError(
            "Only four ActionDiT boundary tensors may be newly initialized: "
            f"observed={excluded}, expected={expected_excluded}"
        )
    actual_action_report = getattr(action_target, "_longwam_strict_backbone_load_report", None)
    if (
        not isinstance(actual_action_report, Mapping)
        or actual_action_report.get("excluded_seed42_tensors") != excluded
    ):
        raise ValueError("Action backbone real loader exclusion report drifted")
    action_report = _validated_actual_load_report(
        label="action backbone",
        actual=actual_action_report,
        expected_path=action_source,
        expected_container="backbone_state_dict",
        target={key: full_action_state[key] for key in backbone_keys},
        source_identity=action_identity,
    )
    action_report["excluded_seed42_tensors"] = [f"action_expert.{name}" for name in excluded]
    return {"video_strict": video_report, "action_backbone_strict": action_report}


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor, got {type(tensor)}")
    cpu = tensor.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(cpu.dtype), "shape": _shape(cpu)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(cpu.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _named_boundary_parameters(model: Any) -> dict[str, nn.Parameter]:
    action = getattr(model, "action_expert", None)
    proprio = getattr(model, "proprio_encoder", None)
    if not isinstance(action, ActionDiT) or not isinstance(proprio, nn.Linear):
        raise ValueError("GR1 requires ActionDiT plus a Linear proprio_encoder")
    values = {
        "action_expert.action_encoder.weight": action.action_encoder.weight,
        "action_expert.action_encoder.bias": action.action_encoder.bias,
        "action_expert.head.weight": action.head.weight,
        "action_expert.head.bias": action.head.bias,
        "proprio_encoder.weight": proprio.weight,
        "proprio_encoder.bias": proprio.bias,
    }
    if tuple(values) != BOUNDARY_NAMES or any(
        not isinstance(value, nn.Parameter) for value in values.values()
    ):
        raise ValueError("GR1 six-tensor random initialization boundary drifted")
    return values


def build_seed42_boundary_report(model: Any, *, seed: int) -> dict[str, Any]:
    if seed != INITIALIZATION_SEED:
        raise ValueError(f"GR1 initialization seed must be {INITIALIZATION_SEED}")
    tensors = []
    for name, parameter in _named_boundary_parameters(model).items():
        tensors.append(
            {
                "name": name,
                "category": (
                    "seed42_action_boundary" if name in ACTION_BOUNDARY_NAMES else "seed42_proprio"
                ),
                "shape": _shape(parameter),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
                "content_sha256": tensor_content_sha256(parameter),
            }
        )
    return {
        "seed": seed,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "content_sha256": canonical_sha256(
            [{"name": item["name"], "content_sha256": item["content_sha256"]} for item in tensors]
        ),
    }


def _category_parameter_maps(model: Any) -> dict[str, dict[str, nn.Parameter]]:
    if not isinstance(model, nn.Module):
        raise ValueError("GR1 optimizer provenance requires an nn.Module model")
    video = getattr(model, "video_expert", None)
    action = getattr(model, "action_expert", None)
    proprio = getattr(model, "proprio_encoder", None)
    dit = getattr(model, "dit", None)
    if not isinstance(video, nn.Module) or not isinstance(action, ActionDiT):
        raise ValueError("GR1 optimizer provenance requires both expert modules")
    if not isinstance(proprio, nn.Linear) or not isinstance(dit, nn.Module):
        raise ValueError("GR1 optimizer provenance requires DiT and proprio modules")

    action_parameters = dict(action.named_parameters())
    action_boundary_local = {name.removeprefix("action_expert.") for name in ACTION_BOUNDARY_NAMES}
    if not action_boundary_local.issubset(action_parameters):
        raise ValueError("Action boundary parameter names are missing")
    maps = {
        "video_strict": {
            f"video_expert.{name}": parameter for name, parameter in video.named_parameters()
        },
        "action_backbone_strict": {
            f"action_expert.{name}": parameter
            for name, parameter in action_parameters.items()
            if name not in action_boundary_local
        },
        "seed42_action_boundary": {
            f"action_expert.{name}": action_parameters[name]
            for name in sorted(action_boundary_local)
        },
        "seed42_proprio": {
            f"proprio_encoder.{name}": parameter for name, parameter in proprio.named_parameters()
        },
    }
    observed_proprio = set(maps["seed42_proprio"])
    if observed_proprio != set(PROPRIO_BOUNDARY_NAMES):
        raise ValueError(f"Proprio random boundary drifted: {sorted(observed_proprio)}")

    seen: dict[int, str] = {}
    for category in OPTIMIZER_CATEGORIES:
        for name, parameter in maps[category].items():
            previous = seen.setdefault(id(parameter), name)
            if previous != name:
                raise ValueError(
                    f"Parameter appears in multiple provenance classes: {previous}, {name}"
                )
            if not parameter.requires_grad:
                raise ValueError(f"Intended optimizer parameter is frozen: {name}")

    expected_dit_ids = {
        id(parameter)
        for category in (
            "video_strict",
            "action_backbone_strict",
            "seed42_action_boundary",
        )
        for parameter in maps[category].values()
    }
    observed_dit_ids = {id(parameter) for parameter in dit.parameters()}
    if observed_dit_ids != expected_dit_ids:
        raise ValueError("model.dit contains missing or unclassified parameters")
    all_trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(all_trainable) != set(seen):
        unoptimized = sorted(
            name for parameter_id, name in all_trainable.items() if parameter_id not in seen
        )
        unexpectedly_classified = sorted(
            name for parameter_id, name in seen.items() if parameter_id not in all_trainable
        )
        raise ValueError(
            "All model trainable parameters must equal the GR1 provenance classes: "
            f"unoptimized={unoptimized[:8]}, classified_but_not_trainable="
            f"{unexpectedly_classified[:8]}"
        )
    return maps


def validate_optimizer_parameter_provenance(model: Any, optimizer: Any) -> dict[str, Any]:
    """Require exact, duplicate-free optimizer coverage by runtime object ID."""

    category_maps = _category_parameter_maps(model)
    expected_by_id = {
        id(parameter): (category, name, parameter)
        for category in OPTIMIZER_CATEGORIES
        for name, parameter in category_maps[category].items()
    }
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, Sequence):
        raise TypeError("Prepared optimizer has no parameter groups")
    optimizer_parameters = [
        parameter for group in param_groups for parameter in group.get("params", [])
    ]
    counts = Counter(id(parameter) for parameter in optimizer_parameters)
    duplicates = sorted(
        expected_by_id[parameter_id][1]
        if parameter_id in expected_by_id
        else f"unclassified_id:{parameter_id}"
        for parameter_id, count in counts.items()
        if count != 1
    )
    missing = sorted(
        name
        for parameter_id, (_category, name, _parameter) in expected_by_id.items()
        if parameter_id not in counts
    )
    extras = sorted(
        f"id:{parameter_id}" for parameter_id in counts if parameter_id not in expected_by_id
    )
    if duplicates or missing or extras or len(counts) != len(expected_by_id):
        raise ValueError(
            "Optimizer parameter-ID coverage must be exact and once-only: "
            f"duplicates={duplicates[:8]}, missing={missing[:8]}, extras={extras[:8]}"
        )

    categories = {}
    coverage = []
    for category in OPTIMIZER_CATEGORIES:
        items = sorted(category_maps[category].items())
        names = [name for name, _parameter in items]
        parameters = [
            {
                "name": name,
                "shape": _shape(parameter),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
                "source": category,
            }
            for name, parameter in items
        ]
        categories[category] = {
            "parameter_count": len(items),
            "numel": sum(int(parameter.numel()) for _name, parameter in items),
            "parameters": parameters,
            "names_sha256": canonical_sha256(names),
            "parameters_sha256": canonical_sha256(parameters),
        }
        coverage.extend({"category": category, "name": name} for name in names)
    return {
        "runtime_identity_policy": "python_parameter_id_exactly_once",
        "all_parameter_ids_unique": True,
        "parameter_count": len(expected_by_id),
        "numel": sum(
            int(parameter.numel()) for _category, _name, parameter in expected_by_id.values()
        ),
        "categories": categories,
        "coverage_sha256": canonical_sha256(coverage),
    }


def build_initialization_manifest(
    *,
    task: str,
    run_id: str,
    composite_expert_sha256: str,
    strict_load_reports: Mapping[str, Any],
    boundary_report: Mapping[str, Any],
    optimizer_report: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(strict_load_reports) != ("video_strict", "action_backbone_strict"):
        raise ValueError("GR1 requires ordered video/action strict-load reports")
    if boundary_report.get("tensor_count") != len(BOUNDARY_NAMES):
        raise ValueError("GR1 manifest requires exactly six random boundary tensors")
    names = [item.get("name") for item in boundary_report.get("tensors", [])]
    if tuple(names) != BOUNDARY_NAMES:
        raise ValueError(f"GR1 random boundary names drifted: {names}")
    categories = optimizer_report.get("categories", {})
    if tuple(categories) != OPTIMIZER_CATEGORIES:
        raise ValueError("GR1 optimizer provenance categories drifted")

    payload = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "task": task,
        "run_id": run_id,
        "composite_expert_sha256": _require_sha256(
            composite_expert_sha256, "composite expert identity"
        ),
        "strict_load_reports": dict(strict_load_reports),
        "seed42_boundary": dict(boundary_report),
        "optimizer": dict(optimizer_report),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def validate_rank_manifest_consensus(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("No rank initialization manifests were gathered")
    expected = records[0]
    for rank, record in enumerate(records):
        if record != expected:
            raise RuntimeError(
                "GR1 initialization manifest or six-tensor content hashes differ "
                f"across ranks (first mismatch rank={rank})"
            )


def rank_consensus_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tensors = manifest.get("seed42_boundary", {}).get("tensors", [])
    hashes = {item.get("name"): item.get("content_sha256") for item in tensors}
    if tuple(hashes) != BOUNDARY_NAMES or any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in hashes.values()
    ):
        raise ValueError("Manifest has an invalid six-tensor hash set")
    return {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "boundary_content_sha256": hashes,
    }


def _validated_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Invalid GR1 initialization manifest schema")
    signed = dict(value)
    observed = signed.pop("manifest_sha256", None)
    expected = canonical_sha256(signed)
    if observed != expected:
        raise ValueError(f"GR1 initialization manifest signature drifted: {observed} != {expected}")
    return value


def load_initialization_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"GR1 initialization manifest is missing or symlinked: {path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read GR1 initialization manifest: {path}") from exc
    return _validated_manifest(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def persist_or_validate_initialization_manifest(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    """Atomically create on fresh launch; validation-only on every resume."""

    manifest_path = Path(path)
    expected_value = _validated_manifest(dict(expected))
    if manifest_path.exists() or manifest_path.is_symlink():
        observed = load_initialization_manifest(manifest_path)
        if observed != expected_value:
            raise ValueError(
                "Existing GR1 initialization manifest drifted; refusing to overwrite it"
            )
        return
    if resume:
        raise ValueError(
            "Formal resume requires the immutable initialization manifest; refusing to recreate it"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.create.lock")
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Concurrent GR1 initialization-manifest creator detected: {lock_path}"
        ) from exc
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if manifest_path.exists() or manifest_path.is_symlink():
            observed = load_initialization_manifest(manifest_path)
            if observed != expected_value:
                raise ValueError("GR1 initialization manifest appeared with different content")
            return
        _write_json_atomic(manifest_path, expected_value)
    finally:
        lock_path.unlink(missing_ok=True)


def initialization_manifest_identity(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    value = load_initialization_manifest(manifest_path)
    payload = manifest_path.read_bytes()
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_sha256": value["manifest_sha256"],
        "schema": value["schema"],
    }
