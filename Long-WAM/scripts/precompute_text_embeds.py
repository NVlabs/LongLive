# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: scripts/precompute_text_embeds.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from longwam.datasets.lerobot.episode_selection import resolve_episode_selection
from longwam.datasets.lerobot.robot_video_dataset import (
    DEFAULT_PROMPT,
    TEXT_CACHE_TAG,
    get_text_cache_path,
)
from longwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from longwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from longwam.utils.config_resolvers import register_default_resolvers
from longwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16
TEXT_CACHE_INVENTORY_SCHEMA = "longwam.text-cache-prompt-inventory/v1"


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world_size = int(os.environ.get("LONGWAM_EXPECTED_WORLD_SIZE", str(world_size)))
    if world_size != expected_world_size:
        raise RuntimeError(
            "Distributed text-cache world-size mismatch: "
            f"expected={expected_world_size}, actual={world_size}."
        )
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    rank_device = (
        torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    )
    rank_tensor = torch.tensor([rank], device=rank_device, dtype=torch.long)
    gathered_ranks = [torch.empty_like(rank_tensor) for _ in range(actual_world_size)]
    dist.all_gather(gathered_ranks, rank_tensor)
    observed_ranks = sorted(int(item.item()) for item in gathered_ranks)
    if observed_ranks != list(range(expected_world_size)):
        raise RuntimeError(
            "Distributed text-cache ranks are incomplete: "
            f"expected={list(range(expected_world_size))}, "
            f"observed={observed_ranks}."
        )
    logger.info(
        "Distributed topology verified: world_size=%d ranks=%s",
        actual_world_size,
        observed_ranks,
    )
    return True, rank, actual_world_size, local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    dataset_specs: list[dict[str, Any]] = []
    cache_dirs: list[Path] = []
    context_lens = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        cache_dir_path = Path(str(cache_dir)).expanduser().resolve()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)

        context_len = node.get("context_len")
        if context_len is None:
            raise ValueError(
                f"Missing `context_len` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )
        context_len = int(context_len)
        context_lens.add(context_len)

        processor = node.get("processor") or {}
        dataset_specs.append(
            {
                "node_path": node_path,
                "dataset_dirs": tuple(str(ds) for ds in raw_dirs),
                "cache_dir": cache_dir_path,
                "context_len": context_len,
                "episode_selection": node.get("episode_selection"),
                "drop_high_level_prob": float(processor.get("drop_high_level_prob", 1.0)),
                "use_zh_instruction": bool(processor.get("use_zh_instruction", False)),
            }
        )

        logger.info(
            "Discovered dataset node `%s` with %d dataset_dirs.",
            node_path,
            len(raw_dirs),
        )

    return dataset_specs, cache_dirs, context_lens


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _canonical_path(value: str | Path) -> str:
    return str(Path(str(value)).expanduser().resolve())


def _jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Value is not JSON-serializable: {type(value).__name__}")


def _json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _selection_config_fingerprint(
    dataset_specs: list[dict[str, Any]],
    override_prompt: str | None,
) -> str:
    datasets = []
    for spec in sorted(dataset_specs, key=lambda item: item["node_path"]):
        selection = _jsonable(spec["episode_selection"])
        if selection is not None:
            if not isinstance(selection, dict):
                raise ValueError(f"episode_selection at {spec['node_path']} must be a mapping.")
            if "manifest_path" in selection:
                selection["manifest_path"] = _canonical_path(selection["manifest_path"])
        datasets.append(
            {
                "node_path": spec["node_path"],
                "dataset_dirs": [_canonical_path(path) for path in spec["dataset_dirs"]],
                "cache_dir": _canonical_path(spec["cache_dir"]),
                "context_len": spec["context_len"],
                "episode_selection": selection,
                "drop_high_level_prob": spec["drop_high_level_prob"],
                "use_zh_instruction": spec["use_zh_instruction"],
            }
        )
    return _json_sha256(
        {
            "datasets": datasets,
            "override_prompt": override_prompt,
        }
    )


def _format_task_prompt(task: str, use_zh_instruction: bool) -> str:
    if "@" in task:
        parts = task.split("@")
        if len(parts) != 2:
            raise ValueError(f"Ambiguous bilingual instruction: {task!r}.")
        task = parts[0] if use_zh_instruction else parts[1]
    return DEFAULT_PROMPT.format(task=task)


def _read_all_tasks_prompts(spec: dict[str, Any]) -> list[str]:
    if spec["drop_high_level_prob"] != 1.0:
        raise ValueError(
            "Prompt inventory currently requires drop_high_level_prob=1.0 so "
            "instruction construction is deterministic."
        )

    dataset_dirs = spec["dataset_dirs"]
    prompts: list[str] = []
    seen = set()
    total_task_references = 0
    episode_catalogs = 0

    for ds_dir in dataset_dirs:
        meta_dir = Path(ds_dir) / "meta"
        episodes_path = meta_dir / "episodes.jsonl"
        tasks_path = meta_dir / "tasks.jsonl"
        task_strings: list[str] = []

        # LeRobot's episode catalog is the authoritative set of instructions
        # actually referenced by samples. Some datasets retain thousands of
        # unused aliases / annotations in tasks.jsonl; encoding those wastes GPU
        # time and storage without preventing a runtime cache miss.
        if episodes_path.is_file():
            episode_catalogs += 1
            with episodes_path.open("r", encoding="utf-8") as handle:
                for line_idx, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    tasks = record.get("tasks")
                    if (
                        not isinstance(tasks, list)
                        or not tasks
                        or any(not isinstance(task, str) or not task for task in tasks)
                    ):
                        raise ValueError(
                            f"`tasks` must be a non-empty list[str] at {episodes_path}:{line_idx}"
                        )
                    task_strings.extend(tasks)
        else:
            if not tasks_path.exists():
                raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
            with tasks_path.open("r", encoding="utf-8") as handle:
                for line_idx, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "task" not in record:
                        raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                    task_strings.append(str(record["task"]))

        for task in task_strings:
            prompt = _format_task_prompt(task, spec["use_zh_instruction"])
            total_task_references += 1
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)

    logger.info(
        "Loaded %d referenced tasks from %d datasets (%d episode catalogs), "
        "deduplicated to %d prompts.",
        total_task_references,
        len(dataset_dirs),
        episode_catalogs,
        len(prompts),
    )
    return prompts


def _selected_task_indices(dataset_root: Path, selection: Any) -> set[int]:
    import pyarrow.parquet as pq

    resolved = resolve_episode_selection(dataset_root, selection)
    with (dataset_root / "meta/info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    with (dataset_root / "meta/episodes.jsonl").open("r", encoding="utf-8") as handle:
        episode_lengths = {
            int(record["episode_index"]): int(record["length"])
            for record in (json.loads(line) for line in handle if line.strip())
        }

    task_indices: set[int] = set()
    for episode_index in tqdm(
        resolved.episode_indices,
        desc="Scanning selected prompt indices",
        unit="episode",
    ):
        chunk = episode_index // int(info["chunks_size"])
        relative_path = str(info["data_path"]).format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        table = pq.ParquetFile(dataset_root / relative_path).read(
            columns=["episode_index", "task_index"]
        )
        if table.num_rows != episode_lengths[episode_index]:
            raise ValueError(
                f"Episode {episode_index} row-count mismatch while collecting prompts."
            )
        episode_values = {
            int(value)
            for value in table["episode_index"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .tolist()
        }
        if episode_values != {episode_index}:
            raise ValueError(
                f"Episode {episode_index} contains inconsistent episode_index values: "
                f"{sorted(episode_values)}."
            )
        task_indices.update(
            int(value)
            for value in table["task_index"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .tolist()
        )
    return task_indices


def _read_selected_prompts(spec: dict[str, Any]) -> list[str]:
    dataset_dirs = spec["dataset_dirs"]
    if len(dataset_dirs) != 1:
        raise ValueError(f"Selected prompt scan at {spec['node_path']} requires one dataset root.")
    if spec["drop_high_level_prob"] != 1.0:
        raise ValueError(
            "Selected prompt scan currently requires drop_high_level_prob=1.0 "
            "so instruction construction is deterministic."
        )

    dataset_root = Path(dataset_dirs[0]).expanduser().resolve()
    needed_indices = _selected_task_indices(dataset_root, spec["episode_selection"])
    prompts: list[str] = []
    found_indices: set[int] = set()
    tasks_path = dataset_root / "meta/tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_index = int(record["task_index"])
            if task_index not in needed_indices:
                continue
            if task_index in found_indices:
                raise ValueError(
                    f"Duplicate task_index={task_index} at {tasks_path}:{line_number}."
                )
            found_indices.add(task_index)
            task = str(record["task"])
            try:
                prompt = _format_task_prompt(task, spec["use_zh_instruction"])
            except ValueError as exc:
                raise ValueError(f"Task {task_index}: {exc}") from exc
            prompts.append(prompt)

    missing = needed_indices - found_indices
    if missing:
        raise ValueError(
            f"Selected episodes reference {len(missing)} unknown task indices; "
            f"first={sorted(missing)[:10]}."
        )
    return prompts


def _collect_prompt_targets(
    dataset_specs: list[dict[str, Any]],
    override_prompt: str | None = None,
) -> dict[str, tuple[Path, ...]]:
    prompt_targets: dict[str, set[Path]] = {}
    for spec in dataset_specs:
        if override_prompt is not None:
            spec_prompts = [override_prompt]
        elif spec["episode_selection"] is None:
            spec_prompts = _read_all_tasks_prompts(spec)
        else:
            spec_prompts = _read_selected_prompts(spec)
        for prompt in spec_prompts:
            prompt_targets.setdefault(prompt, set()).add(spec["cache_dir"])

    ordered_targets = {
        prompt: tuple(sorted(targets, key=str))
        for prompt, targets in sorted(prompt_targets.items())
    }
    logger.info(
        "Collected %d unique prompts with exact cache targets from %d dataset nodes.",
        len(ordered_targets),
        len(dataset_specs),
    )
    return ordered_targets


def _build_cache_inventories(
    prompt_targets: dict[str, tuple[Path, ...]],
    cache_dirs: list[Path],
) -> dict[Path, tuple[str, ...]]:
    inventories: dict[Path, list[str]] = {cache_dir: [] for cache_dir in cache_dirs}
    for prompt, cache_dirs in prompt_targets.items():
        for cache_dir in cache_dirs:
            inventories.setdefault(cache_dir, []).append(prompt)
    return {
        cache_dir: tuple(sorted(prompts))
        for cache_dir, prompts in sorted(inventories.items(), key=lambda item: str(item[0]))
    }


def _is_valid_cache_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _inventory_sha256(prompts: tuple[str, ...], context_len: int) -> str:
    return _json_sha256(
        {
            "cache_tag": TEXT_CACHE_TAG,
            "context_len": context_len,
            "prompts": prompts,
        }
    )


def _get_inventory_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(_canonical_path(str(value)))


def _atomic_json_save(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with tmp_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_prompt_inventory(
    inventory_path: Path,
    prompt_targets: dict[str, tuple[Path, ...]],
    cache_dirs: list[Path],
    context_len: int,
    selection_fingerprint: str,
) -> str:
    canonical_cache_dirs = sorted({_canonical_path(path) for path in cache_dirs})
    allowed_cache_dirs = set(canonical_cache_dirs)
    prompt_entries = []
    for prompt, targets in sorted(prompt_targets.items()):
        target_cache_dirs = sorted({_canonical_path(path) for path in targets})
        if not target_cache_dirs:
            raise ValueError(f"Prompt has no target cache directories: {prompt!r}")
        unexpected = set(target_cache_dirs) - allowed_cache_dirs
        if unexpected:
            raise ValueError(f"Prompt targets caches outside the data config: {sorted(unexpected)}")
        prompt_entries.append(
            {
                "prompt": prompt,
                "target_cache_dirs": target_cache_dirs,
            }
        )

    body = {
        "schema": TEXT_CACHE_INVENTORY_SCHEMA,
        "inventory_path": _canonical_path(inventory_path),
        "cache_tag": TEXT_CACHE_TAG,
        "context_len": context_len,
        "cache_dirs": canonical_cache_dirs,
        "selection_fingerprint": selection_fingerprint,
        "prompts": prompt_entries,
    }
    inventory_sha256 = _json_sha256(body)
    _atomic_json_save(
        {**body, "inventory_sha256": inventory_sha256},
        inventory_path,
    )
    logger.info(
        "Wrote prompt inventory atomically: path=%s prompts=%d sha256=%s",
        inventory_path,
        len(prompt_entries),
        inventory_sha256,
    )
    return inventory_sha256


def _load_prompt_inventory(
    inventory_path: Path,
    cache_dirs: list[Path],
    context_len: int,
    selection_fingerprint: str,
) -> dict[str, tuple[Path, ...]]:
    if not _is_valid_cache_file(inventory_path):
        raise FileNotFoundError(f"Missing or empty prompt inventory: {inventory_path}")
    try:
        with inventory_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prompt inventory JSON: {inventory_path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Prompt inventory must be a JSON object: {inventory_path}")

    expected_keys = {
        "schema",
        "inventory_path",
        "cache_tag",
        "context_len",
        "cache_dirs",
        "selection_fingerprint",
        "prompts",
        "inventory_sha256",
    }
    if set(document) != expected_keys:
        raise ValueError(
            f"Prompt inventory fields changed at {inventory_path}: "
            f"expected={sorted(expected_keys)}, actual={sorted(document)}"
        )
    if document["schema"] != TEXT_CACHE_INVENTORY_SCHEMA:
        raise ValueError(
            f"Prompt inventory schema mismatch at {inventory_path}: {document['schema']!r}"
        )
    expected_inventory_path = _canonical_path(inventory_path)
    if document["inventory_path"] != expected_inventory_path:
        raise ValueError(
            f"Prompt inventory path mismatch: expected={expected_inventory_path}, "
            f"actual={document['inventory_path']!r}"
        )
    if document["cache_tag"] != TEXT_CACHE_TAG:
        raise ValueError(
            f"Prompt inventory cache tag mismatch: expected={TEXT_CACHE_TAG!r}, "
            f"actual={document['cache_tag']!r}"
        )
    if type(document["context_len"]) is not int or document["context_len"] != context_len:
        raise ValueError(
            f"Prompt inventory context_len mismatch: expected={context_len}, "
            f"actual={document['context_len']!r}"
        )

    expected_sha256 = document["inventory_sha256"]
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("Prompt inventory has an invalid inventory_sha256 value.")
    body = {key: value for key, value in document.items() if key != "inventory_sha256"}
    actual_sha256 = _json_sha256(body)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Prompt inventory SHA-256 mismatch at {inventory_path}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    if document["selection_fingerprint"] != selection_fingerprint:
        raise ValueError(
            "Prompt inventory selection fingerprint does not match the current data "
            f"config: expected={selection_fingerprint}, "
            f"actual={document['selection_fingerprint']!r}"
        )

    raw_cache_dirs = document["cache_dirs"]
    expected_cache_dirs = sorted({_canonical_path(path) for path in cache_dirs})
    if (
        not isinstance(raw_cache_dirs, list)
        or any(not isinstance(path, str) for path in raw_cache_dirs)
        or raw_cache_dirs != sorted(set(raw_cache_dirs))
        or any(_canonical_path(path) != path for path in raw_cache_dirs)
        or raw_cache_dirs != expected_cache_dirs
    ):
        raise ValueError(
            "Prompt inventory cache paths do not match the current data config: "
            f"expected={expected_cache_dirs}, actual={raw_cache_dirs!r}"
        )

    raw_prompts = document["prompts"]
    if not isinstance(raw_prompts, list):
        raise ValueError("Prompt inventory `prompts` must be a list.")
    allowed_cache_dirs = set(expected_cache_dirs)
    prompt_targets: dict[str, tuple[Path, ...]] = {}
    for index, entry in enumerate(raw_prompts):
        if not isinstance(entry, dict) or set(entry) != {
            "prompt",
            "target_cache_dirs",
        }:
            raise ValueError(f"Invalid prompt inventory entry at index {index}.")
        prompt = entry["prompt"]
        targets = entry["target_cache_dirs"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Invalid prompt value at inventory index {index}.")
        if prompt in prompt_targets:
            raise ValueError(f"Duplicate prompt in inventory: {prompt!r}")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(path, str) for path in targets)
            or targets != sorted(set(targets))
            or any(_canonical_path(path) != path for path in targets)
            or not set(targets).issubset(allowed_cache_dirs)
        ):
            raise ValueError(f"Invalid target cache paths for prompt at inventory index {index}.")
        prompt_targets[prompt] = tuple(Path(path) for path in targets)

    if list(prompt_targets) != sorted(prompt_targets):
        raise ValueError("Prompt inventory entries are not in canonical prompt order.")
    logger.info(
        "Loaded validated prompt inventory: path=%s prompts=%d sha256=%s",
        inventory_path,
        len(prompt_targets),
        expected_sha256,
    )
    return prompt_targets


def _inspect_cache_inventories(
    inventories: dict[Path, tuple[str, ...]],
    context_len: int,
) -> dict[Path, tuple[Path, ...]]:
    missing_by_cache: dict[Path, tuple[Path, ...]] = {}
    for cache_dir, prompts in inventories.items():
        missing = tuple(
            path
            for prompt in prompts
            if not _is_valid_cache_file(
                path := Path(get_text_cache_path(cache_dir, prompt, context_len))
            )
        )
        missing_by_cache[cache_dir] = missing
        logger.info(
            "Prompt inventory: prompts=%d cached=%d missing=%d sha256=%s cache=%s",
            len(prompts),
            len(prompts) - len(missing),
            len(missing),
            _inventory_sha256(prompts, context_len),
            cache_dir,
        )
    return missing_by_cache


def _get_override_prompt(override_instruction: Any) -> str | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return DEFAULT_PROMPT.format(task=task)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _run(cfg: DictConfig) -> None:
    text_cache_mode = str(cfg.get("text_cache_mode", "encode")).strip().lower()
    if text_cache_mode not in {"encode", "plan", "verify"}:
        raise ValueError(
            f"Unsupported text_cache_mode={text_cache_mode!r}; "
            "expected one of: encode, plan, verify."
        )

    data_cfg = cfg.get("data")
    if data_cfg is None:
        raise ValueError("`cfg.data` is required.")

    dataset_specs, cache_dirs, context_lens = _collect_dataset_settings(data_cfg)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")
    if not dataset_specs:
        raise ValueError("No dataset nodes found under `cfg.data`.")

    context_len = _resolve_context_len(context_lens)
    override_prompt = _get_override_prompt(cfg.get("override_instruction"))
    if override_prompt is not None:
        logger.info("Using override_instruction; each dataset cache receives exactly one prompt.")

    selection_fingerprint = _selection_config_fingerprint(dataset_specs, override_prompt)
    inventory_path = _get_inventory_path(cfg.get("text_cache_inventory_path"))
    if inventory_path is not None and text_cache_mode != "plan":
        prompt_targets = _load_prompt_inventory(
            inventory_path,
            cache_dirs,
            context_len,
            selection_fingerprint,
        )
        logger.info("Skipping dataset prompt scan because a validated inventory was loaded.")
    else:
        prompt_targets = _collect_prompt_targets(dataset_specs, override_prompt)

    if text_cache_mode == "plan" and inventory_path is not None:
        _write_prompt_inventory(
            inventory_path,
            prompt_targets,
            cache_dirs,
            context_len,
            selection_fingerprint,
        )

    inventories = _build_cache_inventories(prompt_targets, cache_dirs)

    if text_cache_mode == "plan":
        _inspect_cache_inventories(inventories, context_len)
        return
    if text_cache_mode == "verify":
        missing_by_cache = _inspect_cache_inventories(inventories, context_len)
        missing_count = sum(len(paths) for paths in missing_by_cache.values())
        if missing_count:
            details = "; ".join(
                f"cache={cache_dir} missing={len(paths)}/{len(inventories[cache_dir])} "
                f"first={list(paths[:5])}"
                for cache_dir, paths in missing_by_cache.items()
                if paths
            )
            raise FileNotFoundError(
                f"Text caches are missing {missing_count} target files: {details}."
            )
        return

    if not prompt_targets:
        logger.warning("No prompts found from tasks.jsonl; nothing to encode.")
        return

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("`cfg.model` is required in encode mode.")
    overwrite = _to_bool(cfg.get("overwrite", True))

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if not is_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone "
            "--nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    enc_id = _model_id_to_enc_id(model_id)
    if enc_id != TEXT_CACHE_TAG:
        raise ValueError(
            f"Text encoder cache tag mismatch: model resolves to {enc_id!r}, "
            f"loader expects {TEXT_CACHE_TAG!r}."
        )

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    cache_dirs = list(inventories)
    stats = {str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0} for cache_dir in cache_dirs}

    prompts = list(prompt_targets)
    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            target_cache_dirs = prompt_targets[prompt]
            fully_cached = all(
                _is_valid_cache_file(Path(get_text_cache_path(cache_dir, prompt, context_len)))
                for cache_dir in target_cache_dirs
            )
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in target_cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = (
                torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            )
            count_tensor = torch.tensor(
                [fully_cached_local, len(prompts)],
                device=reduce_device,
                dtype=torch.long,
            )
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})"
        if is_distributed
        else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    context_i = (
                        context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    )
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    for cache_dir in prompt_targets[prompt]:
                        cache_path = Path(get_text_cache_path(cache_dir, prompt, context_len))
                        key = str(cache_dir)
                        if _is_valid_cache_file(cache_path) and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [
                    stats[str(cache_dir)]["new"],
                    stats[str(cache_dir)]["overwrite"],
                    stats[str(cache_dir)]["skip"],
                ]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@hydra.main(
    config_path=str(
        __import__("longwam.paths", fromlist=["repository_root"]).repository_root() / "configs"
    ),
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig):
    from longwam.paths import setup_model_paths

    setup_model_paths()
    setup_logging(log_level=logging.INFO)
    _run(cfg)


if __name__ == "__main__":
    main()
