# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/robotwin/eval_robotwin_single.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Native RoboTwin/Domino launch helpers used by the common YAML evaluator."""
from pathlib import Path
from typing import Any

POLICY_NAME = "longwam_policy"


def _result_filename_for_task_config(task_config: str) -> str:
    filenames = {
        "demo_clean": "_result_clean.txt",
        "demo_randomized": "_result_random.txt",
        "demo_clean_dynamic": "_result.txt",
        "demo_random_dynamic": "_result.txt",
    }
    try:
        return filenames[str(task_config)]
    except KeyError as exc:
        raise ValueError(f"Unknown RoboTwin/DOMINO setting: {task_config!r}") from exc


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_root}")
    target = policy_root / POLICY_NAME
    source = policy_source_dir.resolve(strict=True)
    if not target.exists() and not target.is_symlink():
        target.symlink_to(source, target_is_directory=True)
    elif not target.is_symlink() or target.resolve() != source:
        raise RuntimeError(f"Policy path conflict: {target}; refusing to replace it")
    return target


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none=True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])
