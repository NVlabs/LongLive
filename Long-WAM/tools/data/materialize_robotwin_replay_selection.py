#!/usr/bin/env python3
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

"""Plan or verify the video files required by the mixed RoboTwin replay subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from longwam.datasets.lerobot.episode_selection import resolve_episode_selection
from longwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

from longwam.paths import repository_root
FW_ROOT = repository_root()
DEFAULT_TASK = "domino"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _load_robotwin_domain(task: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(FW_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    OmegaConf.resolve(cfg)
    return cfg.data.train.domains.robotwin


def _required_files(domain_cfg) -> tuple[dict, list[Path], list[str]]:
    dataset_root = Path(str(domain_cfg.dataset_dirs[0])).expanduser().resolve()
    selection = OmegaConf.to_container(domain_cfg.episode_selection, resolve=True)
    resolved = resolve_episode_selection(dataset_root, selection)
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    video_keys = sorted(
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    )
    if len(video_keys) != 3:
        raise ValueError(f"Expected three RoboTwin video keys, got {video_keys}.")

    video_paths: list[Path] = []
    archive_members: list[str] = []
    for episode_index in resolved.episode_indices:
        episode_chunk = episode_index // int(info["chunks_size"])
        data_relative = str(info["data_path"]).format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        data_path = dataset_root / data_relative
        if not data_path.is_file() or data_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Selected parquet is missing: {data_path}")
        for video_key in video_keys:
            relative = str(info["video_path"]).format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            video_paths.append(dataset_root / relative)
            archive_members.append(f"{dataset_root.name}/{relative}")

    manifest = json.loads(
        Path(str(selection["manifest_path"])).read_text(encoding="utf-8")
    )
    inventory = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "manifest_path": str(Path(str(selection["manifest_path"])).resolve()),
        "manifest_mapping_sha256": manifest["mapping_sha256"],
        "selection_signature": resolved.signature,
        "episode_count": len(resolved.episode_indices),
        "frame_count": sum(resolved.group_frame_counts),
        "video_count": len(video_paths),
        "archive_member_list_sha256": _sha256_lines(archive_members),
        "episode_indices": list(resolved.episode_indices),
    }
    return inventory, video_paths, archive_members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--member-list", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_cfg = _load_robotwin_domain(args.task)
    inventory, video_paths, archive_members = _required_files(domain_cfg)
    if args.verify:
        missing = [
            path
            for path in video_paths
            if not path.is_file() or path.stat().st_size <= 0
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)}/{len(video_paths)} selected videos; "
                f"first={missing[:5]}"
            )

    _atomic_write(args.member_list, "\n".join(archive_members) + "\n")
    _atomic_write(
        args.inventory,
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
    )
    status = "verified" if args.verify else "planned"
    print(
        f"{status}: episodes={inventory['episode_count']} "
        f"frames={inventory['frame_count']} videos={inventory['video_count']} "
        f"selection={inventory['selection_signature']}"
    )


if __name__ == "__main__":
    main()
