# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/datasets/lerobot/base_lerobot_dataset.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import math
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, DefaultDict, Sequence

import numpy as np
import torch
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from .episode_selection import (
    ResolvedPerDatasetEpisodeSelection,
    build_group_ranges,
    resolve_episode_selection,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from longwam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def build_resampled_frame_offsets(
    *,
    start: int,
    size: int,
    sample_stride: float,
) -> list[int]:
    """Map a logical window to nearest source-frame offsets.

    Quantizing once in frame space keeps video, state, and action queries on the
    same source samples even when ``sample_stride`` is fractional (for example,
    5/3 when resampling 16.67 Hz demonstrations to 10 Hz).
    """
    stride = float(sample_stride)
    if not math.isfinite(stride) or stride <= 0:
        raise ValueError(f"`global_sample_stride` must be finite and positive, got {sample_stride}")
    if size <= 0:
        raise ValueError(f"Window size must be positive, got {size}")

    offsets = [round(logical_offset * stride) for logical_offset in range(start, start + size)]
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError(
            "`global_sample_stride` must produce strictly increasing source-frame offsets, "
            f"got stride={sample_stride} and offsets={offsets}"
        )
    return offsets


class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],
        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,  # Excludes the current frame
        obs_size: int = 1,  # should be
        past_obs_size: int = 0,
        # train vs val
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        # sampling
        global_sample_stride: float = 1,
        video_backend: Optional[str] = None,
        episode_selection: Optional[Dict[str, Any]] = None,
        per_dataset_episode_selection: Optional[ResolvedPerDatasetEpisodeSelection] = None,
        image_sample_indices: Optional[Sequence[int]] = None,
        fail_on_sample_error: bool = False,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert (
            past_obs_size >= 0
        )  # v0b: past_obs_size>0 shifts the obs window into the past (streaming clean memory)
        # P4: action_size (the action_chunk) is DECOUPLED from the obs window. It still starts at the
        # current obs (past_action_size==0 -> action deltas range(0, action_size)); it just need not span
        # the whole obs window. v0a/v0b use action_size == obs_size-1 (the coupled default, unchanged);
        # P4 allows a SHORTER chunk (1 <= action_size <= obs_size-1).
        assert 1 <= action_size <= obs_size - 1, (
            f"action_size must satisfy 1 <= action_size <= obs_size-1, got action_size={action_size}, obs_size={obs_size}"
        )

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.fail_on_sample_error = bool(fail_on_sample_error)
        self.processor = None  # Will be set externally
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]

        if episode_selection is not None and per_dataset_episode_selection is not None:
            raise ValueError(
                "Use either manifest `episode_selection` or "
                "`per_dataset_episode_selection`, not both."
            )

        resolved_selection = None
        if episode_selection is not None:
            if len(dataset_dirs) != 1:
                raise ValueError(
                    "`episode_selection` currently requires exactly one dataset directory."
                )
            if val_set_proportion >= 1e-6:
                raise ValueError(
                    "`episode_selection` requires val_set_proportion=0; use an explicit "
                    "manifest for any train/validation split."
                )
            resolved_selection = resolve_episode_selection(dataset_dirs[0], episode_selection)
        if per_dataset_episode_selection is not None:
            if val_set_proportion >= 1e-6:
                raise ValueError("`per_dataset_episode_selection` requires val_set_proportion=0.")
            if len(per_dataset_episode_selection.episode_indices) != len(dataset_dirs):
                raise ValueError(
                    "Per-dataset episode selection does not match the dataset "
                    f"inventory: selections="
                    f"{len(per_dataset_episode_selection.episode_indices)}, "
                    f"datasets={len(dataset_dirs)}."
                )
            for meta, indices in zip(
                metas,
                per_dataset_episode_selection.episode_indices,
                strict=True,
            ):
                if not indices or len(indices) != len(set(indices)):
                    raise ValueError(
                        "Every per-dataset episode allowlist must be non-empty and unique."
                    )
                if any(index < 0 or index >= meta.total_episodes for index in indices):
                    raise ValueError(
                        f"Episode allowlist for {meta.repo_id} exceeds its "
                        f"{meta.total_episodes}-episode inventory."
                    )

        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        obs_frame_offsets = build_resampled_frame_offsets(
            start=-past_obs_size,
            size=obs_size,
            sample_stride=global_sample_stride,
        )
        if image_sample_indices is None:
            image_frame_offsets = obs_frame_offsets
        else:
            image_sample_indices = tuple(int(index) for index in image_sample_indices)
            if not image_sample_indices:
                raise ValueError("`image_sample_indices` must not be empty.")
            if any(index < 0 or index >= obs_size for index in image_sample_indices):
                raise ValueError(
                    "`image_sample_indices` must be within the observation window, "
                    f"got indices={image_sample_indices} and obs_size={obs_size}."
                )
            if any(
                left >= right for left, right in zip(image_sample_indices, image_sample_indices[1:])
            ):
                raise ValueError(
                    "`image_sample_indices` must be strictly increasing, "
                    f"got {image_sample_indices}."
                )
            image_frame_offsets = [obs_frame_offsets[index] for index in image_sample_indices]
        action_frame_offsets = build_resampled_frame_offsets(
            start=-past_action_size,
            size=action_size,
            sample_stride=global_sample_stride,
        )
        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.images.{key}" if key != "default" else "observation.images"
            )
            delta_timestamps[meta["lerobot_key"]] = [offset / fps for offset in image_frame_offsets]

        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.state.{key}" if key != "default" else "observation.state"
            )
            delta_timestamps[meta["lerobot_key"]] = [offset / fps for offset in obs_frame_offsets]

        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [
                offset / fps for offset in action_frame_offsets
            ]

        episodes = {}
        if resolved_selection is not None:
            episodes[metas[0].repo_id] = list(resolved_selection.episode_indices)
        elif per_dataset_episode_selection is not None:
            episodes.update(
                {
                    meta.repo_id: list(indices)
                    for meta, indices in zip(
                        metas,
                        per_dataset_episode_selection.episode_indices,
                        strict=True,
                    )
                }
            )
        elif val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: list(range(meta.total_episodes))})
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                # random shuffle episode indices before splitting
                episode_indices = list(range(meta.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})
                else:
                    episodes.update(
                        {
                            meta.repo_id: [
                                episode_indices[i] for i in range(split_idx, meta.total_episodes)
                            ]
                        }
                    )

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
        if resolved_selection is not None:
            self._dataset_frame_ranges = build_group_ranges(resolved_selection.group_frame_counts)
            self.dataset_group_names = resolved_selection.group_names
            sampling_contract = {
                "episode_selection_signature": resolved_selection.signature,
                "ranges": self._dataset_frame_ranges,
            }
        elif per_dataset_episode_selection is not None:
            self._dataset_frame_ranges = build_group_ranges(
                per_dataset_episode_selection.dataset_frame_counts
            )
            self.dataset_group_names = tuple(Path(path).name for path in dataset_dirs)
            sampling_contract = {
                "per_dataset_episode_selection_signature": (
                    per_dataset_episode_selection.signature
                ),
                "ranges": self._dataset_frame_ranges,
            }
            if self._dataset_frame_ranges != self.multi_dataset.dataset_frame_ranges:
                raise ValueError(
                    "Selected episode metadata frame counts do not match the "
                    "constructed LeRobot datasets."
                )
        else:
            self._dataset_frame_ranges = self.multi_dataset.dataset_frame_ranges
            self.dataset_group_names = tuple(Path(path).name for path in dataset_dirs)
            sampling_contract = {
                "dataset_dirs": [str(Path(path).expanduser().resolve()) for path in dataset_dirs],
                "ranges": self._dataset_frame_ranges,
            }
        if self._dataset_frame_ranges[-1][1] != len(self.multi_dataset):
            raise ValueError(
                "Sampling group ranges do not cover the constructed dataset: "
                f"ranges={self._dataset_frame_ranges}, length={len(self.multi_dataset)}."
            )
        self.sampling_signature = hashlib.sha256(
            json.dumps(sampling_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key]  # [T, action_dim]
        if action.ndim == 1:  # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, (
            f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        )
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1:  # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        assert state.shape[-1] == raw_shape, (
            f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        )
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        lerobot_key = meta["lerobot_key"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3:  # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8)  # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image

    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample

    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    @property
    def dataset_frame_ranges(self) -> tuple[tuple[int, int], ...]:
        """Stable half-open frame ranges for each source dataset."""
        return getattr(
            self,
            "_dataset_frame_ranges",
            self.multi_dataset.dataset_frame_ranges,
        )

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        # Retry with random indices until we successfully load a frame.
        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                if self.fail_on_sample_error:
                    raise
                attempt += 1
                last_exception = err
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}."
            ) from last_exception

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(
                range(episodes_num), desc="Iterating dataset to get normalization"
            ):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key]  # (B, T, dim)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key]  # (B, T, dim)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]

                for future in tqdm(
                    as_completed(futures),
                    total=episodes_num,
                    desc="Iterating dataset to get normalization",
                ):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key]  # (B, T, dim)
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(
                                torch.quantile(cur_state, 0.01, dim=0, keepdim=False)
                            )
                            state_q99[key].append(
                                torch.quantile(cur_state, 0.99, dim=0, keepdim=False)
                            )

                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key]  # (B, T, dim)
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(
                                torch.quantile(cur_action, 0.01, dim=0, keepdim=False)
                            )
                            action_q99[key].append(
                                torch.quantile(cur_action, 0.99, dim=0, keepdim=False)
                            )

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {
            "state": DefaultDict(dict),
            "action": DefaultDict(dict),
            "num_episodes": episodes_num,
            "num_transition": self.multi_dataset.num_frames,
        }
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].

    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)

    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window

    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0

    N, D = x.shape

    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)  # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices  # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
