# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/datasets/lerobot/robot_video_dataset.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import hashlib
import os
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .robocasa_contract import (
    ROBOCASA_LATENT_SLOT_LAYOUT,
    ROBOCASA_LEFT_MAIN_LAYOUT,
    ROBOCASA_RGB_LAYOUTS,
    RoboCasaCameraContractError,
    build_robocasa_camera_layout,
    validate_robocasa_latent_slot_camera_metadata,
    validate_robocasa_latent_slot_cameras,
    validate_robocasa_rgb_camera_metadata,
)
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from longwam.utils.logging_config import get_logger
from longwam.utils import misc
from accelerate import PartialState

logger = get_logger(__name__)


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
TEXT_CACHE_TAG = "wan22ti2v5b"


def get_text_cache_path(cache_dir, prompt: str, context_len: int):
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(
        cache_dir,
        f"{hashed}.t5_len{int(context_len)}.{TEXT_CACHE_TAG}.pt",
    )


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        past_obs_size: int = 0,
        action_chunk: Optional[
            int
        ] = None,  # P4: independent action horizon. None -> num_frames-1 (v0a/v0b default).
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",  # "horizontal", "vertical", "robotwin", named RoboCasa layouts, or None
        override_instruction: Optional[
            str
        ] = None,  # whether to hardcode a specific instruction for all samples, for debugging
        video_backend: Optional[str] = None,
        episode_selection=None,
        per_dataset_episode_selection=None,
        decode_only_sampled_video_frames: bool = False,
        fail_on_sample_error: bool = False,
        lerobot_dataset_cls=None,
    ):
        # P4: action_chunk DECOUPLES the action horizon from num_frames. Default (None) preserves the
        # v0a/v0b coupling action_size=num_frames-1, so existing configs are byte-identical. The action
        # delta_timestamps in BaseLerobotDataset are range(-past_action_size, -past_action_size+action_size)
        # with past_action_size==0 -> range(0, action_size): the chunk ALWAYS starts at the CURRENT obs
        # (t=0) and spans the near-future the policy executes. This is INDEPENDENT of past_obs_size (which
        # only shifts the OBS/video window into the past), so the action<->current-obs alignment is preserved.
        self.action_chunk = (num_frames - 1) if action_chunk is None else int(action_chunk)
        if self.action_chunk <= 0:
            raise ValueError(f"`action_chunk` must be positive, got {self.action_chunk}")

        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {self.action_video_freq_ratio}."
            )
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, got "
                f"{self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "video frames must be divisible by 4 for tokenization, got "
                f"{(self.num_frames - 1) // self.action_video_freq_ratio}"
            )
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.decode_only_sampled_video_frames = bool(decode_only_sampled_video_frames)
        self.fail_on_sample_error = bool(fail_on_sample_error)

        shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        if concat_multi_camera == ROBOCASA_LATENT_SLOT_LAYOUT:
            validate_robocasa_latent_slot_camera_metadata(shape_meta["images"])
        elif concat_multi_camera == ROBOCASA_LEFT_MAIN_LAYOUT:
            validate_robocasa_rgb_camera_metadata(shape_meta["images"])

        if lerobot_dataset_cls is None:
            lerobot_dataset_cls = BaseLerobotDataset
        self.lerobot_dataset = lerobot_dataset_cls(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            obs_size=num_frames,
            past_obs_size=past_obs_size,  # v0b: >0 shifts obs window to [-past_obs_size .. -past_obs_size+num_frames-1]
            action_size=self.action_chunk,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            video_backend=video_backend,
            episode_selection=episode_selection,
            per_dataset_episode_selection=per_dataset_episode_selection,
            image_sample_indices=(
                self.video_sample_indices if self.decode_only_sampled_video_frames else None
            ),
            fail_on_sample_error=self.fail_on_sample_error,
        )

        self.past_obs_size = past_obs_size

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    @property
    def dataset_frame_ranges(self) -> tuple[tuple[int, int], ...]:
        """Stable half-open frame ranges for each source dataset."""
        return self.lerobot_dataset.dataset_frame_ranges

    @property
    def dataset_group_names(self) -> tuple[str, ...]:
        return self.lerobot_dataset.dataset_group_names

    @property
    def sampling_signature(self) -> str:
        return self.lerobot_dataset.sampling_signature

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.decode_only_sampled_video_frames:
                video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            if not self.decode_only_sampled_video_frames:
                video = video[self.video_sample_indices, :, :, :]
            T_video, C, H, W = video.shape
        if self.decode_only_sampled_video_frames:
            expected_video_frames = len(self.video_sample_indices)
            if T_video != expected_video_frames:
                raise ValueError(
                    "Sampled-video query returned an unexpected temporal length: "
                    f"expected {expected_video_frames}, got {T_video}."
                )
            if image_is_pad.shape[0] != expected_video_frames:
                raise ValueError(
                    "Sampled-video padding length mismatch: "
                    f"expected {expected_video_frames}, got {image_is_pad.shape[0]}."
                )
        else:
            image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        latent_slot_mode = self.concat_multi_camera == ROBOCASA_LATENT_SLOT_LAYOUT
        if latent_slot_mode:
            validate_robocasa_latent_slot_cameras(video)
        elif self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif self.concat_multi_camera in ROBOCASA_RGB_LAYOUTS:
            video = build_robocasa_camera_layout(
                video,
                layout=self.concat_multi_camera,
            )
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat(
                    [video[i] for i in range(num_cameras)], dim=-1
                )  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat(
                    [video[i] for i in range(num_cameras)], dim=-2
                )  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin, robocasa, "
                    "robocasa_left_main, robocasa_latent_slots."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        if latent_slot_mode:
            # Keep every source camera at its native 256x256 resolution. The
            # model encodes each view independently before concatenating VAE
            # latents; no RGB resize, crop, or mosaic is permitted here.
            video = self.normalize_transform(video)
            video = video.permute(0, 2, 1, 3, 4)
            video_frame_count = video.shape[2]
        else:
            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)  # [T_video, C, H, W]
            video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W]
            video_frame_count = video.shape[1]

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"]  # [action_chunk, action_dim], starts at the CURRENT obs (t=0)
        # proprio comes from the OBS stream (length num_frames, == obs_size), independent of
        # action_chunk; drop the trailing frame to length num_frames-1. build_inputs indexes the
        # CURRENT obs at proprio[:, current_obs_idx]=proprio[:, past_obs_size].
        proprio = sample["proprio"][:-1, :]  # [num_frames-1, state_dim]
        proprio_is_pad = sample["proprio_is_pad"][:-1]
        if proprio_is_pad.shape[0] != proprio.shape[0]:
            raise ValueError("Proprioception and its padding mask have different lengths")
        if video_frame_count <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        # P4: with action_chunk DECOUPLED from num_frames the action horizon need not be a multiple of
        # the video transitions. The per-video-frame action grouping only exists in the video DiT's
        # `action_conditioned=true` path (LongWAM uses action_conditioned=false; action<->video coupling is
        # purely the MoT attention mask). Only enforce divisibility when action_chunk == num_frames-1,
        # i.e. the v0a/v0b coupled default, so those configs keep their original guard byte-for-byte.
        if (
            self.action_chunk == (self.num_frames - 1)
            and action.shape[0] % (video_frame_count - 1) != 0
        ):
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video_frame_count - 1}"
            )

        task = sample["instruction"]

        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": proprio_is_pad,
        }
        from .processors.longwam_processor import SAMPLE_IDENTITY_KEYS

        for key in SAMPLE_IDENTITY_KEYS:
            if key in sample:
                data[key] = sample[key]
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_text_cache_path(cache_dir, prompt, self.context_len)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except RoboCasaCameraContractError:
            raise
        except Exception as e:
            if self.fail_on_sample_error:
                raise
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
