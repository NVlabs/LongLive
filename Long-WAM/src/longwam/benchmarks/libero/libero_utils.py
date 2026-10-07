# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti.
# SPDX-FileCopyrightText: Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/libero/libero_utils.py
# Source: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py
# Source: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/libero/libero_utils.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import time
import pathlib

import imageio
from PIL import Image, ImageDraw
import numpy as np
from .adapter import get_libero_image, quat2axisangle, invert_gripper_action
from libero.libero import get_libero_path
from longwam.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def get_libero_env(task, resolution, seed, env_num=1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    # Simulator rendering is optional when importing adapters or running diagnostics.
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def load_initial_states(task):
    """Load the installed simulator's NumPy initial states on modern PyTorch."""
    import torch
    path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, map_location="cpu", weights_only=False)




def save_rollout_video(
    rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24
):
    """Saves an MP4 replay of an episode."""
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):
    """Saves an MP4 comparison of ground-truth and predicted future frames for one replanning clip."""
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        if isinstance(gt_frame, dict):
            gt_images = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                gt_images.append(value_array)
            gt_image = np.concatenate(gt_images, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(pred_pil), np.array(gt_pil)], axis=0))
        )

    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path


def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = v > 0.5
    return np.asarray(bin_val, dtype=np.float32)
