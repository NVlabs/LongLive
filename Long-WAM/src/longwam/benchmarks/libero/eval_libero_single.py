# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/libero/eval_libero_single.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/libero/eval_libero_single.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from longwam.benchmarks.libero.eval_latency import summarize_policy_call_latency
from longwam.benchmarks.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from longwam.datasets.lerobot.processors.longwam_processor import LongWAMProcessor
from longwam.utils.pytorch_utils import set_global_seed
from longwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark


from longwam.runtime.libero import (
    _center_crop_resize,
    _denormalize_action,
    _extract_sim_state,
    _get_action_video_freq_ratio,
    _get_num_video_frames,
    _get_past_obs_size,
    _maybe_inject_video_lora,
    _normalize_proprio,
    _obs_to_model_input,
    _predict_action_chunk as _predict_runtime_actions,
)

from longwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from longwam.runtime.policy import build_observation_window as _build_obs_window


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )

def _select_predicted_future_frames(
    pred_video: list[Image.Image], cfg: DictConfig
) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])

def _predict_action_chunk(obs, instruction, model, processor, cfg, *, action_horizon,
                          input_w, input_h, model_device, obs_frame_buffer=None,
                          policy_call_seconds=None, clean_latents=None, image_tensor=None):
    """Optional benchmark video diagnostics and synchronized timing."""
    images = get_libero_image(obs)
    is_cuda = torch.device(model_device).type == "cuda"
    measure = policy_call_seconds is not None
    if measure and is_cuda:
        torch.cuda.synchronize(model_device)
    start = time.perf_counter()
    future = None
    if cfg.EVALUATION.get("visualize_future_video", False):
        if _get_past_obs_size(cfg):
            raise ValueError("Future-video visualization requires a single-frame checkpoint")
        image = image_tensor if image_tensor is not None else _obs_to_model_input(
            obs, cfg, processor, input_w, input_h, model_device, model.torch_dtype
        )[0]
        pred = model.infer_joint(
            prompt=DEFAULT_PROMPT.format(task=instruction), input_image=image,
            proprio=_normalize_proprio(_extract_sim_state(obs), processor),
            action_horizon=action_horizon, num_video_frames=_get_num_video_frames(cfg),
            num_inference_steps=int(cfg.EVALUATION.get("num_inference_steps", 20)),
            text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1)),
            seed=cfg.get("seed"), tiled=bool(cfg.EVALUATION.get("tiled", False)),
        )
        actions = _denormalize_action(pred["action"], processor)[0]
        actions[..., -1] = 1 - 2 * actions[..., -1]
        if cfg.EVALUATION.get("binarize_gripper", False):
            actions[..., -1] = np.sign(actions[..., -1])
        future = _select_predicted_future_frames(pred["video"], cfg)
    else:
        actions = _predict_runtime_actions(
            obs, instruction, model, processor, cfg, action_horizon=action_horizon,
            input_w=input_w, input_h=input_h, model_device=model_device,
            obs_frame_buffer=obs_frame_buffer, clean_latents=clean_latents,
            image_tensor=image_tensor,
        )
    if measure:
        if is_cuda:
            torch.cuda.synchronize(model_device)
        policy_call_seconds.append(time.perf_counter() - start)
    return actions, images, future


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = (
                np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            )
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: LongWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    max_steps_override: Optional[int] = None,
    record_replay: bool = True,
    show_progress: bool = True,
    policy_call_seconds: Optional[list[float]] = None,
    policy=None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = (
        _get_max_steps(cfg.EVALUATION.task_suite_name)
        if max_steps_override is None
        else int(max_steps_override)
    )
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        raise ValueError("The unified runtime uses unblended action execution")

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    diagnostic = {}
    if policy is None:
        # Reuse the shared executor with the caller's already loaded model.
        from longwam.runtime.policy import initialize_policy
        from longwam.runtime.execution import SyncPolicy, SyncSchedule
        from longwam.runtime.worker import _InlineInferenceWorker

        def image(current_obs):
            return _obs_to_model_input(current_obs, cfg, processor, input_w, input_h,
                                       model_device, model.torch_dtype)[0][0]

        def infer(current_obs, instruction, frames, clean):
            actions, images, future = _predict_action_chunk(
                current_obs, instruction, model, processor, cfg,
                action_horizon=action_horizon, input_w=input_w, input_h=input_h,
                model_device=model_device, obs_frame_buffer=frames,
                policy_call_seconds=policy_call_seconds, clean_latents=clean,
                image_tensor=frames[-1].unsqueeze(0),
            )
            diagnostic.update(images=images, future=future)
            return actions

        handler, _ = initialize_policy({
            "execution": "sync", "execute_steps": replan_steps,
            "streaming_vae": False, "hardware": "reference", "optimization": {},
        }, components=(cfg, model, image, infer))
        policy = SyncPolicy(_InlineInferenceWorker(handler),
                            SyncSchedule(action_horizon, replan_steps))
        policy.reset(task_description)
        select = policy.act
    else:
        if visualize_future_video:
            raise ValueError("Future-video diagnostics require an already loaded model")
        from .adapter import policy_batch
        policy.reset()
        select = lambda current_obs: policy.select_action(
            policy_batch(current_obs, task_description)
        )[0].numpy()

    t = 0
    done = False
    pbar = tqdm(
        total=max_steps + num_steps_wait,
        desc=f"Episode {episode_idx + 1}",
        disable=not show_progress,
    )
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        previous_requests = policy.metrics["requests"]
        action = select(obs)
        imgs = get_libero_image(obs)
        if policy.metrics["requests"] != previous_requests:
            predicted_future_frames = diagnostic.get("future")
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [diagnostic["images"].copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
        if record_replay:
            replay_images.append(imgs.copy())
        obs, _, done, _ = env.step(action)
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or current_replan_step == replan_steps:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip[
                    "pred_frames"
                ][:expected_frame_count]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: LongWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    policy=None,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    record_replay = bool(cfg.EVALUATION.get("record_replay", True))
    show_progress = bool(cfg.EVALUATION.get("show_progress", True))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    policy_call_seconds: list[float] = []
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr = (
            run_single_episode(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                record_replay=record_replay,
                show_progress=show_progress,
                policy_call_seconds=policy_call_seconds,
                policy=policy,
            )
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        if record_replay:
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    env.close()
    latency_warmup_calls = int(cfg.EVALUATION.get("latency_warmup_calls", 1))
    results["policy_call_latency"] = summarize_policy_call_latency(
        policy_call_seconds,
        warmup_calls=latency_warmup_calls,
    )
    return results
