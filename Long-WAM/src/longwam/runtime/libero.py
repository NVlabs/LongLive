# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/libero.py
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/libero/eval_libero_single.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""LIBERO model preprocessing/prediction shared by diagnostics and control runtime.

Environment lifecycle, scene seeds and scoring stay in longwam.benchmarks.
"""
import inspect
import logging
import os
from typing import Optional
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from hydra.utils import instantiate
from longwam.datasets.lerobot.processors.longwam_processor import LongWAMProcessor
from longwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from longwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from longwam.benchmarks.libero.adapter import get_libero_image, quat2axisangle, invert_gripper_action
from .policy import build_observation_window as _build_obs_window

def _maybe_inject_video_lora(model: torch.nn.Module, cfg) -> None:
    """Merge an optional video LoRA after checkpoint loading."""
    lora_path = cfg.EVALUATION.get("video_lora_path", None)
    if not lora_path:
        return
    import torch as _torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    video = model.video_expert
    vdtype = next(video.parameters()).dtype

    # Collect target Linear module full-names inside each DiTBlock
    # (self_attn/cross_attn .q/.k/.v/.o and ffn.0/ffn.2) — mirrors LongLive's
    # configure_lora_for_model so the saved adapter keys map exactly.
    targets = []
    for name, module in video.named_modules():
        if module.__class__.__name__ == "DiTBlock":
            for sub_name, sub in module.named_modules(prefix=name):
                if isinstance(sub, _torch.nn.Linear):
                    targets.append(sub_name)
    if not targets:
        raise ValueError("No DiTBlock Linear targets found in video_expert for LoRA injection.")
    rank = int(cfg.EVALUATION.get("video_lora_rank", 128))
    alpha = int(cfg.EVALUATION.get("video_lora_alpha", 128))
    logging.info(
        "[video-LoRA] injecting rank=%d alpha=%d into %d Linear targets from %s",
        rank,
        alpha,
        len(targets),
        lora_path,
    )

    lcfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias="none"
    )
    peft_video = get_peft_model(video, lcfg)

    payload = _torch.load(
        os.path.expanduser(os.path.expandvars(str(lora_path))), map_location="cpu"
    )
    gen_lora = payload.get("generator_lora", payload) if isinstance(payload, dict) else payload
    gen_lora = {k: (v.to(vdtype) if hasattr(v, "to") else v) for k, v in gen_lora.items()}
    res = set_peft_model_state_dict(peft_video, gen_lora)
    missing = getattr(res, "missing_keys", None)
    unexpected = getattr(res, "unexpected_keys", None)
    logging.info(
        "[video-LoRA] load result missing=%s unexpected=%s",
        (len(missing) if missing is not None else "?"),
        (len(unexpected) if unexpected is not None else "?"),
    )

    merged = peft_video.merge_and_unload()
    model.video_expert = merged
    model.mot.mixtures["video"] = merged
    logging.info("[video-LoRA] merged LoRA into video expert weights.")

def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)

def _normalize_proprio(
    proprio: np.ndarray,
    processor: LongWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]

def _image_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: LongWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = obs["images"] if "images" in obs else get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(
            f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}."
        )

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    return x


def _obs_to_model_input(obs, cfg, processor, width, height, device, dtype):
    """Combined input helper for benchmark diagnostics."""
    image = _image_to_model_input(obs, cfg, processor, width, height, device, dtype)
    state = _normalize_proprio(_extract_sim_state(obs), processor)
    images = obs["images"] if "images" in obs else get_libero_image(obs)
    return image, state, images

def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    if "state" in obs:
        return np.asarray(obs["state"], dtype=np.float32)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state

def _denormalize_action(action: torch.Tensor, processor: LongWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()

def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

def _get_past_obs_size(cfg: DictConfig) -> int:
    """Pixel `past_obs_size` for the v0b obs window. 0 (absent) -> v0a (single current frame)."""
    return int(cfg.data.train.get("past_obs_size", 0) or 0)

def _get_action_video_freq_ratio(cfg: DictConfig) -> int:
    return int(cfg.data.train.get("action_video_freq_ratio", 1) or 1)

def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: LongWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    obs_frame_buffer: Optional["list[torch.Tensor]"] = None,
    clean_latents: Optional[torch.Tensor] = None,
    image_tensor: Optional[torch.Tensor] = None,
    text_conditioning=None,
) -> np.ndarray:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image = image_tensor if image_tensor is not None else _image_to_model_input(
        obs, cfg, processor, input_w, input_h, model_device, model.torch_dtype
    )
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    obs_window = None
    past = _get_past_obs_size(cfg)
    if clean_latents is None and past > 0:
        if not obs_frame_buffer:
            raise ValueError("History frames are required by this checkpoint")
        obs_window = _build_obs_window(obs_frame_buffer, past, _get_action_video_freq_ratio(cfg))

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if text_conditioning is not None:
        infer_kwargs.update(text_conditioning(prompt))
    if clean_latents is not None:
        infer_kwargs["clean_latents"] = clean_latents
    elif obs_window is not None:
        infer_kwargs["obs_window"] = obs_window
    if int(getattr(model, "num_imagine_frames", 0)) > 0:
        infer_fn = model.infer_joint_ar
    else:
        infer_fn = model.infer_action
        if "num_video_frames" in inspect.signature(infer_fn).parameters:
            infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    with torch.no_grad():
        pred = infer_fn(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action
