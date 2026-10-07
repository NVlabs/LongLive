# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/robotwin.py
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/robotwin/fastwam_policy/deploy_policy.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""RoboTwin checkpoint preprocessing and prediction; execution is owned by policy.py."""
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import inspect
import os
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from torchvision.transforms import InterpolationMode
from hydra.utils import instantiate
from longwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from longwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from .policy import build_observation_window as _build_obs_window

logger = logging.getLogger(__name__)

def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False

def _resize_training_camera(
    image: np.ndarray,
    *,
    processor_size_hw: tuple[int, int],
    layout_size_hw: tuple[int, int],
) -> torch.Tensor:
    """Apply the same two resize stages used by the RoboTwin training dataset."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image [H,W,3], got {tuple(image.shape)}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    tensor = transforms_F.resize(
        tensor,
        size=list(processor_size_hw),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return transforms_F.resize(
        tensor,
        size=list(layout_size_hw),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

def _maybe_inject_video_lora(model: torch.nn.Module) -> None:
    """Merge an optional video LoRA after checkpoint loading."""
    lora_path = os.environ.get("LONGWAM_VIDEO_LORA_PATH", "").strip()
    if not lora_path or lora_path.lower() in {"none", "null"}:
        return
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    video = model.video_expert
    vdtype = next(video.parameters()).dtype

    # Target every Linear inside each DiTBlock (self_attn/cross_attn .q/.k/.v/.o and
    # ffn.0/ffn.2) — mirrors LongLive's configure_lora_for_model so adapter keys map.
    targets = []
    for name, module in video.named_modules():
        if module.__class__.__name__ == "DiTBlock":
            for sub_name, sub in module.named_modules(prefix=name):
                if isinstance(sub, torch.nn.Linear):
                    targets.append(sub_name)
    if not targets:
        raise ValueError("No DiTBlock Linear targets found in video_expert for LoRA injection.")

    rank = int(os.environ.get("LONGWAM_VIDEO_LORA_RANK", "128"))
    alpha = int(os.environ.get("LONGWAM_VIDEO_LORA_ALPHA", "128"))
    # print() (not logger) so it is visible in RoboTwin's print-based eval stdout/log.
    print(
        f"[video-LoRA] injecting rank={rank} alpha={alpha} into {len(targets)} "
        f"Linear targets from {lora_path}",
        flush=True,
    )

    lcfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias="none"
    )
    peft_video = get_peft_model(video, lcfg)

    payload = torch.load(os.path.expanduser(os.path.expandvars(lora_path)), map_location="cpu")
    gen_lora = payload.get("generator_lora", payload) if isinstance(payload, dict) else payload
    gen_lora = {k: (v.to(vdtype) if hasattr(v, "to") else v) for k, v in gen_lora.items()}
    res = set_peft_model_state_dict(peft_video, gen_lora)
    missing = getattr(res, "missing_keys", None) or []
    unexpected = getattr(res, "unexpected_keys", None) or []
    print(
        f"[video-LoRA] load result: {len(gen_lora)} adapter tensors, "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    # Fail loud if the adapter keys did not map onto the video DiT (invalid experiment).
    if len(unexpected) != 0:
        raise RuntimeError(
            f"[video-LoRA] {len(unexpected)} unexpected adapter keys did NOT map "
            f"onto video_expert — LoRA NOT applied correctly. e.g. {unexpected[:3]}"
        )

    merged = peft_video.merge_and_unload()
    model.video_expert = merged
    model.mot.mixtures["video"] = merged
    print("[video-LoRA] merged LoRA into video expert weights. INJECT_OK", flush=True)

class RobotWinInference:
    """Model components only: no action queue, environment stepping or episode loop."""
    def __init__(self, run, settings):
        from longwam.inference import load_model, TextConditioning
        self.model = load_model(run, settings.checkpoint, device=settings.device,
                                load_text_encoder=not settings.get("text_cache"))
        self.text = TextConditioning(self.model, settings.get("text_cache"),
                                     int(run.data.train.get("context_len", 128)))
        _maybe_inject_video_lora(self.model)
        self.processor = instantiate(run.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(settings.stats)))
        self.action_horizon = int(settings.action_horizon)
        self.num_inference_steps = int(settings.num_inference_steps)
        options = settings.policy_options
        self.sigma_shift = options.get("sigma_shift")
        self.seed = int(settings.seed)
        self.text_cfg_scale = float(options.get("text_cfg_scale", 1.0))
        self.negative_prompt = str(options.get("negative_prompt", ""))
        self.rand_device = str(options.get("rand_device", "cpu"))
        self.tiled = bool(options.get("tiled", False))
        self.past_obs_size = int(run.data.train.get("past_obs_size", 0))
        self.action_video_freq_ratio = int(run.data.train.get("action_video_freq_ratio", 1))
        self._num_video_frames = (int(run.data.train.num_frames) - 1) // self.action_video_freq_ratio + 1
        self.num_imagine_frames = int(getattr(self.model, "num_imagine_frames", 0))
        factor = int(self.model.vae.temporal_downsample_factor)
        expected = self.past_obs_size // self.action_video_freq_ratio // factor + 1
        if int(getattr(self.model, "num_clean_frames", 1)) != expected:
            raise ValueError("RoboTwin checkpoint history and clean latent counts disagree")
        if self.num_imagine_frames > 0 and expected + self.num_imagine_frames != (
            self._num_video_frames - 1
        ) // factor + 1:
            raise ValueError("RoboTwin checkpoint observed/future latent counts disagree")

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {
            "state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        canonical = "images" in observation
        obs_data = observation["images"] if canonical else observation["observation"]
        image_shapes = {
            str(meta["key"]): tuple(int(v) for v in meta["shape"][-2:])
            for meta in self.processor.shape_meta["images"]
        }
        required_shapes = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        if set(image_shapes) != required_shapes:
            raise ValueError(
                "RoboTwin policy expects image keys "
                f"{sorted(required_shapes)}, got {sorted(image_shapes)}."
            )
        head = _resize_training_camera(
            obs_data["cam_high"] if canonical else obs_data["head_camera"]["rgb"],
            processor_size_hw=image_shapes["cam_high"],
            layout_size_hw=(256, 320),
        )
        left = _resize_training_camera(
            obs_data["cam_left_wrist"] if canonical else obs_data["left_camera"]["rgb"],
            processor_size_hw=image_shapes["cam_left_wrist"],
            layout_size_hw=(128, 160),
        )
        right = _resize_training_camera(
            obs_data["cam_right_wrist"] if canonical else obs_data["right_camera"]["rgb"],
            processor_size_hw=image_shapes["cam_right_wrist"],
            layout_size_hw=(128, 160),
        )
        bottom = torch.cat([left, right], dim=2)
        image_tensor = (
            torch.cat([head, bottom], dim=1)
            .unsqueeze(0)
            .to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
        )
        return image_tensor.mul_(2.0).sub_(1.0)

    def _infer_action_chunk(
        self,
        observation: Dict[str, Any],
        instruction: str,
        image_tensor: Optional[torch.Tensor] = None,
        clean_latents: Optional[torch.Tensor] = None,
        obs_frame_buffer=(),
    ) -> np.ndarray:
        if image_tensor is None:
            image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["state"] if "state" in observation else
                                  observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        # Route: k=num_imagine_frames>0 -> infer_joint_ar (action attends M clean obs + k imagined
        # future, partial-denoise latent-only); k=0 -> infer_action exactly (mirror of LIBERO P4 eval).
        use_joint = self.num_imagine_frames > 0
        infer_fn = self.model.infer_joint_ar if use_joint else self.model.infer_action
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_kwargs.update(self.text(prompt))
        if use_joint:
            if clean_latents is not None:
                infer_kwargs["clean_latents"] = clean_latents
            obs_window = None if clean_latents is not None else _build_obs_window(
                list(obs_frame_buffer), self.past_obs_size, self.action_video_freq_ratio
            )
            if obs_window is not None:
                infer_kwargs["obs_window"] = (
                    obs_window  # [1,3,T,H,W]; infer_joint_ar has no num_video_frames
                )
        elif "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        with torch.no_grad():
            pred = infer_fn(**infer_kwargs)

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk
