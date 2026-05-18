# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


wan_default_config = {
    "Wan2.2-TI2V-5B": {
        "resolution": [1280, 704],
        "temporal_compression_ratio": 4,
        "spatial_compression_ratio": 16,
        "num_heads": 24,
        "head_dim": 128,
        "num_transformer_blocks": 30,
        "fps": 24,
    }
}


SECTION_KEYS = (
    "infra",
    "algorithm",
    "training",
    "data",
    "evaluation",
    "inference",
    "logging",
    "checkpoints",
)


def _set_once(config, key, value, source):
    if value is None:
        return
    if key in config and config[key] != value:
        raise ValueError(
            f"{key} is defined more than once with different values: "
            f"{config[key]} vs {value} from {source}."
        )
    config[key] = value


def section_get(config, section_key, key, default=None, aliases=()):
    """Read a grouped config value, falling back to legacy flat names."""
    section = config.get(section_key, None)
    candidate_keys = (key, *aliases)
    if section is not None:
        for candidate in candidate_keys:
            if candidate in section:
                return section[candidate]
    for candidate in candidate_keys:
        if candidate in config:
            return config[candidate]
    return default


def normalize_config(config):
    """Expand grouped release configs into the flat runtime schema.

    The training and inference code historically reads fields such as
    ``config.batch_size`` and ``config.model_kwargs`` directly.  Release
    configs can group those fields for readability, then call this function at
    the entry point to preserve the existing runtime contract.
    """
    for section_key in SECTION_KEYS:
        section = config.get(section_key, None)
        if section is None:
            continue
        for key, value in section.items():
            config[key] = value

    evaluation = config.get("evaluation", None)
    if evaluation is not None:
        if "interval" in evaluation:
            _set_once(config, "generate_interval", evaluation.interval, "evaluation.interval")
            _set_once(config, "vis_interval", evaluation.interval, "evaluation.interval")
        if "num_frames" in evaluation:
            num_frames = evaluation.num_frames
            if isinstance(num_frames, (list, tuple)):
                vis_lengths = list(num_frames)
                inference_num_frames = vis_lengths[0] if vis_lengths else 0
            else:
                inference_num_frames = int(num_frames)
                vis_lengths = [inference_num_frames]
            _set_once(config, "inference_num_frames", inference_num_frames, "evaluation.num_frames")
            _set_once(config, "vis_video_lengths", vis_lengths, "evaluation.num_frames")
        if "use_ema" in evaluation:
            _set_once(config, "vis_ema", evaluation.use_ema, "evaluation.use_ema")

    model_section = config.get("model", None)
    base_model_kwargs = config.get("model_kwargs", None)
    model_kwargs = OmegaConf.create({})
    if base_model_kwargs is not None:
        model_kwargs = OmegaConf.merge(model_kwargs, base_model_kwargs)

    if model_section is not None:
        section_kwargs = model_section.get("kwargs", None)
        if section_kwargs is not None:
            model_kwargs = OmegaConf.merge(model_kwargs, section_kwargs)

        model_name = model_section.get("name", None)
        if model_name is not None:
            model_kwargs.model_name = model_name
            config.model_name = model_name

        _set_once(
            config,
            "num_frame_per_block",
            model_section.get("num_frame_per_block", None),
            "model.num_frame_per_block",
        )

    if "model_name" in config and "model_name" not in model_kwargs:
        model_kwargs.model_name = config.model_name
    if "timestep_shift" in config and "timestep_shift" not in model_kwargs:
        model_kwargs.timestep_shift = config.timestep_shift
    if "timestep_shift" in model_kwargs:
        _set_once(config, "timestep_shift", model_kwargs.timestep_shift, "model_kwargs.timestep_shift")

    model_num_frame_per_block = model_kwargs.get("num_frame_per_block", None)
    if model_num_frame_per_block is not None:
        _set_once(config, "num_frame_per_block", model_num_frame_per_block, "model_kwargs.num_frame_per_block")

    if len(model_kwargs) > 0:
        config.model_kwargs = model_kwargs

    if "wandb_host" not in config:
        config.wandb_host = "https://api.wandb.ai"

    if config.get("trainer", None) == "score_distillation":
        dmd_defaults = {
            "i2v": False,
            "teacher_forcing": False,
            "backward_simulation": True,
            "independent_first_frame": False,
            "num_train_timestep": 1000,
            "denoising_loss_type": "flow",
            "real_guidance_scale": 3.0,
            "fake_guidance_scale": 0.0,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        }
        for key, value in dmd_defaults.items():
            if key not in config:
                config[key] = value
        if "causal" not in config:
            config.causal = bool(config.get("all_causal", True))

    # Causal DMD uses the same Wan backbone for generator/teacher/critic unless
    # a role-specific override is explicitly provided.
    if getattr(config, "all_causal", False) and "model_kwargs" in config:
        for role_key in ("real_model_kwargs", "fake_model_kwargs"):
            if config.get(role_key, None) is None:
                config[role_key] = OmegaConf.create(
                    OmegaConf.to_container(config.model_kwargs, resolve=True)
                )

    return config
