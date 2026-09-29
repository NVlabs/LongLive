# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

import math

from omegaconf import OmegaConf


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


wan_default_config = {
    "Wan2.1-T2V-14B": {
        "resolution": [
            832,
            480
        ],
        "temporal_compression_ratio": 4,
        "spatial_compression_ratio": 8,
        "latent_channels": 16,
        "num_heads": 40,
        "head_dim": 128,
        "num_transformer_blocks": 40,
        "fps": 16,
        "vae_type": "wan2.1",
        "seq_len": 32760
    },
    "Wan2.2-TI2V-5B": {
        "resolution": [
            1280,
            704
        ],
        "temporal_compression_ratio": 4,
        "spatial_compression_ratio": 16,
        "latent_channels": 48,
        "num_heads": 24,
        "head_dim": 128,
        "num_transformer_blocks": 30,
        "fps": 24,
        "vae_type": "wan2.2"
    }
}


CFG_ONLY_SUPPORTED_MODELS = frozenset(
    {
        "Wan2.1-T2V-14B",
        "Wan2.2-TI2V-5B",
    }
)

CFG_ONLY_NATIVE_TIMESTEP_SHIFT = {
    "Wan2.1-T2V-14B": 5.0,
    "Wan2.2-TI2V-5B": 5.0,
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

    model_name = model_kwargs.get("model_name", config.get("model_name", None))
    if model_name in wan_default_config:
        model_defaults = wan_default_config[model_name]
        if "vae_type" in model_defaults and "vae_type" not in config:
            config.vae_type = model_defaults["vae_type"]
        if config.get("trainer") == "score_distillation":
            for flag in ("generator_is_causal", "all_causal", "causal"):
                if flag not in config:
                    config[flag] = False
            if model_name == "Wan2.1-T2V-14B" and config.get("distribution_loss", "dmd") == "dmd":
                if "sfp_training" not in config:
                    config.sfp_training = True

    if len(model_kwargs) > 0:
        config.model_kwargs = model_kwargs

    if "wandb_host" not in config:
        config.wandb_host = "https://api.wandb.ai"

    if "negative_prompt" not in config:
        config.negative_prompt = DEFAULT_NEGATIVE_PROMPT

    if config.get("trainer", None) == "score_distillation":
        all_causal = bool(config.get("all_causal", False))
        is_cfg_only = config.get("distribution_loss", "dmd") == "cfg_guidance"
        distillation_defaults = {
            "i2v": False,
            "teacher_forcing": False,
            "backward_simulation": not is_cfg_only,
            "independent_first_frame": False,
            "num_train_timestep": 1000,
            "denoising_loss_type": "flow",
            "generator_is_causal": False,
            "real_score_is_causal": False if is_cfg_only else all_causal,
        }
        if is_cfg_only:
            distillation_defaults.update({
                "sfp_training": False,
                "teacher_guidance_scale": 5.0,
                "teacher_sampling_steps": 50,
                "cfg_state_source": "teacher_trajectory_cache",
                "cfg_verify_cache_hashes": True,
                "cfg_relative_guidance_max_weight": 16.0,
            })
        else:
            distillation_defaults.update({
                "real_guidance_scale": 3.0,
                "fake_guidance_scale": 0.0,
                "fake_score_is_causal": all_causal,
            })
        for key, value in distillation_defaults.items():
            if key not in config:
                config[key] = value
        if "causal" not in config:
            config.causal = all_causal

    # DMD uses the same Wan backbone kwargs for generator/teacher/critic unless
    # a role-specific checkpoint path is explicitly provided.
    if config.get("trainer", None) == "score_distillation" and "model_kwargs" in config:
        role_keys = ["real_model_kwargs"]
        if config.get("distribution_loss", "dmd") != "cfg_guidance":
            role_keys.append("fake_model_kwargs")
        for role_key in role_keys:
            if config.get(role_key, None) is None:
                config[role_key] = OmegaConf.create(
                    OmegaConf.to_container(config.model_kwargs, resolve=True)
                )

    if (
        config.get("trainer", None) == "score_distillation"
        and config.get("distribution_loss", None) == "cfg_guidance"
    ):
        validate_cfg_only_config(config)

    validate_release_config(config)
    for key in ("model_kwargs", "real_model_kwargs", "fake_model_kwargs"):
        kwargs = config.get(key)
        if kwargs is not None:
            for unused_key in ("local_attn_size", "sink_size", "seq_len"):
                kwargs.pop(unused_key, None)
    return config


def validate_cfg_only_config(config):
    """Fail closed when a CFG-only run accidentally inherits DMD/few-step knobs."""

    model_name = config.get("model_kwargs", {}).get("model_name", None)
    if model_name not in CFG_ONLY_SUPPORTED_MODELS:
        raise ValueError(
            "CFG-only distillation supports "
            f"{sorted(CFG_ONLY_SUPPORTED_MODELS)}, got {model_name}."
        )
    teacher_model_name = config.get("real_model_kwargs", {}).get(
        "model_name", model_name
    )
    if teacher_model_name != model_name:
        raise ValueError(
            "CFG-only student and teacher must use the same Wan model, got "
            f"student={model_name}, teacher={teacher_model_name}."
        )
    if bool(config.get("backward_simulation", True)):
        raise ValueError(
            "CFG-only guidance regression requires backward_simulation=false; "
            "DMD/self-forcing rollout is intentionally disabled."
        )
    if bool(config.get("i2v", False)):
        raise ValueError(
            "The initial CFG-only implementation supports text-to-video only."
        )
    state_source = str(config.get("cfg_state_source", "data"))
    if state_source != "teacher_trajectory_cache":
        raise ValueError("cfg_state_source must be 'teacher_trajectory_cache'")
    if state_source == "teacher_trajectory_cache":
        data_path = config.get("data_path", None)
        if not isinstance(data_path, str) or not data_path.endswith(".json"):
            raise ValueError(
                "teacher_trajectory_cache requires data.data_path to point to "
                "a JSON manifest."
            )
        if bool(config.get("load_raw_video", False)):
            raise ValueError(
                "teacher_trajectory_cache requires load_raw_video=false."
            )
        if config.get("real_score_ckpt", None) is not None:
            raise ValueError(
                "teacher_trajectory_cache already contains teacher targets; "
                "real_score_ckpt must be null."
            )
        if bool(config.get("real_score_quant", False)):
            raise ValueError(
                "teacher_trajectory_cache does not instantiate real_score; "
                "real_score_quant must be false."
            )
        if int(section_get(config, "evaluation", "interval", -1)) > 0:
            raise ValueError(
                "teacher_trajectory_cache training has no VAE/teacher "
                "visualization path; evaluation.interval must be <= 0."
            )
        if bool(
            section_get(config, "evaluation", "before_train", False)
        ):
            raise ValueError(
                "teacher_trajectory_cache training has no VAE/teacher "
                "visualization path; evaluation.before_train must be false."
            )
    verify_shard_hashes = config.get("cfg_verify_cache_hashes", True)
    if not isinstance(verify_shard_hashes, bool):
        raise ValueError("cfg_verify_cache_hashes must be a boolean.")
    if state_source != "teacher_trajectory_cache" and not verify_shard_hashes:
        raise ValueError(
            "cfg_verify_cache_hashes=false is only valid with "
            "cfg_state_source=teacher_trajectory_cache."
        )
    loss_weighting = str(config.get("cfg_distill_loss_weighting", "uniform"))
    if loss_weighting not in {"uniform", "flow", "relative_guidance"}:
        raise ValueError(
            "cfg_distill_loss_weighting must be 'uniform', 'flow', or "
            f"'relative_guidance', got {loss_weighting!r}."
        )
    relative_epsilon = float(
        config.get("cfg_relative_guidance_epsilon", 1.0e-8)
    )
    if not math.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise ValueError("cfg_relative_guidance_epsilon must be positive.")
    relative_max_weight = float(
        config.get("cfg_relative_guidance_max_weight", 16.0)
    )
    if not math.isfinite(relative_max_weight) or relative_max_weight <= 0:
        raise ValueError(
            "cfg_relative_guidance_max_weight must be positive."
        )

    teacher_scale = float(config.get("teacher_guidance_scale", 0.0))
    if teacher_scale <= 1.0:
        raise ValueError(
            "teacher_guidance_scale must be > 1 for CFG distillation, got "
            f"{teacher_scale}."
        )

    inference_steps = int(section_get(config, "inference", "sampling_steps", 0))
    teacher_steps = int(config.get("teacher_sampling_steps", 0))
    if inference_steps != 50 or teacher_steps != 50:
        raise ValueError(
            "CFG-only distillation is fixed to the native 50->50 "
            "schedule: "
            f"inference.sampling_steps={inference_steps}, "
            f"teacher_sampling_steps={teacher_steps}."
        )
    timestep_shift = float(
        config.get("model_kwargs", {}).get(
            "timestep_shift",
            config.get("timestep_shift", 0.0),
        )
    )
    expected_timestep_shift = CFG_ONLY_NATIVE_TIMESTEP_SHIFT[model_name]
    if timestep_shift != expected_timestep_shift:
        raise ValueError(
            f"CFG-only distillation for {model_name} must preserve "
            f"timestep_shift={expected_timestep_shift:g}, "
            f"got {timestep_shift}."
        )
    inference_guidance = float(
        section_get(config, "inference", "guidance_scale", -1.0)
    )
    if inference_guidance != 1.0:
        raise ValueError(
            "The distilled student must be evaluated with guidance_scale=1, "
            f"got {inference_guidance}."
        )

    if bool(config.get("generator_is_causal", False)) != bool(
        config.get("real_score_is_causal", False)
    ):
        raise ValueError(
            "CFG-only student and teacher must use the same causal/non-causal mode."
        )

    adapter = config.get("adapter", None)
    if adapter is None or adapter.get("type", None) != "lora":
        raise ValueError("CFG-only distillation requires a LoRA adapter config.")
    if bool(adapter.get("apply_to_critic", False)):
        raise ValueError(
            "CFG-only distillation has no critic; set adapter.apply_to_critic=false."
        )


def validate_release_config(config):
    """Reject removed research modes instead of silently changing the recipe."""
    name = config.get("model_kwargs", {}).get("model_name")
    if name not in wan_default_config:
        raise ValueError(f"LongLive-Plug supports Wan2.1-T2V-14B and Wan2.2-TI2V-5B, got {name!r}")
    for flag in ("all_causal", "causal", "generator_is_causal", "fake_score_is_causal",
                 "real_score_is_causal", "i2v", "teacher_forcing", "independent_first_frame",
                 "model_quant", "generator_quant", "real_score_quant", "fake_score_quant"):
        if config.get(flag, False):
            raise ValueError(f"{flag}=true is outside the LongLive-Plug non-AR release")
    if config.get("trainer", "score_distillation") != "score_distillation":
        raise ValueError("Only LoRA score distillation is supported")
    if int(config.get("sequence_parallel_size", 1)) != 1:
        raise ValueError("Sequence parallelism is not part of the four released recipes")
    for flag in ("multi_shot_sink", "multi_shot_rope_offset", "chunks_per_shot", "debug_score_pred_videos", "torch_compile"):
        if config.get(flag, False):
            raise ValueError(f"{flag} is outside the full-sequence release")
    if config.get("denoising_loss_type", "flow") != "flow":
        raise ValueError("DMD uses the flow denoising loss")
    loss = config.get("distribution_loss", "dmd")
    if loss not in ("dmd", "cfg_guidance"):
        raise ValueError(f"Unsupported distillation objective: {loss}")
    if loss == "dmd" and not config.get("backward_simulation", True):
        raise ValueError("The DMD recipe requires backward_simulation=true")
    adapter = config.get("adapter")
    if adapter is None or adapter.get("type") != "lora":
        raise ValueError("LongLive-Plug training requires a LoRA adapter")
    if bool(adapter.get("apply_to_critic", False)) != (loss == "dmd"):
        raise ValueError("DMD requires a critic LoRA; CFG-only must not enable it")
    for key in ("generator_ckpt", "real_score_ckpt", "fake_score_ckpt"):
        if config.get(key) is not None:
            raise ValueError(f"Use model_kwargs.model_dir for base weights; {key} is not supported")
    if loss == "dmd" and config.get("image_or_video_shape") is not None:
        frames = int(config.image_or_video_shape[1])
        for key in ("min_num_training_frames", "num_training_frames", "slice_last_frames"):
            if int(config.get(key, frames)) != frames:
                raise ValueError(f"{key} must match the full training clip length ({frames})")
    for key in ("real_model_kwargs", "fake_model_kwargs"):
        if config.get(key, {}).get("model_name", name) != name:
            raise ValueError("Student, teacher and critic must use the same base model")
