# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/policy.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
# Changes: Added Unitree G1 inference/deployment integration while retaining the shared runtime.

"""Checkpoint-backed inference worker using the release model and processors."""

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def initialize_policy(settings, components=None):
    import torch
    from omegaconf import OmegaConf
    from longwam.evaluation import native_config
    from longwam.inference import load_model, TextConditioning
    from .streaming_vae import StreamingVAE
    from .backends import prepare_backend

    settings = OmegaConf.create(settings)
    if components is None:
        from longwam.utils.pytorch_utils import set_global_seed

        set_global_seed(int(settings.get("seed", 0)), get_worker_init_fn=False)
    generic = settings.get("robot") or settings.get("benchmark") is None
    runtime = None if components is not None or generic else native_config(settings)
    if components is not None:
        runtime, model, image, infer = components
    elif generic:
        runtime, model, image, infer = load_robot_components(settings)
    elif settings.benchmark == "libero":
        from . import libero as adapter

        model = load_model(runtime, settings.checkpoint, device=settings.device,
                           load_text_encoder=not settings.get("text_cache"))
        text = TextConditioning(model, settings.get("text_cache"),
                                int(runtime.data.train.get("context_len", 128)))
        adapter._maybe_inject_video_lora(model, runtime)
        processor = adapter.instantiate(runtime.data.train.processor).eval()
        processor.set_normalizer_from_stats(
            adapter.load_dataset_stats_from_json(str(settings.stats))
        )
        height, width = map(int, runtime.data.train.video_size)

        def image(obs):
            return adapter._image_to_model_input(
                obs,
                cfg=runtime,
                processor=processor,
                width=width,
                height=height,
                device=str(settings.device),
                dtype=model.torch_dtype,
            )[0]

        def infer(obs, instruction, frames, clean):
            return adapter._predict_action_chunk(
                obs,
                instruction,
                model,
                processor,
                runtime,
                action_horizon=int(settings.action_horizon),
                input_w=width,
                input_h=height,
                model_device=str(settings.device),
                obs_frame_buffer=frames,
                clean_latents=clean,
                image_tensor=frames[-1].unsqueeze(0),
                text_conditioning=text,
            )
    elif settings.benchmark in {"robotwin2", "domino"}:
        from .robotwin import RobotWinInference

        components = RobotWinInference(runtime, settings)
        model = components.model
        image = lambda obs: components._build_robotwin_image_tensor(obs)[0]

        def infer(obs, instruction, frames, clean):
            return components._infer_action_chunk(
                obs, instruction, image_tensor=frames[-1].unsqueeze(0),
                obs_frame_buffer=frames, clean_latents=clean
            )
    else:
        raise ValueError("runtime supports LIBERO and RoboTwin 2")

    if (settings.streaming_vae or settings.hardware != "reference") and int(
        getattr(model, "num_imagine_frames", 0)
    ) <= 0:
        raise ValueError("runtime requires a video-action checkpoint with imagined future frames")
    options = dict(settings.get("optimization", {}))
    if settings.get("hardware", "reference") != "reference":
        # Match the released processor and VAE rather than hard-code LIBERO's grid.
        height, width = map(int, runtime.data.train.video_size)
        stride = int(runtime.data.train.get("action_video_freq_ratio", 1))
        past = int(runtime.data.train.get("past_obs_size", 0))
        group = int(model.vae.temporal_downsample_factor)
        spatial = int(model.vae.upsampling_factor)
        patch = model.video_expert.patch_size
        options.setdefault(
            "video_rope_grid_size",
            (
                (past // stride // group + 1 + model.num_imagine_frames) // patch[0],
                height // spatial // patch[1],
                width // spatial // patch[2],
            ),
        )
    if hasattr(model, "configure_inference_optimizations"):
        prepare_backend(model, settings.get("hardware", "reference"), options)
    elif settings.get("hardware", "reference") != "reference" or options:
        raise ValueError("Optimized execution requires a model with optimization hooks")
    past = int(runtime.data.train.get("past_obs_size", 0))
    ratio = int(runtime.data.train.get("action_video_freq_ratio", 1))
    frames = deque(maxlen=past + 1)
    session = (
        StreamingVAE(model, past_steps=past, sample_stride=ratio)
        if settings.streaming_vae
        else None
    )
    episode, instruction, current, last_step, next_anchor = 0, "", None, -1, None

    def handle(operation, payload):
        nonlocal episode, instruction, current, last_step, next_anchor
        if operation == "reset":
            episode, instruction = int(payload["episode"]), str(payload["instruction"])
            frames.clear()
            current, last_step, next_anchor = None, -1, None
            if session:
                session.reset()
            return None
        if int(payload["episode"]) != episode:
            raise RuntimeError("Observation/request belongs to another episode")
        if operation == "observe":
            step = int(payload["step"])
            if step != last_step + 1:
                raise ValueError("Observations must arrive at consecutive control steps")
            current, last_step = payload["observation"], step
            frame = image(current).detach()
            frames.append((step, frame))
            if session and session.anchor is None and next_anchor is not None:
                session.begin(next_anchor, list(frames))
            if session and session.anchor is not None:
                first = session.anchor - past
                if step <= session.anchor and (step - first) % ratio == 0:
                    session.feed(step, frame)
            return None
        if operation != "infer" or int(payload["anchor"]) != last_step:
            raise ValueError("Inference must use the current observation anchor")
        anchor = int(payload["anchor"])
        clean = None
        if session:
            if session.anchor != anchor:
                session.begin(anchor, list(frames))
            clean = session.finish(anchor)
        with torch.no_grad():
            actions = infer(current, instruction, [f for _, f in frames], clean)
        # Preencode the available prefix of the next window outside graph replay.
        next_anchor = anchor + int(
            settings.trigger_stride if settings.execution == "async" else settings.execute_steps
        )
        return {"episode": episode, "anchor": anchor, "actions": actions}

    return handle, {
        "execution": settings.execution,
        "hardware": settings.hardware,
        "history_steps": past,
        "streaming_vae": bool(session),
    }


def _create_execution(settings, *, normalized=False):
    from omegaconf import OmegaConf
    from .settings import normalize_settings
    from .execution import AsyncPolicy, Schedule, SyncPolicy, SyncSchedule
    from .worker import SpawnedInferenceProcess
    from longwam.evaluation import native_config

    settings = settings if normalized else normalize_settings(settings)
    if settings.get("action_horizon") is None:
        if settings.get("robot") or settings.get("benchmark") is None:
            from longwam.inference import load_run_config
            settings.action_horizon = int(load_run_config(
                settings.model_config, domain=settings.get("domain")
            ).data.train.action_chunk)
        else:
            settings.action_horizon = native_config(settings).EVALUATION.action_horizon
        settings = normalize_settings(settings)
    if settings.execution == "async":
        schedule = Schedule(
            settings.action_horizon, settings.execute_steps, settings.trigger_stride
        )
        policy_type = AsyncPolicy
    else:
        schedule = SyncSchedule(settings.action_horizon, settings.execute_steps)
        policy_type = SyncPolicy
    worker = SpawnedInferenceProcess(
        initialize_policy,
        OmegaConf.to_container(settings, resolve=True),
        name="longwam-inference",
    )
    return policy_type(worker, schedule, timeout=float(settings.get("inference_timeout", 600)))


def load_robot_components(settings):
    """Construct model/image/infer closures for the existing shared GPU handler."""
    from collections.abc import Mapping
    from .robots import make_adapter
    import torch
    import torchvision.transforms.functional as TF
    from hydra.utils import instantiate
    from longwam.inference import load_model, load_run_config, TextConditioning
    from longwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from longwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    robot = settings.get("robot")
    adapter = make_adapter(robot, dict(settings.adapter_options)) if robot else None
    run = load_run_config(settings.model_config, domain=settings.get("domain"))
    data = run.data.train
    if robot == "g1":
        adapter.validate_checkpoint(data)
    past, ratio = int(data.get("past_obs_size", 0)), int(data.get("action_video_freq_ratio", 1))
    if past < 0 or ratio <= 0 or past % ratio:
        raise ValueError("Robot history requires nonnegative past_obs_size divisible by sample stride")
    if int(run.model.get("longwam_past_obs_size", 0)) != past:
        raise ValueError("Checkpoint model and data history lengths disagree")
    processor = instantiate(data.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(settings.stats)))
    shape = processor.shape_meta
    if len(shape["state"]) != 1 or len(shape["action"]) != 1:
        raise ValueError("Robot adapters require one merged state and action field")
    state_key, action_key = shape["state"][0]["key"], shape["action"][0]["key"]
    def raw_dimension(meta):
        raw = meta.get("raw_shape", meta["shape"])
        return (raw,) if isinstance(raw, int) else tuple(raw)

    if adapter and (
        raw_dimension(shape["state"][0]) != (adapter.state_dim,)
        or raw_dimension(shape["action"][0]) != (adapter.action_dim,)
    ):
        raise ValueError("Robot action mode/dimensions disagree with checkpoint shape_meta")
    camera_names = [meta["key"] for meta in shape["images"]]
    if adapter and set(camera_names) != set(adapter.camera_keys):
        raise ValueError("Robot cameras disagree with checkpoint shape_meta")
    if robot == "yam" and camera_names != ["top", "left_wrist", "right_wrist"]:
        raise ValueError("YAM camera order must be top, left_wrist, right_wrist")
    layout = str(data.concat_multi_camera)
    if layout not in {"horizontal", "vertical", "robotwin", "robotwin_right"} or (
        layout == "robotwin_right" and robot != "g1"
    ) or (
        robot == "yam" and layout != "robotwin"
    ) or (robot == "franka" and layout not in {"horizontal", "vertical"}):
        raise ValueError("Robot checkpoint camera layout is unsupported")
    height, width = map(int, data.video_size)
    if robot == "yam" and (height, width) != (384, 320):
        raise ValueError("YAM requires the trained 384x320 mosaic")
    torch.manual_seed(int(settings.seed))
    model = load_model(
        run, settings.checkpoint, device=settings.device,
        load_text_encoder=not settings.get("text_cache"),
    )
    text = TextConditioning(model, settings.get("text_cache"), int(data.get("context_len", 128)))
    expected_clean = past // ratio // int(model.vae.temporal_downsample_factor) + 1
    if int(model.num_clean_frames) != expected_clean:
        raise ValueError("Robot history and model clean latent counts disagree")

    def image(obs):
        cameras = []
        for index, meta in enumerate(shape["images"]):
            frame = torch.from_numpy(obs["images"][meta["key"]]).permute(2, 0, 1).unsqueeze(0)
            transforms = processor.val_transforms
            camera_transforms = (
                transforms[meta["key"]] if isinstance(transforms, Mapping) else transforms
            )
            for transform in camera_transforms:
                frame = transform(frame)
            frame = frame[0]
            target_h, target_w = map(int, meta["shape"][-2:])
            if robot == "franka" and tuple(frame.shape[-2:]) != (target_h, target_w):
                scale = max(target_h / frame.shape[1], target_w / frame.shape[2])
                frame = TF.resize(
                    frame, [round(frame.shape[1] * scale), round(frame.shape[2] * scale)],
                    antialias=True,
                )
                frame = TF.center_crop(frame, [target_h, target_w])
            elif tuple(frame.shape[-2:]) != (target_h, target_w):
                frame = TF.resize(frame, [target_h, target_w], antialias=True)
            if layout in {"robotwin", "robotwin_right"}:
                frame = TF.resize(frame, [256, 320] if index == 0 else [128, 160], antialias=True)
            cameras.append(frame)
        if layout == "robotwin_right":
            # The missing left wrist is black in RGB space, not normalized zero.
            cameras.insert(1, torch.zeros_like(cameras[1]))
        frame = (
            torch.cat([cameras[0], torch.cat(cameras[1:], dim=2)], dim=1)
            if layout in {"robotwin", "robotwin_right"}
            else torch.cat(cameras, dim=2 if layout == "horizontal" else 1)
        )
        if tuple(frame.shape[-2:]) != (height, width):
            raise ValueError("Processed cameras do not match checkpoint video_size")
        return frame.mul(2).sub(1).to(device=model.device, dtype=model.torch_dtype)

    def infer(obs, instruction, frames, clean):
        state = {"state": {state_key: torch.as_tensor(obs["state"]).unsqueeze(0)}}
        state = processor.action_state_transform(state)
        state = processor.normalizer.forward(state)
        proprio = processor.action_state_merger.forward(state)["state"]
        kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=instruction), "input_image": frames[-1].unsqueeze(0),
            "proprio": proprio, "action_horizon": int(settings.action_horizon),
            "num_inference_steps": int(settings.num_inference_steps), "seed": int(settings.seed),
            "rand_device": str(settings.policy_options.get("rand_device", "cpu")),
            "text_cfg_scale": float(settings.policy_options.get("text_cfg_scale", 1.0)),
            "tiled": bool(settings.policy_options.get("tiled", False)),
            "negative_prompt": str(settings.policy_options.get("negative_prompt", "")),
            "sigma_shift": settings.policy_options.get("sigma_shift"),
        }
        kwargs.update(text(kwargs["prompt"]))
        if clean is not None:
            kwargs["clean_latents"] = clean
        elif past:
            kwargs["obs_window"] = build_observation_window(frames, past, ratio)
        infer_fn = model.infer_joint_ar if getattr(model, "num_imagine_frames", 0) > 0 else model.infer_action
        prediction = infer_fn(**kwargs)["action"].to("cpu", torch.float32)
        restored = processor.action_state_merger.backward({
            "action": prediction.unsqueeze(0), "state": proprio.unsqueeze(0),
        })
        restored = processor.normalizer.backward(restored)
        for transform in reversed(processor.action_state_transforms or []):
            restored = transform.backward(restored)
        return restored["action"][action_key][0].numpy()

    return run, model, image, infer


def build_observation_window(
    obs_frame_buffer: "list[torch.Tensor]",
    past_obs_size: int,
    action_video_freq_ratio: int,
):
    """Build the v0b past->current observation window matching `RobotVideoDataset` sampling.

    Mirrors training exactly:
      * The dataset obs window spans raw-frame deltas [-past_obs_size .. ] (delta_timestamps in
        `base_lerobot_dataset`), then samples `video_sample_indices = range(0, num_frames, freq)`.
      * The clean (past->current) region is the first M latent frames = VAE of the first
        T_sampled = past_obs_size // freq + 1 sampled frames, i.e. raw deltas
        [-past_obs_size, -past_obs_size+freq, ..., -freq, 0] -> the last `past_obs_size + 1`
        raw frames, sub-sampled every `freq`, ending on the CURRENT frame.
      * Episode-start padding repeats the OLDEST available frame (matches lerobot
        `_get_query_indices`: `max(ep_start, min(ep_end-1, idx+delta))` clamps out-of-range
        past deltas to `ep_start`).

    Args:
        obs_frame_buffer: list of processed model-input frames, each [3,H,W], most-recent last.
            Must contain at least one frame (the current obs).
        past_obs_size: pixel `past_obs_size` (P). 0 -> returns None (v0a single-frame path).
        action_video_freq_ratio: sampling stride (`freq`).

    Returns:
        obs_window tensor [1, 3, T_sampled, H, W] with T_sampled = P // freq + 1, or None when
        past_obs_size == 0 (v0a: caller falls back to single `input_image`).
    """
    import torch

    if past_obs_size <= 0:
        return None
    if len(obs_frame_buffer) == 0:
        raise ValueError("`obs_frame_buffer` must contain at least the current observation frame.")
    if past_obs_size % action_video_freq_ratio != 0:
        raise ValueError(
            f"past_obs_size ({past_obs_size}) must be divisible by action_video_freq_ratio "
            f"({action_video_freq_ratio}) so the clean/noised boundary lands on a latent-frame edge."
        )

    needed = past_obs_size + 1  # raw frames at deltas [-P .. 0]
    # Pad-at-start by repeating the oldest available frame (repeat-oldest, matches lerobot).
    if len(obs_frame_buffer) < needed:
        pad = [obs_frame_buffer[0]] * (needed - len(obs_frame_buffer))
        raw_window = pad + list(obs_frame_buffer)
    else:
        raw_window = list(obs_frame_buffer[-needed:])
    # Sub-sample every `freq`: indices [0, freq, 2*freq, ..., P] -> deltas [-P, -P+freq, ..., 0].
    sampled = raw_window[0::action_video_freq_ratio]
    expected_t = past_obs_size // action_video_freq_ratio + 1
    if len(sampled) != expected_t:
        raise ValueError(
            f"obs window sampling produced {len(sampled)} frames, expected {expected_t} "
            f"(past_obs_size={past_obs_size}, action_video_freq_ratio={action_video_freq_ratio})."
        )
    # Stack to [1, 3, T_sampled, H, W].
    temporal_dim = sampled[0].ndim - 2  # [C,H,W] or view-preserving [V,C,H,W]
    window = torch.stack(sampled, dim=temporal_dim).unsqueeze(
        0
    )  # frames are [3,H,W] -> [3,T,H,W] -> [1,3,T,H,W]
    return window
