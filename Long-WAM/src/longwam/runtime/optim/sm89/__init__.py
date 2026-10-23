# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Opt-in SM89 BF16 execution with the validated inference protocol."""

from functools import wraps
import inspect

import torch

from .profiles import apply_profile

PROFILE = "graph_first_prefix_direct"


class SM89State:
    prepared = True

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.profile = PROFILE

    def to_dict(self):
        result = {key: value for key, value in self.telemetry.items() if key != "wrappers"}
        result.update(
            prepared=True,
            hardware="rtx4090",
            required_compute_capability="8.9",
            model_quant=False,
            kv_quant_enabled=False,
            preprocessing="Existing runtime history buffer and preprocessing",
        )
        result["graphs"] = {
            name: wrapper.telemetry() for name, wrapper in self.telemetry["wrappers"].items()
        }
        return result


def prepare_rtx4090(model):
    """Prepare once, after strict checkpoint/adapter loading and model.eval()."""
    if getattr(model, "_inference_optimization_prepared", False):
        state = getattr(model, "_sm89_optimization_state", None)
        if state is not None and state.profile == PROFILE:
            return state
        raise RuntimeError("Cannot mix or reconfigure prepared inference backends")
    if torch.device(model.device).type != "cuda":
        raise ValueError("rtx4090 requires a CUDA-resident model")
    actual = tuple(torch.cuda.get_device_capability(model.device))
    if actual != (8, 9):
        raise ValueError(f"rtx4090 requires compute capability 8.9, got {actual}")
    if model.training or model.joint_denoise:
        raise ValueError("rtx4090 currently supports evaluation-only sequential IDM")
    if (model.num_clean_frames, model.num_imagine_frames) != (4, 2):
        raise ValueError("rtx4090 currently requires four clean and two future latent frames")
    if model.video_expert.patch_embedding.weight.dtype != torch.bfloat16:
        raise ValueError("rtx4090 requires the original BF16 video backbone")
    if model.action_expert.blocks[0].self_attn.q.weight.dtype != torch.bfloat16:
        raise ValueError("rtx4090 requires the original BF16 action backbone")
    if model.mot.kv_cache_quantizer is not None:
        raise ValueError("rtx4090 prohibits KV quantization")
    previous = getattr(model, "_inference_optimization_config", None)
    if previous is not None and previous.enabled:
        raise ValueError("rtx4090 cannot mix preconfigured optimization policies")
    signature = inspect.signature(model.infer_joint_ar)
    model.configure_inference_optimizations({"required_compute_capability": "8.9"})
    telemetry = apply_profile(model, PROFILE)
    # The native runtime normalizes each new frame into its history buffer.
    # The study's raw-history batch-upload adapter is a separate frontend.
    model._wam4090_batch_rgb_upload = False
    telemetry["batch_rgb_upload"] = False
    infer = model.infer_joint_ar

    def parameter_bindings():
        return tuple(
            (
                name,
                id(value),
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
                value.device,
            )
            for name, value in model.named_parameters()
        )

    bindings = parameter_bindings()

    @wraps(infer)
    def checked_inference(*args, **kwargs):
        if parameter_bindings() != bindings:
            raise RuntimeError("Model parameter bindings changed; create a fresh prepared model")
        if args:
            raise ValueError("rtx4090 inference requires explicit keyword arguments")
        if kwargs.get("action_horizon") != 32 or kwargs.get("num_inference_steps") != 10:
            raise ValueError("rtx4090 validation currently covers H32 and ten action steps")
        video_steps = kwargs.get("video_num_inference_steps")
        if video_steps is None:
            video_steps = model.imagine_infer_steps
        if int(video_steps) != 4:
            raise ValueError("rtx4090 validation currently covers four video steps")
        if float(kwargs.get("text_cfg_scale", 1.0)) != 1.0:
            raise ValueError("rtx4090 currently requires conditional CFG=1")
        if kwargs.get("clean_latents") is not None or kwargs.get("tiled", False):
            raise ValueError("rtx4090 currently requires full fresh observation VAE encoding")
        if str(kwargs.get("rand_device", "cpu")) != "cpu":
            raise ValueError("rtx4090 validation currently uses original CPU noise generation")
        video = kwargs.get("obs_window")
        if (
            not isinstance(video, torch.Tensor)
            or video.ndim != 5
            or tuple(video.shape[:3]) != (1, 3, 13)
        ):
            raise ValueError("rtx4090 requires a batch-one, thirteen-frame observation window")
        if video.dtype != torch.bfloat16 or video.device.type != "cuda":
            raise ValueError("rtx4090 observation tensors must retain the original CUDA BF16 input")
        return infer(**kwargs)

    checked_inference.__signature__ = signature
    model.infer_joint_ar = checked_inference
    state = SM89State(telemetry)
    model._inference_optimization_prepared = True
    model._inference_optimization_state = state
    model._sm89_optimization_state = state
    return state
