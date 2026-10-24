# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Install the strict default or the explicitly selected BF16 fast mode.

Keep the wrapper installation order used by the recorded strict candidate.
Full-model equivalence must still be checked on the submitted source version.
"""

from .context_cache import RequestContextCache
from .first_pass_prefix import FirstPassPrefix
from .gemm_shapes import GROUPS, VideoGemmShapes
from .rounded_pointwise import RoundedPointwise
from .static_graph import StaticGraphCallable

PROFILE = "graph_first_prefix_direct"
FAST_PROFILE = "graph_clean_prefix"
GRAPH_METHODS = (
    "_encode_video_latents",
    "_predict_video_noise_with_clean_prefix",
    "mot.prefill_video_cache_with_clean_prefix",
    "_predict_action_noise_with_cache",
)
FAST_GRAPH_METHODS = (
    "_encode_video_latents",
    "_prefill_video_clean_prefix",
    "_predict_video_noise_with_clean_prefix",
    "mot.prefill_video_cache_with_clean_prefix",
    "_predict_action_noise_with_cache",
)


def validate_fast_profile(model):
    """Check causal-prefix admission before installing any wrappers or flags."""
    if model.training or model.joint_denoise:
        raise ValueError("BF16 clean-prefix fast mode requires sequential evaluation")
    if (model.num_clean_frames, model.num_imagine_frames) != (4, 2):
        raise ValueError("BF16 clean-prefix fast mode requires P48 and two future frames")
    if (
        tuple(model.video_expert.patch_size)[0] != 1
        or model.video_expert.video_attention_mask_mode != "per_frame_causal"
    ):
        raise ValueError("BF16 clean-prefix reuse requires frame-local causal video attention")
    if model.mot.kv_cache_quantizer is not None:
        raise ValueError("BF16 clean-prefix fast mode prohibits KV quantization")
    if model.mot.mixtures["video"] is not model.video_expert:
        raise ValueError("Video stack and MoT must share the same original expert")
    if getattr(model, "_video_clean_prefix_wavefront_enabled", False):
        raise ValueError("Cannot mix fast mode with an already configured clean-prefix path")


def apply_profile(model, name: str) -> dict:
    """Retain BF16, the original inference schedule and unmerged adapters."""
    import torch

    if name not in (PROFILE, FAST_PROFILE):
        raise ValueError("Unknown SM89 execution profile")
    fast = name == FAST_PROFILE
    if fast:
        validate_fast_profile(model)
    telemetry = {
        "name": name,
        "quantization": False,
        "merged_lora": False,
        "numerical_candidate": True,
        "batch_rgb_upload": False,
        "wrappers": {},
    }
    model._wam4090_batch_rgb_upload = False
    wrappers = telemetry["wrappers"]
    if fast:
        # The original BF16 implementation supports this causal schedule.
        # Splitting token-row shapes changes rounding; selection is explicit.
        model._video_clean_prefix_wavefront_enabled = True
        telemetry["clean_prefix_scope"] = "One request, with fresh VAE and context"
    else:
        wrappers["video_gemm_shapes"] = VideoGemmShapes(model, GROUPS, reuse_buffers=True)
        wrappers["rounded_pointwise"] = RoundedPointwise(model)

    def resident(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=model.device)
        if isinstance(value, tuple):
            return tuple(resident(item) for item in value)
        raise TypeError("Unexpected RoPE constant container")

    # Transfer the existing bits; never regenerate or cast RoPE values.
    telemetry["resident_rope_constants"] = []
    for expert_name in ("action_expert", "video_expert"):
        expert = getattr(model, expert_name)
        expert.freqs = resident(expert.freqs)
        telemetry["resident_rope_constants"].append(expert_name)
    # Encoding retains the original cast from these FP32 scales to latent dtype.
    model.vae.scale = [value.to(device=model.device) for value in model.vae.scale]
    telemetry["resident_vae_scale_constants"] = True
    wrappers["request_context_projections"] = RequestContextCache(
        model, capture_projections=True
    )
    for method in FAST_GRAPH_METHODS if fast else GRAPH_METHODS:
        parent_name, _, leaf = method.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = StaticGraphCallable(getattr(parent, leaf), name=method)
        setattr(parent, leaf, wrapper)
        wrappers[method] = wrapper
    if not fast:
        wrappers["first_pass_prefix"] = FirstPassPrefix(
            model, wrappers["video_gemm_shapes"], direct_retention=True
        )
    return telemetry
