# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Install the single opt-in SM89 candidate, without research profile dispatch.

Keep the wrapper installation order used by the recorded strict candidate.
Full-model equivalence must still be checked on the submitted source version.
"""

from .context_cache import RequestContextCache
from .first_pass_prefix import FirstPassPrefix
from .gemm_shapes import GROUPS, VideoGemmShapes
from .rounded_pointwise import RoundedPointwise
from .static_graph import StaticGraphCallable

PROFILE = "graph_first_prefix_direct"
GRAPH_METHODS = (
    "_encode_video_latents",
    "_predict_video_noise_with_clean_prefix",
    "mot.prefill_video_cache_with_clean_prefix",
    "_predict_action_noise_with_cache",
)


def apply_profile(model, name: str) -> dict:
    """Retain BF16, the original inference schedule and unmerged adapters."""
    import torch

    if name != PROFILE:
        raise ValueError(f"The SM89 backend supports only {PROFILE}")
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
    for method in GRAPH_METHODS:
        parent_name, _, leaf = method.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = StaticGraphCallable(getattr(parent, leaf), name=method)
        setattr(parent, leaf, wrapper)
        wrappers[method] = wrapper
    wrappers["first_pass_prefix"] = FirstPassPrefix(
        model, wrappers["video_gemm_shapes"], direct_retention=True
    )
    return telemetry
