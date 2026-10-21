# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Retain causal clean K/V from the first original full video prediction.

The first video pass executes its unchanged block stack and original GEMM
shapes. Later passes reuse only that request's causally closed clean K/V.
No separate clean prefill is run. All four video updates remain in the native
scheduler, and every request begins with a fresh full observation VAE.
"""

from __future__ import annotations


class FirstPassPrefix:
    def __init__(self, model, shapes, *, capture=True, direct_retention=False):
        from .static_graph import StaticGraphCallable

        if model.training or model.joint_denoise:
            raise ValueError("First-pass prefix retention requires sequential evaluation")
        if (model.num_clean_frames, model.num_imagine_frames) != (4, 2):
            raise ValueError("First-pass prefix retention requires P48 and two future frames")
        if model.video_expert.video_attention_mask_mode != "per_frame_causal":
            raise ValueError("First-pass prefix retention requires causal video attention")
        if model.mot.kv_cache_quantizer is not None:
            raise ValueError("First-pass prefix retention prohibits KV quantization")
        if model.mot.mixtures["video"] is not model.video_expert:
            raise ValueError("Video stack and MoT must share the same original expert")
        if getattr(model, "_video_clean_prefix_wavefront_enabled", False):
            raise ValueError("Do not mix first-pass retention with separate clean prefill")
        self.model, self.shapes = model, shapes
        self.direct_retention = bool(direct_retention)
        self.raw_predict = model._predict_video_noise
        self.raw_imagine = model._imagine_future_latents
        self.active = self.ready = self.recording = False
        self.steps = self.requests = self.prefills = 0
        self.observed = {}
        self.hooks = []
        self.original_attention_forwards = []
        for index, block in enumerate(model.video_expert.blocks):

            def frequency(_module, args, index=index):
                if self.recording:
                    self.observed.setdefault(index, {})["freqs"] = args[1]

            def key(_module, _args, output, index=index):
                if self.recording:
                    self.observed.setdefault(index, {})["k"] = output

            def value(_module, _args, output, index=index):
                if self.recording:
                    self.observed.setdefault(index, {})["v"] = output

            if self.direct_retention:
                # Clone the unchanged attention body with a model-local lookup
                # that observes the original second RoPE result (K). This saves
                # a repeated rotation without changing any attention operation.
                import ast
                import inspect
                import textwrap
                import types
                from functools import update_wrapper

                original = block.self_attn.forward
                function = original.__func__
                tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "rope_apply"
                ]
                rotations = [
                    node.targets[0].id
                    for node in tree.body[0].body
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "rope_apply"
                ]
                if len(calls) != 2 or rotations != ["q", "k"]:
                    raise ValueError("Original attention Q/K rotation order changed")
                rope = function.__globals__["rope_apply"]

                def observed_rope(*args, _rope=rope, _index=index, **kwargs):
                    result = _rope(*args, **kwargs)
                    if self.recording:
                        entry = self.observed.setdefault(_index, {})
                        count = entry.get("rotations", 0) + 1
                        entry["rotations"] = count
                        if count == 2:
                            entry["k"] = result
                    return result

                namespace = dict(function.__globals__, rope_apply=observed_rope)
                clone = types.FunctionType(
                    function.__code__,
                    namespace,
                    function.__name__,
                    function.__defaults__,
                    function.__closure__,
                )
                clone.__kwdefaults__ = function.__kwdefaults__
                update_wrapper(clone, function)
                self.original_attention_forwards.append(original)
                block.self_attn.forward = types.MethodType(clone, block.self_attn)
                self.hooks.append(block.self_attn.v.register_forward_hook(value))
            else:
                self.hooks.extend(
                    (
                        block.self_attn.register_forward_pre_hook(frequency),
                        block.self_attn.norm_k.register_forward_hook(key),
                        block.self_attn.v.register_forward_hook(value),
                    )
                )
        self.full = (
            StaticGraphCallable(self.full_prediction, name="first_full_video_and_clean_kv")
            if capture
            else self.full_prediction
        )
        model._predict_video_noise = self.predict
        model._imagine_future_latents = self.imagine
        model.mot.prefill_video_cache = self.prefill

    def full_prediction(self, *args, **kwargs):
        from longwam.models.wan22.wan_video_dit import rope_apply

        self.observed = {}
        self.recording = True
        try:
            prediction = self.raw_predict(*args, **kwargs)
        finally:
            self.recording = False
        clean, _, total = self.shapes.geometry
        cache = []
        for index, block in enumerate(self.model.video_expert.blocks):
            entry = self.observed[index]
            expected_keys = (
                {"k", "v", "rotations"} if self.direct_retention else {"k", "v", "freqs"}
            )
            if set(entry) != expected_keys or int(entry["k"].shape[1]) != total:
                raise RuntimeError("First full pass did not expose original full-sequence K/V")
            attention = block.self_attn
            # Repeat the exact original pointwise RoPE over the original full
            # sequence before slicing. Original normalized K and V are captured
            # by hooks; no Linear, normalization or LoRA is bypassed.
            if self.direct_retention:
                if entry["rotations"] != 2:
                    raise RuntimeError(
                        "First full attention did not execute exactly two original rotations"
                    )
                rotated = entry["k"]
            else:
                rotated = rope_apply(
                    entry["k"],
                    entry["freqs"],
                    attention.num_heads,
                    attention.rope_compute_dtype,
                    attention.rope_direct_output,
                )
            cache.append(
                {"k": rotated[:, :clean].contiguous(), "v": entry["v"][:, :clean].contiguous()}
            )
        if self.direct_retention:
            # The original installer creates stable arenas during warmup. Its
            # copies are captured with the first full prediction, so each replay
            # refreshes all clean K/V before any future prediction consumes it.
            self.model.mot.install_video_clean_prefix_cache(cache, total_sequence_length=total)
            return prediction
        return {"prediction": prediction, "cache": cache}

    def imagine(self, *args, **kwargs):
        if self.active:
            raise RuntimeError("First-pass prefix retention is not reentrant")
        requested = kwargs.get("num_inference_steps")
        steps = self.model.imagine_infer_steps if requested is None else int(requested)
        if steps != 4:
            raise ValueError("First-pass prefix retention currently validates four video updates")
        self.active = True
        self.ready = False
        self.steps = 0
        try:
            result = self.raw_imagine(*args, **kwargs)
            if self.steps != 4:
                raise RuntimeError("Native video schedule did not execute four predictions")
            self.ready = True
            self.requests += 1
            return result
        finally:
            self.active = False

    def predict(self, *args, **kwargs):
        import torch.nn.functional as F

        if not self.active:
            raise RuntimeError(
                "Retained-prefix video prediction must run inside a fresh native imagination"
            )
        latent = args[0] if args else kwargs["latents_video"]
        clean_frames = int(self.model.num_clean_frames)
        self.shapes.configure(latent[:, :, :clean_frames])
        if self.steps == 0:
            result = self.full(*args, **kwargs)
            if self.direct_retention:
                prediction = result
            else:
                self.model.mot.install_video_clean_prefix_cache(
                    result["cache"], total_sequence_length=self.shapes.geometry[2]
                )
                prediction = result["prediction"]
        elif self.steps < 4:
            future = self.model._predict_video_noise_with_clean_prefix(*args, **kwargs)
            # The unchanged parent uses only [:,:,clean_frames:]. These clean
            # output slots are unobserved and never enter the scheduler update.
            prediction = F.pad(future, (0, 0, 0, 0, clean_frames, 0))
        else:
            raise RuntimeError("Unexpected extra video denoising step")
        self.steps += 1
        return prediction

    def prefill(self, *args, **kwargs):
        if self.active or not self.ready or self.steps != 4:
            raise RuntimeError(
                "Final video prefill requires this request's completed four-step imagination"
            )
        try:
            result = self.model.mot.prefill_video_cache_with_clean_prefix(*args, **kwargs)
            self.prefills += 1
            return result
        finally:
            self.ready = False

    def telemetry(self):
        return {
            "requests": self.requests,
            "final_prefills": self.prefills,
            "retain_original_rotated_keys": self.direct_retention,
            "prefix_copies_inside_first_graph": self.direct_retention,
            "first_full_graph": self.full.telemetry() if hasattr(self.full, "telemetry") else None,
            "scope": "Fresh first full video pass per request; no separate clean prefill; all four native updates retained",
        }
