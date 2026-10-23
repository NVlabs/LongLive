# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Request-local common subexpressions, computed by original module forwards.

Only task/state-conditioned text projections are reused. Noise, observations,
VAE latents, video self-attention KV and predictions are never memoized.
Stable buffers let captured kernels observe each request's fresh projections.
"""

from __future__ import annotations


class RequestContextCache:
    def __init__(self, model, *, capture_projections=False):
        self.model = model
        self.original_infer = model.infer_joint_ar
        self.original_append = model._append_proprio_to_context
        self.buffers = {}
        self.raw = {}
        self.contracts = {}
        self.dependencies = {}
        self.active = self.ready = False
        self.preparations = self.raw_calls = self.python_hits = 0
        self.request_preparations = 0
        self.experts = []
        for name in ("video_expert", "action_expert"):
            expert = getattr(model, name)
            embedding = name + ".text_embedding"
            self._install(expert.text_embedding, embedding)
            blocks = []
            for index, block in enumerate(expert.blocks):
                prefix = f"{name}.blocks.{index}.cross_attn"
                self._install(block.cross_attn.k, prefix + ".k", embedding)
                self._install(block.cross_attn.v, prefix + ".v", embedding)
                self._install(block.cross_attn.norm_k, prefix + ".norm_k", prefix + ".k")
                blocks.append(prefix)
            self.experts.append((embedding, blocks))
        model._append_proprio_to_context = self.append
        model.infer_joint_ar = self.run
        self.prepare = self._prepare_values
        self.projection_graph = None
        if capture_projections:
            self.enable_capture()

    def enable_capture(self):
        if self.buffers or self.projection_graph is not None:
            raise RuntimeError("Projection capture must be selected before first inference")
        from .static_graph import StaticGraphCallable

        self.projection_graph = StaticGraphCallable(
            self._prepare_values, name="request_context_projections"
        )
        self.prepare = self.projection_graph

    def _install(self, module, key, dependency=None):
        self.raw[key] = module.forward
        self.dependencies[key] = dependency

        def cached(value, *args, **kwargs):
            if args or kwargs or not self.active or not self.ready:
                raise RuntimeError("Context projection used outside its prepared inference request")
            contract = (tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
            if contract != self.contracts[key]:
                raise ValueError("Context projection input layout changed")
            parent = self.dependencies[key]
            if parent is not None:
                from .static_graph import same_expression

                if not same_expression(value, self.buffers[parent]):
                    raise ValueError(
                        f"Cross-attention received a different context expression: {key}"
                    )
            self.python_hits += 1
            return self.buffers[key]

        module.forward = cached

    def _compute(self, key, value):
        result = self.raw[key](value)
        self.raw_calls += 1
        self.contracts[key] = (tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
        if key not in self.buffers:
            self.buffers[key] = result.detach().clone()
        else:
            target = self.buffers[key]
            if (target.shape, target.dtype, target.device) != (
                result.shape,
                result.dtype,
                result.device,
            ):
                raise ValueError("Resident context buffer contract changed; recapture required")
            target.copy_(result)
        return self.buffers[key]

    def _prepare_values(self, context):
        for embedding, blocks in self.experts:
            projected = self._compute(embedding, context)
            for prefix in blocks:
                k = self._compute(prefix + ".k", projected)
                self._compute(prefix + ".norm_k", k)
                self._compute(prefix + ".v", projected)
        return tuple(self.buffers.values())

    def append(self, *args, **kwargs):
        import torch

        if not self.active or self.request_preparations or torch.is_grad_enabled():
            raise RuntimeError("Only one inference context per request is supported")
        context, mask = self.original_append(*args, **kwargs)
        self.prepare(context)
        self.preparations += 1
        self.request_preparations += 1
        self.ready = True
        return context, mask

    def run(self, *args, **kwargs):
        if self.active or self.model.training:
            raise RuntimeError("Context cache is inference-only and cannot be nested")
        if float(kwargs.get("text_cfg_scale", 1.0)) != 1.0 or kwargs.get("proprio") is None:
            raise ValueError("This candidate requires one conditional context with proprio")
        self.active = True
        self.ready = False
        self.request_preparations = 0
        try:
            result = self.original_infer(*args, **kwargs)
            if self.request_preparations != 1:
                raise RuntimeError("Inference did not prepare exactly one context")
            return result
        finally:
            self.active = self.ready = False

    def telemetry(self):
        return {
            "requests_prepared": self.preparations,
            "original_projection_python_calls_excluding_replay": self.raw_calls,
            "python_cache_hits_excluding_graph_replay": self.python_hits,
            "resident_projection_buffers": len(self.buffers),
            "resident_bytes": sum(t.numel() * t.element_size() for t in self.buffers.values()),
            "cache_scope": "Within each request only; all original forwards rerun for every new request",
            "lora_policy": "Original module.forward includes unmerged LoRA; never reads Linear.weight as a replacement",
            "projection_graph": self.projection_graph.telemetry()
            if self.projection_graph
            else None,
        }
