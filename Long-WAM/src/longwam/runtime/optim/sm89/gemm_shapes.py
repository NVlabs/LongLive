# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Retain original token-row GEMM geometry while reusing a causal prefix.

This does not change weights, arithmetic dtype, contraction dimensions or any
selected input row. Extra zero rows reproduce the original kernel workload
shape; their outputs are discarded. Full-model byte validation remains required.
"""

from __future__ import annotations


GROUPS = frozenset(
    {"time", "self_qkv", "self_o", "cross_q", "cross_o", "ffn_in", "ffn_out", "head"}
)


def group_of(name):
    if name.startswith(("time_embedding.", "time_projection.")):
        return "time"
    if name.endswith((".self_attn.q", ".self_attn.k", ".self_attn.v")):
        return "self_qkv"
    if name.endswith(".self_attn.o"):
        return "self_o"
    if name.endswith(".cross_attn.q"):
        return "cross_q"
    if name.endswith(".cross_attn.o"):
        return "cross_o"
    if name.endswith(".ffn.0"):
        return "ffn_in"
    if name.endswith(".ffn.2"):
        return "ffn_out"
    if name.startswith("head."):
        return "head"
    return None


class VideoGemmShapes:
    def __init__(self, model, groups, *, reuse_buffers=False):
        import torch

        self.model = model
        self.groups = frozenset(groups)
        self.geometry = None
        self.reuse_buffers = bool(reuse_buffers)
        self.arenas = {}
        if not self.groups or not self.groups <= GROUPS:
            raise ValueError("Unknown or empty video GEMM group selection")
        if (model.num_clean_frames, model.num_imagine_frames) != (4, 2) or tuple(
            model.video_expert.patch_size
        )[0] != 1:
            raise ValueError(
                "Shape-preserving prefix currently requires frame-local P48/V4/A10 IDM"
            )
        self.raw = {}
        self.installed = {}
        self.padded_calls = 0
        self.padded_elements = 0
        for name, module in model.video_expert.named_modules():
            if isinstance(module, torch.nn.Linear) and group_of(name) in self.groups:
                original = module.forward
                self.raw[name] = (module, original)

                def padded(value, *args, _original=original, **kwargs):
                    if self.geometry is None or value.ndim not in (2, 3):
                        return _original(value, *args, **kwargs)
                    clean, future, total = self.geometry
                    rows = int(value.shape[-2])
                    if rows not in (clean, future):
                        return _original(value, *args, **kwargs)
                    if value.ndim == 3 and value.shape[0] != 1:
                        raise ValueError("Prefix GEMM geometry requires batch one")
                    if torch.is_grad_enabled():
                        raise RuntimeError("Prefix GEMM geometry is inference-only")
                    import torch.nn.functional as F

                    before = 0 if rows == clean else clean
                    after = total - before - rows
                    if self.reuse_buffers:
                        # Reuse only constant-zero padding. Active rows are
                        # overwritten on every call. Separate clean/future
                        # arenas keep discarded rows permanently zero.
                        shape = (*value.shape[:-2], total, value.shape[-1])
                        key = (shape, value.dtype, value.device, before, rows)
                        if key not in self.arenas:
                            self.arenas[key] = torch.zeros(
                                shape, dtype=value.dtype, device=value.device
                            )
                        padded = self.arenas[key]
                        padded[..., before : before + rows, :].copy_(value)
                    else:
                        padded = F.pad(value, (0, 0, before, after))
                    result = _original(padded, *args, **kwargs)
                    self.padded_calls += 1
                    self.padded_elements += (total - rows) * int(value.shape[-1])
                    return result[..., before : before + rows, :]

                module.forward = padded
                self.installed[name] = padded
        if not self.raw:
            raise ValueError("No selected video Linear modules found")
        self.original_prefix = model._prefill_video_clean_prefix

        def configured_prefix(*args, **kwargs):
            value = args[0] if args else kwargs["clean_latents"]
            self.configure(value)
            return self.original_prefix(*args, **kwargs)

        self.configured_prefix = configured_prefix
        model._prefill_video_clean_prefix = configured_prefix

    def configure(self, clean_latents):
        if clean_latents.ndim != 5 or clean_latents.shape[0] != 1 or clean_latents.shape[2] != 4:
            raise ValueError("Unexpected clean-latent prefix geometry")
        _, patch_h, patch_w = self.model.video_expert.patch_size
        height, width = map(int, clean_latents.shape[-2:])
        if height % patch_h or width % patch_w:
            raise ValueError("Latent size is not divisible by the original patch")
        tokens = (height // patch_h) * (width // patch_w)
        geometry = (4 * tokens, 2 * tokens, 6 * tokens)
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError("Video geometry changed; construct a new profile")
        self.geometry = geometry

    def restore(self):
        """Diagnostic-only restoration, before installing any captured wrappers."""
        if self.model._prefill_video_clean_prefix is not self.configured_prefix:
            raise RuntimeError("Cannot restore GEMM wrappers after graph installation")
        for name, (module, original) in self.raw.items():
            if module.forward is not self.installed[name]:
                raise RuntimeError("Another wrapper replaced a selected Linear")
            module.forward = original
        self.model._prefill_video_clean_prefix = self.original_prefix

    def telemetry(self):
        return {
            "groups": sorted(self.groups),
            "clean_future_full_rows": list(self.geometry) if self.geometry else None,
            "reuse_constant_zero_arenas": self.reuse_buffers,
            "padding_arenas": len(self.arenas),
            "padding_arena_bytes": sum(
                value.numel() * value.element_size() for value in self.arenas.values()
            ),
            "selected_linear_modules": len(self.raw),
            "padded_python_calls_excluding_graph_replay": self.padded_calls,
            "extra_input_elements_excluding_graph_replay": self.padded_elements,
            "policy": "Original Linear.forward on original full token-row shape; discard only added zero rows",
        }
