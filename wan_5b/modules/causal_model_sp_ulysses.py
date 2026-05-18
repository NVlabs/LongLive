# Copyright 2024-2025 LongLive Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Causal Wan2.2-TI2V-5B model with Ulysses-style sequence parallelism.

This inference-only variant mirrors the regular CausalWanModel parameter layout
so existing generator checkpoints can be loaded into ``model.*`` keys.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from wan_5b.modules.attention import attention
from wan_5b.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)
from wan_5b.modules.causal_model import causal_rope_apply, MultiShotT2VCrossAttention
from wan_5b.distributed.sp_ulysses_inference import (
    get_sp_rank,
    get_sp_world_size,
    is_sp_enabled,
    sp_all_gather,
    sp_scatter,
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)


QK_QUANT_SIM = os.environ.get("QK_QUANT_SIM", "0") == "1"
QK_QUANT_RATIO = float(os.environ.get("QK_QUANT_RATIO", "0.28125"))


try:
    import torch.cuda.nvtx as nvtx
    NVTX_ENABLED = True
except Exception:
    nvtx = None
    NVTX_ENABLED = False


class NVTXRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if NVTX_ENABLED:
            nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc, tb):
        if NVTX_ENABLED:
            nvtx.range_pop()


def _get_d_quant(head_dim: int, ratio: float = QK_QUANT_RATIO) -> int:
    d_quant = max(8, round(head_dim * ratio))
    return ((d_quant + 7) // 8) * 8


def _compute_ulysses_frame_info(num_frames, local_seq_len, sp_world_size, sp_rank):
    if num_frames == 1:
        return 1, local_seq_len, 0
    assert num_frames % sp_world_size == 0, (
        f"Ulysses SP requires num_frames ({num_frames}) divisible by "
        f"world_size ({sp_world_size})."
    )
    local_frames = num_frames // sp_world_size
    frame_seqlen = local_seq_len // local_frames
    frame_offset = sp_rank * local_frames
    return local_frames, frame_seqlen, frame_offset


class UlyssesCausalWanSelfAttention(nn.Module):
    """Causal self-attention with Ulysses sequence/head exchange."""

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0,
                 qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.global_sink_size = 0
        self.qk_norm = qk_norm
        self.eps = eps

        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if non_neg_vals else -1
        self.max_attention_size = 28160 if max_local == -1 else max_local * 880

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache=None,
                current_start=0, cache_start=None, t_scale=1.0,
                use_relative_rope=False, method="linear",
                original_seq_len=None, temporal_offset=0.0):
        b, s_local = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        with NVTXRange("ulysses_qkv_proj"):
            q = self.norm_q(self.q(x)).view(b, s_local, n, d)
            k = self.norm_k(self.k(x)).view(b, s_local, n, d)
            v = self.v(x).view(b, s_local, n, d)

        if kv_cache is None:
            return self._forward_no_cache(q, k, v, grid_sizes, freqs, t_scale, method, original_seq_len)
        return self._forward_with_cache(
            q, k, v, grid_sizes, freqs, kv_cache, current_start, cache_start,
            t_scale, use_relative_rope, method, original_seq_len, temporal_offset,
        )

    def _exchange_qkv(self, q, k, v):
        if QK_QUANT_SIM and is_sp_enabled():
            d = q.shape[-1]
            d_quant = _get_d_quant(d)
            q_heads = F.pad(ulysses_seq_to_head(q[..., :d_quant]), (0, d - d_quant))
            k_heads = F.pad(ulysses_seq_to_head(k[..., :d_quant]), (0, d - d_quant))
            v_heads = ulysses_seq_to_head(v)
        else:
            q_heads = ulysses_seq_to_head(q)
            k_heads = ulysses_seq_to_head(k)
            v_heads = ulysses_seq_to_head(v)
        return q_heads, k_heads, v_heads

    def _forward_no_cache(self, q, k, v, grid_sizes, freqs, t_scale, method, original_seq_len):
        with NVTXRange("ulysses_seq_to_head"):
            q_heads, k_heads, v_heads = self._exchange_qkv(q, k, v)

        with NVTXRange("ulysses_rope"):
            roped_q = rope_apply(
                q_heads, grid_sizes, freqs, t_scale=t_scale,
                method=method, original_seq_len=original_seq_len,
            ).type_as(v_heads)
            roped_k = rope_apply(
                k_heads, grid_sizes, freqs, t_scale=t_scale,
                method=method, original_seq_len=original_seq_len,
            ).type_as(v_heads)

        with NVTXRange("ulysses_attention"):
            out_heads = attention(roped_q, roped_k, v_heads, causal=True)
        with NVTXRange("ulysses_head_to_seq"):
            out = ulysses_head_to_seq(out_heads)
        with NVTXRange("ulysses_out_proj"):
            return self.o(out.flatten(2))

    def _forward_with_cache(self, q, k, v, grid_sizes, freqs, kv_cache,
                            current_start, cache_start, t_scale,
                            use_relative_rope, method, original_seq_len,
                            temporal_offset):
        sp_world_size = get_sp_world_size() if is_sp_enabled() else 1
        s_total = q.shape[1] * sp_world_size
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        current_end = current_start + s_total

        with NVTXRange("ulysses_seq_to_head"):
            q_heads, k_heads, v_heads = self._exchange_qkv(q, k, v)

        with NVTXRange("ulysses_rope_cached"):
            roped_q = causal_rope_apply(
                q_heads, grid_sizes, freqs, start_frame=current_start_frame,
                t_scale=t_scale, method=method, original_seq_len=original_seq_len,
                temporal_offset=temporal_offset,
            ).type_as(v_heads)
            if use_relative_rope:
                key_to_cache = k_heads
            else:
                key_to_cache = causal_rope_apply(
                    k_heads, grid_sizes, freqs, start_frame=current_start_frame,
                    t_scale=t_scale, method=method, original_seq_len=original_seq_len,
                    temporal_offset=temporal_offset,
                ).type_as(v_heads)

        with NVTXRange("ulysses_cache_update"):
            k_full, v_full = self._update_cache_and_get_kv(
                key_to_cache, v_heads, kv_cache, current_start, current_end, frame_seqlen,
            )

        if use_relative_rope:
            raise NotImplementedError("use_relative_rope is not implemented for SP inference.")

        with NVTXRange("ulysses_cached_attention"):
            out_heads = attention(roped_q, k_full, v_full, causal=False)
        with NVTXRange("ulysses_head_to_seq"):
            out = ulysses_head_to_seq(out_heads)
        with NVTXRange("ulysses_out_proj"):
            out = self.o(out.flatten(2))
        return out, (current_end, kv_cache["local_end_index"].item(), None)

    def _effective_sink(self, kv_cache, frame_seqlen):
        sink_tokens = self.sink_size * frame_seqlen
        global_sink_tokens = getattr(self, "global_sink_size", 0) * frame_seqlen
        pinned_start_t = kv_cache.get("pinned_start", None)
        if pinned_start_t is not None and hasattr(pinned_start_t, "item"):
            pinned_start = pinned_start_t.item()
            pinned_len = kv_cache["pinned_len"].item()
        else:
            pinned_start = -1
            pinned_len = 0
        has_pinned = pinned_start >= 0 and pinned_len > 0
        if has_pinned and pinned_start == global_sink_tokens:
            effective_sink = global_sink_tokens + pinned_len
        elif has_pinned:
            effective_sink = global_sink_tokens
        else:
            effective_sink = max(global_sink_tokens, sink_tokens)
        return effective_sink, pinned_start, pinned_len, has_pinned

    def _update_cache_and_get_kv(self, k_new, v_new, kv_cache, current_start,
                                 current_end, frame_seqlen):
        b, s_new, _, _ = k_new.shape
        kv_cache_size = kv_cache["k"].shape[1]
        global_end_prev = kv_cache["global_end_index"].item()
        local_end_prev = kv_cache["local_end_index"].item()
        is_recompute = current_end <= global_end_prev and current_start > 0

        effective_sink, pinned_start, pinned_len, has_pinned = self._effective_sink(kv_cache, frame_seqlen)
        need_roll = (
            self.local_attn_size != -1
            and current_end > global_end_prev
            and s_new + local_end_prev > kv_cache_size
        )

        if need_roll:
            num_evicted = s_new + local_end_prev - kv_cache_size
            num_rolled = local_end_prev - num_evicted - effective_sink
            if num_rolled > 0:
                kv_cache["k"][:, effective_sink:effective_sink + num_rolled] = (
                    kv_cache["k"][
                        :, effective_sink + num_evicted:effective_sink + num_evicted + num_rolled
                    ].clone()
                )
                kv_cache["v"][:, effective_sink:effective_sink + num_rolled] = (
                    kv_cache["v"][
                        :, effective_sink + num_evicted:effective_sink + num_evicted + num_rolled
                    ].clone()
                )
            if has_pinned and pinned_start >= effective_sink and "pinned_start" in kv_cache:
                kv_cache["pinned_start"].sub_(num_evicted)
                pinned_start -= num_evicted
            local_end_new = local_end_prev + (current_end - global_end_prev) - num_evicted
        else:
            local_end_new = local_end_prev + (current_end - global_end_prev)

        local_start_new = local_end_new - s_new
        write_start = max(local_start_new, effective_sink) if is_recompute else local_start_new
        write_offset = max(0, write_start - local_start_new)
        write_len = max(0, local_end_new - write_start)
        if write_len > 0:
            kv_cache["k"][:, write_start:local_end_new] = k_new[:, write_offset:write_offset + write_len]
            kv_cache["v"][:, write_start:local_end_new] = v_new[:, write_offset:write_offset + write_len]

        if not is_recompute:
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_new)

        window_start = max(0, local_end_new - self.max_attention_size)
        prepend_sink = effective_sink > 0 and window_start > 0
        prepend_pinned = has_pinned and pinned_start >= effective_sink and pinned_start < window_start

        if prepend_sink and prepend_pinned:
            extra = effective_sink + pinned_len
            effective_local_size = max(0, self.max_attention_size - extra)
            local_window_start = max(effective_sink, local_end_new - effective_local_size)
            k_full = torch.cat([
                kv_cache["k"][:, :effective_sink],
                kv_cache["k"][:, pinned_start:pinned_start + pinned_len],
                kv_cache["k"][:, local_window_start:local_end_new],
            ], dim=1)
            v_full = torch.cat([
                kv_cache["v"][:, :effective_sink],
                kv_cache["v"][:, pinned_start:pinned_start + pinned_len],
                kv_cache["v"][:, local_window_start:local_end_new],
            ], dim=1)
        elif prepend_sink:
            effective_local_size = max(0, self.max_attention_size - effective_sink)
            local_window_start = max(effective_sink, local_end_new - effective_local_size)
            k_full = torch.cat([
                kv_cache["k"][:, :effective_sink],
                kv_cache["k"][:, local_window_start:local_end_new],
            ], dim=1)
            v_full = torch.cat([
                kv_cache["v"][:, :effective_sink],
                kv_cache["v"][:, local_window_start:local_end_new],
            ], dim=1)
        elif prepend_pinned:
            effective_local_size = max(0, self.max_attention_size - pinned_len)
            local_window_start = max(0, local_end_new - effective_local_size)
            k_full = torch.cat([
                kv_cache["k"][:, pinned_start:pinned_start + pinned_len],
                kv_cache["k"][:, local_window_start:local_end_new],
            ], dim=1)
            v_full = torch.cat([
                kv_cache["v"][:, pinned_start:pinned_start + pinned_len],
                kv_cache["v"][:, local_window_start:local_end_new],
            ], dim=1)
        else:
            k_full = kv_cache["k"][:, window_start:local_end_new]
            v_full = kv_cache["v"][:, window_start:local_end_new]

        return k_full, v_full


class UlyssesCausalWanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, local_attn_size=-1,
                 sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = UlyssesCausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = MultiShotT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                kv_cache=None, crossattn_cache=None, current_start=0,
                cache_start=None, num_frames=None, local_frame_offset=0,
                t_scale=1.0, use_relative_rope=False, method="linear",
                original_seq_len=None, temporal_offset=0.0):
        num_frames_total = e.shape[1] if num_frames is None else num_frames
        local_seq_len = x.shape[1]
        if is_sp_enabled() and local_seq_len > 0:
            actual_local_frames, local_frame_seqlen, _ = _compute_ulysses_frame_info(
                num_frames_total, local_seq_len, get_sp_world_size(), get_sp_rank()
            )
            e_local = e[:, local_frame_offset:local_frame_offset + actual_local_frames]
        else:
            actual_local_frames = num_frames_total
            local_frame_seqlen = local_seq_len // actual_local_frames if actual_local_frames > 0 else 0
            e_local = e

        e_chunks = (self.modulation.unsqueeze(1) + e_local).chunk(6, dim=2)
        x_normed = self.norm1(x)
        if actual_local_frames > 0 and local_frame_seqlen > 0:
            x_mod = x_normed.unflatten(dim=1, sizes=(actual_local_frames, local_frame_seqlen))
            x_mod = (x_mod * (1 + e_chunks[1]) + e_chunks[0]).flatten(1, 2)
        else:
            x_mod = x_normed

        self_attn_result = self.self_attn(
            x_mod, seq_lens, grid_sizes, freqs, kv_cache, current_start, cache_start,
            t_scale, use_relative_rope, method, original_seq_len, temporal_offset,
        )
        if kv_cache is not None and isinstance(self_attn_result, tuple):
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        if actual_local_frames > 0 and local_frame_seqlen > 0:
            y_mod = y.unflatten(dim=1, sizes=(actual_local_frames, local_frame_seqlen))
            x = x + (y_mod * e_chunks[2]).flatten(1, 2)
        else:
            x = x + y

        if x.shape[1] > 0:
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            x_normed = self.norm2(x)
            if actual_local_frames > 0 and local_frame_seqlen > 0:
                x_mod = x_normed.unflatten(dim=1, sizes=(actual_local_frames, local_frame_seqlen))
                y = self.ffn((x_mod * (1 + e_chunks[4]) + e_chunks[3]).flatten(1, 2))
                y_mod = y.unflatten(dim=1, sizes=(actual_local_frames, local_frame_seqlen))
                x = x + (y_mod * e_chunks[5]).flatten(1, 2)
            else:
                x = x + self.ffn(x_normed)

        return (x, cache_update_info) if cache_update_info is not None else x


class UlyssesCausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, local_frame_offset=0, actual_local_frames=None,
                frame_seqlen_global=None):
        num_frames_total = e.shape[1]
        local_seq_len = x.shape[1]
        if is_sp_enabled() and actual_local_frames is not None:
            local_frames = actual_local_frames
            local_seqlen = local_seq_len // local_frames if local_frames > 0 else frame_seqlen_global
            e_local = e[:, local_frame_offset:local_frame_offset + local_frames]
        else:
            local_frames = num_frames_total
            local_seqlen = local_seq_len // local_frames if local_frames > 0 else 0
            e_local = e

        if local_frames == 0 or local_seq_len == 0:
            b, _, _ = x.shape
            return torch.empty(
                b, 0, local_seqlen if local_seqlen else 1,
                math.prod(self.patch_size) * self.out_dim,
                dtype=x.dtype, device=x.device,
            )

        e_chunks = (self.modulation.unsqueeze(1) + e_local).chunk(2, dim=2)
        x = self.norm(x).unflatten(dim=1, sizes=(local_frames, local_seqlen))
        return self.head(x * (1 + e_chunks[1]) + e_chunks[0])


class UlyssesSPCausalWanModel(ModelMixin, ConfigMixin):
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["UlyssesCausalWanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, model_type="ti2v", patch_size=(1, 2, 2), text_len=512,
                 in_dim=48, dim=3072, ffn_dim=14336, freq_dim=256,
                 text_dim=4096, out_dim=48, num_heads=24, num_layers=30,
                 local_attn_size=-1, sink_size=0, num_frame_per_block=1,
                 qk_norm=True, cross_attn_norm=True, eps=1e-6):
        super().__init__()
        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            UlyssesCausalWanAttentionBlock(
                dim, ffn_dim, num_heads, local_attn_size, sink_size,
                qk_norm, cross_attn_norm, eps,
            )
            for _ in range(num_layers)
        ])
        self.head = UlyssesCausalHead(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)

        self.init_weights()
        self.gradient_checkpointing = False
        self.num_frame_per_block = num_frame_per_block
        self.t_scale = 1.0
        self.use_relative_rope = False
        self.rope_method = "linear"
        self.original_seq_len = None
        self.rope_temporal_offset = 0.0
        self.kv_quant_config = None

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None,
                kv_cache=None, crossattn_cache=None, current_start=0,
                cache_start=0):
        sp_world_size = get_sp_world_size() if is_sp_enabled() else 1
        sp_rank = get_sp_rank() if is_sp_enabled() else 0
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        with NVTXRange("ulysses_patch_embedding"):
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat(x)

        num_frames = t.shape[1] if t.dim() > 1 else grid_sizes[0, 0].item()
        frame_seqlen = x.shape[1] // num_frames
        if is_sp_enabled():
            assert self.num_heads % sp_world_size == 0
            assert x.shape[1] % sp_world_size == 0
            with NVTXRange("ulysses_scatter_input"):
                x = sp_scatter(x, dim=1)
            actual_local_frames, local_frame_seqlen, local_frame_offset = _compute_ulysses_frame_info(
                num_frames, x.shape[1], sp_world_size, sp_rank
            )
        else:
            actual_local_frames = num_frames
            local_frame_seqlen = frame_seqlen
            local_frame_offset = 0

        with NVTXRange("ulysses_time_embedding"):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        with NVTXRange("ulysses_text_embedding"):
            context_lens = None
            context = self.text_embedding(torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
            if clip_fea is not None:
                context = torch.concat([self.img_emb(clip_fea), context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            num_frames=num_frames,
            local_frame_offset=local_frame_offset,
            t_scale=self.t_scale,
            use_relative_rope=self.use_relative_rope,
            method=self.rope_method,
            original_seq_len=self.original_seq_len,
            temporal_offset=self.rope_temporal_offset,
        )
        for block_idx, block in enumerate(self.blocks):
            kwargs.update({
                "kv_cache": kv_cache[block_idx] if kv_cache else None,
                "crossattn_cache": crossattn_cache[block_idx] if crossattn_cache else None,
                "current_start": current_start,
                "cache_start": cache_start,
            })
            result = block(x, **kwargs)
            x = result[0] if kv_cache is not None and isinstance(result, tuple) else result

        x = self.head(
            x,
            e.unflatten(dim=0, sizes=t.shape).unsqueeze(2),
            local_frame_offset=local_frame_offset,
            actual_local_frames=actual_local_frames,
            frame_seqlen_global=frame_seqlen,
        )
        if is_sp_enabled():
            x = sp_all_gather(x, dim=2 if num_frames == 1 else 1)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        pt, ph, pw = self.patch_size
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            x_i = x[i, :f, :h * w, :].reshape(f, h, w, pt, ph, pw, c)
            x_i = x_i.permute(6, 0, 3, 1, 4, 2, 5).reshape(c, f * pt, h * ph, w * pw)
            output.append(x_i)
        return output

    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)
