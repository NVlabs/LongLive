# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
# # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from transformers.models.x_clip.modeling_x_clip import x_clip_loss
from wan_5b.modules.attention import attention
from wan_5b.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WanCrossAttention,
    rope_params,
    sinusoidal_embedding_1d,
    WanCrossAttention,
    flash_attention
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

# wan 5b model compilation for flexattention
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


from utils.position_embedding_utils import (
    compute_temporal_freqs as _compute_temporal_freqs,
    select_temporal_offset_for_sample,
)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, t_scale=1.0,
                      method="linear", original_seq_len=None,
                      temporal_offset=0.0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        temporal_offset_i = select_temporal_offset_for_sample(
            temporal_offset, i, f, start_frame=start_frame)
        temporal_freqs = _compute_temporal_freqs(freqs[0], f, start_frame, t_scale, x.device,
                                                  method=method, original_seq_len=original_seq_len,
                                                  temporal_offset=temporal_offset_i)
        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

class MultiShotT2VCrossAttention(WanCrossAttention):

    def forward(self, x, context, context_lens, is_teacher_forcing=False, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B * num_chunks, L2, C]
            context_lens(Tensor): Shape [B] or [B * num_chunks]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        # Original batch size (videos)
        b_orig, L1, C = x.size()
        n, d = self.num_heads, self.head_dim

        # Effective batch size for cross-attention (videos * chunks)
        b_ctx = context.size(0)
        assert b_ctx % b_orig == 0, f"context batch ({b_ctx}) must be a multiple of x batch ({b_orig})"
        num_chunks = b_ctx // b_orig

        # Prepare context_lens for [B * num_chunks] if needed
        if context_lens is not None and context_lens.numel() == b_orig:
            context_lens = context_lens.repeat_interleave(num_chunks)
        elif context_lens is not None:
            assert context_lens.numel() == b_ctx, \
                f"context_lens must have length {b_orig} or {b_ctx}, got {context_lens.numel()}"
        # Helper to run standard cross-attention on a given x_chunk of shape [B * num_chunks, L_chunk, C]
        def _cross_attend(x_chunk):
            b_eff, L_chunk, _ = x_chunk.size()

            # compute query, key, value
            q = self.norm_q(self.q(x_chunk)).view(b_eff, -1, n, d)

            if crossattn_cache is not None:
                if not crossattn_cache["is_init"]:
                    crossattn_cache["is_init"] = True
                    k = self.norm_k(self.k(context)).view(b_eff, -1, n, d)
                    v = self.v(context).view(b_eff, -1, n, d)
                    crossattn_cache["k"] = k
                    crossattn_cache["v"] = v
                else:
                    k = crossattn_cache["k"]
                    v = crossattn_cache["v"]
            else:
                k = self.norm_k(self.k(context)).view(b_eff, -1, n, d)
                v = self.v(context).view(b_eff, -1, n, d)

            # compute attention
            x_attn = flash_attention(q, k, v, k_lens=context_lens)

            # output projection
            x_attn = x_attn.flatten(2)
            x_attn = self.o(x_attn)
            return x_attn

        if not is_teacher_forcing:
            # -------------------------------
            # Regular multi-shot: all tokens attend text, we just chunk along L1
            # x: [B, L1, C] -> [B * num_chunks, L1 / num_chunks, C]
            # -------------------------------
            assert L1 % num_chunks == 0, \
                f"L1 ({L1}) must be divisible by num_chunks ({num_chunks})"
            tokens_per_chunk = L1 // num_chunks

            x_chunked = x.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_chunked = x_chunked.reshape(b_ctx, tokens_per_chunk, C)

            x_attn = _cross_attend(x_chunked)  # [B * num_chunks, tokens_per_chunk, C]

            # reshape back to [B, L1, C]
            x_attn = x_attn.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_attn = x_attn.reshape(b_orig, L1, C)
            return x_attn

        # -------------------------------
        # Teacher forcing:
        # x is typically [B, 2 * L_tf, C], where the first half is clean and
        # the second half is noisy. Apply multi-shot cross-attention to both
        # halves.
        # -------------------------------
        assert L1 % 2 == 0, f"In teacher-forcing mode, L1 ({L1}) should be even."
        half = L1 // 2
        x_clean = x[:, :half, :]       # [B, L_tf, C]
        x_noisy = x[:, half:, :]       # [B, L_tf, C]

        def _chunk_and_attend(x_part):
            L_part = x_part.size(1)
            assert L_part % num_chunks == 0, \
                f"Segment length ({L_part}) must be divisible by num_chunks ({num_chunks})"
            tokens_per_chunk = L_part // num_chunks

            # [B, L_part, C] -> [B * num_chunks, L_part / num_chunks, C]
            x_chunked = x_part.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_chunked = x_chunked.reshape(b_ctx, tokens_per_chunk, C)

            x_attn = _cross_attend(x_chunked)  # [B * num_chunks, tokens_per_chunk, C]
            x_attn = x_attn.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_attn = x_attn.reshape(b_orig, L_part, C)
            return x_attn

        x_clean_attn = _chunk_and_attend(x_clean)
        x_noisy_attn = _chunk_and_attend(x_noisy)

        # Reassemble the full sequence from cross-attended clean and noisy halves.
        x_out = torch.cat([x_clean_attn, x_noisy_attn], dim=1)  # [B, L1, C]
        return x_out


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size if local_attn_size != -1 else 24
        self.sink_size = sink_size
        self.global_sink_size = 0
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 24 * 880 if local_attn_size == -1 else local_attn_size * 880

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        t_scale=1.0,
        use_relative_rope=False,
        method="linear",
        original_seq_len=None,
        temporal_offset=0.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            t_scale (float): Temporal RoPE interpolation scale. <1.0 compresses positions.
            use_relative_rope (bool): If True, store raw K in cache and apply RoPE
                with window-relative positions at attention time.
            method (str): RoPE method. This release supports "linear".
            original_seq_len (int): Unused by the release linear RoPE path.
            temporal_offset (float): Multi-shot RoPE offset (shot_index * phi).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # Teacher-forcing training doubles sequence length with clean/noisy halves.
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs, t_scale=t_scale,
                                    method=method, original_seq_len=original_seq_len,
                                    temporal_offset=temporal_offset).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs, t_scale=t_scale,
                                    method=method, original_seq_len=original_seq_len,
                                    temporal_offset=temporal_offset).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                x = x[:, :, :(-padded_length)] if padded_length > 0 else x
                x = x.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs, t_scale=t_scale,
                                         method=method, original_seq_len=original_seq_len,
                                         temporal_offset=temporal_offset).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs, t_scale=t_scale,
                                       method=method, original_seq_len=original_seq_len,
                                       temporal_offset=temporal_offset).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                x = x[:, :, :(-padded_length)] if padded_length > 0 else x
                x = x.transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            num_new_frames = grid_sizes[0][0].item()
            h, w = grid_sizes[0][1].item(), grid_sizes[0][2].item()
            num_new_tokens = q.shape[1]
            current_end = current_start + num_new_tokens

            if not use_relative_rope:
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame, t_scale=t_scale,
                    method=method, original_seq_len=original_seq_len,
                    temporal_offset=temporal_offset).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame, t_scale=t_scale,
                    method=method, original_seq_len=original_seq_len,
                    temporal_offset=temporal_offset).type_as(v)
                key_to_cache = roped_key
            else:
                key_to_cache = k

            sink_tokens = self.sink_size * frame_seqlen
            global_sink_tokens = getattr(self, "global_sink_size", 0) * frame_seqlen
            is_quantized_cache = kv_cache.get("quantized", False)
            if is_quantized_cache:
                kv_cache_size = kv_cache["max_blocks"] * kv_cache["block_token_size"]
            else:
                kv_cache_size = kv_cache["k"].shape[1]

            # ----- global + multi-shot pinned-sink support -----
            # Two protection mechanisms (independent, both optional):
            #   * global_sink_tokens: first N frames are permanently anchored
            #     (set via global_sink_size; never moves, always attended).
            #   * pinned region (pinned_start/pinned_len): multi-shot sink put
            #     on a scene cut. The pinned chunk lives at its original buffer
            #     position; rolling shifts non-pinned data around it.
            # effective_sink = leading buffer prefix that rolling MUST keep:
            #   pinned right after global (pinned_start == global_sink_tokens)
            #     -> global_sink_tokens + pinned_len
            #   pinned elsewhere (floating)
            #     -> global_sink_tokens
            #   no pinned
            #     -> max(global_sink_tokens, sink_tokens)  # legacy compat
            pinned_start_t = kv_cache.get("pinned_start", None)
            pinned_len_val = 0
            if pinned_start_t is not None and hasattr(pinned_start_t, 'item'):
                pinned_start_val = pinned_start_t.item()
                pinned_len_val = kv_cache["pinned_len"].item()
            else:
                pinned_start_val = -1
            has_pinned = pinned_start_val >= 0 and pinned_len_val > 0
            if has_pinned and pinned_start_val == global_sink_tokens:
                effective_sink = global_sink_tokens + pinned_len_val
            elif has_pinned:
                effective_sink = global_sink_tokens
            else:
                effective_sink = max(global_sink_tokens, sink_tokens)

            cache_update_info = None
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - effective_sink

                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                if is_quantized_cache:
                    from utils.quant import dequantize_kv_cache, k_smooth

                    max_blks = int(kv_cache["max_blocks"])
                    blk_sz = int(kv_cache["block_token_size"])
                    temp_k = dequantize_kv_cache(
                        kv_cache["k"], max_blks, self.num_heads, blk_sz, v.dtype, v.device
                    )
                    temp_v = dequantize_kv_cache(
                        kv_cache["v"], max_blks, self.num_heads, blk_sz, v.dtype, v.device
                    )
                else:
                    temp_k = kv_cache["k"].clone()
                    temp_v = kv_cache["v"].clone()
                
                temp_k[:, effective_sink:effective_sink + num_rolled_tokens] = \
                    temp_k[:, effective_sink + num_evicted_tokens:effective_sink + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, effective_sink:effective_sink + num_rolled_tokens] = \
                    temp_v[:, effective_sink + num_evicted_tokens:effective_sink + num_evicted_tokens + num_rolled_tokens].clone()
                
                temp_k[:, local_start_index:local_end_index] = (
                    k_smooth(key_to_cache) if is_quantized_cache else key_to_cache
                )
                temp_v[:, local_start_index:local_end_index] = v

                # When pinned is "floating" (lives outside effective_sink), the
                # rolling shifted non-pinned data left by num_evicted_tokens;
                # the pinned anchor must follow that shift to keep tracking the
                # same data. When pinned sits inside effective_sink (i.e. right
                # after the global region), it is part of the protected prefix
                # and rolling does not move it.
                pinned_shift = num_evicted_tokens if (has_pinned and pinned_start_val >= effective_sink) else 0

                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": effective_sink,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "new_k": key_to_cache,
                    "new_v": v,
                    "current_end": current_end,
                    "pinned_shift": pinned_shift,
                }

            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                if is_quantized_cache:
                    from utils.quant import dequantize_kv_cache, k_smooth

                    if local_start_index == 0:
                        temp_k = k_smooth(key_to_cache)
                        temp_v = v
                    else:
                        max_blks = int(kv_cache["max_blocks"])
                        blk_sz = int(kv_cache["block_token_size"])
                        temp_k = dequantize_kv_cache(
                            kv_cache["k"], max_blks, self.num_heads, blk_sz, v.dtype, v.device
                        )
                        temp_v = dequantize_kv_cache(
                            kv_cache["v"], max_blks, self.num_heads, blk_sz, v.dtype, v.device
                        )
                else:
                    temp_k = kv_cache["k"].clone()
                    temp_v = kv_cache["v"].clone()
                temp_k[:, local_start_index:local_end_index] = (
                    k_smooth(key_to_cache) if is_quantized_cache else key_to_cache
                )
                temp_v[:, local_start_index:local_end_index] = v

                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "new_k": key_to_cache,
                    "new_v": v,
                    "current_end": current_end,
                    "pinned_shift": 0,
                }

            window_start = max(0, local_end_index - self.max_attention_size)

            # Build the K/V actually attended over.
            # Cases:
            #   (a) prepend_sink  : effective_sink > 0 and out of window
            #                       -> prepend [:effective_sink] (covers global
            #                          and any pinned-merged-to-front)
            #   (b) prepend_pinned: a floating pinned region (pinned_start
            #                       >= effective_sink) lives outside the window
            #                       -> additionally prepend that pinned slice
            #   (c) otherwise     : plain sliding window
            # Note (a) and (b) are not mutually exclusive: when global is
            # enabled AND there is a separate floating pinned region outside
            # the window, both prefixes must be prepended.
            prepend_sink = effective_sink > 0 and window_start > 0
            prepend_pinned = (
                has_pinned and pinned_start_val >= effective_sink
                and pinned_start_val < window_start
            )

            if prepend_sink and prepend_pinned:
                # [global+sink] + [pinned] + [local window]
                extra = effective_sink + pinned_len_val
                effective_local_size = self.max_attention_size - extra
                local_window_start = max(effective_sink, local_end_index - effective_local_size)
                window_k = torch.cat([
                    temp_k[:, :effective_sink],
                    temp_k[:, pinned_start_val:pinned_start_val + pinned_len_val],
                    temp_k[:, local_window_start:local_end_index],
                ], dim=1)
                window_v = torch.cat([
                    temp_v[:, :effective_sink],
                    temp_v[:, pinned_start_val:pinned_start_val + pinned_len_val],
                    temp_v[:, local_window_start:local_end_index],
                ], dim=1)
            elif prepend_sink:
                effective_local_size = self.max_attention_size - effective_sink
                local_window_start = max(effective_sink, local_end_index - effective_local_size)
                window_k = torch.cat([temp_k[:, :effective_sink], temp_k[:, local_window_start:local_end_index]], dim=1)
                window_v = torch.cat([temp_v[:, :effective_sink], temp_v[:, local_window_start:local_end_index]], dim=1)
            elif prepend_pinned:
                effective_local_size = self.max_attention_size - pinned_len_val
                local_window_start = max(0, local_end_index - effective_local_size)
                window_k = torch.cat(
                    [temp_k[:, pinned_start_val:pinned_start_val + pinned_len_val],
                     temp_k[:, local_window_start:local_end_index]], dim=1)
                window_v = torch.cat(
                    [temp_v[:, pinned_start_val:pinned_start_val + pinned_len_val],
                     temp_v[:, local_window_start:local_end_index]], dim=1)
            else:
                window_k = temp_k[:, window_start:local_end_index]
                window_v = temp_v[:, window_start:local_end_index]

            if use_relative_rope:
                if prepend_sink:
                    # Sink and local window tokens get separate RoPE in a
                    # virtual contiguous layout: [sink_frames | local_frames].
                    sink_frame_count = effective_sink // frame_seqlen
                    local_tokens = window_k.shape[1] - effective_sink
                    local_frame_count = local_tokens // frame_seqlen
                    combined_frames = sink_frame_count + local_frame_count

                    sink_grid = torch.tensor(
                        [[sink_frame_count, h, w]], device=q.device, dtype=torch.long
                    ).expand(b, -1)
                    roped_sink_k = causal_rope_apply(
                        window_k[:, :effective_sink], sink_grid, freqs,
                        start_frame=0, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                    local_grid = torch.tensor(
                        [[local_frame_count, h, w]], device=q.device, dtype=torch.long
                    ).expand(b, -1)
                    roped_local_k = causal_rope_apply(
                        window_k[:, effective_sink:], local_grid, freqs,
                        start_frame=sink_frame_count, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                    roped_window_k = torch.cat([roped_sink_k, roped_local_k], dim=1)

                    q_start_frame = combined_frames - num_new_frames
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs,
                        start_frame=q_start_frame, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)
                else:
                    window_tokens = window_k.shape[1]
                    window_frames = window_tokens // frame_seqlen

                    window_grid_sizes = torch.tensor(
                        [[window_frames, h, w]], device=q.device, dtype=torch.long
                    ).expand(b, -1)

                    roped_window_k = causal_rope_apply(
                        window_k, window_grid_sizes, freqs,
                        start_frame=0, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                    q_start_frame = window_frames - num_new_frames
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs,
                        start_frame=q_start_frame, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                x = attention(roped_query, roped_window_k, window_v)
            else:
                x = attention(roped_query, window_k, window_v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = MultiShotT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        t_scale=1.0,
        use_relative_rope=False,
        method="linear",
        original_seq_len=None,
        temporal_offset=0.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            t_scale (float): Temporal RoPE interpolation scale. <1.0 compresses positions.
            use_relative_rope (bool): If True, use window-relative RoPE positions in KV cache path.
            method (str): RoPE method. This release supports "linear".
            original_seq_len (int): Unused by the release linear RoPE path.
            temporal_offset (float): Multi-shot RoPE offset (shot_index * phi).
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, t_scale=t_scale,
            use_relative_rope=use_relative_rope,
            method=method, original_seq_len=original_seq_len,
            temporal_offset=temporal_offset)
        
        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        is_tf = (x.shape[1] == seq_lens[0].item() * 2)
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache, is_teacher_forcing=is_tf)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already formatted as
            # (current_end, local_end_index, cache_update_info).
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video with causal attention.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video), 'i2v' (image-to-video), or 'ti2v'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
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

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads,
                                  local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self._block_mask_batch_size = 0

        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = False
        self.t_scale = 1.0
        self.use_relative_rope = False
        self.rope_method = "linear"
        self.original_seq_len = None
        self.rope_temporal_offset = 0.0
        self.kv_quant_config = None

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 31,
        frame_seqlen: int = 880, num_frame_per_block=3,
        batch_size=None,
    ) -> BlockMask:

        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length
            return (q_idx == kv_idx) | (is_real_q & is_real_k & (kv_idx < ends[q_idx]))

        block_mask = create_block_mask(attention_mask, B=batch_size, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 31,
        frame_seqlen: int = 880, num_frame_per_block=1,
        batch_size=None,
    ) -> BlockMask:
        """
        Block-wise causal mask. The mask is defined only by the AR chunk size:
        a token can attend to all tokens before the end of its current
        num_frame_per_block chunk.
        """
        print(f"num_frame_per_block: {num_frame_per_block}")
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        block_size = frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            # Apply only to real tokens in [0, total_length); padding keeps only
            # self-loops.
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            # End position of the block containing the current token.
            current_block_end = ((q_idx // block_size) + 1) * block_size

            clean_mask = is_real_q & is_real_k & (kv_idx < current_block_end)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask

        block_mask = create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int = 31,
        frame_seqlen: int = 880,
        num_frame_per_block: int = 1,
        batch_size: int | None = None,
    ):
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        attention_block_size = frame_seqlen * num_frame_per_block

        # Use pure mathematical coordinates; do not introduce external tensor
        # lookup tables here.
        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            # ==========================================
            # 1. Clean-frame mask.
            # ==========================================
            is_clean_q = q_idx < clean_ends

            # End position of the block containing the current token.
            clean_block_idx = q_idx // attention_block_size
            current_clean_block_end = (clean_block_idx + 1) * attention_block_size

            clean_mask = (
                is_clean_q
                & (kv_idx < current_clean_block_end)
            )

            # ==========================================
            # 2. Noisy-frame mask.
            # ==========================================
            is_noisy_q = q_idx >= clean_ends

            noisy_rel_idx = q_idx - clean_ends
            block_index = noisy_rel_idx // attention_block_size

            # C1: noisy tokens in the same block.
            noisy_block_start = clean_ends + (block_index * attention_block_size)
            noisy_block_end = noisy_block_start + attention_block_size
            C1 = (kv_idx >= noisy_block_start) & (kv_idx < noisy_block_end)

            # C2: clean context tokens from previous blocks.
            context_end_for_noisy = block_index * attention_block_size

            C2 = kv_idx < context_end_for_noisy
            noise_mask = is_noisy_q & (C1 | C2)

            # ==========================================
            # 3. Final merge.
            # ==========================================
            eye_mask = q_idx == kv_idx
            return eye_mask | (is_real_q & is_real_k & (clean_mask | noise_mask))

        # _compile=True is required here. Triton compiles the mathematical
        # formula above directly into a memory-efficient block-sparse matrix.
        block_mask = create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(block_mask)
        
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask_natural(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        sp_size: int = 1,
        batch_size: int | None = None,
    ):
        """Teacher-Forcing attention mask for the *natural* interleaved layout
        produced directly by `all_to_all(scatter=head, gather=seq)`:

            [r0_clean, r0_noisy, r1_clean, r1_noisy, ..., r_{sp-1}_clean, r_{sp-1}_noisy]

        Semantically equivalent to :func:`_prepare_teacher_forcing_mask` (which
        assumes the [all_clean; all_noisy] layout), but the mask decodes
        ``is_noisy`` and ``global_frame`` directly from the token index, so
        :func:`distributed_flex_attention` no longer has to reshape/permute
        tokens after all_to_all.
        """
        assert num_frames % sp_size == 0, (
            f"num_frames ({num_frames}) must be divisible by sp_size ({sp_size}) "
            f"for natural TF layout"
        )

        F_local = num_frames // sp_size
        clean_half = F_local * frame_seqlen            # per-rank, clean side
        per_rank_len = 2 * clean_half                  # per-rank, clean + noisy
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            # ---- decode q ----
            r_q = q_idx // per_rank_len
            in_rank_q = q_idx % per_rank_len
            is_noisy_q = in_rank_q >= clean_half
            side_q = in_rank_q % clean_half              # offset within clean/noisy half
            global_f_q = r_q * F_local + side_q // frame_seqlen
            block_q = global_f_q // num_frame_per_block

            # ---- decode k ----
            r_k = kv_idx // per_rank_len
            in_rank_k = kv_idx % per_rank_len
            is_noisy_k = in_rank_k >= clean_half
            side_k = in_rank_k % clean_half
            global_f_k = r_k * F_local + side_k // frame_seqlen
            block_k = global_f_k // num_frame_per_block

            # 1. clean_q -> clean_k: blockwise causal.
            clean2clean = (
                (~is_noisy_q) & (~is_noisy_k)
                & (block_k <= block_q)
            )

            # 2. noisy_q -> clean_k: strictly earlier blocks.
            noisy2clean = (
                is_noisy_q & (~is_noisy_k)
                & (block_k < block_q)
            )

            # 3. noisy_q -> noisy_k: only tokens within the same block.
            noisy2noisy = (
                is_noisy_q & is_noisy_k
                & (block_k == block_q)
            )

            eye_mask = q_idx == kv_idx
            return eye_mask | (
                is_real_q & is_real_k
                & (clean2clean | noisy2clean | noisy2noisy)
            )

        block_mask = create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[TF mask natural] sp_size={sp_size} F_local={F_local} "
                f"clean_half={clean_half} per_rank_len={per_rank_len} "
                f"total_length={total_length} "
                f"num_frame_per_block={num_frame_per_block}"
            )
            print(block_mask)

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                is_quantized = cache.get("quantized", False)
                
                if update_info["action"] == "roll_and_insert":
                    # Apply the rolling update.
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    if is_quantized:
                        from utils.quant import clone_quantized_tensor, quantize_to_fp4

                        blk_sz = int(cache["block_token_size"])
                        sink_blks = sink_tokens // blk_sz
                        evict_blks = num_evicted_tokens // blk_sz
                        roll_blks = num_rolled_tokens // blk_sz

                        for i in range(roll_blks):
                            src = sink_blks + evict_blks + i
                            dst = sink_blks + i
                            cache["k"][dst] = clone_quantized_tensor(cache["k"][src])
                            cache["v"][dst] = clone_quantized_tensor(cache["v"][src])

                        start_blk = local_start_index // blk_sz
                        n_insert_blks = (local_end_index - local_start_index) // blk_sz
                        head_dim = new_k.shape[-1]
                        for bi in range(n_insert_blks):
                            blk_idx = start_blk + bi
                            ts = bi * blk_sz
                            te = ts + blk_sz
                            k_block = new_k[0, ts:te, :, :].reshape(-1, head_dim).contiguous()
                            v_block = new_v[0, ts:te, :, :].reshape(-1, head_dim).contiguous()
                            cache["k"][blk_idx] = quantize_to_fp4(k_block, self.kv_quant_config)
                            cache["v"][blk_idx] = quantize_to_fp4(v_block, self.kv_quant_config)
                    else:
                        # Roll cached tokens.
                        cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                        # Insert the new key/value tensors.
                        cache["k"][:, local_start_index:local_end_index] = new_k
                        cache["v"][:, local_start_index:local_end_index] = new_v

                    # If a pinned multi-shot sink lives outside position 0,
                    # the rolling shifted everything left by num_evicted_tokens;
                    # pinned_start must follow so it tracks the same data.
                    pinned_shift = update_info.get("pinned_shift", 0)
                    if pinned_shift > 0 and "pinned_start" in cache:
                        cache["pinned_start"].sub_(pinned_shift)

                elif update_info["action"] == "direct_insert":
                    # Insert directly.
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    if is_quantized:
                        from utils.quant import quantize_to_fp4

                        blk_sz = int(cache["block_token_size"])
                        start_blk = local_start_index // blk_sz
                        n_insert_blks = (local_end_index - local_start_index) // blk_sz
                        head_dim = new_k.shape[-1]
                        for bi in range(n_insert_blks):
                            blk_idx = start_blk + bi
                            ts = bi * blk_sz
                            te = ts + blk_sz
                            k_block = new_k[0, ts:te, :, :].reshape(-1, head_dim).contiguous()
                            v_block = new_v[0, ts:te, :, :].reshape(-1, head_dim).contiguous()
                            cache["k"][blk_idx] = quantize_to_fp4(k_block, self.kv_quant_config)
                            cache["v"][blk_idx] = quantize_to_fp4(v_block, self.kv_quant_config)
                    else:
                        # Insert the new key/value tensors.
                        cache["k"][:, local_start_index:local_end_index] = new_k
                        cache["v"][:, local_start_index:local_end_index] = new_v
            
            # Update cache indices.
            kv_cache[block_index]["global_end_index"].fill_(current_end)
            kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (880 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(x)

        # time embeddings
        if t.dim() == 1:
            raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")

        bt = t.size(0)
        t_len = t.size(1)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, t_len)).type_as(x))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, F, 6, C

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            t_scale=self.t_scale,
            use_relative_rope=self.use_relative_rope,
            method=self.rope_method,
            original_seq_len=self.original_seq_len,
            temporal_offset=self.rope_temporal_offset,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache updates from every block.
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Keep only basic metadata for later blocks, without the
                    # concrete cache-update payload.
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Keep only basic metadata for later blocks, without the
                    # concrete cache-update payload.
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result

        # Apply all cache updates after every block has run.
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        # Recreate mask when batch size changes to avoid Triton broadcasting bug
        current_batch_size = x.shape[0]
        if self.block_mask is None or self._block_mask_batch_size != current_batch_size:
            self._block_mask_batch_size = current_batch_size
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        batch_size=current_batch_size,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        batch_size=current_batch_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        batch_size=current_batch_size,
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        max_len = int(seq_lens.max().item())
        assert max_len > 0, "Token sequence length is zero after patch embedding"
        # Pad all samples to the batch max length instead of the first sample length
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")
        bt = t.size(0)
        t_len = t.size(1)
        t_ori_shape = t.shape
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, t_len)).type_as(x))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, F, 6, C

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros(t_ori_shape, device=t.device, dtype=t.dtype)
            bt_clean = aug_t.size(0)
            t_clean_len = aug_t.size(1)
            aug_t = aug_t.flatten()
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t).unflatten(0, (bt_clean, t_clean_len)).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(2, (6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            t_scale=self.t_scale,
            method=self.rope_method,
            original_seq_len=self.original_seq_len,
            temporal_offset=self.rope_temporal_offset,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
