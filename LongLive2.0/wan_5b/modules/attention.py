# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import functools
import os
import torch

# FA4: preferred on Blackwell (sm_100). Imported via `flash_attn.cute`. The
# inference loop on GB200 runs this path when LLV2_USE_FA4 is unset or "1".
try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func
    FLASH_ATTN_4_AVAILABLE = True
except Exception:
    FLASH_ATTN_4_AVAILABLE = False

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# TE 2.13 ships a `DotProductAttention` Module whose cuDNN-backed
# FusedAttention path is Blackwell sm_100-tuned. Default it ON when TE is
# importable; couple it with `NVTE_FLASH_ATTN=0` in the launch env so TE
# picks cuDNN FusedAttention instead of falling back to flash-attn 2 (which
# is what the original baseline used and is the kernel we want to replace).
try:
    from transformer_engine.pytorch.attention import (
        DotProductAttention as _TE_DPA,
    )
    TE_DPA_AVAILABLE = True
except Exception:
    TE_DPA_AVAILABLE = False

# Default off — iter-5 (run-20260520-022256-fa4) showed FA4 4.0.0b4 + torch
# 2.12 + cute-DSL is currently quality-FAIL on this NVFP4 pipeline (max|Δ|≈9
# vs threshold 5e-3) and ~6% slower in steady-state. Re-enable with
# LLV2_USE_FA4=1 once we have a clean qlive_fa4 baseline + working
# torch.compile interop.
_USE_FA4 = os.environ.get("LLV2_USE_FA4", "0") == "1"
_USE_TE_ATTN = os.environ.get("LLV2_USE_TE_ATTN", "0") == "1"
# iter-32: FA3 default-off. Initial sm_100 build only JIT'd common head_dim
# templates; less-common shapes throw "no kernel image is available". Rebuild
# FA3 with TORCH_CUDA_ARCH_LIST=10.0+PTX then flip this to 1.
_USE_FA3 = os.environ.get("LLV2_USE_FA3", "0") == "1"


@functools.lru_cache(maxsize=16)
def _get_te_dpa(
    num_heads: int,
    head_dim: int,
    attn_mask_type: str,
    window_left: int,
    window_right: int,
) -> "torch.nn.Module":
    """Cached TE DotProductAttention instance keyed by attention shape +
    masking. Constructed lazily and reused across forward calls. TE's DPA
    object is light at __init__ (no params); the cuDNN dispatch happens in
    forward.
    """
    ws = (window_left, window_right)
    return _TE_DPA(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        attention_dropout=0.0,
        attn_mask_type=attn_mask_type,
        window_size=ws,
        qkv_format="thd",  # varlen — flat tokens + cu_seqlens
    ).cuda()


import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)

    # TE DotProductAttention (cuDNN FusedAttention, sm_100-tuned). Opt-in via
    # LLV2_USE_TE_ATTN=1 + NVTE_FLASH_ATTN=0 in the launch env (the latter
    # stops TE from dispatching internally to flash-attn 2 — which is what
    # we're trying to replace).
    #
    # iter-6 unit test (agent/te_dpa_unit_test.py, ran in qlive env outside
    # any TE FP8 autocast scope) showed `padding`+`window=(-1,-1)` matches
    # FA2 (causal=False) at max|Δ|=3e-5 (bf16 rounding). But iter-6 in the
    # full pipeline gave video PSNR = 10.4 dB — the math goes wrong because
    # the model's TE-wrapped Linear forwards open a `te.fp8_autocast(...)`
    # scope, and the DPA inside that scope tries to run FP8 attention without
    # calibrated scales. Wrapping the DPA call in `fp8_autocast(enabled=False)`
    # forces it to bf16 cuDNN attention, which is what the unit test verified.
    if _USE_TE_ATTN and TE_DPA_AVAILABLE:
        n_q = q.size(1)  # after flatten(0,1), q is [Lq_total, n, d]; size(1)=n
        d = q.size(2)
        ws_left = -1 if window_size[0] is None or window_size[0] < 0 else int(window_size[0])
        ws_right = -1 if window_size[1] is None or window_size[1] < 0 else int(window_size[1])
        mask_type = "padding_causal" if causal else "padding"
        if q_scale is not None and softmax_scale is None:
            softmax_scale = float(q_scale) / (d ** 0.5)
        dpa = _get_te_dpa(n_q, d, mask_type, ws_left, ws_right)
        # iter-6b confirmed wrapping each DPA call in a fp8_autocast(enabled=False)
        # context is (a) a no-op for correctness (latent drift unchanged) and
        # (b) a recompile trap for dynamo (medians thrash between 1272 and 1859
        # across prompts). Just call DPA directly.
        out = dpa(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_kv=lk,
        )
        if out.dim() == 2:
            out = out.view(-1, n_q, d)
        x = out.unflatten(0, (b, lq))
    # FA4 (Blackwell sm_100): preferred when available unless caller pins
    # version=2/3 or env var disables. iter-5.
    elif (version is None or version == 4) and _USE_FA4 and FLASH_ATTN_4_AVAILABLE:
        # FA4 uses None for "no window"; FA2 used (-1, -1).
        ws = (
            None if window_size[0] is None or window_size[0] < 0 else window_size[0],
            None if window_size[1] is None or window_size[1] < 0 else window_size[1],
        )
        out = _fa4_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=ws,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        x = out.unflatten(0, (b, lq))
    elif (version == 3 or (version is None and _USE_FA3)) and FLASH_ATTN_3_AVAILABLE:
        # iter-32: FA3 (built from hopper/ source). Returns a single tensor
        # at default `return_attn_probs=False`, NOT a (out, lse) tuple — the
        # original `[0]` here was indexing into dim-0 of the output, giving a
        # bogus (24, 128) slice. Use the return value directly. window_size
        # supported by FA3 (default (-1, -1)); thread the caller's value
        # through so local-attention windows work.
        out = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        x = out.unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
