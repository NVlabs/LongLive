# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/runtime/test_streaming_vae.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Compare real causal VAE full and window-scoped streaming computations on CPU."""

from types import SimpleNamespace

import pytest
import torch

from longwam.runtime.streaming_vae import StreamingVAE
from longwam.models.wan22.wan_video_vae import VideoVAE_, VideoVAE38_


@pytest.mark.parametrize("patched", [False, True])
def test_causal_window_matches_full_encoding_and_reset(patched):
    torch.manual_seed(7)
    cls = VideoVAE38_ if patched else VideoVAE_
    vae = cls(
        dim=4,
        z_dim=2,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
    ).eval()
    frames = torch.randn(1, 3, 9, 16, 16)
    scale = [torch.zeros(2), torch.ones(2)]
    with torch.no_grad():
        reference = vae.encode(frames, scale)
    wrapper = SimpleNamespace(temporal_downsample_factor=4, model=vae)
    model = SimpleNamespace(
        vae=wrapper,
        num_clean_frames=3,
        _stream_video_latents_start=vae.stream_encode_start,
        _stream_video_latents_step=vae.stream_encode_step,
        _stream_video_latents_finish=lambda: vae.stream_encode_finish(scale),
    )
    session = StreamingVAE(model, past_steps=8, sample_stride=1)
    session.begin(8, [(i, frames[0, :, i]) for i in range(3)])
    for i in range(3, 9):
        session.feed(i, frames[0, :, i])
    torch.testing.assert_close(session.finish(8), reference, rtol=1e-5, atol=1e-5)
    session.begin(8, [(i, frames[0, :, i]) for i in range(9)])
    torch.testing.assert_close(session.finish(8), reference, rtol=1e-5, atol=1e-5)


def test_incomplete_windows_and_invalid_cadence_fail():
    fake = SimpleNamespace(vae=SimpleNamespace(temporal_downsample_factor=4))
    with pytest.raises(ValueError, match="complete"):
        StreamingVAE(fake, past_steps=3, sample_stride=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("length", [588, 720, 882, 1080, 1470, 1800, 2646, 3240])
def test_segmented_attention_matches_fp32_reference_and_cuda_graph(length):
    from longwam.runtime.optim.action_segmented_attention import action_segmented_attention

    torch.manual_seed(7)
    q = torch.randn(1, 32, 3072, device="cuda", dtype=torch.bfloat16)
    kv, vv = [torch.randn(1, length, 3072, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    ka, va = [torch.randn_like(q) for _ in range(2)]
    mask = torch.rand(32, length + 32, device="cuda") > 0.2
    mask[0] = False
    k, v = torch.cat([kv, ka], dim=1), torch.cat([vv, va], dim=1)
    reshape = lambda x: x.reshape(1, -1, 24, 128).transpose(1, 2).float()
    scores = reshape(q) @ reshape(k).transpose(-1, -2) / (128**0.5)
    scores = scores.masked_fill(~mask, -torch.inf)
    reference = (
        (torch.softmax(scores, dim=-1).nan_to_num() @ reshape(v)).transpose(1, 2).reshape_as(q)
    )
    result = action_segmented_attention(q, kv, vv, ka, va, mask)
    torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=0.005)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = action_segmented_attention(q, kv, vv, ka, va, mask)
    graph.replay()
    torch.testing.assert_close(captured, result, rtol=0, atol=0)
