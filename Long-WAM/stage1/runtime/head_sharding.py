# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot-video integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/runtime/head_sharding.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

# Export adaptation: standalone helper imports / package-relative root; algorithm unchanged.
"""Lossless-head-partitioned KV storage for the robot-video evaluator.

Only evaluation uses this adapter. Q/K normalization still sees every channel;
independent attention heads and their BF16 caches are partitioned over ranks.
The attention output is gathered in its original order before the unchanged
output projection. No temporal context, CFG branch, or precision is removed.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    # Full release-file hashes, including attribution headers; upstream bodies are unchanged.
    'wan_5b/modules/causal_model.py': '23e7dc1e1ff77ac6058a36c61ba1d7a56caa83721b4a05cfc78d17e8065d6f8f',
    'pipeline/causal_diffusion_inference.py': '92f2b06f06476f99238728bcfd46388589affe0fe6a46824db03088ea31faa24',
}


def partition_heads(q, k, v, cache):
    import torch.distributed as dist
    if cache is None:
        raise RuntimeError('head-sharded adapter is evaluation-only')
    if cache.get('quantized', False):
        raise RuntimeError('quantized evaluation is forbidden')
    size, rank = dist.get_world_size(), dist.get_rank()
    if q.shape[2] % size:
        raise RuntimeError('attention heads must divide evenly over evaluation ranks')
    width = q.shape[2] // size
    if cache['k'].shape[2] != width:
        raise RuntimeError('cache head partition disagrees with attention')
    return tuple(t[:, :, rank * width:(rank + 1) * width].contiguous() for t in (q, k, v))


def gather_heads(x):
    import torch
    import torch.distributed as dist
    shards = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, x.contiguous())
    return torch.cat(shards, dim=2)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('reviewed source insertion point changed')
    return source.replace(old, new)


def transformed_forward(original):
    source = textwrap.dedent(inspect.getsource(original))
    source = replace_once(source, 'q, k, v = qkv_fn(x)',
                          'q, k, v = _robot_video_partition_heads(*qkv_fn(x), kv_cache)')
    source = replace_once(source, 'x = x.flatten(2)',
                          'x = _robot_video_gather_heads(x).flatten(2)')
    namespace = dict(original.__globals__, _robot_video_partition_heads=partition_heads,
                     _robot_video_gather_heads=gather_heads)
    exec(compile(source, str(Path(__file__).resolve()) + ':reviewed_forward', 'exec'), namespace)
    return namespace[original.__name__]


def transformed_cache(original):
    source = textwrap.dedent(inspect.getsource(original))
    old = 'num_heads = wan_default_config[self.model_name]["num_heads"]'
    source = replace_once(source, old, old + '\n    num_heads = _robot_video_local_heads(num_heads)')
    namespace = dict(original.__globals__, _robot_video_local_heads=local_heads)
    exec(compile(source, str(Path(__file__).resolve()) + ':reviewed_cache', 'exec'), namespace)
    return namespace[original.__name__]


def local_heads(heads):
    import torch.distributed as dist
    size = dist.get_world_size()
    if heads % size:
        raise RuntimeError('invalid head topology')
    return heads // size


def install():
    for relative, expected in PINS.items():
        path = ROOT / 'third_party/LongLive' / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'unreviewed evaluator source: {path}')
    from wan_5b.modules.causal_model import CausalWanSelfAttention
    from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
    original = CausalWanSelfAttention.forward
    if getattr(original, '_robot_video_head_sharded', False):
        return
    changed = transformed_forward(original)
    changed._robot_video_head_sharded = True
    changed._robot_video_original = original
    CausalWanSelfAttention.forward = changed
    cls = CausalDiffusionInferencePipeline
    cls._initialize_kv_cache = transformed_cache(cls._initialize_kv_cache)


def parity_check(device, dtype):
    """Exercise actual attention, rolling cache writes and gather on all ranks."""
    import torch
    import torch.distributed as dist
    from wan_5b.modules import causal_model as cm
    cls = cm.CausalWanSelfAttention
    original = getattr(cls.forward, '_robot_video_original', cls.forward)
    sharded = transformed_forward(original)
    size, rank = dist.get_world_size(), dist.get_rank()
    # Small context, but the real model's head dimension and normalization.
    heads, dim, tokens = 24, 128, 12
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(20260925)
        module = cls(heads * dim, heads, local_attn_size=3).to(device=device, dtype=dtype).eval()
        module.max_attention_size = tokens
        def cache(nheads):
            return dict(k=torch.zeros(1, tokens, nheads, dim, device=device, dtype=dtype),
                        v=torch.zeros(1, tokens, nheads, dim, device=device, dtype=dtype),
                        quantized=False, global_end_index=torch.tensor([0], device=device),
                        local_end_index=torch.tensor([0], device=device),
                        pinned_start=torch.tensor([-1], device=device), pinned_len=torch.tensor([0], device=device))
        full, shard = cache(heads), cache(heads // size)
        freqs = torch.cat([cm.rope_params(64, dim - 4 * (dim // 6)),
                           cm.rope_params(64, 2 * (dim // 6)),
                           cm.rope_params(64, 2 * (dim // 6))], dim=1).to(device)
        maximum = 0.0
        cm._CURRENT_GRID_META.clear()
        with torch.no_grad():
            for start in (0, 4, 8, 12):
                x = torch.randn(1, 4, heads * dim, device=device, dtype=dtype)
                kwargs = dict(seq_lens=torch.tensor([4], device=device),
                              grid_sizes=torch.tensor([[1, 2, 2]], device=device),
                              freqs=freqs, block_mask=None, current_start=start)
                reference, ref_update = original(module, x, kv_cache=full, **kwargs)
                actual, update = sharded(module, x, kv_cache=shard, **kwargs)
                error = float((actual - reference).abs().max())
                maximum = max(maximum, error)
                torch.testing.assert_close(actual, reference,
                    rtol=0.02 if dtype == torch.bfloat16 else 1e-8,
                    atol=0.004 if dtype == torch.bfloat16 else 1e-9)
                cm.CausalWanModel._apply_cache_updates(None, [full], [(0, ref_update)])
                cm.CausalWanModel._apply_cache_updates(None, [shard], [(0, update)])
                width = heads // size
                for key in ('k', 'v'):
                    torch.testing.assert_close(shard[key], full[key][:, :, rank * width:(rank + 1) * width],
                                               rtol=0.02 if dtype == torch.bfloat16 else 1e-8,
                                               atol=0.004 if dtype == torch.bfloat16 else 1e-9)
        del module, full, shard
    return dict(status='passed', rank=rank, world_size=size, max_absolute_error=maximum,
                dtype=str(dtype), cases=['first_chunk', 'append', 'full_window', 'roll'])
