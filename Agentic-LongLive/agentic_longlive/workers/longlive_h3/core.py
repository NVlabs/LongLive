# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongLive-H3 sampler: an exact KV-cached form of LongLive-H3's chunked few-step sampler on one GPU.

LongLive-H3 is a few-step chunked autoregressive distillation of MiniMax-H3 (video + audio). Its reference sampler
generates a shot chunk by chunk (12 chunks for 8 s), with 3 denoising steps per chunk, and re-evaluates every past
chunk at the context clock 0.999 in each forward pass. This module keeps the same packed layout, rotary positions,
paired audio/video clocks, sigma grids, RNG draw order and update rule, and adds two exact shortcuts:

  * a past chunk only attends to the text, chunk 0 and its two predecessors, which are fixed once generated, so its
    keys/values are the same in every later forward pass: they are computed once ("clean pass") and cached;
  * the per-block AdaLN modulations only depend on the clock, and a shot uses a handful of fixed clocks, so they are
    precomputed once and the per-block ``adaln_proj`` weights (26 GB) are dropped; the cache then fits on one 80 GB GPU.

Memory pictures (the agent's G / M images) enter as a pinned chunk right after the text, which every chunk attends to,
noised to the context clock 0.999 like H3's keyframe / reference anchors:

  * sink layout: picture j at ``text_len + 40 * ref_seconds[j]``, generated media shifted by
    ``40 * video_start_seconds`` (H3 counts 40 rotary ticks per second): the pictures are short shots before a cut;
  * ref2va layout: picture j at ``text_len + j``, generated media shifted by the number of pictures.
"""

from __future__ import annotations

import gc
import math
import time
from fractions import Fraction

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

VIDEO, TEXT, AUDIO = 0, 1, 2  # H3 token tags
CTX = 0.999  # clock of context rows (past chunks, memory pictures)
TICKS_PER_SECOND = 40.0  # H3's rotary clock
MEMORY_NOISE_SEED = 20261019  # noise mixed into the memory latents at the context clock
FPS = 24


class Geometry:
    """Latent geometry of one shot: ``frames = 17k + 5`` video frames at ``width`` x ``height``."""

    def __init__(self, frames: int = 192, width: int = 1344, height: int = 768):
        if (frames - 5) % 17:
            raise ValueError("frames must be 17k + 5")
        self.frames, self.width, self.height = frames, width, height
        self.k = (frames - 5) // 17
        self.n_lat = 5 * self.k + 2
        self.n_audio = math.ceil(Fraction(frames * 5, 3))
        self.video_edges = tuple(range(0, 5 * self.k + 1, 5)) + (self.n_lat,)
        self.audio_edges = (0, *(math.ceil(Fraction(i * 85, 3)) for i in range(1, self.k + 1)), self.n_audio)
        self.lh, self.lw = height // 16, width // 16
        self.spatial_rows = (height // 32) * (width // 32)
        self.blocks = len(self.video_edges) - 1

    def chunk_rows(self, b):
        v = (self.video_edges[b + 1] - self.video_edges[b]) * self.spatial_rows
        a = 2 * (self.audio_edges[b + 1] - self.audio_edges[b])
        return v + a


def sigma_grids(device, nfe=3):
    """(video, audio) sigma grids of the few-step sampler (shifted 1 -> 0 grids, nfe + 1 points)."""
    base = torch.tensor([(nfe - s) / nfe for s in range(nfe + 1)], dtype=torch.float64, device=device)
    return (12 * base / (1 + 11 * base)).float(), (3 * base / (1 + 2 * base)).float()


def patchify(latent):
    """(24, F, lh, lw) -> (F * spatial_rows, 96), the sampler's row order."""
    c, f, h, w = latent.shape
    return latent.reshape(1, c, f, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(f * (h // 2) * (w // 2), c * 4)


def unpack(video, audio, g: Geometry):
    """Sampler rows -> (video latents (1, 24, n_lat, lh, lw), audio latents (2, 32, n_audio))."""
    value = video.reshape(1, g.n_lat, g.lh // 2, g.lw // 2, 24, 1, 2, 2)
    value = value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, g.n_lat, g.lh, g.lw).contiguous()
    audio_latents = audio.reshape(2, g.n_audio, 32).permute(0, 2, 1).contiguous()
    return value, audio_latents


def positions(g: Geometry, text_len: int, mem_offsets, shift: float):
    """Rotary positions (text (T, 3), memory (M, 3), audio (2, n_audio, 3), video (n_lat, R, 3)), float64.

    Text / audio / video come from H3's packed layout; video times follow the paired audio/video clock from
    ``text_len + shift``; memory picture j sits at ``text_len + mem_offsets[j]`` on the first latent frame's grid."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    res = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.full((text_len,), TEXT, dtype=torch.long), num_latent_frames=g.n_lat,
        latent_height=g.lh, latent_width=g.lw, num_audio_latents=g.n_audio, patch_size=(1, 2, 2), keyframe_anchors=())
    pos, video_idx, audio_idx, text_idx = res[0], res[2], res[3], res[4]
    text = pos[text_idx].clone()
    audio = pos[audio_idx].reshape(2, g.n_audio, 3).clone()
    video = pos[video_idx].reshape(g.n_lat, g.spatial_rows, 3).clone()
    origin = float(text_len) if not shift else float(text_len) + float(shift)
    temporal = torch.tensor([origin + i // 5 * (85.0 / 3.0) + 5.0 / 3.0 * sum((1, 4, 4, 4, 4)[:i % 5])
                             for i in range(g.n_lat)], dtype=video.dtype)
    video[:, :, 0] = temporal[:, None]
    if shift:
        audio[:, :, 0] += float(shift)
    frame_grid = video[0, :, 1:]
    mem = torch.zeros(len(mem_offsets), g.spatial_rows, 3, dtype=video.dtype)
    for j, off in enumerate(mem_offsets):
        mem[j, :, 0] = float(text_len) + float(off)
        mem[j, :, 1:] = frame_grid
    return text, mem.reshape(-1, 3), audio, video


def _rope(x, cos, sin):
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    return _apply_rotary_emb(x, cos, sin)


class KVRuntime:
    """One shot at a time on one GPU. Keys/values per layer live in a (cap, heads, head_dim) buffer laid out as
    [text | memory | chunk 0 | rolling chunks (<= 2, oldest first) | current]."""

    def __init__(self, model, device, nfe=3):
        self.m, self.dev, self.nfe = model, device, nfe
        attn = model.transformer_blocks[0].attn
        self.heads, self.head_dim = attn.heads, attn.head_dim
        self.hdtype = attn.to_k.weight.dtype  # block stack / text dtype (bf16); proj_in & co. are fp32 modules
        vg, ag = sigma_grids(device, nfe)
        clocks = [torch.tensor(CTX, dtype=torch.float32, device=device)]
        clocks += [1 - vg[s] for s in range(nfe)] + [1 - ag[s] for s in range(nfe)]
        self.table = torch.unique(torch.stack(clocks), sorted=True)
        with torch.no_grad():
            dtype = model.time_embedder.linear_1.weight.dtype
            temb = model.time_embedder(model.time_proj(self.table).to(dtype))
            self.mods = []
            for blk in model.transformer_blocks:
                self.mods.append(tuple(t.contiguous() for t in blk.adaln_proj(temb)))
                for prm in blk.adaln_proj.parameters():  # free the storage even if a loader hook still holds the module
                    prm.data = torch.empty(0, device=prm.device, dtype=prm.dtype)
                blk.adaln_proj = None
            shift, scale = model.norm_out.linear(F.silu(temb).to(model.norm_out.linear.weight.dtype)).chunk(2, dim=-1)
            self.final = (shift.contiguous(), scale.contiguous())
        gc.collect()
        torch.cuda.empty_cache()
        self.K = self.V = None

    def clock_index(self, value):
        hit = (self.table == value).nonzero()
        if hit.numel() != 1:
            raise ValueError(f"clock {float(value)} not in the precomputed table")
        return int(hit[0, 0])

    # ------------------------------------------------------------------ cache
    def alloc(self, cap):
        self.free()
        shape = (cap, self.heads, self.head_dim)
        dtype = self.hdtype
        n = len(self.m.transformer_blocks)
        self.K = [torch.empty(shape, dtype=dtype, device=self.dev) for _ in range(n)]
        self.V = [torch.empty(shape, dtype=dtype, device=self.dev) for _ in range(n)]
        self.past, self.fixed, self.rolling = 0, 0, []

    def free(self):
        self.K = self.V = None

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, hidden, pos, clock_idx, tags, store=None, outputs=None):
        """hidden (1, n, D) input rows; pos (n, 3); clock_idx/tags (n,). store: None | 'fixed' | 'rolling'.
        outputs: None or (n_audio_rows, n_video_rows) -> returns (video_out, audio_out) for the [audio | video] rows."""
        m = self.m
        n = hidden.shape[1]
        L0 = self.past
        idx = (clock_idx * 3 + tags.clamp(min=0)).view(1, -1)
        cos, sin = m.rope(pos)
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        drop = len(self.rolling) >= 2 and store == "rolling"
        for li, blk in enumerate(m.transformer_blocks):
            sa, ca, ga, sf, cf, gf = self.mods[li]
            x = blk.norm1(hidden) * (1 + ca[idx]) + sa[idx]
            attn = blk.attn
            q = attn.norm_q(attn.to_q(x).unflatten(-1, (attn.heads, -1)))
            k = attn.norm_k(attn.to_k(x).unflatten(-1, (attn.heads, -1)))
            v = attn.to_v(x).unflatten(-1, (attn.heads, -1))
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            Kb, Vb = self.K[li], self.V[li]
            Kb[L0:L0 + n] = k[0]
            Vb[L0:L0 + n] = v[0]
            with sdpa_kernel(backends):
                o = F.scaled_dot_product_attention(q.transpose(1, 2), Kb[:L0 + n].unsqueeze(0).transpose(1, 2),
                                                   Vb[:L0 + n].unsqueeze(0).transpose(1, 2), dropout_p=0.0)
            o = o.transpose(1, 2).flatten(2, 3).type_as(q)
            hidden = hidden + ga[idx] * attn.to_out[1](attn.to_out[0](o))
            x = blk.norm2(hidden) * (1 + cf[idx]) + sf[idx]
            hidden = hidden + gf[idx] * blk.ff(x)
            if drop:  # keep [fixed | rolling[1] | new]; rows of rolling[0] leave the window
                old = self.rolling[0]
                for buf in (Kb, Vb):
                    tail = buf[self.fixed + old:L0 + n].clone()
                    buf[self.fixed:self.fixed + tail.shape[0]] = tail
        if store == "fixed":
            self.past = self.fixed = L0 + n
        elif store == "rolling":
            if drop:
                self.past = L0 + n - self.rolling.pop(0)
            else:
                self.past = L0 + n
            self.rolling.append(n)
        if outputs is None:
            return None
        shift, scale = self.final
        ti = clock_idx.view(1, -1)
        h = m.norm_out.norm(hidden) * (1 + scale[ti]) + shift[ti]
        na, nv = outputs
        audio = m.audio_proj_out(h[:, :na].to(m.audio_proj_out.weight.dtype))
        video = m.proj_out(h[:, na:na + nv].to(m.proj_out.weight.dtype))
        return video, audio

    # ------------------------------------------------------------------ embed
    @torch.no_grad()
    def text_hidden(self, prompt_embeds):
        m = self.m
        one = m.context_embedder(prompt_embeds.to(m.context_embedder.weight.dtype))
        return m.token_refiner(one)

    @torch.no_grad()
    def media_hidden(self, audio_rows, video_rows):
        """audio_rows (na, 32), video_rows (nv, 96) float32 -> (1, na + nv, D)."""
        m = self.m
        a = m.audio_proj_in(audio_rows.to(m.audio_proj_in.weight.dtype))
        v = m.proj_in(video_rows.to(m.proj_in.weight.dtype))
        return torch.cat((a.to(self.hdtype), v.to(self.hdtype)), dim=0).unsqueeze(0)


@torch.no_grad()
def generate(rt: KVRuntime, g: Geometry, prompt_embeds, seed, *, mem_latents=None, mem_offsets=(), shift=0.0,
             text_tags=None, timing=None, mem_keep=None):
    """Sample one shot with the KV cache.

    prompt_embeds (1, T, 5120); mem_latents None or (24, n_mem, lh, lw) normalized picture latents at
    ``text_len + mem_offsets[j]``; shift: start of the generated media after the text, in ticks; mem_keep None or a
    bool tensor over the n_mem * spatial_rows memory rows (foreground-only pictures: the other rows never enter the
    cache). Returns (video (n_lat * R, 96), audio (2 * n_audio, 32)) float32 rows; ``timing`` (dict) gets stage times."""
    dev = rt.dev

    def sync():
        torch.cuda.synchronize(dev)

    t = timing if timing is not None else {}
    T = prompt_embeds.shape[1]
    n_mem = 0 if mem_latents is None else mem_latents.shape[1]
    tpos, mpos, apos, vpos = positions(g, T, list(mem_offsets)[:n_mem], shift)
    R = g.spatial_rows
    cmax = max(g.chunk_rows(b) for b in range(g.blocks))
    rt.alloc(T + n_mem * R + g.chunk_rows(0) + 3 * cmax)
    ctx_idx = rt.clock_index(torch.tensor(CTX, dtype=torch.float32, device=dev))
    t0 = time.perf_counter()
    # text (and memory) pass
    tags = (text_tags.to(dev).long() if text_tags is not None else torch.full((T,), TEXT, dtype=torch.long, device=dev))
    rt.forward(rt.text_hidden(prompt_embeds.to(dev)), tpos.to(dev), torch.full((T,), ctx_idx, device=dev), tags, store="fixed")
    if n_mem:
        gen_m = torch.Generator(device=dev).manual_seed(MEMORY_NOISE_SEED)
        lat = mem_latents.to(dev).float()
        eps = torch.randn(lat.shape, generator=gen_m, device=dev, dtype=torch.float32)
        ctx = torch.tensor(CTX, dtype=torch.float32, device=dev)
        rows = patchify(ctx * lat + (1 - ctx) * eps)
        row_idx = torch.full((rows.shape[0],), rt.clock_index(ctx), device=dev)
        if mem_keep is not None:
            keep = mem_keep.to(torch.bool).reshape(-1)
            rows, mpos, row_idx = rows[keep.to(dev)], mpos[keep.cpu()], row_idx[keep.to(dev)]
            t["memory_rows"] = int(rows.shape[0])
        hid = rt.m.proj_in(rows.to(rt.m.proj_in.weight.dtype)).to(rt.hdtype).unsqueeze(0)
        rt.forward(hid, mpos.to(dev), row_idx,
                   torch.full((rows.shape[0],), VIDEO, dtype=torch.long, device=dev), store="fixed")
    sync()
    t["prefill_s"] = time.perf_counter() - t0
    generator = torch.Generator(device=dev).manual_seed(int(seed))

    def draw(shape):
        return torch.stack([torch.randn(shape, generator=generator, device=dev, dtype=torch.float32)])

    def draw_video(frames):
        native = draw((24, frames, g.lh, g.lw))
        return native.reshape(1, 24, frames, g.lh // 2, 2, g.lw // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(1, frames * R, 96)

    def draw_audio(frames):
        return draw((2, 32, frames)).transpose(2, 3)

    video_noise, audio_noise = draw_video(g.n_lat), draw_audio(g.n_audio)
    video_ctx, audio_ctx = draw_video(g.n_lat), draw_audio(g.n_audio)
    vgrid, agrid = sigma_grids(dev, rt.nfe)
    clean = torch.tensor(CTX, dtype=torch.float32, device=dev)
    vpos_d, apos_d = vpos.to(dev), apos.to(dev)
    v_out, a_out = [], []
    t["noise_s"] = t["clean_s"] = 0.0
    for b in range(g.blocks):
        v0, v1 = g.video_edges[b] * R, g.video_edges[b + 1] * R
        a0, a1 = g.audio_edges[b], g.audio_edges[b + 1]
        na, nv = 2 * (a1 - a0), v1 - v0
        pos = torch.cat((apos_d[:, a0:a1].reshape(-1, 3), vpos_d.reshape(-1, 3)[v0:v1]))
        tags = torch.cat((torch.full((na,), AUDIO, dtype=torch.long, device=dev), torch.full((nv,), VIDEO, dtype=torch.long, device=dev)))
        vcur, acur = video_noise[:, v0:v1], audio_noise[:, :, a0:a1]
        ts = time.perf_counter()
        for s in range(rt.nfe):
            ci = torch.cat((torch.full((na,), rt.clock_index(1 - agrid[s]), device=dev),
                            torch.full((nv,), rt.clock_index(1 - vgrid[s]), device=dev)))
            hid = rt.media_hidden(acur[0].reshape(-1, 32), vcur[0])
            vel_v, vel_a = rt.forward(hid, pos, ci, tags, outputs=(na, nv))
            vel_v = vel_v.float()
            vel_a = vel_a.float().reshape(1, 2, a1 - a0, 32)
            vclean = vcur + vgrid[s] * vel_v
            aclean = acur + agrid[s] * vel_a
            vcur = (1 - vgrid[s + 1]) * vclean + vgrid[s + 1] * draw_video(vclean.shape[1] // R)
            acur = (1 - agrid[s + 1]) * aclean + agrid[s + 1] * draw_audio(aclean.shape[2])
            if not bool(torch.isfinite(vcur).all() & torch.isfinite(acur).all()):
                raise FloatingPointError("non-finite LongLive-H3 rollout")
        sync()
        t["noise_s"] += time.perf_counter() - ts
        if b == 0:
            t["first_chunk_s"] = time.perf_counter() - t0
        v_out.append(vcur)
        a_out.append(acur)
        if b < g.blocks - 1:  # clean pass: the finished chunk at the context clock enters the cache
            ts = time.perf_counter()
            pv = clean * vcur + (1 - clean) * video_ctx[:, v0:v1]
            pa = clean * acur + (1 - clean) * audio_ctx[:, :, a0:a1]
            rt.forward(rt.media_hidden(pa[0].reshape(-1, 32), pv[0]), pos, torch.full((na + nv,), ctx_idx, device=dev),
                       tags, store="fixed" if b == 0 else "rolling")
            sync()
            t["clean_s"] += time.perf_counter() - ts
    rt.free()
    t["rollout_s"] = time.perf_counter() - t0
    return torch.cat(v_out, dim=1)[0], torch.cat(a_out, dim=2).flatten(1, 2)[0]


class VAEs:
    """Video + audio VAE kept on the CPU (pinned) and moved to the GPU only to decode (the cache is freed then)."""

    def __init__(self, base, device):
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        self.dev = device
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(base, subfolder="vae", dtype=torch.float32, local_files_only=True).eval().requires_grad_(False)
        self.avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(base, subfolder="audio_vae", dtype=torch.float32, local_files_only=True).eval().requires_grad_(False)
        self.cpu = {}
        for name, mod in (("v", self.vae), ("a", self.avae)):
            for k, p in list(mod.named_parameters()) + list(mod.named_buffers()):
                p.data = p.data.pin_memory()
                self.cpu[(name, k)] = p.data

    def _to(self, mod, name, gpu):
        for k, p in list(mod.named_parameters()) + list(mod.named_buffers()):
            p.data = p.data.to(self.dev, non_blocking=True) if gpu else self.cpu[(name, k)]

    @torch.no_grad()
    def decode(self, video, audio, g: Geometry):
        """Sampler rows -> (frames (F, H, W, 3) uint8, waveform (2, samples) float32, sample rate)."""
        value, audio_latents = unpack(video, audio, g)
        dev = self.dev
        self._to(self.vae, "v", True)
        try:
            vae = self.vae
            mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
            with torch.autocast("cuda", dtype=torch.float16):
                decoded = vae.decode(value.to(dev) * std + mean, return_dict=False)[0]
            if decoded.shape != (1, 3, g.frames, g.height, g.width):
                raise ValueError(f"unexpected decoded video geometry: {tuple(decoded.shape)}")
            if not bool(torch.isfinite(decoded).all()):
                raise ValueError("non-finite VAE video output")
            pm = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1, 1)
            ps = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1, 1)
            frames = ((decoded.float() * ps + pm).clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 3, 0).cpu().contiguous()
            del decoded
        finally:
            self._to(self.vae, "v", False)
            torch.cuda.empty_cache()
        self._to(self.avae, "a", True)
        try:
            av = self.avae
            mean = torch.tensor(av.config.latents_mean, device=dev).view(1, -1, 1)
            std = torch.tensor(av.config.latents_std, device=dev).view(1, -1, 1)
            wav = av.decode(audio_latents.to(dev) * std + mean, return_dict=False)[0].float()
            rate = int(av.config.sampling_rate)
            count = round(g.frames * rate / FPS)
            if wav.ndim != 3 or wav.shape[:2] != (2, 1) or wav.shape[-1] < count or not torch.isfinite(wav).all():
                raise ValueError(f"bad audio decode {tuple(wav.shape)}")
            wav = wav[:, 0, :count].cpu().contiguous()
        finally:
            self._to(self.avae, "a", False)
            torch.cuda.empty_cache()
        return frames, wav, rate


def load_transformer(directory, device):
    """The LongLive-H3 transformer (``<directory>/transformer``, BF16) on ``device``."""
    from diffusers import MiniMaxH3Transformer3DModel
    model = MiniMaxH3Transformer3DModel.from_pretrained(directory, subfolder="transformer", dtype=torch.bfloat16,
                                                        device_map={"": device}, low_cpu_mem_usage=True, local_files_only=True)
    if len(model.transformer_blocks) != 50 or len(model.token_refiner.refiner_blocks) != 2:
        raise ValueError("expected the MiniMax-H3 architecture (50 blocks, 2 token-refiner blocks)")
    return model.eval().requires_grad_(False)
