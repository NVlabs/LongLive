# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongLive-2.0 (Wan2.2-TI2V-5B, block-causal AR, multi-shot attention sink) with sink memory.

Sink injection (Method, "Sink injection"):
  * each memory image is fitted to the video resolution, held still for 29 pixel frames and encoded by the frozen Wan
    VAE into one 8-latent block;
  * G is written at frame 0 with its caption and stays the permanent global sink. Every M is written once, one block
    after the previous M on a reference clock, conditioned on the revealed prompt it was drawn from and attending to G
    and the two previous M, so its K/V carry the memory history (:class:`MemoryChain`). Text enters through
    cross-attention, so memory K/V do not depend on the new prompt and are prefilled ahead of time on an auxiliary GPU;
  * shot i starts from the window [G, M_{i-2}, M_{i-1}] as a *new shot*: an 8-frame RoPE gap follows the newest M and
    the first chunk is prefixed with LongLive's scene-cut text. That chunk is pinned as the shot sink and the window
    rolls to [G, first chunk, recent chunks], so M sets the appearance at the start and then leaves;
  * scene routing (:mod:`..agent.router`) picks [G, M_{i-2}, M_{i-1}], [G, M_loc] or [G] when the prompt is revealed;
    the newest M of the most recent known locations is kept prefilled under the current G (parked on the decode
    GPU), so a recall only swaps blocks;
  * generated video K/V is never carried to the next shot; latents are decoded block by block on another GPU while
    the next blocks are denoised.

Without memory (``injection: none``) the backend runs LongLive-2.0's own multi-shot inference (the paper's
"built-in memory" baseline): K/V carry across shots, the first chunk of the video and of every shot are sinks, and
the RoPE phase advances by ``multi_shot_rope_offset`` per shot.

params:
  longlive_root     the LongLive2.0 project (default: ../LongLive2.0 next to this project)
  wan_models        directory containing Wan2.2-TI2V-5B/ (default: <longlive_root>/wan_models)
  config            LongLive inference yaml (default: <longlive_root>/configs/inference.yaml)
  generator_ckpt    model_bf16.pt from Efficient-Large-Model/LongLive-2.0-5B
  device / aux_device / decode_device   generator, memory prefill and streaming VAE decode GPUs
  blocks_per_shot   8 (8 blocks x 8 latents = 253 frames, 10.5 s at 24 fps); fps 24; memory_pixel_frames 29
  scene_routing     llm (default) | none: always inject [G, M_{i-2}, M_{i-1}]
  route_llm_url / route_llm_model / route_llm_style   the router (defaults: AGL_ROUTE_URL / AGL_ROUTE_MODEL env)
  route_decisions   replay recorded routing decisions (JSON file or directory with <story>.json)
  recall_prefill    true (default): keep [G, M_loc] of known locations prefilled (prefill GPU computes, decode GPU
                    stores them when it is a third GPU); recall_prefill_cap: at most this many locations, the most
                    recently drawn (default 8, about 2.6 GB each)
  attn_backend      fa2 (default) | fa4 | cudnn: self-attention kernel of the DiT (faster kernels on Hopper/Blackwell)
  async_decode_tail false (default) | true: the next shot starts while the last blocks of this one are decoded
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import queue
import sys
import tempfile
import threading
import time
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent.router import SceneRouter, load_decisions
from ..memory.snapshot import MemorySnapshot
from .base import ShotResult, VideoBackend, concat_shots

_PROJECT = Path(__file__).resolve().parents[3]


def _no_grad(fn):
    """Grad mode is thread-local and the backend is driven from several threads: guard every entry point."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch

        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


class _ThreadLocalMeta(MutableMapping):
    """Per-thread stand-in for LongLive's module-level ``_CURRENT_GRID_META``.

    LongLive publishes the current chunk's KV-cache indices into one module-level dict right before each DiT call
    (single-thread inference). Here the generator (main thread) and the memory prefill (prefill thread) run
    concurrently, so each thread gets its own dict.
    """

    def __init__(self):
        self._local = threading.local()

    @property
    def _data(self) -> Dict[str, int]:
        data = getattr(self._local, "data", None)
        if data is None:
            data = self._local.data = {}
        return data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def _isolate_threads() -> None:
    import wan_5b.modules.causal_model as causal_model

    if not isinstance(causal_model._CURRENT_GRID_META, _ThreadLocalMeta):
        causal_model._CURRENT_GRID_META = _ThreadLocalMeta()
    if os.environ.get("LLV2_FREQS_I_CACHE", "0") == "1":
        raise RuntimeError("LLV2_FREQS_I_CACHE=1 shares a RoPE cache across engines; unset it")


def _install_attention(kind: str) -> None:
    """Self-attention kernel of the DiT window attention (no masks): fa2 is LongLive's own FlashAttention-2;
    fa4 (FlashAttention-4, CuTe DSL, sm_90/sm_100) and cudnn (torch SDPA cuDNN) are faster on H100/B200.
    Cross-attention (variable text lengths) keeps FA2."""
    if kind in (None, "", "fa2"):
        return
    import torch
    import wan_5b.modules.causal_model as cm

    orig = getattr(cm, "_agl_orig_attention", cm.attention)
    cm._agl_orig_attention = orig
    if kind == "fa4":
        from flash_attn.cute import flash_attn_func as fa4

        def kernel(q, k, v, causal, scale):
            out = fa4(q, k, v, softmax_scale=scale, causal=causal)
            return out[0] if isinstance(out, tuple) else out
        layout = "bshd"
    elif kind == "cudnn":
        # the aten op, not sdpa_kernel(): that context flips process-global flags, which would also restrict the
        # VAE's SDPA running concurrently in the decode thread
        def kernel(q, k, v, causal, scale):
            return torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, False, 0.0, causal, False,
                                                                       scale=scale)[0]
        layout = "bhsd"
    else:
        raise ValueError(f"attn_backend must be fa2|fa4|cudnn, got {kind!r}")

    def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
                  window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, fa_version=None):
        if q_lens is not None or k_lens is not None or q_scale is not None or dropout_p or tuple(window_size) != (-1, -1):
            return orig(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
                        q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic,
                        dtype=dtype, fa_version=fa_version)
        if layout == "bshd":
            return kernel(q.to(dtype), k.to(dtype), v.to(dtype), causal, softmax_scale).type(q.dtype)
        qb, kb, vb = (x.to(dtype).transpose(1, 2) for x in (q, k, v))
        return kernel(qb, kb, vb, causal, softmax_scale).transpose(1, 2).type(q.dtype)

    cm.attention = attention


@contextlib.contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@dataclass
class SinkMemory:
    """Prefilled memory: per-block K/V [G, M...] (one (k, v) per transformer block) and where the video starts."""

    version: int
    watermark: int
    blocks_kv: List[List[Tuple[Any, Any]]]
    video_start: int  # frame position on the reference clock right after the newest memory block
    has_g: bool
    device: str
    scene: str = ""  # location of the newest M
    m_entries: Tuple[Dict[str, Any], ...] = ()  # the snapshot's M images (for the routing bank)
    scene_descriptions: Optional[Dict[str, str]] = None
    g_key: Optional[Tuple] = None
    encode_s: float = 0.0
    prefill_s: float = 0.0

    @property
    def n_blocks(self) -> int:
        return len(self.blocks_kv)


class _Engine:
    """One copy of the LongLive pipeline (DiT + text encoder + VAE) on one device."""

    @_no_grad
    def __init__(self, args, device: str, workdir: Path, ckpt: Optional[str], vae_device: Optional[str]):
        import torch
        from pipeline import CausalDiffusionInferencePipeline
        from utils.nvfp4_checkpoint import unwrap_generator_state_dict

        self.torch = torch
        self.device = torch.device(device)
        with torch.cuda.device(self.device), _cwd(workdir):
            pipe = CausalDiffusionInferencePipeline(args, device=self.device)
        if bool(getattr(args, "model_quant", False)):
            # NVFP4 generator (LongLive-2.0 release path); never cast the pipeline with .to(dtype) afterwards
            from utils.inference_utils import setup_nvfp4_pipeline

            if ckpt:
                args.generator_ckpt = ckpt
            with torch.cuda.device(self.device):
                setup_nvfp4_pipeline(pipe, args, self.device)
        else:
            if ckpt:
                state = torch.load(ckpt, map_location="cpu")
                pipe.generator.load_state_dict(unwrap_generator_state_dict(state, use_ema=False), strict=True)
                del state
            pipe.text_encoder.to(dtype=torch.bfloat16)
            pipe.vae.to(dtype=torch.bfloat16)
            pipe = pipe.to(dtype=torch.bfloat16)
        pipe.generator.to(device=self.device)
        pipe.text_encoder.to(device=self.device)
        self.vae_device = torch.device(vae_device or device)
        self.vae_lock = threading.Lock()  # memory-image encoding and the stream decoder share the VAE feature cache
        pipe.vae.to(device=self.vae_device)
        pipe.vae.mean = pipe.vae.mean.to(self.vae_device)
        pipe.vae.std = pipe.vae.std.to(self.vae_device)
        self.pipe = pipe
        self.F = pipe.num_frame_per_block
        self.seq = pipe.frame_seq_length
        self.shape = list(args.image_or_video_shape)  # [B, F, C, H, W]
        dit = pipe._dit_model
        dit.local_attn_size = pipe.local_attn_size
        pipe._set_all_modules_max_attention_size(pipe.local_attn_size)
        pipe._set_all_modules_sink_size(pipe.sink_size)
        pipe._set_all_modules_global_sink_size(pipe.sink_size)
        dit.use_relative_rope = False
        dit.rope_temporal_offset = 0.0
        with torch.cuda.device(self.device):
            pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)
            pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)

    def reset_cache(self) -> None:
        for cache in self.pipe.kv_cache_pos:
            cache["global_end_index"].fill_(0)
            cache["local_end_index"].fill_(0)
            cache["pinned_start"].fill_(-1)
            cache["pinned_len"].zero_()
        self.reset_cross()

    def reset_cross(self) -> None:
        for cache in self.pipe.crossattn_cache_pos:
            cache["is_init"] = False

    def encode_text(self, texts: List[str]) -> Dict[str, Any]:
        with self.torch.cuda.device(self.device):
            return self.pipe.text_encoder(text_prompts=texts)

    def forward(self, latents, cond, timestep, start_frame: int):
        return self.pipe.generator(
            noisy_image_or_video=latents, conditional_dict=cond, timestep=timestep,
            kv_cache=self.pipe.kv_cache_pos, crossattn_cache=self.pipe.crossattn_cache_pos,
            current_start=start_frame * self.seq, cache_start=start_frame * self.seq,
        )

    @_no_grad
    def encode_image(self, path: str, pixel_frames: int):
        """Image -> one block of VAE latents [1, F, C, H, W] (the image held still for ``pixel_frames``)."""
        import numpy as np
        from PIL import Image, ImageOps

        torch = self.torch
        _, _, _, h, w = self.shape
        with Image.open(path) as image:
            image = ImageOps.fit(image.convert("RGB"), (w * 16, h * 16), Image.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        frame = torch.from_numpy(array).permute(2, 0, 1)  # [3, H, W]
        video = frame[None, :, None].repeat(1, 1, pixel_frames, 1, 1).to(device=self.vae_device, dtype=torch.bfloat16)
        with self.vae_lock, torch.cuda.device(self.vae_device):
            latent = self.pipe.vae.encode_to_latent(video)
        if latent.shape[1] != self.F:
            raise ValueError(f"memory image encoded to {latent.shape[1]} latent frames, expected {self.F}")
        return latent.to(device=self.device, dtype=torch.bfloat16)

    @_no_grad
    def write_block(self, latent, text: str, position: int, context: List[List[Tuple[Any, Any]]]):
        """Write one clean memory block at frame ``position`` after the ``context`` blocks (which keep the positions
        they were written at) and return its per-layer K/V."""
        torch = self.torch
        tokens = self.F * self.seq
        n = len(context)
        if (n + 1) * self.F > self.pipe.local_attn_size:
            raise ValueError("memory context does not fit the attention window")
        self.reset_cache()
        for j, block in enumerate(context):
            for cache, (k, v) in zip(self.pipe.kv_cache_pos, block):
                cache["k"][:, j * tokens:(j + 1) * tokens].copy_(k, non_blocking=True)
                cache["v"][:, j * tokens:(j + 1) * tokens].copy_(v, non_blocking=True)
        for cache in self.pipe.kv_cache_pos:
            cache["global_end_index"].fill_(position * self.seq)
            cache["local_end_index"].fill_(n * tokens)
        with torch.cuda.device(self.device):
            self.reset_cross()
            self.pipe._dit_model.rope_temporal_offset = 0.0
            cond = self.encode_text([text])
            t0 = torch.zeros([1, self.F], device=self.device, dtype=torch.float32)
            self.forward(latent, cond, t0, position)
            self.reset_cross()
        return [(c["k"][:, n * tokens:(n + 1) * tokens].clone(), c["v"][:, n * tokens:(n + 1) * tokens].clone())
                for c in self.pipe.kv_cache_pos]

    @_no_grad
    def load_blocks(self, blocks: List[List[Tuple[Any, Any]]], video_start: int) -> None:
        """Copy memory K/V into the generator's cache; the next write lands at ``video_start`` on the clock."""
        tokens = self.F * self.seq
        self.reset_cache()
        for j, block in enumerate(blocks):
            for cache, (k, v) in zip(self.pipe.kv_cache_pos, block):
                cache["k"][:, j * tokens:(j + 1) * tokens].copy_(k, non_blocking=True)
                cache["v"][:, j * tokens:(j + 1) * tokens].copy_(v, non_blocking=True)
        for cache in self.pipe.kv_cache_pos:
            cache["global_end_index"].fill_(video_start * self.seq)
            cache["local_end_index"].fill_(len(blocks) * tokens)
        self.torch.cuda.synchronize(self.device)


class _StreamDecoder:
    """Decodes 8-latent blocks on the decode device while the next blocks are denoised."""

    def __init__(self, engine: _Engine, prev_done: Optional[threading.Event] = None):
        import torch

        self.torch = torch
        self._lock = engine.vae_lock
        self._prev_done = prev_done  # async decode tail: the previous shot's decoder must finish first
        self.done = threading.Event()
        self.vae = engine.pipe.vae
        self.device = engine.vae_device
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._frames: List[Any] = []
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self.block_s: List[float] = []

    def begin(self) -> None:
        self._frames, self._error, self.block_s, self._first = [], None, [], True
        self._thread = threading.Thread(target=self._run, name="vae-decode", daemon=True)
        self._thread.start()

    def push(self, latents) -> None:
        self._queue.put(latents.detach().to(self.device, non_blocking=True))

    def end(self):
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self.torch.cat(self._frames, dim=0)  # [T, H, W, 3] uint8 (cpu)

    @_no_grad
    def _run(self) -> None:
        torch = self.torch
        if self._prev_done is not None:
            self._prev_done.wait()
        self._lock.acquire()
        try:
            from wan_5b.modules.vae2_2 import unpatchify

            model = self.vae.model
            with torch.cuda.device(self.device):
                model.clear_cache()
                mean = self.vae.mean.to(self.device, torch.bfloat16)
                inv_std = 1.0 / self.vae.std.to(self.device, torch.bfloat16)
                while True:
                    latents = self._queue.get()
                    if latents is None:
                        break
                    td = time.monotonic()
                    z = latents.permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
                    z = z / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
                    x = model.conv2(z)
                    outs = []
                    for i in range(x.shape[2]):
                        model._conv_idx = [0]
                        outs.append(model.decoder(x[:, :, i:i + 1], feat_cache=model._feat_map,
                                                  feat_idx=model._conv_idx, first_chunk=self._first))
                        self._first = False
                    pixels = unpatchify(torch.cat(outs, 2), patch_size=2).float().clamp_(-1, 1)
                    self._frames.append(((pixels[0].permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).cpu())
                    self.block_s.append(time.monotonic() - td)
                model.clear_cache()
        except BaseException as exc:
            self._error = exc
        finally:
            self._lock.release()
            self.done.set()


class MemoryChain:
    """Reference-clock memory K/V shared across memory versions.

    G is written at frame 0. Every new M block is written once, one block after the newest older M on the clock,
    attending to G and the two newest older M blocks; its K/V is reused by later versions, so an M block carries the
    memory history before it. Refinement rounds of the same watermark replace each other. The chain restarts when G
    changes or the clock would run past the RoPE table (``max_frames``).
    """

    def __init__(self, frames_per_block: int, max_frames: int, keep: int = 4):
        self.F = frames_per_block
        self.max_frames = max_frames
        self.keep = keep
        self.g_key: Optional[Tuple] = None
        self.g_kv = None
        self.entries: List[Dict[str, Any]] = []  # oldest -> newest: key, watermark, pos, kv
        self.restarts = 0
        self.lock = threading.Lock()

    def build(self, write, g: Optional[Tuple[Tuple, Any, str]], ms: List[Tuple[Tuple, int, Any, str]],
              video_frames: int) -> Tuple[List[Any], int, Dict[str, Any]]:
        """``write(latent, text, pos, context) -> kv``; g = (key, latent, text); ms = [(key, watermark, latent, text)]
        oldest first. Returns per-block K/V [G, M...], the video start frame and bookkeeping info."""
        with self.lock:
            g_key = g[0] if g else None
            restart = g_key != self.g_key or self.g_kv is None and g is not None
            last = max([e["pos"] for e in self.entries], default=0)
            if last + (len(ms) + 1) * self.F + video_frames > self.max_frames:
                restart = True
            if restart:
                self.entries, self.g_key, self.g_kv = [], g_key, None
                self.restarts += 1
                if g is not None:
                    self.g_kv = write(g[1], g[2], 0, [])
            base = [self.g_kv] if self.g_kv is not None else []
            blocks, written, newest_pos = list(base), [], None
            for key, watermark, latent, text in ms:
                entry = next((e for e in self.entries if e["key"] == key), None)
                if entry is None:
                    kept = [e for e in self.entries if e["watermark"] < watermark]
                    pos = kept[-1]["pos"] + self.F if kept else (self.F if base else 0)
                    kv = write(latent, text, pos, base + [e["kv"] for e in kept[-2:]])
                    entry = {"key": key, "watermark": watermark, "pos": pos, "kv": kv}
                    self.entries = (kept + [entry])[-self.keep:]
                    written.append(watermark)
                blocks.append(entry["kv"])
                newest_pos = entry["pos"]
            video_start = (newest_pos + self.F) if newest_pos is not None else (self.F if base else 0)
            return blocks, video_start, {"restart": restart, "written": written, "video_start": video_start,
                                         "positions": [e["pos"] for e in self.entries]}


class LongLiveBackend(VideoBackend):
    name = "longlive"

    def __init__(self, cfg, run, out_dir, tracer):
        super().__init__(cfg, run, out_dir, tracer)
        self.builtin = self.injection == "none"  # LongLive-2.0's own multi-shot memory (baseline)
        self.main: Optional[_Engine] = None
        self.aux: Optional[_Engine] = None
        self._latents: Dict[str, Any] = {}
        self._latents_lock = threading.Lock()
        self._frame = 0  # builtin mode: next frame position of the continuous stream
        self._chain: Optional[MemoryChain] = None
        self._route_info: Optional[Dict[str, Any]] = None
        self._recall_kv: Dict[Tuple, Any] = {}
        self._recall_g: Optional[Tuple] = None
        self._recall_lock = threading.Lock()
        # MP4 encoding is file I/O, not streaming work: keep it off the shot's critical path
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mp4-writer")
        self._writes: List[Any] = []

    # ----------------------------------------------------------------- setup
    def start(self, setup) -> None:
        from omegaconf import OmegaConf

        params = self.params
        root = Path(params.get("longlive_root") or _PROJECT / "LongLive2.0").resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from utils.config import normalize_config

        _isolate_threads()
        _install_attention(str(params.get("attn_backend") or "fa2"))
        workdir = Path(tempfile.mkdtemp(prefix="agl_longlive_"))
        wan = Path(params.get("wan_models") or root / "wan_models").resolve()
        (workdir / "wan_models").symlink_to(wan, target_is_directory=True)
        args = normalize_config(OmegaConf.load(params.get("config") or root / "configs/inference.yaml"))
        for key, value in (params.get("overrides") or {}).items():
            OmegaConf.update(args, key, value, merge=True)
        args.distributed = False
        device = params.get("device", "cuda:0")
        aux_device = params.get("aux_device", "cuda:1")
        decode_device = params.get("decode_device", aux_device)
        ckpt = params.get("generator_ckpt")
        t0 = time.monotonic()
        self.main = _Engine(args, device, workdir, ckpt, vae_device=decode_device)
        self.tracer.log("longlive_engine_ready", role="main", device=device, seconds=round(time.monotonic() - t0, 2))
        if not self.builtin and self.run.runtime.aux_prefill and not self.run.runtime.replay_memory:
            t0 = time.monotonic()
            self.aux = _Engine(args, aux_device, workdir, ckpt, vae_device=aux_device)
            self.tracer.log("longlive_engine_ready", role="aux", device=aux_device,
                            seconds=round(time.monotonic() - t0, 2))
        self.decoder = _StreamDecoder(self.main)
        self.async_tail = bool(params.get("async_decode_tail", False))
        self._last_dec: Optional[_StreamDecoder] = None
        self._tailpool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode-tail")
        self.blocks_per_shot = int(params.get("blocks_per_shot", 8))
        self.fps = float(params.get("fps", 24))
        self.pixel_frames = int(params.get("memory_pixel_frames", 29))
        # LongLive marks a new shot with the scene-cut text and a +multi_shot_rope_offset jump of the RoPE positions
        self.phi = float(getattr(args.inference, "multi_shot_rope_offset", 0.0) or 0.0)
        self.cut_prefix = str(getattr(args.inference, "scene_cut_prefix", None) or "The scene transitions. ")
        self.router: Optional[SceneRouter] = None
        if not self.builtin and str(params.get("scene_routing", "llm")) == "llm":
            decisions = None
            if params.get("route_decisions"):
                decisions = load_decisions(params["route_decisions"], Path(self.run.story).stem)
            self.router = SceneRouter(
                url=str(params.get("route_llm_url") or os.environ.get("AGL_ROUTE_URL") or "http://127.0.0.1:31001/v1"),
                model=str(params.get("route_llm_model") or os.environ.get("AGL_ROUTE_MODEL")
                          or "Qwen/Qwen3.6-35B-A3B-FP8"),
                style=str(params.get("route_llm_style", "sglang")), decisions=decisions)
        self.recall_prefill = bool(params.get("recall_prefill", True)) and self.router is not None \
            and self.aux is not None
        self.recall_cap = int(params.get("recall_prefill_cap", 8))
        if params.get("warmup", True):
            t0 = time.monotonic()
            self._warmup()
            self.tracer.log("longlive_warmup", seconds=round(time.monotonic() - t0, 2))

    @_no_grad
    def _warmup(self) -> None:
        """One dummy block through the generator, the stream decoder and the aux engine (kernel setup, allocator),
        so that the first measured shot runs at steady-state speed. Caches are reset afterwards."""
        for engine in [e for e in (self.main, self.aux) if e is not None]:
            torch = engine.torch
            with torch.cuda.device(engine.device):
                _, _, C, H, W = engine.shape
                engine.reset_cache()
                cond = engine.encode_text(["warm-up"])
                latents = torch.zeros([1, engine.F, C, H, W], device=engine.device, dtype=torch.bfloat16)
                t = torch.full([1, engine.F], 500.0, device=engine.device, dtype=torch.float32)
                engine.forward(latents, cond, t, 0)
                engine.forward(latents, cond, t * 0, 0)
                if engine is self.main:
                    self.decoder.begin()
                    self.decoder.push(latents)
                    self.decoder.end()
                engine.reset_cache()
                torch.cuda.synchronize(engine.device)

    # --------------------------------------------------------------- prefill
    def _latent(self, engine: _Engine, path: str):
        key = f"{engine.device}:{path}:{os.path.getmtime(path)}"
        with self._latents_lock:
            if key in self._latents:
                return self._latents[key]
        latent = engine.encode_image(path, self.pixel_frames)
        with self._latents_lock:
            self._latents[key] = latent
            if len(self._latents) > 64:
                self._latents.pop(next(iter(self._latents)))
        return latent

    def memory_stats(self) -> Dict[str, Any]:
        torch = self.main.torch
        stats = {}
        for role, engine in (("main", self.main), ("aux", self.aux)):
            if engine is not None:
                stats[f"{role}_alloc_gb"] = round(torch.cuda.memory_allocated(engine.device) / 2**30, 2)
        return stats

    @staticmethod
    def _text(block) -> str:
        """An M block is written with the revealed prompt it was drawn from, G with its caption."""
        return block.prompt if (block.role == "M" and block.prompt) else block.caption

    @_no_grad
    def prefill(self, snapshot: MemorySnapshot, on_aux: bool = True) -> Optional[SinkMemory]:
        if self.builtin or not snapshot.blocks:
            return None
        engine = self.aux if (on_aux and self.aux is not None) else self.main
        torch = engine.torch
        if self._chain is None or self._chain_device != str(engine.device):
            self._chain, self._chain_device = MemoryChain(engine.F, int(engine.pipe._dit_model.freqs.shape[0])), \
                str(engine.device)
        t0 = time.monotonic()
        g = snapshot.g
        g_item = ((g.path, os.path.getmtime(g.path)), self._latent(engine, g.path), self._text(g)) if g else None
        ms = [((m.path, os.path.getmtime(m.path)), m.watermark, self._latent(engine, m.path), self._text(m))
              for m in snapshot.m]
        torch.cuda.synchronize(engine.device)
        t1 = time.monotonic()
        video_frames = self.blocks_per_shot * engine.F + int(self.phi)
        blocks, video_start, info = self._chain.build(engine.write_block, g_item, ms, video_frames)
        torch.cuda.synchronize(engine.device)
        t2 = time.monotonic()
        self.tracer.log("memory_chain", version=snapshot.version, watermark=snapshot.watermark, **info)
        return SinkMemory(
            snapshot.version, snapshot.watermark, blocks, video_start, g is not None, str(engine.device),
            scene=snapshot.m[-1].scene if snapshot.m else "",
            m_entries=tuple({"scene": m.scene, "prompt": m.prompt, "watermark": int(m.watermark), "path": m.path,
                             "caption": m.caption} for m in snapshot.m),
            scene_descriptions={r.name: r.description for r in snapshot.references if r.kind == "scene" and r.name},
            g_key=g_item[0] if g_item else None, encode_s=t1 - t0, prefill_s=t2 - t1)

    def after_prefill(self, snapshot: MemorySnapshot, memory: Optional[SinkMemory]) -> None:
        """Prefill thread, once a version is usable: keep [G, M_loc] of the most recent known locations (newest M per
        location, at most ``recall_prefill_cap``) prefilled under the current G, so a recall at the boundary only swaps
        blocks. Blocks of locations that left the set are freed; they are stored on the decode GPU when that is a
        third GPU, so the prefill GPU's memory stays flat."""
        if not (self.recall_prefill and memory is not None and memory.has_g):
            return
        entries = [m for _, m in self.router._bank] + list(memory.m_entries)
        newest: Dict[str, Dict[str, Any]] = {}
        for m in entries:
            if m.get("scene") and int(m.get("watermark", -1)) >= 0:
                old = newest.get(m["scene"])
                if old is None or int(m["watermark"]) >= int(old["watermark"]):
                    newest[m["scene"]] = m
        cands = sorted(newest.values(), key=lambda m: int(m["watermark"]))[-self.recall_cap:]
        engine, g_kv = self.aux, memory.blocks_kv[0]
        torch = engine.torch
        dec = torch.device(self.main.vae_device)
        store = dec if dec not in (engine.device, self.main.device) else engine.device
        keep = {(m["path"], self._text_of(m)) for m in cands}
        with self._recall_lock:
            if self._recall_g != memory.g_key:  # blocks prefilled against an older G are useless
                self._recall_kv, self._recall_g = {}, memory.g_key
            for key in [k for k in self._recall_kv if k not in keep]:
                del self._recall_kv[key]
            todo = [m for m in cands if (m["path"], self._text_of(m)) not in self._recall_kv]
        for m in todo:
            kv = engine.write_block(self._latent(engine, m["path"]), self._text_of(m), engine.F, [g_kv])
            torch.cuda.synchronize(engine.device)
            if store != engine.device:
                kv = [(k.to(store, non_blocking=True), v.to(store, non_blocking=True)) for k, v in kv]
                torch.cuda.synchronize(store)
            with self._recall_lock:
                if self._recall_g != memory.g_key:
                    return
                self._recall_kv[(m["path"], self._text_of(m))] = kv
        self.tracer.log("recall_prefill", version=snapshot.version, new=len(todo), cached=len(self._recall_kv),
                        store=str(store), store_alloc_gb=round(torch.cuda.memory_allocated(store) / 2**30, 2))

    @staticmethod
    def _text_of(entry: Dict[str, Any]) -> str:
        return entry.get("prompt") or entry.get("caption", "")

    # ----------------------------------------------------------------- route
    def _route(self, memory: SinkMemory, prompt: str, shot: int) -> Tuple[Optional[SinkMemory], Dict[str, Any]]:
        """Scene routing at the boundary: keep [G, M_{i-2}, M_{i-1}], recall [G, M_loc] or drop to [G]."""
        engine = self.main
        self.router.register(shot, memory.m_entries, memory.scene_descriptions)
        n_m = memory.n_blocks - (1 if memory.has_g else 0)
        if n_m <= 0:
            return memory, {"decision": "keep", "note": "no M"}
        current = dict(memory.m_entries[-1]) if memory.m_entries else {"scene": memory.scene}
        decision = self.router.route(shot, prompt, current)
        entry = decision.pop("entry", None)
        info = {k: v for k, v in decision.items() if k != "entry"}
        if decision["decision"] == "keep":
            return memory, info
        g = list(memory.blocks_kv[:1]) if memory.has_g else []
        if decision["decision"] == "drop":
            if not g:
                return None, info
            return dataclasses.replace(memory, blocks_kv=g, video_start=engine.F, scene=""), info
        t0 = time.monotonic()
        text = self._text_of(entry)
        kv = None
        if g and self.recall_prefill:
            with self._recall_lock:
                if self._recall_g == memory.g_key:
                    kv = self._recall_kv.get((entry["path"], text))
        if kv is not None:
            kv = [(k.to(engine.device, non_blocking=True), v.to(engine.device, non_blocking=True)) for k, v in kv]
        else:  # [G, M_loc]: the recalled block right after G, attending to it
            kv = engine.write_block(self._latent(engine, entry["path"]), text, engine.F if g else 0, g)
        engine.torch.cuda.synchronize(engine.device)
        info["recall"] = {"scene": entry["scene"], "watermark": int(entry["watermark"]), "path": entry["path"],
                          "prefilled": kv is not None and self.recall_prefill, "seconds": round(time.monotonic() - t0, 4)}
        return dataclasses.replace(memory, blocks_kv=g + [kv], video_start=(engine.F if g else 0) + engine.F,
                                   scene=entry["scene"]), info

    # -------------------------------------------------------------- generate
    @_no_grad
    def generate(self, shot: int, prompt: str, memory: Optional[SinkMemory], seed: int) -> ShotResult:
        engine, torch = self.main, self.main.torch
        t0 = time.monotonic()
        route_info = None
        with torch.cuda.device(engine.device):
            if self.builtin:
                start_frame = self._frame
                if shot > 0 and self.phi:
                    engine.pipe._dit_model.rope_temporal_offset = shot * self.phi
                engine.reset_cross()
                cut = pin_first = shot > 0
                boundary = "builtin"
            else:
                if memory is not None and self.router is not None and shot > 0:
                    memory, route_info = self._route(memory, prompt, shot)
                if memory is None:
                    engine.reset_cache()
                    start_frame, cut, phase, boundary = 0, False, 0.0, "none"
                else:
                    engine.load_blocks(memory.blocks_kv, memory.video_start)
                    start_frame = memory.video_start
                    # a new shot after the memory blocks: scene-cut text (from the second shot on) and the RoPE gap
                    cut, phase, boundary = shot > 0, self.phi, "cut"
                engine.reset_cross()
                pin_first = True  # the shot's own first chunk is its sink, as in LongLive's multi-shot training
                engine.pipe._dit_model.rope_temporal_offset = phase
            torch.cuda.synchronize(engine.device)
            t_kv = time.monotonic()
            text = prompt.rstrip() + self.text_suffix if self.text_suffix else prompt
            cond = engine.encode_text([text])
            first_cond = engine.encode_text([self.cut_prefix + text]) if cut else cond
            torch.cuda.synchronize(engine.device)
            t_text = time.monotonic()
            _, _, C, H, W = engine.shape
            total = self.blocks_per_shot * engine.F
            gen = torch.Generator(device=engine.device).manual_seed(int(seed))
            noise = torch.randn([1, total, C, H, W], generator=gen, device=engine.device, dtype=torch.bfloat16)
            dec = self.decoder
            if self.async_tail:
                dec = _StreamDecoder(self.main, prev_done=self._last_dec.done if self._last_dec is not None else None)
                self._last_dec = dec
            dec.begin()
            block_s = []
            for b in range(self.blocks_per_shot):
                tb = time.monotonic()
                latents = noise[:, b * engine.F:(b + 1) * engine.F]
                scheduler = engine.pipe._initialize_sample_scheduler(noise)
                frame = start_frame + b * engine.F
                block_cond = first_cond if b == 0 else cond
                if b == 1 and first_cond is not cond:
                    engine.reset_cross()  # cross-attention K/V is cached per prompt
                timestep = None
                for t in scheduler.timesteps:
                    timestep = t * torch.ones([1, engine.F], device=engine.device, dtype=torch.float32)
                    flow, _ = engine.forward(latents, block_cond, timestep, frame)
                    latents = scheduler.step(flow, t, latents, return_dict=False)[0]
                engine.forward(latents, block_cond, timestep * 0, frame)  # clean K/V of this block
                if b == 0 and pin_first and (shot > 0 or not self.builtin):
                    engine.pipe._pin_current_chunk(engine.pipe.kv_cache_pos, engine.F)
                dec.push(latents)
                torch.cuda.synchronize(engine.device)
                block_s.append(time.monotonic() - tb)
            t_dit = time.monotonic()
            if self.async_tail:
                frames = None
                self._writes.append(self._tailpool.submit(lambda d=dec, s=shot: self._write(s, d.end())))
            else:
                frames = dec.end()  # waits only for the last block(s): earlier blocks were decoded meanwhile
            if self.builtin:
                self._frame = start_frame + total
        t_end = time.monotonic()
        if frames is None:
            path = self.out_dir / "shots" / f"shot_{shot:03d}.mp4"
            last = self.out_dir / "shots" / f"shot_{shot:03d}_last.jpg"
            n_frames = 4 * total - 3
        else:
            path, last = self._write(shot, frames)
            n_frames = int(frames.shape[0])
        if route_info is not None and route_info.get("decision") != "keep":
            boundary += "+" + route_info["decision"]
        result = ShotResult(
            shot=shot, video_path=str(path), num_frames=n_frames, fps=self.fps, last_frame=str(last),
            timings={"kv_prepare": t_kv - t0, "video": t_end - t_kv, "text": t_text - t_kv, "dit": sum(block_s),
                     "decode_tail": t_end - t_dit},
            meta={"memory_version": memory.version if memory else None,
                  "memory_blocks": memory.n_blocks if memory else 0,
                  "prefill_device": memory.device if memory else None,
                  "prefill_s": round(memory.prefill_s, 4) if memory else None,
                  "block_s": [round(s, 4) for s in block_s], "boundary": boundary,
                  "rope_offset": float(engine.pipe._dit_model.rope_temporal_offset), "video_start": int(start_frame),
                  "memory_scene": memory.scene if memory else None, "route": route_info})
        self.shots.append(result)
        return result

    # ----------------------------------------------------------------- output
    def _write(self, shot: int, frames) -> Tuple[Path, Path]:
        from PIL import Image

        out = self.out_dir / "shots"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"shot_{shot:03d}.mp4"
        array = frames.numpy()
        last = out / f"shot_{shot:03d}_last.jpg"
        Image.fromarray(array[-1]).save(last, quality=92)
        self._writes.append(self._writer.submit(self._encode, path, array))
        return path, last

    def _encode(self, path: Path, array) -> None:
        import imageio.v2 as imageio

        tmp = path.with_suffix(".part.mp4")
        imageio.mimwrite(tmp, array, fps=self.fps, codec="libx264", quality=8, macro_block_size=8)
        tmp.replace(path)

    def flush(self) -> None:
        for future in list(self._writes):
            future.result()

    def finish(self) -> Dict[str, Any]:
        self.flush()
        return concat_shots(self.out_dir, self.shots)

    def close(self) -> None:
        self._writer.shutdown(wait=True)
        self._tailpool.shutdown(wait=True)
