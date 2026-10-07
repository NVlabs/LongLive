# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/streaming_vae.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Window-scoped causal encoding, independent of simulator and denoising."""

import torch


class StreamingVAE:
    """Encode the next observation window while its frames arrive.

    Every window starts with one frame, followed by exact temporal groups.
    Reinitialize at planning boundaries to match full-window causal padding.
    No video-KV prefill or shared-memory IPC is performed here.
    """

    def __init__(self, model, *, past_steps, sample_stride):
        self.model = model
        self.past_steps, self.sample_stride = int(past_steps), int(sample_stride)
        self.group = int(model.vae.temporal_downsample_factor)
        if self.past_steps < 0 or self.sample_stride <= 0 or self.group <= 0:
            raise ValueError("Invalid VAE observation cadence")
        if self.past_steps % (self.sample_stride * self.group):
            raise ValueError("Streaming history must contain complete temporal VAE groups")
        self.anchor = None
        self._pending = []
        self._count = 0

    def reset(self):
        self.model.vae.model.clear_cache()
        self.anchor = None
        self._pending.clear()
        self._count = 0

    def begin(self, anchor, history):
        self.reset()
        if not history:
            raise ValueError("A streaming window requires at least one observation")
        self.anchor = int(anchor)
        first = self.anchor - self.past_steps
        frames = dict(history)
        earliest = min(frames)
        latest = max(frames)
        for step in range(first, min(latest, self.anchor) + 1, self.sample_stride):
            if step < 0:
                frame = frames[earliest]
            else:
                if step not in frames:
                    raise ValueError(f"Missing observed frame {step} in the streaming window")
                frame = frames[step]
            self.feed(step, frame)

    @torch.no_grad()
    def feed(self, step, frame):
        if self.anchor is None:
            return
        expected = self.anchor - self.past_steps + self._count * self.sample_stride
        if step != expected:
            if step > expected:
                raise ValueError(f"Skipped sampled observation {expected}")
            return
        if step > self.anchor:
            return
        self._count += 1
        self._pending.append(frame)
        count = 1 if self._count == 1 else self.group
        if len(self._pending) == count:
            chunk = torch.stack(self._pending, dim=1).unsqueeze(0)
            if self._count == 1:
                self.model._stream_video_latents_start(chunk)
            else:
                self.model._stream_video_latents_step(chunk)
            self._pending.clear()

    @torch.no_grad()
    def finish(self, anchor):
        expected = self.past_steps // self.sample_stride + 1
        if anchor != self.anchor or self._count != expected or self._pending:
            raise RuntimeError("Streaming VAE window is incomplete or has the wrong anchor")
        result = self.model._stream_video_latents_finish()
        self.anchor = None
        if result.ndim != 5 or result.shape[2] != self.model.num_clean_frames:
            raise ValueError("Streaming VAE latent window differs from the model contract")
        return result
