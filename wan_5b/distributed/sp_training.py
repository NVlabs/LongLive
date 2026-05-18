# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""LongLive sequence-parallel training utilities.

This module collects the LongLive-specific SP training pieces that are layered
on top of Wan2.2's distributed Ulysses helpers: SP/DP process-group routing,
autograd-safe all-to-all for FlexAttention, distributed teacher-forcing
FlexAttention, and chunk-halo VAE preparation.
"""

import math

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from utils.config import wan_default_config


DEFAULT_SP_VAE_HALO_LATENTS = 28

_sp_group = None
_dp_group = None
_compiled_flex_attention = None


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            _flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    return _compiled_flex_attention


def set_sequence_parallel_group(group):
    """Set the SP group used by SP rank/world-size and all-to-all helpers."""
    global _sp_group
    _sp_group = group


def get_sequence_parallel_group():
    return _sp_group


def set_data_parallel_group(group):
    """Set the DP group whose ranks own the same sequence chunk."""
    global _dp_group
    _dp_group = group


def get_data_parallel_group():
    return _dp_group


def get_sp_rank():
    if _sp_group is not None:
        return dist.get_rank(_sp_group)
    return dist.get_rank()


def get_sp_world_size():
    if _sp_group is not None:
        return dist.get_world_size(_sp_group)
    return dist.get_world_size()


def resolve_sequence_parallel_group(group=None):
    if group is not None:
        return group
    return _sp_group


def _world_size(group=None):
    return dist.get_world_size(group) if group is not None else dist.get_world_size()


def _all_to_all_list_impl(x, scatter_dim, gather_dim, group=None, **kwargs):
    world_size = _world_size(group)
    if world_size <= 1:
        return x
    inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(u) for u in inputs]
    dist.all_to_all(outputs, inputs, group=group, **kwargs)
    return torch.cat(outputs, dim=gather_dim).contiguous()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    group = resolve_sequence_parallel_group(group)
    if _world_size(group) <= 1:
        return x
    return _all_to_all_list_impl(x, scatter_dim, gather_dim, group=group, **kwargs)


class AllToAllWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return _all_to_all_list_impl(
            input_tensor, scatter_dim, gather_dim, group=group
        )

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _all_to_all_list_impl(
                grad_output, ctx.gather_dim, ctx.scatter_dim, group=ctx.group
            ),
            None,
            None,
            None,
        )


def all_to_all_with_grad(x, scatter_dim, gather_dim, group=None):
    group = resolve_sequence_parallel_group(group)
    if _world_size(group) <= 1:
        return x
    return AllToAllWithGrad.apply(x, scatter_dim, gather_dim, group)


def all_gather(tensor, group=None):
    group = resolve_sequence_parallel_group(group)
    world_size = _world_size(group)
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def gather_forward(input, dim, group=None):
    group = resolve_sequence_parallel_group(group)
    if _world_size(group) == 1:
        return input
    return torch.cat(all_gather(input, group=group), dim=dim).contiguous()


def distributed_flex_attention(
    roped_q: torch.Tensor,
    roped_k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    pad_multiple: int = 128,
):
    """Distributed FlexAttention over SP-sharded heads/sequences.

    Inputs use the local SP layout [B, L_local, N_local, D]. The function
    gathers sequence tokens across the SP group, applies block-sparse
    FlexAttention, then scatters the output back to the original local layout.
    """
    roped_q = all_to_all_with_grad(roped_q, scatter_dim=2, gather_dim=1)
    roped_k = all_to_all_with_grad(roped_k, scatter_dim=2, gather_dim=1)
    v = all_to_all_with_grad(v, scatter_dim=2, gather_dim=1)

    batch_size, global_len, num_local_heads, head_dim = roped_q.shape
    padded_length = math.ceil(global_len / pad_multiple) * pad_multiple - global_len
    if padded_length > 0:
        zeros_q = torch.zeros(
            batch_size,
            padded_length,
            num_local_heads,
            head_dim,
            device=roped_q.device,
            dtype=v.dtype,
        )
        zeros_k = torch.zeros(
            batch_size,
            padded_length,
            num_local_heads,
            head_dim,
            device=roped_k.device,
            dtype=v.dtype,
        )
        zeros_v = torch.zeros(
            batch_size,
            padded_length,
            num_local_heads,
            head_dim,
            device=v.device,
            dtype=v.dtype,
        )
        roped_q = torch.cat([roped_q, zeros_q], dim=1)
        roped_k = torch.cat([roped_k, zeros_k], dim=1)
        v = torch.cat([v, zeros_v], dim=1)

    x = _get_compiled_flex_attention()(
        query=roped_q.transpose(2, 1),
        key=roped_k.transpose(2, 1),
        value=v.transpose(2, 1),
        block_mask=block_mask,
    )
    if padded_length > 0:
        x = x[:, :, :-padded_length]

    x = x.transpose(2, 1)
    return all_to_all_with_grad(x, scatter_dim=1, gather_dim=2)


class SequenceParallelHelper:
    def __init__(self, trainer):
        self.trainer = trainer

    @property
    def config(self):
        return self.trainer.config

    @property
    def device(self):
        return self.trainer.device

    @property
    def dtype(self):
        return self.trainer.dtype

    @property
    def sp_size(self):
        return int(getattr(self.trainer, "sequence_parallel_size", 1))

    @property
    def sp_group(self):
        return getattr(self.trainer, "sp_group", None)

    @property
    def vae_halo_latents(self):
        return int(getattr(self.config, "vae_halo_latents", DEFAULT_SP_VAE_HALO_LATENTS))

    def enabled(self):
        return self.sp_size > 1 and self.sp_group is not None

    def sp_root_global_rank(self):
        global_rank = dist.get_rank()
        return (global_rank // self.sp_size) * self.sp_size

    def local_sp_rank(self):
        return get_sp_rank()

    def _chunk_tensor(self, tensor, dim):
        return tensor.chunk(self.sp_size, dim=dim)[self.local_sp_rank()].contiguous()

    def partition_training_inputs(
        self,
        *,
        image_or_video_shape,
        clean_latent=None,
        conditional_dict=None,
        clean_latent_is_sharded=False,
    ):
        if not self.enabled():
            return clean_latent, conditional_dict, image_or_video_shape

        image_or_video_shape = list(image_or_video_shape)
        batch_size = (
            clean_latent.shape[0]
            if clean_latent is not None
            else int(image_or_video_shape[0])
        )

        def chunk_dict(d):
            if d is None:
                return None
            out = {}
            for key, value in d.items():
                if value is None or not torch.is_tensor(value):
                    out[key] = value
                    continue
                value = value.reshape(batch_size, -1, *value.shape[1:])
                num_segments = value.shape[1]
                if num_segments == 1:
                    out[key] = value
                elif num_segments % self.sp_size == 0:
                    out[key] = self._chunk_tensor(value, dim=1)
                else:
                    raise ValueError(
                        f"SP chunking failed for key={key}: "
                        f"num_segments={num_segments} must be 1 or divisible by "
                        f"sp_size={self.sp_size}"
                    )
                out[key] = out[key].reshape(-1, *value.shape[2:]).contiguous()
            return out

        if clean_latent is not None and not clean_latent_is_sharded:
            clean_latent = self._chunk_tensor(clean_latent, dim=1)

        conditional_dict = chunk_dict(conditional_dict)
        image_or_video_shape[1] = image_or_video_shape[1] // self.sp_size
        return clean_latent, conditional_dict, image_or_video_shape

    def chunk_if_needed(self, tensor, *, dim, already_sharded=False):
        if tensor is None or not self.enabled() or already_sharded:
            return tensor
        return self._chunk_tensor(tensor, dim=dim)

    def build_loss_mask(self, batch, clean_latent, clean_latent_is_sharded):
        num_valid_latent_frames = batch.get("num_valid_latent_frames", None)
        if num_valid_latent_frames is None:
            return None
        num_valid_latent_frames = num_valid_latent_frames.to(device=self.device)
        _, local_or_global_frames = clean_latent.shape[:2]
        if clean_latent_is_sharded:
            frame_start = self.local_sp_rank() * local_or_global_frames
            frame_indices = torch.arange(
                frame_start,
                frame_start + local_or_global_frames,
                device=self.device,
            ).unsqueeze(0)
        else:
            frame_indices = torch.arange(
                local_or_global_frames, device=self.device
            ).unsqueeze(0)
        return (frame_indices < num_valid_latent_frames.unsqueeze(1)).float()

    def partition_loss_mask(self, loss_mask, *, already_sharded=False):
        if loss_mask is None:
            return None, None
        if self.enabled() and not already_sharded:
            loss_mask = self._chunk_tensor(loss_mask, dim=1)
        if not self.enabled():
            return loss_mask, None
        global_valid = loss_mask.sum()
        dist.all_reduce(global_valid, op=dist.ReduceOp.SUM, group=self.sp_group)
        return loss_mask, global_valid

    def latent_range_to_raw_window(self, latent_start, latent_end):
        assert latent_end > latent_start
        ratio = int(
            wan_default_config[self.config.model_kwargs.model_name][
                "temporal_compression_ratio"
            ]
        )
        if latent_start == 0:
            raw_start = 0
            pseudo_prefix_latents = 0
        else:
            raw_start = ratio * (latent_start - 1)
            pseudo_prefix_latents = 1
        raw_end = 1 + ratio * (latent_end - 1)
        return raw_start, raw_end, pseudo_prefix_latents

    def chunk_halo_meta(self, *, sp_rank, total_latent_frames, total_raw_frames):
        if total_latent_frames % self.sp_size != 0:
            raise ValueError(
                f"total_latent_frames={total_latent_frames} must be divisible by "
                f"sequence_parallel_size={self.sp_size}"
            )
        local_latent_frames = total_latent_frames // self.sp_size
        keep_start = sp_rank * local_latent_frames
        keep_end = keep_start + local_latent_frames
        halo_start = max(0, keep_start - self.vae_halo_latents)
        raw_start, raw_end, pseudo_prefix_latents = self.latent_range_to_raw_window(
            halo_start, keep_end
        )
        raw_start = max(0, raw_start)
        raw_end = min(int(total_raw_frames), raw_end)
        drop_latents = pseudo_prefix_latents + (keep_start - halo_start)
        return {
            "sp_rank": int(sp_rank),
            "keep_start": int(keep_start),
            "keep_end": int(keep_end),
            "halo_start": int(halo_start),
            "raw_start": int(raw_start),
            "raw_end": int(raw_end),
            "raw_frames": int(raw_end - raw_start),
            "drop_latents": int(drop_latents),
            "local_latent_frames": int(local_latent_frames),
        }

    def chunk_halo_metas(self, *, total_latent_frames, total_raw_frames):
        return [
            self.chunk_halo_meta(
                sp_rank=sp_rank,
                total_latent_frames=total_latent_frames,
                total_raw_frames=total_raw_frames,
            )
            for sp_rank in range(self.sp_size)
        ]

    def scatter_frame_windows_for_chunk_halo(self, root_frames):
        global_rank = dist.get_rank()
        sp_rank = global_rank % self.sp_size
        root_global_rank = self.sp_root_global_rank()

        batch_size, channels, total_raw_frames, height, width = tuple(root_frames.shape)
        total_latent_frames = int(list(self.config.image_or_video_shape)[1])
        metas = self.chunk_halo_metas(
            total_latent_frames=total_latent_frames,
            total_raw_frames=total_raw_frames,
        )
        rank_meta = metas[sp_rank]

        if global_rank == root_global_rank:
            local_frames = None
            for send_meta in metas:
                raw_window = root_frames[
                    :,
                    :,
                    send_meta["raw_start"]:send_meta["raw_end"],
                    :,
                    :,
                ]
                dst_global_rank = root_global_rank + int(send_meta["sp_rank"])
                if dst_global_rank == global_rank:
                    local_frames = raw_window.contiguous()
                    continue
                send_buffer = raw_window.contiguous()
                dist.send(send_buffer, dst=dst_global_rank)
                del send_buffer
            if local_frames is None:
                raise RuntimeError("SP-VAE chunk_halo root did not build local frames")
        else:
            local_frames = torch.empty(
                (batch_size, channels, rank_meta["raw_frames"], height, width),
                device=self.device,
                dtype=root_frames.dtype,
            )
            dist.recv(local_frames, src=root_global_rank)
            local_frames = local_frames.contiguous()

        return local_frames, rank_meta

    def sync_batch(self, batch, step):
        if not self.enabled():
            return batch

        global_rank = dist.get_rank()
        root_global_rank = self.sp_root_global_rank()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if "frames" in batch:
            frames_for_scatter = batch["frames"]
            if global_rank == root_global_rank:
                frames_for_scatter = frames_for_scatter.to(device, non_blocking=True)
            local_frames, meta = self.scatter_frame_windows_for_chunk_halo(
                frames_for_scatter
            )
            batch["frames"] = local_frames
            batch["sp_vae_chunk_meta"] = meta

        if global_rank == root_global_rank:
            info_obj = [{
                "prompts": batch["prompts"],
                "idx": batch["idx"],
                "num_valid_latent_frames": batch.get("num_valid_latent_frames", None),
            }]
        else:
            info_obj = [None]
        dist.broadcast_object_list(info_obj, src=root_global_rank, group=self.sp_group)

        batch["prompts"] = info_obj[0]["prompts"]
        batch["idx"] = info_obj[0]["idx"]
        if info_obj[0]["num_valid_latent_frames"] is not None:
            batch["num_valid_latent_frames"] = info_obj[0]["num_valid_latent_frames"]
        return batch

    def broadcast_tensor_from_root(self, root_tensor, *, shape):
        root_global_rank = self.sp_root_global_rank()
        if dist.get_rank() == root_global_rank:
            tensor = root_tensor.contiguous()
        else:
            tensor = torch.empty(shape, device=self.device, dtype=self.dtype)
        dist.broadcast(tensor, src=root_global_rank, group=self.sp_group)
        return tensor

    def encode_raw_video_latents(self, batch, *, batch_size):
        if self.enabled():
            meta = batch.get("sp_vae_chunk_meta", None)
            if meta is None:
                raise RuntimeError(
                    "SP chunk-halo VAE requires sync_batch to attach sp_vae_chunk_meta."
                )
            total_latent_frames = int(list(self.config.image_or_video_shape)[1])
            latent_tail_shape = tuple(list(self.config.image_or_video_shape)[2:])
            first_latent_shape = (batch_size, 1, *latent_tail_shape)

            with torch.no_grad():
                frames = batch["frames"].to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
                chunk_latent = self.trainer.model.vae.encode_to_latent(frames).to(
                    device=self.device, dtype=self.dtype
                )

                drop = int(meta["drop_latents"])
                local_latent_frames = int(meta["local_latent_frames"])
                keep_end = drop + local_latent_frames
                if chunk_latent.shape[1] < keep_end:
                    raise RuntimeError(
                        "SP-VAE chunk_halo encode produced too few latent frames: "
                        f"chunk_latent={tuple(chunk_latent.shape)} drop={drop} "
                        f"local_latent_frames={local_latent_frames} meta={meta}"
                    )
                clean_latent = chunk_latent[:, drop:keep_end].contiguous()
                if clean_latent.shape[1] != total_latent_frames // self.sp_size:
                    raise RuntimeError(
                        "SP-VAE chunk_halo local latent shape mismatch: "
                        f"clean_latent={tuple(clean_latent.shape)} meta={meta}"
                    )

                image_latent_root = (
                    clean_latent[:, 0:1] if int(meta["keep_start"]) == 0 else None
                )
                image_latent = self.broadcast_tensor_from_root(
                    image_latent_root, shape=first_latent_shape
                )

                del chunk_latent
            return clean_latent, image_latent, True

        frames = batch["frames"].to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            clean_latent = self.trainer.model.vae.encode_to_latent(frames).to(
                device=self.device, dtype=self.dtype
            )
        image_latent = clean_latent[:, 0:1]
        return clean_latent, image_latent, False
