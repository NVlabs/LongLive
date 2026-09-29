# Copyright 2024-2025 LongLive Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel causal diffusion inference pipeline for Wan2.2-TI2V-5B."""

from typing import List, Optional
import types

import torch
import torch.distributed as dist

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from utils.config import wan_default_config
from utils.scheduler import FlowMatchScheduler, SchedulerInterface

from wan_5b.distributed.sp_ulysses_inference import (
    get_sp_world_size,
    init_sequence_parallel,
    is_sp_enabled,
    sp_print,
)


def _model_kw(model_kwargs, key, default=None):
    if isinstance(model_kwargs, dict):
        return model_kwargs.get(key, default)
    return getattr(model_kwargs, key, default)


class SPWanDiffusionWrapper5B(torch.nn.Module):
    """Wan 5B diffusion wrapper backed by the Ulysses SP causal model."""

    def __init__(
        self,
        model_name="Wan2.2-TI2V-5B",
        timestep_shift=5.0,
        local_attn_size=-1,
        sink_size=0,
        num_frame_per_block=1,
        t_scale=1.0,
        rope_method="linear",
        original_seq_len=None,
    ):
        super().__init__()
        if model_name != "Wan2.2-TI2V-5B":
            raise ValueError(f"SP inference only supports Wan2.2-TI2V-5B, got {model_name}")

        from wan_5b.modules.causal_model_sp_ulysses import UlyssesSPCausalWanModel

        sp_print("Using Ulysses SP model for Wan2.2-TI2V-5B")
        self.model = UlyssesSPCausalWanModel(
            model_type="ti2v",
            in_dim=48,
            dim=3072,
            ffn_dim=14336,
            out_dim=48,
            num_heads=24,
            num_layers=30,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
        )
        self.model.eval()
        self.model.t_scale = t_scale
        self.model.rope_method = rope_method
        self.model.original_seq_len = original_seq_len

        self.uniform_timestep = False
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 28160
        self._compiled_model_call = None
        self.get_scheduler()

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def configure_torch_compile(
        self,
        *,
        backend: str = "inductor",
        mode: str | None = "max-autotune-no-cudagraphs",
        fullgraph: bool = False,
        dynamic: bool | None = False,
        options: dict | None = None,
        suppress_errors: bool = True,
    ) -> bool:
        from utils.torch_compile_utils import configure_module_call_torch_compile

        self._compiled_model_call = configure_module_call_torch_compile(
            self.model,
            name="SPWanDiffusionWrapper5B.model",
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
            suppress_errors=suppress_errors,
        )
        return self._compiled_model_call is not None

    def _call_model(self, *args, **kwargs):
        if self._compiled_model_call is not None:
            return self._compiled_model_call(*args, **kwargs)
        return self.model(*args, **kwargs)

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        **_,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
        flow_pred = self._call_model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start if current_start is not None else 0,
            cache_start=cache_start if cache_start is not None else 0,
        ).permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0


class CausalDiffusionInferencePipelineSP(CausalDiffusionInferencePipeline):
    """LongLive2.0 diffusion inference pipeline using Ulysses sequence parallelism."""

    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
        sp_group=None,
        dp_rank: int = 0,
    ):
        if dist.is_initialized():
            init_sequence_parallel(group=sp_group)
        self.dp_rank = dp_rank
        if generator is None:
            model_kwargs = getattr(args, "model_kwargs", {})
            generator = SPWanDiffusionWrapper5B(
                model_name=_model_kw(model_kwargs, "model_name", "Wan2.2-TI2V-5B"),
                timestep_shift=_model_kw(model_kwargs, "timestep_shift", 5.0),
                local_attn_size=_model_kw(model_kwargs, "local_attn_size", -1),
                sink_size=_model_kw(model_kwargs, "sink_size", 0),
                num_frame_per_block=getattr(args, "num_frame_per_block", 1),
                t_scale=getattr(args, "t_scale", 1.0),
                rope_method=getattr(args, "rope_method", "linear"),
                original_seq_len=getattr(args, "original_seq_len", None),
            )
        super().__init__(
            args=args,
            device=device,
            generator=generator,
            text_encoder=text_encoder,
            vae=vae,
        )
        if self.quantize_kv:
            raise ValueError("kv_quant is not supported in Ulysses SP inference.")
        sp_print(
            f"SP diffusion pipeline initialized: nfpb={self.num_frame_per_block}, "
            f"sp_world={get_sp_world_size() if is_sp_enabled() else 1}, dp_rank={dp_rank}"
        )

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize head-split KV caches for Ulysses SP."""
        kv_cache_pos = []
        kv_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        sp_world_size = get_sp_world_size() if is_sp_enabled() else 1
        if num_heads % sp_world_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by sp_world_size ({sp_world_size})"
            )
        cache_heads = num_heads // sp_world_size

        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 3 * self.num_frame_per_block * self.frame_seq_length
        block_token_size = self.num_frame_per_block * self.frame_seq_length
        max_blocks = kv_cache_size // block_token_size

        sp_print(
            f"Initializing SP KV cache: size={kv_cache_size}, heads/rank={cache_heads}, "
            f"block_token_size={block_token_size}"
        )
        for _ in range(self.num_transformer_blocks):
            entry = {
                "k": torch.zeros(
                    [batch_size, kv_cache_size, cache_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, cache_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "quantized": False,
                "block_token_size": block_token_size,
                "max_blocks": max_blocks,
                "num_heads": cache_heads,
                "num_filled_blocks": 0,
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
            }
            kv_cache_pos.append({key: value.clone() if torch.is_tensor(value) else value for key, value in entry.items()})
            kv_cache_neg.append({key: value.clone() if torch.is_tensor(value) else value for key, value in entry.items()})

        self.kv_cache_pos = kv_cache_pos
        self.kv_cache_neg = kv_cache_neg
