# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from utils.wan_5b_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from utils.i2v_conditioning import (
    _overwrite_i2v_context,
    _zero_i2v_context_timestep,
)
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
from torchvision.io import write_video



class SelfForcingTrainingPipeline:
    def __init__(self,
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 denoising_step_list: Optional[List[int]] = None,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 sampling_steps: Optional[int] = None,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 multi_shot_sink: bool = False,
                 scene_cut_prefix: str = "[SCENE_CUT]",
                 multi_shot_rope_offset: float = 0.0,
                 frame_seq_length: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        if denoising_step_list is None:
            if sampling_steps is None:
                raise ValueError("sampling_steps is required when denoising_step_list is not provided")
            denoising_step_list = self._build_default_denoising_step_list(sampling_steps)
        self.denoising_step_list = torch.as_tensor(denoising_step_list, dtype=torch.long)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = self.generator.model.num_layers
        if not dist.is_initialized() or dist.get_rank() == 0:   
            print(f"num_transformer_blocks: {self.num_transformer_blocks}")
        if frame_seq_length is not None:
            self.frame_seq_length = frame_seq_length
        else:
            self.frame_seq_length = 880
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"frame_seq_length: {self.frame_seq_length}")
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.sampling_steps = sampling_steps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.multi_shot_sink = multi_shot_sink
        self.global_sink_size = sink_size if multi_shot_sink else 0
        self.scene_cut_prefix = scene_cut_prefix
        self.multi_shot_rope_offset = multi_shot_rope_offset
        self.kv_cache_size = num_max_frames * self.frame_seq_length

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[SelfForcingTrainingPipeline] kv_cache_size={self.kv_cache_size}, "
                f"local_attn_size={local_attn_size}, sink_size={sink_size}, "
                f"auto_global_sink_size={self.global_sink_size}, multi_shot_sink={multi_shot_sink}"
            )

    def _build_default_denoising_step_list(self, sampling_steps):
        shift = getattr(self.scheduler, "shift", 1.0)
        num_train_timesteps = getattr(self.scheduler, "num_train_timesteps", 1000)
        sigmas = torch.linspace(1.0, 0.0, int(sampling_steps) + 1, dtype=torch.float32)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        return torch.cat([
            (sigmas * num_train_timesteps).to(torch.long),
            torch.zeros(1, dtype=torch.long),
        ])

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Chunk generation method tailored for sequential training
        
        Args:
            noise: noise tensor for a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: dictionary of conditional information
            kv_cache: externally provided KV cache (defaults to self.kv_cache1 if None)
            crossattn_cache: externally provided cross-attention cache (defaults to self.crossattn_cache if None)
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            return_sim_step: whether to return simulation step info
            
        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
        """
        batch_size, chunk_frames, num_channels, height, width = noise.shape
        
        # Compute block configuration
        if not self.independent_first_frame or chunk_frames % self.num_frame_per_block == 0:
            assert chunk_frames % self.num_frame_per_block == 0
            num_blocks = chunk_frames // self.num_frame_per_block
            all_num_frames = [self.num_frame_per_block] * num_blocks
        else:
            # Handle the case of an independent first frame
            assert (chunk_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (chunk_frames - 1) // self.num_frame_per_block
            all_num_frames = [1] + [self.num_frame_per_block] * num_blocks
            
        # Prepare output tensor
        output = torch.zeros_like(noise)

        # Build per-block conditional dicts for prompt switching
        prompt_embeds = conditional_dict["prompt_embeds"]
        num_prompts = prompt_embeds.shape[0]
        num_segments = num_prompts // batch_size
        if num_segments > 1:
            prompt_embeds_per_block = prompt_embeds.reshape(
                batch_size, num_segments, *prompt_embeds.shape[1:])
            conditional_dict_list = [
                {"prompt_embeds": prompt_embeds_per_block[:, i]}
                for i in range(num_segments)
            ]
        else:
            conditional_dict_list = None

        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        
        # Determine gradient-enabled range — disable everywhere when requires_grad=False
        if not requires_grad:
            start_gradient_frame_index = chunk_frames  # Out of range: no gradients anywhere
        else:
            start_gradient_frame_index = 0
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[SeqTrain-Pipeline] start_gradient_frame_index={start_gradient_frame_index}, chunk_frames={chunk_frames}")
        
        # Generate block by block
        local_start_frame = 0
        for block_index, current_num_frames in enumerate(all_num_frames):
            if conditional_dict_list is not None:
                block_cond = conditional_dict_list[min(block_index, len(conditional_dict_list) - 1)]
                for cache_idx in range(self.num_transformer_blocks):
                    self.crossattn_cache[cache_idx]["is_init"] = False
            else:
                block_cond = conditional_dict

            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]
            
            # Spatial denoising loop
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep
                
                if not exit_flag:
                    # Intermediate steps: no gradients
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                        
                        # Add noise for the next step
                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final step may require gradients
                    enable_grad = local_start_frame >= start_gradient_frame_index

                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                    break
            
            # Record output
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred
            
            # Update cache with context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                )
            
            local_start_frame += current_num_frames
        
        # Compute returned timestep information
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        
        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        
        return output, denoised_timestep_from, denoised_timestep_to

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 21,
            sampling_steps: Optional[int] = None,
            **conditional_dict
    ) -> torch.Tensor:
        # Apply local_attn_size / sink_size overrides before inference,
        # matching what CausalDiffusionInferencePipeline does.
        prev_state = self._apply_attn_overrides()
        try:
            return self._inference_with_trajectory_inner(
                noise=noise,
                initial_latent=initial_latent,
                return_sim_step=return_sim_step,
                slice_last_frames=slice_last_frames,
                sampling_steps=sampling_steps,
                **conditional_dict,
            )
        finally:
            self._restore_attn_overrides(prev_state)

    def _apply_attn_overrides(self):
        """Save current model attention state and apply pipeline overrides."""
        model = self.generator.model
        prev = {
            "local_attn_size": getattr(model, "local_attn_size", -1),
            "rope_temporal_offset": getattr(model, "rope_temporal_offset", 0.0),
            "max_attention_sizes": {},
            "sink_sizes": {},
            "global_sink_sizes": {},
        }
        for name, module in model.named_modules():
            if hasattr(module, "max_attention_size"):
                prev["max_attention_sizes"][name] = module.max_attention_size
            if hasattr(module, "sink_size"):
                prev["sink_sizes"][name] = module.sink_size
            if hasattr(module, "global_sink_size"):
                prev["global_sink_sizes"][name] = module.global_sink_size

        model.local_attn_size = self.local_attn_size
        model.rope_temporal_offset = 0.0
        self._set_all_modules_max_attention_size(self.local_attn_size)
        if self.sink_size is not None and self.sink_size >= 0:
            self._set_all_modules_sink_size(self.sink_size)
        self._set_all_modules_global_sink_size(self.global_sink_size)

        return prev

    def _restore_attn_overrides(self, prev):
        """Restore model attention state saved by _apply_attn_overrides."""
        model = self.generator.model
        model.local_attn_size = prev["local_attn_size"]
        model.rope_temporal_offset = prev["rope_temporal_offset"]
        for name, module in model.named_modules():
            if name in prev["max_attention_sizes"]:
                try:
                    module.max_attention_size = prev["max_attention_sizes"][name]
                except Exception:
                    pass
            if name in prev["sink_sizes"]:
                try:
                    module.sink_size = prev["sink_sizes"][name]
                except Exception:
                    pass
            if name in prev["global_sink_sizes"]:
                try:
                    module.global_sink_size = prev["global_sink_sizes"][name]
                except Exception:
                    pass

    @staticmethod
    def _is_scene_cut_from_mask(scene_cut_mask, block_index: int) -> bool:
        if scene_cut_mask is None or block_index <= 0:
            return False
        if block_index >= len(scene_cut_mask):
            return False
        value = scene_cut_mask[block_index]
        if torch.is_tensor(value):
            return bool(value.item())
        return bool(value)

    def _set_all_modules_sink_size(self, sink_size_value: int):
        """Override sink_size on all submodules that define it."""
        model = self.generator.model
        if hasattr(model, "sink_size"):
            model.sink_size = sink_size_value
        for _name, module in model.named_modules():
            if hasattr(module, "sink_size"):
                try:
                    module.sink_size = sink_size_value
                except Exception:
                    pass

    def _set_all_modules_global_sink_size(self, value: int):
        """Override global_sink_size on all submodules; create the attribute if missing."""
        setattr(self.generator.model, "global_sink_size", value)
        for _, module in self.generator.model.named_modules():
            try:
                setattr(module, "global_sink_size", value)
            except Exception:
                pass

    def _pin_current_chunk(self, kv_cache, current_num_frames):
        """Mark the current chunk's buffer position as pinned for multi-shot sink.

        The pinned region REPLACES the original sink on the next rolling event.
        No data is copied here — relocation happens inside the attention layer
        during rolling, ensuring zero duplication.
        """
        chunk_tokens = current_num_frames * self.frame_seq_length
        pin_len = min(self.sink_size * self.frame_seq_length, chunk_tokens)

        for block_cache in kv_cache:
            local_end = block_cache["local_end_index"].item()
            chunk_start = local_end - chunk_tokens
            block_cache["pinned_start"].fill_(chunk_start)
            block_cache["pinned_len"].fill_(pin_len)

    def _inference_with_trajectory_inner(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 21,
            sampling_steps: Optional[int] = None,
            **conditional_dict
    ) -> torch.Tensor:
        from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        batch_size, num_frames, num_channels, height, width = noise.shape
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        clamp_i2v_first_chunk = self.independent_first_frame and initial_latent is not None
        if clamp_i2v_first_chunk and num_input_frames != 1:
            raise ValueError(
                f"i2v first-chunk clamp expects one conditioning latent frame, got {num_input_frames}."
            )

        if not self.independent_first_frame or clamp_i2v_first_chunk:
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_output_frames = (
            num_frames if clamp_i2v_first_chunk else num_frames + num_input_frames
        )
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Build per-block conditional dicts for prompt switching
        prompt_embeds = conditional_dict["prompt_embeds"]
        num_prompts = prompt_embeds.shape[0]
        num_segments = num_prompts // batch_size
        if num_segments > 1:
            prompt_embeds_per_block = prompt_embeds.reshape(
                batch_size, num_segments, *prompt_embeds.shape[1:])
            conditional_dict_list = [
                {"prompt_embeds": prompt_embeds_per_block[:, i]}
                for i in range(num_segments)
            ]
        else:
            conditional_dict_list = None

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None and not clamp_i2v_first_chunk:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            init_cond = conditional_dict_list[0] if conditional_dict_list is not None else conditional_dict
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=init_cond,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        # --- UniPC scheduler setup ---
        # Priority: function arg > pipeline attribute > len(denoising_step_list)
        if sampling_steps is None:
            sampling_steps = self.sampling_steps if self.sampling_steps is not None else len(self.denoising_step_list)
        shift = self.scheduler.shift
        num_train_timesteps = self.scheduler.num_train_timesteps
        ref_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
        ref_scheduler.set_timesteps(sampling_steps, device=noise.device, shift=shift)
        unipc_timesteps = ref_scheduler.timesteps
        num_denoising_steps = len(unipc_timesteps)

        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        if slice_last_frames == -1:
            # -1 means keep full sequence trainable.
            start_gradient_frame_index = 0
        else:
            start_gradient_frame_index = num_output_frames - slice_last_frames

        scene_cut_mask = conditional_dict.pop("scene_cut_mask", None)
        current_shot_index = 0
        phi = self.multi_shot_rope_offset
        self.generator.model.rope_temporal_offset = 0.0

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)
        for block_index, current_num_frames in enumerate(all_num_frames):
            if phi != 0.0 and self._is_scene_cut_from_mask(scene_cut_mask, block_index):
                current_shot_index += 1
                self.generator.model.rope_temporal_offset = current_shot_index * phi
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(
                        f"[training] multi-shot RoPE: shot_index={current_shot_index}, "
                        f"temporal_offset={self.generator.model.rope_temporal_offset:.4f}"
                    )

            if conditional_dict_list is not None:
                block_cond = conditional_dict_list[min(block_index, len(conditional_dict_list) - 1)]
                for cache_idx in range(self.num_transformer_blocks):
                    self.crossattn_cache[cache_idx]["is_init"] = False
            else:
                block_cond = conditional_dict

            first_i2v_block = clamp_i2v_first_chunk and block_index == 0
            noise_start_frame = (
                current_start_frame
                if clamp_i2v_first_chunk
                else current_start_frame - num_input_frames
            )
            latents = noise[
                :,
                noise_start_frame:noise_start_frame + current_num_frames,
            ]

            # re-init scheduler per chunk (internal state is consumed during stepping)
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=noise.device, shift=shift)

            # Step 3.1: Spatial denoising loop (UniPC multi-step)
            for index, t in enumerate(sample_scheduler.timesteps):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                timestep = t * torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.float32)
                if first_i2v_block:
                    latents = _overwrite_i2v_context(
                        latents, initial_latent, num_input_frames
                    )
                    timestep = _zero_i2v_context_timestep(
                        timestep, num_input_frames
                    )
                if not exit_flag:
                    with torch.no_grad():
                        flow_pred, _ = self.generator(
                            noisy_image_or_video=latents,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        latents = sample_scheduler.step(
                            flow_pred, t, latents, return_dict=False)[0]
                        if first_i2v_block:
                            latents = _overwrite_i2v_context(
                                latents, initial_latent, num_input_frames
                            )
                else:
                    if current_start_frame < start_gradient_frame_index:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        with torch.no_grad():
                            flow_pred, denoised_pred = self.generator(
                                noisy_image_or_video=latents,
                                conditional_dict=block_cond,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True
                        flow_pred, denoised_pred = self.generator(
                            noisy_image_or_video=latents,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    if first_i2v_block:
                        denoised_pred = _overwrite_i2v_context(
                            denoised_pred, initial_latent, num_input_frames
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with context noise to update the cache
            context_timestep = torch.ones(
                [batch_size, current_num_frames], device=noise.device, dtype=torch.long) * self.context_noise
            if first_i2v_block:
                context_timestep = _zero_i2v_context_timestep(
                    context_timestep, num_input_frames
                )
            # add context noise
            context_noise = torch.randn_like(denoised_pred.flatten(0, 1))
            if first_i2v_block:
                context_noise = context_noise.unflatten(0, denoised_pred.shape[:2])
                context_noise[:, :num_input_frames] = 0
                context_noise = context_noise.flatten(0, 1)
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                context_noise,
                context_timestep.reshape(1, -1) * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            if first_i2v_block:
                denoised_pred = _overwrite_i2v_context(
                    denoised_pred, initial_latent, num_input_frames
                )
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.3b: pin KV on scene cut for multi-shot sink.
            if self.multi_shot_sink and scene_cut_mask is not None:
                is_cut = (
                    block_index > 0
                    and block_index < len(scene_cut_mask)
                    and scene_cut_mask[block_index]
                )
                if is_cut:
                    self._pin_current_chunk(self.kv_cache1, current_num_frames)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == num_denoising_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - unipc_timesteps[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - unipc_timesteps[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - unipc_timesteps[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Get the actual number of heads and head dimension from model
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads
        
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "pinned_start": torch.tensor([0], dtype=torch.long, device=device),
                "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        # Get the actual number of heads and head dimension from model
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache

    def clear_kv_cache(self):
        """
        Zero out all tensors in KV cache and cross-attention cache instead of setting them to None.
        This preserves memory allocation while clearing old information, avoiding reallocation overhead.
        """

        # Clear KV cache
        if getattr(self, "kv_cache1", None) is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()
                if "pinned_start" in blk:
                    blk["pinned_start"].zero_()
                if "pinned_len" in blk:
                    blk["pinned_len"].zero_()

        # Clear cross-attention cache
        if getattr(self, "crossattn_cache", None) is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set a unified upper bound for all submodules that contain the max_attention_size attribute.
        local_attn_size_value == -1 indicates global attention (use Wan's default token limit 32760).
        Otherwise set to local_attn_size_value * frame_seq_length.
        """
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        # Root module
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                _ = getattr(self.generator.model, "max_attention_size")
            except Exception:
                pass
            setattr(self.generator.model, "max_attention_size", target_size)

        # Child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass
