# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from pipeline.streaming_training import StreamingTrainingPipeline
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.memory import log_gpu_memory


class StreamingSwitchTrainingPipeline(StreamingTrainingPipeline):
    """Training pipeline supporting mid-video prompt switching.

    Use case: In a single roll-out, the first half of the video uses prompt-1. Once the
    switch frame is reached, only the high-level self-attention KV cache is refreshed and
    all cross-attention caches are reset, then generation continues with prompt-2 for the
    remaining frames.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_sink = getattr(args, "global_sink", False)

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        switch_frame_index: Optional[int] = None,
        switch_conditional_dict: Optional[dict] = None,
        switch_recache_frames: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Chunk generation method tailored for sequential training with prompt switching.

        Args:
            noise: noise of a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: initial conditional information
            kv_cache: external KV cache
            crossattn_cache: external cross-attention cache
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            switch_frame_index: switch frame index (relative to chunk start)
            switch_conditional_dict: conditional info after switching
            switch_recache_frames: frames used to recache during switch (the 21 frames before switch_index)
            return_sim_step: whether to return simulation step info

        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
        """

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-DMDSwitch] generate_chunk_with_cache called")
            print(f"[SeqTrain-DMDSwitch] switch_frame_index={switch_frame_index}, switch_conditional_dict={switch_conditional_dict is not None}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-DMDSwitch: Before switch chunk generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # If no switch info, fall back to the parent implementation
        if switch_conditional_dict is None or switch_frame_index is None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-DMDSwitch] No switch info, using parent class implementation")
            return super().generate_chunk_with_cache(
                noise=noise,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                requires_grad=requires_grad,
                return_sim_step=return_sim_step,
            )
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-DMDSwitch] Switch will occur at relative frame {switch_frame_index}")
        
        batch_size, chunk_frames, num_channels, height, width = noise.shape
        assert chunk_frames % self.num_frame_per_block == 0
        num_blocks = chunk_frames // self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks

        # Prepare output
        output = torch.zeros_like(noise)
        
        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        
        # Determine the gradient-enabled range
        if not requires_grad:
            start_gradient_frame_index = chunk_frames  # Out of range: no gradients anywhere
        else:
            start_gradient_frame_index = switch_frame_index
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-DMDSwitch] start_gradient_frame_index={start_gradient_frame_index}")
        
        local_start_frame = 0
        self.generator.model.local_attn_size = int(self.local_attn_size)
        self._set_all_modules_max_attention_size(int(self.local_attn_size))

        using_second = False
        cond_in_use = conditional_dict
        for block_index, current_num_frames in enumerate(all_num_frames):
            if (not using_second) and (local_start_frame >= switch_frame_index):
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-DMDSwitch] Triggering switch at local_frame={local_start_frame}, switch_index={switch_frame_index}")
                
                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"SeqTrain-DMDSwitch: Before cache refresh for switch", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
                    
                self._recache_after_switch(output[:, :local_start_frame, ...], current_start_frame+local_start_frame, switch_conditional_dict, local_start_frame, switch_recache_frames)
                
                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"SeqTrain-DMDSwitch: After cache refresh for switch", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
                
                cond_in_use = switch_conditional_dict
                using_second = True
                
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-DMDSwitch] Switch completed, using_second={using_second}")
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                prompt_type = "second" if using_second else "first"
                print(f"[SeqTrain-DMDSwitch] Processing block {block_index}: frames {local_start_frame}-{local_start_frame + current_num_frames}, using {prompt_type} prompt")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and block_index == 0:
                log_gpu_memory(f"SeqTrain-DMDSwitch: Before first block generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
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
                            conditional_dict=cond_in_use,
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
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-DMDSwitch] Block {block_index} final step: enable_grad={enable_grad}, local_frame={local_start_frame}")
                    
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                    break
            
            # Record output
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred
            
            # Update cache using context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=cond_in_use,
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

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict, local_start_frame=None, switch_recache_frames=None):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        if current_start_frame == 0:
            return

        if switch_recache_frames is not None:
            frames_to_recache = torch.cat([switch_recache_frames, output], dim=1)[:, -21:, ...]
            num_recache_frames = frames_to_recache.shape[1]
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-DMDSwitch] Using external switch_recache_frames (previous_frames): {frames_to_recache.shape}")
        else:
            # Determine how to fetch frames based on whether local_start_frame is provided
            if local_start_frame is not None:
                # Chunk mode: output is the current chunk's output; use relative coordinates
                num_recache_frames = min(local_start_frame, 21)
                frames_to_recache = output[:, -num_recache_frames:]
            else:
                # Full sequence mode: output is the complete sequence; use absolute coordinates
                num_recache_frames = min(current_start_frame, 21)
                frames_to_recache = output[:, -num_recache_frames:]
            
        batch_size, num_recache_frames, c, h, w = frames_to_recache.shape
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            print(f"num_recache_frames: {num_recache_frames}, current_start_frame: {current_start_frame}, local_start_frame: {local_start_frame}")
        
        # Create an appropriate BlockMask for recomputation
        device = frames_to_recache.device
        
        # Use the standard blockwise causal mask
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=21
        )
        
        # Prepare time steps
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.context_noise
        
        # Set the new block_mask
        self.generator.model.block_mask = block_mask
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"current_start_frame: {current_start_frame}, num_recache_frames: {num_recache_frames}")
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=(current_start_frame - num_recache_frames) * self.frame_seq_length,
            )

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False