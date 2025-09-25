# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import time
import torch
import torch.distributed as dist
from typing import Tuple, Dict, Any, Optional, List
from einops import rearrange

from utils.debug_option import DEBUG, LOG_GPU_MEMORY, DEBUG_GRADIENT
from utils.memory import log_gpu_memory
from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline


class StreamingTrainingModel:
    """
    A model wrapper specifically for streaming/serialized training.

    This class wraps existing models (DMD, DMDSwitch, etc.) and provides a unified
    interface for streaming training. Main features:
    1. Manage streaming generation state
    2. Reuse KV cache and cross-attention cache
    3. Support prompt switching for DMD Switch
    4. Provide chunk-wise loss computation
    5. Support overlapping frames to ensure continuity
    """
    
    def __init__(self, base_model, config):
        """
        Initialize the streaming training model.

        Args:
            base_model: underlying model (DMD, DMDSwitch, etc.)
            config: configuration object
        """
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, 'image_or_video_shape', None)
        
        # Streaming training configuration
        self.chunk_size = getattr(config, "streaming_chunk_size", 21)  # Fixed chunk size used for loss computation
        self.max_length = getattr(config, "streaming_max_length", 57)
        self.possible_max_length = getattr(config, "streaming_possible_max_length", None)
        self.min_new_frame = getattr(config, "streaming_min_new_frame", 18)

        # Get required components from the underlying model
        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func
        
        # Fetch model configuration
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(base_model.inference_pipeline, 'frame_seq_length', 1560)
        
        # Initialize inference pipeline
        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None:
            base_model._initialize_inference_pipeline()
            self.inference_pipeline = base_model.inference_pipeline
        
        # Streaming state
        self.reset_state()  
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] streamingTrainingModel initialized:")
            print(f"[StreamingTrain-Model] chunk_size={self.chunk_size}, max_length={self.max_length}")
            print(f"[StreamingTrain-Model] min_new_frame={self.min_new_frame}")
            print(f"[StreamingTrain-Model] base_model type: {type(self.base_model).__name__}")

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply special encoding to the first frame, following the logic in _run_generator.

        Args:
            frames: frame sequence [batch_size, num_frames, C, H, W]

        Returns:
            processed_frames: processed frame sequence where the first frame is re-encoded as an image latent
        """
        total_frames = frames.shape[1]
        
        if total_frames <= 1:
            # Only one or zero frames, return as is
            return frames
        
        # Determine the range to process: last 21 frames
        process_frames = min(21, total_frames)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Processing first frame encoding for loss: total_frames={total_frames}, processing last {process_frames} frames")
            
        with torch.no_grad():
            # Decode the frames to be processed into pixels
            frames_to_decode = frames[:, :-(process_frames - 1), ...]
            pixels = self.base_model.vae.decode_to_pixel(frames_to_decode)
            
            # Take the last frame's pixel representation
            last_frame_pixel = pixels[:, -1:, ...].to(self.dtype)
            last_frame_pixel = rearrange(last_frame_pixel, "b t c h w -> b c t h w")
            
            # Re-encode as image latent
            image_latent = self.base_model.vae.encode_to_latent(last_frame_pixel).to(self.dtype)
            
        remaining_frames = frames[:, -(process_frames - 1):, ...]
        processed_frames = torch.cat([image_latent, remaining_frames], dim=1)
                
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Processed first frame encoding: {frames.shape} -> {processed_frames.shape}")
            
        return processed_frames

    def reset_state(self):
        """Reset streaming training state"""
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Resetting streaming training state")
            
        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "has_switched": False,  # Track whether prompt has been switched
            "previous_frames": None,  # Store last generated frames for overlap (up to 21)
            "temp_max_length": None,  # Temporary max length for the current sequence
        }

        self.inference_pipeline.clear_kv_cache()
        
    
    def _should_switch_prompt(self, chunk_start_frame: int, chunk_size: int) -> bool:
        """Determine whether to switch prompt (DMDSwitch only)"""
        # Check if the model supports switching (DMDSwitch)
        from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline
        if not isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Not a switch pipeline, no switching")
            return False
        
        # If already switched, do not switch again
        if self.state.get("has_switched", False):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Already switched, not switching again")
            return False
        
        switch_info = self.state["conditional_info"].get("switch_info", {})
        switch_frame_index = switch_info.get("switch_frame_index")
        
        if switch_frame_index is None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] No switch_frame_index, not switching")
            return False
        
        # Check if the switch point falls within the current chunk range [chunk_start_frame, chunk_start_frame + chunk_size)
        chunk_end_frame = chunk_start_frame + chunk_size
        should_switch = chunk_start_frame <= switch_frame_index < chunk_end_frame
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Switch check: switch_frame={switch_frame_index}, chunk=[{chunk_start_frame}, {chunk_end_frame}), should_switch={should_switch}")
            
        return should_switch

    def _get_current_conditional_dict(self, chunk_start_frame: int) -> dict:
        """Get the conditional_dict to use for the current chunk"""
        cond_info = self.state["conditional_info"]
        
        # Check whether it has switched already or should switch now
        switch_info = cond_info.get("switch_info", {})
        if switch_info:
            switch_frame_index = switch_info.get("switch_frame_index")
            if switch_frame_index is not None:
                if self.state.get("has_switched", False) or chunk_start_frame >= switch_frame_index:
                    # If already switched, or current frame has reached the switch point, use the switched prompt
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[StreamingTrain-Model] Using switch conditional_dict for chunk starting at frame {chunk_start_frame}")
                    return switch_info.get("switch_conditional_dict", cond_info["conditional_dict"])
        
        # Otherwise use the original prompt
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Using original conditional_dict for chunk starting at frame {chunk_start_frame}")
        return cond_info["conditional_dict"]
    
    def _generate_chunk(
        self, 
        noise_chunk: torch.Tensor,
        chunk_start_frame: int,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Generate a single chunk.

        Args:
            noise_chunk: noise input [batch_size, chunk_frames, C, H, W]
            chunk_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting timestep for denoising
            denoised_timestep_to: ending timestep for denoising
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before generate chunk {chunk_start_frame}", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] _generate_chunk: chunk_start_frame={chunk_start_frame}, chunk_size={noise_chunk.shape[1]}")
            print(f"[StreamingTrain-Model] requires_grad={requires_grad}")
            
        # Get the conditional_dict to use now
        current_conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        
        # Prepare generation parameters
        kwargs = {
            "noise": noise_chunk,
            "conditional_dict": current_conditional_dict,
            "current_start_frame": chunk_start_frame,
            "requires_grad": requires_grad,
            "return_sim_step": False,
        }
        
        # Add switching logic for DMDSwitch models
        if isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            switch_info = self.state["conditional_info"].get("switch_info", {})
            if switch_info and self._should_switch_prompt(chunk_start_frame, noise_chunk.shape[1]):
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[StreamingTrain-Model] Switching prompt at frame {switch_info['switch_frame_index']}")
                
                # Compute the relative switch frame index (relative to current chunk start)
                relative_switch_index = max(0, switch_info["switch_frame_index"] - chunk_start_frame)
                kwargs["switch_frame_index"] = relative_switch_index
                kwargs["switch_conditional_dict"] = switch_info["switch_conditional_dict"]
                
                # Pass previous_frames for recache when switching
                if self.state["previous_frames"] is not None:
                    kwargs["switch_recache_frames"] = self.state["previous_frames"]
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[StreamingTrain-Model] Passed previous_frames for switch recache: {self.state['previous_frames'].shape}")
                
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[StreamingTrain-Model] Adding switch parameters: relative_switch_index={relative_switch_index}")
                
                # Mark switched to avoid switching again in later chunks
                self.state["has_switched"] = True
        
        # Call the pipeline-specific method to generate a chunk
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Calling pipeline.generate_chunk_with_cache")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before pipeline.generate_chunk_with_cache", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        output, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.generate_chunk_with_cache(**kwargs)
        # if DEBUG:
        #     # Inspect timestep info
        #     print(f"[DEBUG-SeqModel-GenChunk] denoised_timestep_from: {denoised_timestep_from}")
        #     print(f"[DEBUG-SeqModel-GenChunk] denoised_timestep_to: {denoised_timestep_to}")

        #     print(f"output shape: {output.shape}")
        #     with torch.no_grad():
        #         if self.state["previous_frames"] is not None:
        #             output_vis = torch.cat([self.state["previous_frames"], output], dim=1)
        #         else:
        #             output_vis = output
        #         video = self.base_model.vae.decode_to_pixel(output_vis, use_cache=False)
        #         video = (video * 0.5 + 0.5).clamp(0, 1)
        #         video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        #         video_tensor = torch.from_numpy(video[0].astype("uint8"))
        #         from torchvision.io import write_video
        #         write_video(f"debug_save/output_{chunk_start_frame}_to_{chunk_start_frame+output.shape[1]}_denoise_{denoised_timestep_from}_{denoised_timestep_to}_rank{dist.get_rank()}.mp4", video_tensor, fps=16)
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After pipeline.generate_chunk_with_cache", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        return output, denoised_timestep_from, denoised_timestep_to
    
    def setup_sequence(
        self,
        conditional_dict: Dict,
        unconditional_dict: Dict,
        initial_latent: Optional[torch.Tensor] = None,
        switch_conditional_dict: Optional[Dict] = None,
        switch_frame_index: Optional[int] = None,
        temp_max_length: Optional[int] = None,
    ):
        """Set up a new sequence"""
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before setup_sequence", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Setting up new sequence:")
            print(f"[StreamingTrain-Model] image_or_video_shape={self.image_or_video_shape}")
            print(f"[StreamingTrain-Model] initial_latent shape: {initial_latent.shape if initial_latent is not None else None}")
            print(f"[StreamingTrain-Model] switch_frame_index={switch_frame_index}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.image_or_video_shape[0]
        if self.inference_pipeline.kv_cache1 is None:
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] init kv_cache1: {self.inference_pipeline.kv_cache1[0]['k'].shape}")

        if self.inference_pipeline.crossattn_cache is None:
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] init crossattn_cache: {self.inference_pipeline.crossattn_cache[0]['k'].shape}")
        
        # Reset state
        self.reset_state()
        self.state["temp_max_length"] = temp_max_length
        
        # if self.possible_max_length is not None:
        #     # Ensure all processes select the same length
        #     if dist.is_initialized():
        #         if dist.get_rank() == 0:
        #             import random
        #             selected_idx = random.randint(0, len(self.possible_max_length) - 1)
        #         else:
        #             selected_idx = 0
        #         selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
        #         dist.broadcast(selected_idx_tensor, src=0)
        #         selected_idx = selected_idx_tensor.item()
        #     else:
        #         import random
        #         selected_idx = random.randint(0, len(self.possible_max_length) - 1)
            
        #     self.state["temp_max_length"] = self.possible_max_length[selected_idx]
        # else:
        #     self.state["temp_max_length"] = self.max_length
            
        #     if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        #         print(f"[StreamingTrain-Model] Selected temporary max length: {self.state['temp_max_length']} (from {self.possible_max_length})")
        
        # Prepare initial sequence
        if initial_latent is not None:
            self.state["current_length"] = initial_latent.shape[1]
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Starting with initial_latent, length={self.state['current_length']}")
        else:
            self.state["current_length"] = 0
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Starting with empty sequence")
        
        # Save conditional information
        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
        }
        
        # DMDSwitch related information
        if switch_conditional_dict is not None and switch_frame_index is not None:
            self.state["conditional_info"]["switch_info"] = {
                "switch_conditional_dict": switch_conditional_dict,
                "switch_frame_index": switch_frame_index,
            }
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] DMDSwitch info saved: switch_frame_index={switch_frame_index}")
        
        # Handle cache updates for initial_latent
        if initial_latent is not None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Initializing cache with initial_latent")
            
            # Use initial latent to update cache
            timestep = torch.zeros([batch_size, initial_latent.shape[1]], device=self.device, dtype=torch.int64)
            with torch.no_grad():
                self.inference_pipeline.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.inference_pipeline.kv_cache1,
                    crossattn_cache=self.inference_pipeline.crossattn_cache,
                    current_start=0
                )
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"StreamingTrain-Model: After initial latent processing", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] No initial latent")

    def can_generate_more(self) -> bool:
        """Check whether more chunks can be generated"""
        current_length = self.state["current_length"]
        temp_max_length = self.state.get("temp_max_length")
        can_generate = current_length < temp_max_length and (current_length + self.min_new_frame) <= temp_max_length
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] can_generate_more: current_length={current_length}, temp_max_length={temp_max_length}, global_max_length={self.max_length}, can_generate={can_generate}")
            
        return can_generate
    def generate_next_chunk(self, requires_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate the next chunk, supporting overlap to ensure temporal continuity.

        Args:
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: the full generated chunk (including overlap frames)
            info: generation info (including timestep, gradient_mask, etc.)
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] generate_next_chunk called: requires_grad={requires_grad}")
            
        # DEBUG: inspect the generator model gradient state
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            gen_training_mode = self.generator.training
            gen_params_requiring_grad = sum(1 for p in self.generator.parameters() if p.requires_grad)
            gen_params_total = sum(1 for p in self.generator.parameters())
            print(f"[DEBUG-SeqModel] Generator training mode: {gen_training_mode}")
            print(f"[DEBUG-SeqModel] Generator params requiring grad: {gen_params_requiring_grad}/{gen_params_total}")
            
        if not self.can_generate_more():
            raise ValueError("Cannot generate more chunks")
        
        current_length = self.state["current_length"]
        batch_size = self.image_or_video_shape[0]
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Generating chunk: current_length={current_length}")
        
        # Check if previous_frames can be used for overlap and auto-compute overlap frame count
        previous_frames = self.state.get("previous_frames")
        if previous_frames is not None:
            # Randomly select number of new frames (min=min_new_frame, max=chunk_size, step=3)
            max_new_frames = min(self.state["temp_max_length"] - current_length + 1, self.chunk_size)
            possible_new_frames = list(range(self.min_new_frame, max_new_frames, 3))
            
            # Ensure all processes choose the same random value
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(possible_new_frames) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(possible_new_frames) - 1)
            
            new_frames_to_generate = possible_new_frames[selected_idx]

            # Auto-compute required overlap frames to ensure the final chunk has 21 frames
            overlap_frames = self.chunk_size - new_frames_to_generate
            if overlap_frames > 0 and overlap_frames <= previous_frames.shape[1]:
                overlap_frames_to_use = overlap_frames
            else:
                # If overlap can't be used, generate a full chunk_size without overlap
                overlap_frames_to_use = 0
                new_frames_to_generate = self.chunk_size
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] With auto overlap: generating {new_frames_to_generate} new frames, reusing {overlap_frames_to_use} overlap frames")
        else:
            overlap_frames_to_use = 0
            new_frames_to_generate = self.chunk_size
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] First chunk: generating {new_frames_to_generate} frames (no overlap)")
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Random frame selection: selected={new_frames_to_generate}")
            print(f"[StreamingTrain-Model] Auto overlap calculation: overlap_frames={overlap_frames_to_use}")
        
        # Sample noise for new frames
        noise_chunk = torch.randn(
            [batch_size, new_frames_to_generate, *self.image_or_video_shape[2:]],
            device=self.device,
            dtype=self.dtype
        )
        
        # Generate new frames - note chunk_start_frame should consider overlap
        generated_new_frames, denoised_timestep_from, denoised_timestep_to = self._generate_chunk(
            noise_chunk=noise_chunk,
            chunk_start_frame=current_length,
            requires_grad=requires_grad,
        )

        # Build the full chunk for loss computation
        if previous_frames is not None:
            # Concatenate specified overlap frames and newly generated frames
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)
        else:
            full_chunk = generated_new_frames
        
        # Update state - save the last 21 frames as previous_frames for the next chunk
        # The frames saved here should be those before _process_first_frame_encoding
        frames_to_save = full_chunk.detach().clone()[:, -self.chunk_size:, ...]
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Saved last {frames_to_save.shape[1]} frames as previous_frames")
        

        # Process first-frame encoding (if there is overlap)
        if previous_frames is not None:
            full_chunk = self._process_first_frame_encoding(full_chunk)
        
        if previous_frames is not None:
            # Create gradient_mask: only newly generated frames require gradients
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            # Overlap frames do not compute gradients; new frames do
            gradient_mask[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate, ...] = True
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Built chunk with auto overlap: shape={full_chunk.shape}")
                print(f"[StreamingTrain-Model] Gradient mask: {new_frames_to_generate} frames will have gradients out of {full_chunk.shape[1]}")
        else:
            # For the first chunk, all frames are newly generated
            gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)

        self.state["current_length"] += new_frames_to_generate  # Increase only by newly generated frames
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Updated state: current_length={self.state['current_length']}")
            if self.state["previous_frames"] is not None:
                print(f"[StreamingTrain-Model] Saved {self.state['previous_frames'].shape[1]} frames as previous_frames for next chunk")

        self.state["previous_frames"] = frames_to_save

        # Return info
        info = {
            "denoised_timestep_from": denoised_timestep_from,
            "denoised_timestep_to": denoised_timestep_to,
            "chunk_start_frame": current_length,  # Start frame position in the full sequence
            "chunk_frames": full_chunk.shape[1],  # Chunk size used for loss (fixed 21 frames)
            "new_frames_generated": new_frames_to_generate,
            "current_length": self.state["current_length"],
            "gradient_mask": gradient_mask,  # Mask frames that do not require gradients for loss computation
            "overlap_frames_used": overlap_frames_to_use,
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[StreamingTrain-Model] current_training_chunk: ({self.state['current_length'] - new_frames_to_generate} -> {self.state['current_length']})/{self.state['temp_max_length']}")
        return full_chunk, info
    
    def compute_generator_loss(self,
        chunk: torch.Tensor,
        chunk_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the generator loss.

        Args:
            chunk: generated chunk
            chunk_info: chunk metadata

        Returns:
            loss: loss value
            log_dict: log dictionary
        """
        _t_loss_start = time.time()
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before compute generator loss", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        # Fetch conditional_dict for loss computation
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        unconditional_dict = self.state["conditional_info"]["unconditional_dict"]
        
        # Fetch gradient_mask to compute loss only on newly generated frames
        gradient_mask = chunk_info.get("gradient_mask", None)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Using conditional_dict and unconditional_dict for loss calculation at frame {chunk_start_frame}")
        
        # Compute DMD loss
        dmd_loss, dmd_log_dict = self.base_model.compute_distribution_matching_loss(
            image_or_video=chunk,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,  # Pass gradient_mask
            denoised_timestep_from=chunk_info["denoised_timestep_from"],
            denoised_timestep_to=chunk_info["denoised_timestep_to"]
        )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After DMD loss computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Update log dict
        dmd_log_dict.update({
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", chunk.shape[1]),
        })
        
        return dmd_loss, dmd_log_dict
    
    def _clear_cache_gradients(self):
        """
        Clear possible gradient references in KV cache and cross-attention cache.
        This is important for preventing memory leaks, especially before critic training.
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Clearing cache gradients")
            
        # Clear gradient refs in KV cache
        if hasattr(self.inference_pipeline, 'kv_cache1') and self.inference_pipeline.kv_cache1 is not None:
            for cache_block in self.inference_pipeline.kv_cache1:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()
        
        # Clear gradient refs in cross-attention cache        
        if hasattr(self.inference_pipeline, 'crossattn_cache') and self.inference_pipeline.crossattn_cache is not None:
            for cache_block in self.inference_pipeline.crossattn_cache:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Cache gradients cleared")

    def compute_critic_loss(self, chunk: torch.Tensor, chunk_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute critic loss.

        Args:
            chunk: generated chunk
            chunk_info: chunk metadata

        Returns:
            loss: loss value
            log_dict: log dictionary
        """
        _t_loss_start = time.time()
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before compute critic loss", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] compute_critic_loss: chunk_shape={chunk.shape}")
            for k, v in chunk_info.items():
                if k == "gradient_mask":
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v[0, :, 0, 0, 0]}")
                else:
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v}")
            print(f"[StreamingTrain-Model] chunk requires_grad: {chunk.requires_grad}")
            
        # Critical fix: ensure chunk has no gradient connections
        if chunk.requires_grad:
            chunk = chunk.detach()

        # Critical fix: clear gradient references inside caches
        self._clear_cache_gradients()
        
        # Force clear CUDA cache to ensure previous graphs are released
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After chunk detachment and cache cleanup", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        # Fetch conditional_dict for loss computation
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        
        # Fetch gradient_mask to compute loss only on newly generated frames
        gradient_mask = chunk_info.get("gradient_mask", None)
        
        batch_size, num_frame = chunk.shape[:2]
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Preparing critic loss: batch_size={batch_size}, num_frame={num_frame}")
        
        # Use the same timestep range logic as non-streaming training
        denoised_timestep_from = chunk_info.get("denoised_timestep_from", None)
        denoised_timestep_to = chunk_info.get("denoised_timestep_to", None)
        
        min_timestep = denoised_timestep_to if (getattr(self.base_model, 'ts_schedule', False) and denoised_timestep_to is not None) else getattr(self.base_model, 'min_score_timestep')
        max_timestep = denoised_timestep_from if (getattr(self.base_model, 'ts_schedule_max', False) and denoised_timestep_from is not None) else getattr(self.base_model, 'num_train_timestep')
        
        # Randomly select time steps
        critic_timestep = self.base_model._get_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=getattr(self.base_model, 'num_frame_per_block', 3),
            uniform_timestep=True  # Set to True to match non-streaming training
        ).to(self.device)
        
        # Apply the same timestep shift logic as non-streaming training
        if getattr(self.base_model, 'timestep_shift') > 1:
            timestep_shift = self.base_model.timestep_shift
            critic_timestep = timestep_shift * \
                (critic_timestep / 1000) / (1 + (timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        
        critic_timestep = critic_timestep.clamp(self.base_model.min_step, self.base_model.max_step)
        
        # Sample noise
        critic_noise = torch.randn_like(chunk)
        
        # Add noise to chunk
        noisy_chunk = self.scheduler.add_noise(
            chunk.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Added noise, timestep range: [{critic_timestep.min().item()}, {critic_timestep.max().item()}]")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before fake score computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute fake prediction
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_chunk,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After fake score computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        # Compute denoising loss
        denoising_loss_type = getattr(self.base_model.args, 'denoising_loss_type', 'mse')
        if denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Using flow-based denoising loss")
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Using MSE-based denoising loss")

        gradient_mask_flat = gradient_mask.flatten(0, 1) if gradient_mask is not None else None
        denoising_loss = self.denoising_loss_func(
            x=chunk.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat  # Pass gradient_mask
        )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After denoising loss computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Critic loss computed: {denoising_loss.item()}")

        # Critical: clean up intermediate variables after critic loss
        del conditional_dict, critic_noise, noisy_chunk, pred_fake_image
        if 'flow_pred' in locals():
            del flow_pred
        if 'pred_fake_noise' in locals():
            del pred_fake_noise

        # Build log dict
        critic_log_dict = {
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", num_frame),
        }

        return denoising_loss, critic_log_dict
    
    def get_sequence_length(self) -> int:
        """Get current sequence length"""
        return self.state.get("current_length", 0) 