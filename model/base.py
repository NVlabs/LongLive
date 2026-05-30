# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch
import math

from pipeline import SelfForcingTrainingPipeline
from utils.config import section_get
from utils.loss import get_denoising_loss
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def build_default_denoising_step_list(sampling_steps, num_train_timesteps=1000, shift=1.0, include_zero=True):
    sigmas = torch.linspace(1.0, 0.0, int(sampling_steps) + 1, dtype=torch.float32)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = (sigmas * num_train_timesteps).to(torch.long)
    if include_zero:
        timesteps = torch.cat([timesteps, torch.zeros(1, dtype=torch.long)])
    return timesteps


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        print("args.model_kwargs.model_name", args.model_kwargs.model_name)
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.denoising_step_list = None
        if getattr(args, "denoising_step_list", None) is not None:
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if getattr(args, "warp_denoising_step", False):
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
        elif getattr(args, "sampling_steps", None) is not None:
            self.denoising_step_list = build_default_denoising_step_list(
                sampling_steps=args.sampling_steps,
                num_train_timesteps=getattr(args, "num_train_timestep", self.scheduler.num_train_timesteps),
                shift=getattr(args, "timestep_shift", self.scheduler.shift),
                include_zero=True,
            )

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.2-TI2V-5B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.2-TI2V-5B")
        self.local_attn_size = section_get(
            args,
            "inference",
            "local_attn_size",
            getattr(args, "model_kwargs", {}).get("local_attn_size", -1),
            aliases=("inference_local_attn_size",),
        )
        all_causal = getattr(args, "all_causal", False)
        score_is_causal = all_causal

        model_name = args.model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported in this release, got {model_name}")
        if not dist.is_initialized() or dist.get_rank() == 0:
            tag = "all-causal 5B mode" if all_causal else "Wan2.2-TI2V-5B"
            print(f"Using {tag}")

        # Generator
        generator_is_causal = getattr(args, "generator_is_causal", True)
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=generator_is_causal)
        self.generator.model.requires_grad_(True)

        # Real Score
        real_kwargs = getattr(args, "real_model_kwargs", {"model_name": self.real_model_name})
        self.real_score = WanDiffusionWrapper(**real_kwargs, is_causal=score_is_causal)
        self.real_score.model.requires_grad_(False)

        # Fake Score
        fake_kwargs = getattr(args, "fake_model_kwargs", {"model_name": self.fake_model_name})
        self.fake_score = WanDiffusionWrapper(**fake_kwargs, is_causal=score_is_causal)
        self.fake_score.model.requires_grad_(True)

        # Text Encoder & VAE
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame and not getattr(self.args, "i2v", False):
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(getattr(args, "denoising_loss_type", "flow"))()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        slice_last_frames: int = 21,
        noise=None,
        clean_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
            - slice_last_frames: number of frames to keep from the end.
            - noise: optional pre-sampled noise.
            - clean_latent: a tensor [B, F, C, H, W] for off-policy mode (backward_simulation=False).
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - gradient_mask: boolean tensor or None.
            - denoised_timestep_from: int or None.
            - denoised_timestep_to: int or None.
        """
        use_backward_simulation = getattr(self.args, "backward_simulation", True)

        if use_backward_simulation:
            return self._run_generator_backward_simulation(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
                noise=noise,
            )
        else:
            assert clean_latent is not None, "clean_latent is required when backward_simulation=False"
            return self._run_generator_off_policy(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
            )

    def _run_generator_off_policy(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Off-policy generator: add noise to clean_latent at each timestep in
        denoising_step_list, randomly pick one, and run the generator for a
        single forward pass.  Returns (pred_x0, gradient_mask,
        denoised_timestep_from, denoised_timestep_to).
        """
        batch_size, num_frame = image_or_video_shape[:2]
        denoising_step_list = self.denoising_step_list.to(self.device)

        # Build noisy versions at every timestep in the schedule
        simulated_noisy_input = []
        for ts in denoising_step_list:
            rand_noise = torch.randn_like(clean_latent)
            noisy_timestep = ts * torch.ones(
                [batch_size, num_frame], device=self.device, dtype=torch.long)

            if ts.item() != 0:
                noisy_image = self.scheduler.add_noise(
                    clean_latent.flatten(0, 1),
                    rand_noise.flatten(0, 1),
                    noisy_timestep.flatten(0, 1),
                ).unflatten(0, (batch_size, num_frame))
            else:
                noisy_image = clean_latent
            simulated_noisy_input.append(noisy_image)

        simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)  # [B, T, F, C, H, W]

        # Randomly sample a step index [B, F], uniform within each block
        num_steps = len(denoising_step_list)
        generator_is_causal = getattr(self.args, "generator_is_causal", True)
        if not generator_is_causal:
            # Bidirectional generator: all frames must share the same timestep
            index = torch.randint(0, num_steps, [batch_size, 1],
                                  device=self.device, dtype=torch.long).expand(-1, num_frame).contiguous()
        else:
            index = torch.randint(0, num_steps, [batch_size, num_frame],
                                  device=self.device, dtype=torch.long)
            # Make the index the same within every block
            if self.independent_first_frame and not getattr(self.args, "i2v", False):
                idx_rest = index[:, 1:]
                idx_rest = idx_rest.reshape(batch_size, -1, self.num_frame_per_block)
                idx_rest[:, :, 1:] = idx_rest[:, :, 0:1]
                index = torch.cat([index[:, :1], idx_rest.reshape(batch_size, -1)], dim=1)
            else:
                index = index.reshape(batch_size, -1, self.num_frame_per_block)
                index[:, :, 1:] = index[:, :, 0:1]
                index = index.reshape(batch_size, -1)

        # Gather the noisy input corresponding to the sampled index
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(batch_size, 1, num_frame, 1, 1, 1).expand(
                -1, -1, -1, *image_or_video_shape[2:])
        ).squeeze(1)  # [B, F, C, H, W]

        timestep = denoising_step_list[index]  # [B, F]
        context_frames = int(initial_latent.shape[1]) if initial_latent is not None else 0
        if context_frames > 0:
            if context_frames >= num_frame:
                raise ValueError(
                    f"initial_latent has {context_frames} frames but training clip has {num_frame}."
                )
            noisy_input[:, :context_frames] = initial_latent.to(
                device=noisy_input.device,
                dtype=noisy_input.dtype,
            )
            timestep[:, :context_frames] = 0

        # Single forward pass through the generator
        _, pred_x0 = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep.float(),
            clean_x=clean_latent if getattr(self.args, "teacher_forcing", False) else None,
        )
        pred_x0 = pred_x0.to(self.dtype)

        # Derive denoised_timestep_from / to from the sampled index for ts_schedule
        # Use the first batch element's first block index as the representative scalar
        rep_idx = index[0, 0].item()
        denoised_timestep_from = denoising_step_list[rep_idx].item()
        if rep_idx + 1 < num_steps:
            denoised_timestep_to = denoising_step_list[rep_idx + 1].item()
        else:
            denoised_timestep_to = 0

        gradient_mask = None
        if context_frames > 0:
            pred_x0[:, :context_frames] = initial_latent.to(
                device=pred_x0.device,
                dtype=pred_x0.dtype,
            )
            gradient_mask = torch.ones_like(pred_x0, dtype=torch.bool)
            gradient_mask[:, :context_frames] = False
        return pred_x0, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _run_generator_backward_simulation(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        slice_last_frames: int = 21,
        noise=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        On-policy generator via backward simulation (original path).
        """
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        noise_shape = image_or_video_shape.copy()

        separate_first_frame = self.independent_first_frame and not getattr(self.args, "i2v", False)
        min_num_frames = (self.min_num_training_frames - 1) if separate_first_frame else self.min_num_training_frames
        max_num_frames = self.num_training_frames - 1 if separate_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if separate_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        noise_shape[1] = num_generated_frames
        if noise is not None:
            noise = noise[:, :num_generated_frames]
        else:
            noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=noise,
            slice_last_frames=slice_last_frames,
            **conditional_dict,
        )

        if slice_last_frames != -1 and pred_image_or_video.shape[1] > slice_last_frames:
            with torch.no_grad():
                if slice_last_frames > 1:
                    latent_to_decode = pred_image_or_video[:, :-(slice_last_frames - 1), ...]
                else:
                    latent_to_decode = pred_image_or_video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            if slice_last_frames > 1:
                last_frames = pred_image_or_video[:, -(slice_last_frames - 1):, ...]
                pred_image_or_video_sliced = torch.cat([image_latent, last_frames], dim=1)
            else:
                pred_image_or_video_sliced = image_latent
            if num_generated_frames != min_num_frames:
                gradient_mask = torch.ones_like(pred_image_or_video_sliced, dtype=torch.bool)
                if self.independent_first_frame:
                    gradient_mask[:, :1] = False
                else:
                    gradient_mask[:, :self.num_frame_per_block] = False
            else:
                gradient_mask = None
        else:
            pred_image_or_video_sliced = pred_image_or_video
            if num_generated_frames != min_num_frames:
                gradient_mask = torch.ones_like(pred_image_or_video_sliced, dtype=torch.bool)
                if self.independent_first_frame:
                    gradient_mask[:, :1] = False
                else:
                    gradient_mask[:, :self.num_frame_per_block] = False
            else:
                gradient_mask = None

        pred_image_or_video_sliced = pred_image_or_video_sliced.to(self.dtype)
        return pred_image_or_video_sliced, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        slice_last_frames: int = 21,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        generator_is_causal = getattr(self.args, "generator_is_causal", True)
        if not generator_is_causal:
            return self._bidirectional_backward_simulation(
                noise=noise,
                slice_last_frames=slice_last_frames,
                **conditional_dict,
            )

        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict, slice_last_frames=slice_last_frames
        )

    def _bidirectional_backward_simulation(
        self,
        noise: torch.Tensor,
        slice_last_frames: int = 21,
        **conditional_dict: dict
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Backward simulation for bidirectional (non-causal) generator.
        All frames are processed at once at each denoising step — no KV cache,
        no block-by-block processing.
        """
        from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        batch_size, num_frames = noise.shape[:2]

        # Resolve single-segment prompt for bidirectional model
        prompt_embeds = conditional_dict["prompt_embeds"]
        num_segments = prompt_embeds.shape[0] // batch_size
        if num_segments > 1:
            prompt_embeds = prompt_embeds.reshape(
                batch_size, num_segments, *prompt_embeds.shape[1:])[:, 0]
        cond = {**conditional_dict, "prompt_embeds": prompt_embeds}

        # Setup UniPC scheduler
        sampling_steps = getattr(self.args, "sampling_steps", None) or len(self.denoising_step_list)
        shift = self.scheduler.shift
        num_train_timesteps = self.scheduler.num_train_timesteps
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(sampling_steps, device=noise.device, shift=shift)
        unipc_timesteps = sample_scheduler.timesteps
        num_denoising_steps = len(unipc_timesteps)

        # Pick a random exit step (synchronized across ranks)
        last_step_only = getattr(self.args, "last_step_only", False)
        if last_step_only:
            exit_step = num_denoising_steps - 1
        else:
            exit_step_t = torch.randint(0, num_denoising_steps, (1,), device=self.device)
            dist.broadcast(exit_step_t, src=0)
            exit_step = exit_step_t.item()

        # Multi-step denoising loop (full-sequence bidirectional forward each step)
        latents = noise
        for index, t in enumerate(unipc_timesteps):
            timestep = t * torch.ones(
                [batch_size, num_frames], device=noise.device, dtype=torch.float32)

            if index < exit_step:
                with torch.no_grad():
                    flow_pred, _ = self.generator(
                        noisy_image_or_video=latents,
                        conditional_dict=cond,
                        timestep=timestep,
                    )
                    latents = sample_scheduler.step(
                        flow_pred, t, latents, return_dict=False)[0]
            else:
                # Exit step: forward with gradient
                flow_pred, denoised_pred = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=cond,
                    timestep=timestep,
                )
                break

        # Compute denoised_timestep_from / to
        if exit_step == num_denoising_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - unipc_timesteps[exit_step].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - unipc_timesteps[exit_step + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - unipc_timesteps[exit_step].cuda()).abs(), dim=0).item()

        return denoised_pred, denoised_timestep_from, denoised_timestep_to

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        local_attn_size = section_get(
            self.args,
            "inference",
            "local_attn_size",
            getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1),
            aliases=("inference_local_attn_size",),
        )
        sink_size = section_get(
            self.args,
            "inference",
            "sink_size",
            getattr(self.args, "model_kwargs", {}).get("sink_size", 0),
            aliases=("inference_sink_size",),
        )
        multi_shot_sink = section_get(self.args, "inference", "multi_shot_sink", False)
        multi_shot_rope_offset = section_get(
            self.args,
            "inference",
            "multi_shot_rope_offset",
            0.0,
        )
        scene_cut_prefix = section_get(self.args, "inference", "scene_cut_prefix", "[SCENE_CUT]")
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # do not use self.num_training_frames, because it is changed by generator_loss and critic_loss
        num_training_frames = getattr(self.args, "num_training_frames")
        if local_attn_size == -1:
            kv_cache_size = num_training_frames
        else:
            kv_cache_size = min(local_attn_size + slice_last_frames, num_training_frames)
        frame_seq_length = math.prod(self.args.image_or_video_shape[-2:]) // 4
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            same_step_across_blocks=getattr(
                self.args, "same_step_across_blocks", getattr(self, "same_step_across_blocks", False)
            ),
            last_step_only=getattr(self.args, "last_step_only", False),
            num_max_frames=kv_cache_size,
            context_noise=getattr(self.args, "context_noise", 0),
            sampling_steps=getattr(self.args, "sampling_steps", None),
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            multi_shot_sink=multi_shot_sink,
            scene_cut_prefix=scene_cut_prefix,
            multi_shot_rope_offset=multi_shot_rope_offset,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
            model_name=getattr(self.args, "model_kwargs", {}).get("model_name", None),
            frame_seq_length=frame_seq_length,
        )
