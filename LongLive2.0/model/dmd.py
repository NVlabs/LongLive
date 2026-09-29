# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import time

from model.base import SelfForcingModel
import torch.distributed as dist
from utils.i2v_conditioning import (
    _get_i2v_context_frames,
    _i2v_loss_mask_like,
    _overwrite_i2v_context,
    _zero_i2v_context_timestep,
)


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 21)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            for diffusion_model in (self.generator, self.real_score, self.fake_score):
                if hasattr(diffusion_model.model, "num_frame_per_block"):
                    diffusion_model.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame and not getattr(args, "i2v", False):
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = getattr(args, "num_train_timestep", 1000)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    @staticmethod
    def _slice_block_cond_dict(cond_dict, batch_size, new_num_segments):
        """Slice a block-wise conditional dict to keep only the last `new_num_segments` segments."""
        pe = cond_dict["prompt_embeds"]
        orig_segs = pe.shape[0] // batch_size
        if orig_segs > new_num_segments:
            pe = pe.reshape(batch_size, orig_segs, *pe.shape[1:])[:, -new_num_segments:]
            return {**cond_dict, "prompt_embeds": pe.reshape(batch_size * new_num_segments, *pe.shape[2:])}
        return cond_dict

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        clean_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_x
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                clean_x=clean_x
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_x
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            clean_x=clean_x
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # NOTE: Changed the normalizer for causal teacher — per-block normalization
        if normalization:
            p_real = (estimated_clean_image_or_video - pred_real_image)

            B, F, C, H, W = p_real.shape
            if dist.get_rank() == 0:
                print(f"p_real: {p_real.shape}")
            if (
                self.independent_first_frame
                and not getattr(self.args, "i2v", False)
                and (F - 1) % self.num_frame_per_block == 0
            ):
                p_real_tail = p_real[:, 1:]
                p_real_blocks = p_real_tail.view(
                    B,
                    (F - 1) // self.num_frame_per_block,
                    self.num_frame_per_block,
                    C,
                    H,
                    W,
                )
                normalizer_tail = torch.abs(p_real_blocks).mean(dim=[2, 3, 4, 5], keepdim=True)
                normalizer = torch.ones_like(p_real)
                normalizer[:, 1:] = normalizer_tail.expand_as(p_real_blocks).reshape(
                    B, F - 1, C, H, W
                )
            else:
                p_real_blocks = p_real.view(B, F // self.num_frame_per_block, self.num_frame_per_block, C, H, W)
                normalizer = torch.abs(p_real_blocks).mean(dim=[2, 3, 4, 5], keepdim=True)
                normalizer = normalizer.expand_as(p_real_blocks).reshape(B, F, C, H, W)


            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        clean_x: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        context_frames = _get_i2v_context_frames(image_or_video, initial_latent)
        original_latent = _overwrite_i2v_context(
            image_or_video, initial_latent, context_frames
        )
        if clean_x is not None:
            clean_x = _overwrite_i2v_context(clean_x, initial_latent, context_frames)

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                # t2v keeps the original NVFP4 behavior (one shared timestep across
                # all frames); i2v uses per-block timesteps.
                uniform_timestep=not getattr(self.args, "i2v", False)
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)
            timestep = _zero_i2v_context_timestep(timestep, context_frames)

            noise = torch.randn_like(image_or_video)
            if context_frames > 0:
                noise[:, :context_frames] = 0
            noisy_latent = self.scheduler.add_noise(
                original_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))
            noisy_latent = _overwrite_i2v_context(
                noisy_latent, initial_latent, context_frames
            )

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_x=clean_x
            )

        context_mask = _i2v_loss_mask_like(original_latent, context_frames)
        if context_mask is not None:
            gradient_mask = context_mask if gradient_mask is None else gradient_mask & context_mask

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        num_gen_frames = image_or_video_shape[1]
        sampled_noise = torch.randn(
            [image_or_video_shape[0], num_gen_frames, *image_or_video_shape[2:]],
            device=self.device, dtype=self.dtype)
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames,
            noise=sampled_noise,
            clean_latent=clean_latent
        )
        gen_time = time.time() - _t_gen_start
        # Step 2: Compute the DMD loss
        _t_loss_start = time.time()
        if getattr(self.args, "teacher_forcing", False):
            if getattr(self.args, "backward_simulation", True):
                score_clean_x = pred_image.detach()
            else:
                score_clean_x = clean_latent
        else:
            score_clean_x = None
        _bs = pred_image.shape[0]
        _new_segs = pred_image.shape[1] // self.num_frame_per_block
        if not getattr(self.args, "generator_is_causal", True):
            _new_segs = 1
        conditional_dict = self._slice_block_cond_dict(conditional_dict, _bs, _new_segs)
        unconditional_dict = self._slice_block_cond_dict(unconditional_dict, _bs, _new_segs)
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            clean_x=score_clean_x,
            initial_latent=initial_latent if pred_image.shape[1] == image_or_video_shape[1] else None,
        )
        try:
            loss_val = dmd_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start

        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time
        })

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # Step 1: Run generator on backward simulated noisy input
        _t_gen_start = time.time()
        with torch.no_grad():
            num_gen_frames = image_or_video_shape[1]
            sampled_noise = torch.randn(
                [image_or_video_shape[0], num_gen_frames, *image_or_video_shape[2:]],
                device=self.device, dtype=self.dtype)
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
                noise=sampled_noise,
                clean_latent=clean_latent
            )
        gen_time = time.time() - _t_gen_start
        score_initial_latent = (
            initial_latent
            if initial_latent is not None and generated_image.shape[1] == image_or_video_shape[1]
            else None
        )
        context_frames = _get_i2v_context_frames(generated_image, score_initial_latent)
        batch_size, num_frame = generated_image.shape[:2]

        _new_segs = num_frame // self.num_frame_per_block
        if not getattr(self.args, "generator_is_causal", True):
            _new_segs = 1
        conditional_dict = self._slice_block_cond_dict(conditional_dict, batch_size, _new_segs)

        if getattr(self.args, "teacher_forcing", False):
            if getattr(self.args, "backward_simulation", True):
                score_clean_x = generated_image
            else:
                score_clean_x = clean_latent
        else:
            score_clean_x = None
        if score_clean_x is not None:
            score_clean_x = _overwrite_i2v_context(
                score_clean_x, score_initial_latent, context_frames
            )
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            # t2v keeps the original NVFP4 behavior (one shared timestep across
            # all frames); i2v uses per-block timesteps.
            uniform_timestep=not getattr(self.args, "i2v", False)
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)
        critic_timestep = _zero_i2v_context_timestep(critic_timestep, context_frames)

        critic_noise = torch.randn_like(generated_image)
        if context_frames > 0:
            critic_noise[:, :context_frames] = 0
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        noisy_generated_image = _overwrite_i2v_context(
            noisy_generated_image, score_initial_latent, context_frames
        )

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            clean_x=score_clean_x
        )

        # Step 3: Compute the denoising loss for the fake critic
        if getattr(self.args, "denoising_loss_type", "flow") == "flow":
            from utils.wan_5b_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            gradient_mask=(
                _i2v_loss_mask_like(generated_image, context_frames).flatten(0, 1)
                if context_frames > 0 else None
            ),
            flow_pred=flow_pred,
        )

        try:
            loss_val = denoising_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start
        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time
        }

        return denoising_loss, critic_log_dict
