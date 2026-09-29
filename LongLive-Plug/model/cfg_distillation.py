# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Classifier-free-guidance distillation without step distillation.

The frozen teacher is evaluated twice at the same noisy latent, once with the
text condition and once with the configured negative prompt.  The trainable
student is evaluated once and regresses the teacher's guided flow field:

    v_cfg = v_uncond + scale * (v_cond - v_uncond)

Only the guidance field is distilled.  This module does not unroll a short
student trajectory, use a consistency target, or alter the inference schedule.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model.base import BaseModel
from utils.config import CFG_ONLY_SUPPORTED_MODELS
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder


def combine_cfg_predictions(
    conditional_prediction: torch.Tensor,
    unconditional_prediction: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine teacher predictions using Wan's standard CFG convention.

    ``guidance_scale=1`` is exactly the conditional prediction.  This differs
    from code that names the *extra* extrapolation coefficient ``w``; in that
    convention ``w = guidance_scale - 1``.
    """

    if conditional_prediction.shape != unconditional_prediction.shape:
        raise ValueError(
            "Conditional and unconditional predictions must have identical "
            f"shapes, got {tuple(conditional_prediction.shape)} and "
            f"{tuple(unconditional_prediction.shape)}."
        )
    if guidance_scale < 1.0:
        raise ValueError(
            f"teacher_guidance_scale must be >= 1, got {guidance_scale}."
        )
    conditional_fp32 = conditional_prediction.float()
    unconditional_fp32 = unconditional_prediction.float()
    return unconditional_fp32 + guidance_scale * (
        conditional_fp32 - unconditional_fp32
    )


class CFGGuidanceDistillation(BaseModel):
    """Distill a guided Wan teacher into one conditional LoRA forward."""

    def _initialize_models(self, args, device):
        base_model_kwargs = getattr(args, "model_kwargs", {}) or {}
        model_name = base_model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
        if model_name not in CFG_ONLY_SUPPORTED_MODELS:
            raise ValueError(
                "CFG-only distillation supports "
                f"{sorted(CFG_ONLY_SUPPORTED_MODELS)}, got {model_name}."
            )

        self.generator = WanDiffusionWrapper(
            **base_model_kwargs,
            is_causal=False,
        )
        self.generator.model.requires_grad_(True)

        # A trajectory-cache run has already materialized the frozen teacher's
        # CFG targets. Do not allocate another 5B backbone (or a VAE) merely to
        # leave it unused throughout LoRA optimization.
        self.real_score = None

        # There is deliberately no learned fake score / DMD critic in this
        # objective.  Keeping this attribute explicit makes accidental use fail
        # loudly in trainer code instead of allocating a third 5B backbone.
        self.fake_score = None

        model_dir = base_model_kwargs.get("model_dir", None)
        self.text_encoder = WanTextEncoder(
            model_name=model_name,
            model_dir=model_dir,
        )
        self.text_encoder.requires_grad_(False)
        self.vae = None

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 1))
        self.teacher_guidance_scale = float(
            getattr(args, "teacher_guidance_scale", 5.0)
        )
        if self.teacher_guidance_scale <= 1.0:
            raise ValueError(
                "CFG-only distillation requires teacher_guidance_scale > 1; "
                f"got {self.teacher_guidance_scale}."
            )

        self.loss_weighting = str(
            getattr(args, "cfg_distill_loss_weighting", "uniform")
        )
        if self.loss_weighting not in {"uniform", "flow", "relative_guidance"}:
            raise ValueError(
                "cfg_distill_loss_weighting must be 'uniform', 'flow', or "
                "'relative_guidance', got "
                f"{self.loss_weighting!r}."
            )
        self.cfg_state_source = str(getattr(args, "cfg_state_source", "data"))
        self.relative_guidance_epsilon = float(
            getattr(args, "cfg_relative_guidance_epsilon", 1.0e-8)
        )
        self.relative_guidance_max_weight = float(
            getattr(args, "cfg_relative_guidance_max_weight", 16.0)
        )
        if (
            not math.isfinite(self.relative_guidance_epsilon)
            or self.relative_guidance_epsilon <= 0
        ):
            raise ValueError("cfg_relative_guidance_epsilon must be positive.")
        if (
            not math.isfinite(self.relative_guidance_max_weight)
            or self.relative_guidance_max_weight <= 0
        ):
            raise ValueError(
                "cfg_relative_guidance_max_weight must be positive."
            )

        if self.num_frame_per_block > 1:
            for diffusion_model in (self.generator, self.real_score):
                if (
                    diffusion_model is not None
                    and hasattr(diffusion_model.model, "num_frame_per_block")
                ):
                    diffusion_model.model.num_frame_per_block = (
                        self.num_frame_per_block
                    )

        if getattr(args, "gradient_checkpointing", False):
            # The teacher always runs under no_grad and does not benefit from
            # activation checkpointing.
            self.generator.enable_gradient_checkpointing()

    @staticmethod
    def _single_segment_cond_dict(cond_dict: dict, batch_size: int) -> dict:
        """Collapse repeated block prompts for a bidirectional full-clip pass."""

        prompt_embeds = cond_dict.get("prompt_embeds")
        if prompt_embeds is None or prompt_embeds.shape[0] == batch_size:
            return cond_dict
        if prompt_embeds.shape[0] % batch_size != 0:
            raise ValueError(
                "prompt_embeds leading dimension must be divisible by batch size."
            )
        num_segments = prompt_embeds.shape[0] // batch_size
        prompt_embeds = prompt_embeds.reshape(
            batch_size, num_segments, *prompt_embeds.shape[1:]
        )
        selected = prompt_embeds[:, -1:]
        if not torch.allclose(
            prompt_embeds,
            selected.expand_as(prompt_embeds),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError(
                "Bidirectional CFG distillation received multiple different "
                "prompt segments. Use one prompt per training clip."
            )
        return {
            **cond_dict,
            "prompt_embeds": selected[:, 0].reshape(
                batch_size, *prompt_embeds.shape[2:]
            ),
        }

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: Optional[dict],
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        cached_noisy_latent: Optional[torch.Tensor] = None,
        cached_guided_target: Optional[torch.Tensor] = None,
        cached_timestep: Optional[torch.Tensor] = None,
        cached_guidance_energy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Regress the frozen teacher's guided flow at a data-trajectory point."""

        if initial_latent is not None or getattr(self.args, "i2v", False):
            raise NotImplementedError(
                "The initial CFG-only release supports text-to-video training only."
            )

        cached_values = (
            cached_noisy_latent,
            cached_guided_target,
            cached_timestep,
            cached_guidance_energy,
        )
        if any(value is None for value in cached_values):
            raise ValueError(
                "teacher_trajectory_cache requires cached noisy latent, "
                "guided target, timestep, and guidance energy."
            )
        if clean_latent is not None:
            raise ValueError(
                "teacher_trajectory_cache must not also receive clean_latent."
            )
        noisy_latent = cached_noisy_latent.to(
            device=self.device, dtype=self.dtype
        )
        guided_target = cached_guided_target.to(
            device=self.device, dtype=torch.float32
        )
        batch_size, num_frames = noisy_latent.shape[:2]
        expected_shape = tuple(image_or_video_shape)
        if tuple(noisy_latent.shape) != expected_shape:
            raise ValueError(
                f"Cached noisy latent has shape {tuple(noisy_latent.shape)}; "
                f"expected {expected_shape}."
            )
        if tuple(guided_target.shape) != expected_shape:
            raise ValueError(
                f"Cached guided target has shape {tuple(guided_target.shape)}; "
                f"expected {expected_shape}."
            )
        timestep = cached_timestep.to(
            device=self.device, dtype=torch.float32
        )
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(batch_size, num_frames)
        elif tuple(timestep.shape) == (batch_size, 1):
            timestep = timestep.expand(batch_size, num_frames)
        elif tuple(timestep.shape) != (batch_size, num_frames):
            raise ValueError(
                "Cached timestep must have shape [B], [B,1], or [B,F], "
                f"got {tuple(timestep.shape)}."
            )
        guidance_energy = cached_guidance_energy.to(
            device=self.device, dtype=torch.float32
        )
        if guidance_energy.ndim == 1 and batch_size == 1:
            guidance_energy = guidance_energy.unsqueeze(0)
        if tuple(guidance_energy.shape) != (batch_size, num_frames):
            raise ValueError(
                "Cached guidance energy must have shape [B,F], got "
                f"{tuple(guidance_energy.shape)}."
            )
        tensors_to_check = {
            "cached noisy latent": noisy_latent,
            "cached guided target": guided_target,
            "cached timestep": timestep,
            "cached guidance energy": guidance_energy,
        }
        for label, value in tensors_to_check.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{label} contains non-finite values.")
        if (guidance_energy < 0).any():
            raise ValueError("Cached guidance energy must be non-negative.")
        teacher_guidance_energy = guidance_energy

        conditional_dict = self._single_segment_cond_dict(
            conditional_dict, batch_size
        )

        student_flow, _ = self.generator(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )

        per_frame_loss = F.mse_loss(
            student_flow.float(), guided_target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))
        median_floor = None
        if self.loss_weighting == "flow":
            weights = self.scheduler.training_weight(timestep).unflatten(
                0, (batch_size, num_frames)
            )
            per_frame_loss = per_frame_loss * weights
        elif self.loss_weighting == "relative_guidance":
            # A per-sample median floor prevents nearly-zero low-noise
            # guidance vectors from dominating their neighboring frames. The
            # explicit cap also bounds a whole low-energy sample's gradient.
            median_floor = teacher_guidance_energy.median(dim=1).values.clamp(
                min=self.relative_guidance_epsilon
            )
            denominator = torch.maximum(
                teacher_guidance_energy,
                median_floor[:, None],
            )
            relative_weights = denominator.reciprocal().clamp(
                max=self.relative_guidance_max_weight
            )
            per_frame_loss = per_frame_loss * relative_weights

        if loss_mask is not None:
            if tuple(loss_mask.shape) != (batch_size, num_frames):
                raise ValueError(
                    "loss_mask must have shape [batch, frames], got "
                    f"{tuple(loss_mask.shape)}."
                )
            mask = loss_mask.to(per_frame_loss)
            loss = (per_frame_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = per_frame_loss.mean()

        with torch.no_grad():
            student_error = student_flow.float() - guided_target.float()
            log_dict = {
                "cfg_distill_loss": loss.detach(),
                "teacher_guidance_delta_norm": teacher_guidance_energy.mean().sqrt()
                / max(self.teacher_guidance_scale - 1.0, 1.0),
                "student_teacher_rmse": student_error.square().mean().sqrt(),
                "timestep_mean": timestep.float().mean(),
            }
            if median_floor is not None:
                log_dict["relative_guidance_median_floor"] = median_floor.mean()
                log_dict["relative_guidance_weight_max"] = (
                    relative_weights.max()
                )
        return loss, log_dict
