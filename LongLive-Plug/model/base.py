# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch import nn
import torch.distributed as dist
import torch

from utils.config import wan_default_config
from utils.loss import get_denoising_loss
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def build_role_model_kwargs(base_kwargs, role_name, explicit_kwargs=None):
    """Resolve teacher/critic weights for the same supported Wan backbone."""
    resolved = dict(base_kwargs if explicit_kwargs is None else explicit_kwargs)
    resolved.setdefault("model_name", role_name)
    if resolved["model_name"] != role_name or role_name != base_kwargs["model_name"]:
        raise ValueError("Student, teacher and critic must use the same base model")
    return resolved


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        print("args.model_kwargs.model_name", args.model_kwargs.model_name)
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    def _initialize_models(self, args, device):
        base_model_kwargs = getattr(args, "model_kwargs", {}) or {}
        model_name = base_model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
        self.real_model_name = getattr(args, "real_name", model_name)
        self.fake_model_name = getattr(args, "fake_name", model_name)
        if model_name not in wan_default_config:
            raise ValueError(f"Unsupported Wan model for LongLive DMD: {model_name}")
        unsupported_roles = [
            name
            for name in (self.real_model_name, self.fake_model_name)
            if name not in wan_default_config
        ]
        if unsupported_roles:
            raise ValueError(
                "Unsupported Wan role model(s) for LongLive DMD: "
                + ", ".join(unsupported_roles)
            )
        # Generator
        self.generator = WanDiffusionWrapper(**base_model_kwargs, is_causal=False)
        self.generator.model.requires_grad_(True)

        # Real Score
        real_kwargs = build_role_model_kwargs(
            base_model_kwargs,
            self.real_model_name,
            getattr(args, "real_model_kwargs", None),
        )
        self.real_score = WanDiffusionWrapper(**real_kwargs, is_causal=False)
        self.real_score.model.requires_grad_(False)

        # Fake Score
        fake_kwargs = build_role_model_kwargs(
            base_model_kwargs,
            self.fake_model_name,
            getattr(args, "fake_model_kwargs", None),
        )
        self.fake_score = WanDiffusionWrapper(**fake_kwargs, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        # Text Encoder & VAE
        model_dir = base_model_kwargs.get("model_dir", None)
        self.text_encoder = WanTextEncoder(model_name=model_name, model_dir=model_dir)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(
            model_name=model_name,
            model_dir=model_dir,
            vae_type=getattr(args, "vae_type", None),
        )
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(self, min_timestep, max_timestep, batch_size, num_frame):
        """Sample one noise level per video, shared by every frame."""
        return torch.randint(
            min_timestep, max_timestep, [batch_size, 1],
            device=self.device, dtype=torch.long,
        ).repeat(1, num_frame)


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(getattr(args, "denoising_loss_type", "flow"))()

    @staticmethod
    def _single_segment_cond_dict(cond_dict: dict, batch_size: int) -> dict:
        """Collapse block-wise prompt embeddings for a bidirectional full-clip pass."""
        prompt_embeds = cond_dict.get("prompt_embeds", None)
        if prompt_embeds is None or prompt_embeds.shape[0] == batch_size:
            return cond_dict
        num_segments = prompt_embeds.shape[0] // batch_size
        if num_segments <= 1 or prompt_embeds.shape[0] % batch_size != 0:
            return cond_dict
        prompt_embeds = prompt_embeds.reshape(batch_size, num_segments, *prompt_embeds.shape[1:])
        selected_prompt = prompt_embeds[:, -1:]
        if not torch.allclose(prompt_embeds, selected_prompt.expand_as(prompt_embeds), rtol=1e-4, atol=1e-4):
            raise ValueError(
                "Bidirectional full-clip rollout received multiple different prompt segments. "
                "Use one prompt per full-sequence video."
            )
        return {**cond_dict, "prompt_embeds": selected_prompt[:, 0].reshape(batch_size, *prompt_embeds.shape[2:])}

    def _run_generator(self, image_or_video_shape, conditional_dict, noise=None):
        """Run a full-sequence UniPC rollout with one differentiable exit step."""
        noise_shape = image_or_video_shape.copy()
        min_num_frames = self.min_num_training_frames
        max_num_frames = self.num_training_frames
        assert min_num_frames == max_num_frames == noise_shape[1]
        assert max_num_frames % self.num_frame_per_block == 0
        num_blocks = max_num_frames // self.num_frame_per_block
        # Preserve the recipe's RNG consumption, including its fixed-length draw.
        num_generated_blocks = torch.randint(num_blocks, num_blocks + 1, (1,), device=self.device)
        if dist.is_initialized():
            dist.broadcast(num_generated_blocks, src=0)
        num_generated_frames = num_generated_blocks.item() * self.num_frame_per_block
        noise_shape[1] = num_generated_frames
        if noise is not None:
            noise = noise[:, :num_generated_frames]
        else:
            noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        prediction, timestep_from, timestep_to = self._consistency_backward_simulation_noncausal(
            noise=noise, **conditional_dict,
        )
        return prediction.to(self.dtype), None, timestep_from, timestep_to


    def _consistency_backward_simulation_noncausal(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """Full-sequence backward simulation for a bidirectional Wan student.

        Each denoise step sees the whole clip with bidirectional attention.
        """
        from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        batch_size, num_frames = noise.shape[:2]
        conditional_dict = self._single_segment_cond_dict(conditional_dict, batch_size)

        sampling_steps = self.args.sampling_steps
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=noise.device, shift=self.scheduler.shift)
        denoising_timesteps = sample_scheduler.timesteps
        num_denoising_steps = len(denoising_timesteps)

        if getattr(self.args, "last_step_only", False):
            exit_index = num_denoising_steps - 1
        else:
            exit_index_tensor = torch.randint(
                0,
                num_denoising_steps,
                (1,),
                device=noise.device,
                dtype=torch.long,
            )
            if dist.is_initialized():
                dist.broadcast(exit_index_tensor, src=0)
            exit_index = int(exit_index_tensor.item())

        latents = noise

        denoised_pred = None
        for index, timestep_value in enumerate(denoising_timesteps):
            timestep = timestep_value * torch.ones(
                [batch_size, num_frames],
                device=noise.device,
                dtype=torch.float32,
            )

            if index != exit_index:
                with torch.no_grad():
                    flow_pred, _ = self.generator(
                        noisy_image_or_video=latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                    )
                    latents = sample_scheduler.step(
                        flow_pred,
                        timestep_value,
                        latents,
                        return_dict=False,
                    )[0]
            else:
                # Preserve the caller's grad mode.  In particular, critic_loss
                # deliberately runs the generator rollout under no_grad; forcing
                # gradients back on here connects the critic loss to generator
                # LoRA parameters and contaminates the pending generator update.
                _, denoised_pred = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                break

        if denoised_pred is None:
            raise RuntimeError("Non-causal backward simulation did not produce a denoised prediction.")

        scheduler_timesteps = self.scheduler.timesteps.to(noise.device)
        if exit_index == num_denoising_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps - denoising_timesteps[exit_index]).abs(),
                dim=0,
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (scheduler_timesteps - denoising_timesteps[exit_index + 1]).abs(),
                dim=0,
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps - denoising_timesteps[exit_index]).abs(),
                dim=0,
            ).item()

        return denoised_pred, denoised_timestep_from, denoised_timestep_to
