from typing import List, Optional
import os
import torch

from utils.scheduler import FlowMatchScheduler
from utils.config import wan_default_config
from utils.wan_model_utils import (
    resolve_wan_model_dir,
)

from wan_5b.modules.tokenizers import HuggingfaceTokenizer
from wan_5b.modules.model import WanModel
from wan_5b.modules.vae2_1 import _video_vae as _video_vae_2_1
from wan_5b.modules.vae2_2 import _video_vae as _video_vae_2_2
from wan_5b.modules.t5 import umt5_xxl


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "Wan2.2-TI2V-5B",
        model_dir: Optional[str] = None,
        t5_checkpoint: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        base_dir = resolve_wan_model_dir(model_name, model_dir)
        t5_checkpoint = t5_checkpoint or os.path.join(
            base_dir, "models_t5_umt5-xxl-enc-bf16.pth"
        )
        tokenizer_dir = tokenizer_dir or os.path.join(base_dir, "google", "umt5-xxl")

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(t5_checkpoint, map_location='cpu', weights_only=False)
        )

        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_dir, seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "Wan2.2-TI2V-5B",
        model_dir: Optional[str] = None,
        vae_checkpoint: Optional[str] = None,
        vae_type: Optional[str] = None,
    ):
        super().__init__()
        base_dir = resolve_wan_model_dir(model_name, model_dir)
        if vae_type is None:
            vae_type = wan_default_config.get(model_name, {}).get("vae_type", "wan2.2")
        vae_type = str(vae_type).lower().replace("_", ".")

        if vae_type in ("wan2.1", "wan21", "2.1"):
            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075,
                -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632,
                -0.1922, -0.9497, 0.2503, -0.2921,
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558,
                1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579,
                1.6382, 1.1253, 2.8251, 1.9160,
            ]
            vae_checkpoint = vae_checkpoint or os.path.join(base_dir, "Wan2.1_VAE.pth")
            video_vae = _video_vae_2_1(pretrained_path=vae_checkpoint, z_dim=16)
        elif vae_type in ("wan", "wan2.2", "wan22", "2.2", ""):
            mean = [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ]
            std = [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ]
            vae_checkpoint = vae_checkpoint or os.path.join(base_dir, "Wan2.2_VAE.pth")
            video_vae = _video_vae_2_2(pretrained_path=vae_checkpoint)
        else:
            raise ValueError(
                f"Unknown Wan VAE type '{vae_type}'. Expected wan2.1 or wan2.2."
            )

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        self.model = video_vae.eval().requires_grad_(False)

    def _decode_device_dtype(self, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return fallback.device, fallback.dtype
        return first_param.device, first_param.dtype

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype

        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        device, dtype = self._decode_device_dtype(latent)
        zs = latent.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B",
            model_dir=None,
            timestep_shift=8.0,
            is_causal=False,
            num_frame_per_block=1,
    ):
        super().__init__()

        self.model_name = model_name
        self.model_dir = resolve_wan_model_dir(model_name, model_dir)
        self.is_causal = bool(is_causal)
        if is_causal:
            raise ValueError("LongLive-Plug uses non-AR Wan backbones")
        self.model = WanModel.from_pretrained(self.model_dir)
        self.model.eval()
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)


    def _infer_seq_len(self, noisy_image_or_video: torch.Tensor) -> int:
        patch_size = getattr(self.model, "patch_size", (1, 2, 2))
        num_frames, height, width = noisy_image_or_video.shape[1], noisy_image_or_video.shape[3], noisy_image_or_video.shape[4]
        return (
            (num_frames // patch_size[0])
            * (height // patch_size[1])
            * (width // patch_size[2])
        )

    def enable_gradient_checkpointing(self) -> None:
        try:
            self.model.enable_gradient_checkpointing()
        except TypeError as exc:
            # Older WanModel implementations define
            # _set_gradient_checkpointing(module, value=False), while recent
            # diffusers calls it with enable=... and gradient_checkpointing_func=....
            if "_set_gradient_checkpointing" not in str(exc) or "enable" not in str(exc):
                raise
            if hasattr(self.model, "_set_gradient_checkpointing"):
                self.model._set_gradient_checkpointing(self.model, True)
            else:
                self.model.gradient_checkpointing = True


    def _call_model(self, *args, **kwargs):
        result = self.model(*args, **kwargs)
        return torch.stack(result) if isinstance(result, list) else result


    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(self, noisy_image_or_video, conditional_dict, timestep, clean_x=None, **kwargs):
        if clean_x is not None or any(v is not None for v in kwargs.values()):
            raise ValueError("Only full-sequence text-to-video diffusion is supported")
        flow_pred = self._call_model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep[:, 0], context=conditional_dict["prompt_embeds"],
            seq_len=self._infer_seq_len(noisy_image_or_video),
        ).permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def get_scheduler(self) -> FlowMatchScheduler:
        return self.scheduler
