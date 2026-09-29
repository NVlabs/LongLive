import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from wan_5b.modules.vae2_2 import (
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    count_conv3d,
    patchify,
    unpatchify,
)


def _extract_checkpoint_state_dict(raw):
    state = raw
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "gen_model" in state:
        state = state["gen_model"]
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like state_dict.")
    return state


def _map_lightvae_key_to_wanvae(key):
    def _map_resnet_tail(tail):
        if tail.startswith("norm1."):
            return "residual.0." + tail[len("norm1."):]
        if tail.startswith("conv1."):
            return "residual.2." + tail[len("conv1."):]
        if tail.startswith("norm2."):
            return "residual.3." + tail[len("norm2."):]
        if tail.startswith("conv2."):
            return "residual.6." + tail[len("conv2."):]
        if tail.startswith("conv_shortcut."):
            return "shortcut." + tail[len("conv_shortcut."):]
        return tail

    if key.startswith("dynamic_feature_projection_heads."):
        return None

    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "conv1.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "conv2.", 1)

    if key.startswith("encoder.conv_in."):
        return key.replace("encoder.conv_in.", "encoder.conv1.", 1)
    if key.startswith("encoder.mid_block.resnets.0."):
        tail = key[len("encoder.mid_block.resnets.0."):]
        return "encoder.middle.0." + _map_resnet_tail(tail)
    if key.startswith("encoder.mid_block.attentions.0."):
        return key.replace("encoder.mid_block.attentions.0.", "encoder.middle.1.", 1)
    if key.startswith("encoder.mid_block.resnets.1."):
        tail = key[len("encoder.mid_block.resnets.1."):]
        return "encoder.middle.2." + _map_resnet_tail(tail)
    if key.startswith("encoder.norm_out."):
        return key.replace("encoder.norm_out.", "encoder.head.0.", 1)
    if key.startswith("encoder.conv_out."):
        return key.replace("encoder.conv_out.", "encoder.head.2.", 1)

    if key.startswith("encoder.down_blocks."):
        parts = key.split(".")
        if len(parts) >= 6 and parts[3] == "resnets":
            tail = ".".join(parts[5:])
            return f"encoder.downsamples.{parts[2]}.downsamples.{parts[4]}." + _map_resnet_tail(tail)
        if len(parts) >= 7 and parts[3] == "downsampler" and parts[4] == "resample":
            return f"encoder.downsamples.{parts[2]}.downsamples.2.resample.{parts[5]}." + ".".join(parts[6:])
        if len(parts) >= 6 and parts[3] == "downsampler" and parts[4] == "time_conv":
            return f"encoder.downsamples.{parts[2]}.downsamples.2.time_conv." + ".".join(parts[5:])

    if key.startswith("decoder.conv_in."):
        return key.replace("decoder.conv_in.", "decoder.conv1.", 1)
    if key.startswith("decoder.mid_block.resnets.0."):
        tail = key[len("decoder.mid_block.resnets.0."):]
        return "decoder.middle.0." + _map_resnet_tail(tail)
    if key.startswith("decoder.mid_block.attentions.0."):
        return key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.", 1)
    if key.startswith("decoder.mid_block.resnets.1."):
        tail = key[len("decoder.mid_block.resnets.1."):]
        return "decoder.middle.2." + _map_resnet_tail(tail)
    if key.startswith("decoder.norm_out."):
        return key.replace("decoder.norm_out.", "decoder.head.0.", 1)
    if key.startswith("decoder.conv_out."):
        return key.replace("decoder.conv_out.", "decoder.head.2.", 1)

    if key.startswith("decoder.up_blocks."):
        parts = key.split(".")
        if len(parts) >= 6 and parts[3] == "resnets":
            tail = ".".join(parts[5:])
            return f"decoder.upsamples.{parts[2]}.upsamples.{parts[4]}." + _map_resnet_tail(tail)
        if len(parts) >= 7 and parts[3] == "upsampler" and parts[4] == "resample":
            return f"decoder.upsamples.{parts[2]}.upsamples.3.resample.{parts[5]}." + ".".join(parts[6:])
        if len(parts) >= 6 and parts[3] == "upsampler" and parts[4] == "time_conv":
            return f"decoder.upsamples.{parts[2]}.upsamples.3.time_conv." + ".".join(parts[5:])

    return key


def _normalize_vae_state_dict(raw_state):
    state = _extract_checkpoint_state_dict(raw_state)
    normalized = {}
    for key, value in state.items():
        mapped_key = _map_lightvae_key_to_wanvae(key)
        if mapped_key is None:
            continue
        normalized[mapped_key] = value
    return normalized


def infer_lightvae_pruning_rate_from_ckpt(vae_path, full_decoder_conv1_out=1024):
    if vae_path is None or not os.path.exists(vae_path):
        return None
    try:
        raw_state = torch.load(vae_path, map_location="cpu")
        state = _extract_checkpoint_state_dict(raw_state)
    except Exception as exc:
        logging.warning("Failed to load checkpoint for pruning-rate inference: %s", exc)
        return None

    weight = None
    if isinstance(state, dict):
        if "decoder.conv_in.weight" in state:
            weight = state["decoder.conv_in.weight"]
        elif "decoder.conv1.weight" in state:
            weight = state["decoder.conv1.weight"]

    if weight is None:
        try:
            normalized_state = _normalize_vae_state_dict(state)
            weight = normalized_state.get("decoder.conv1.weight", None)
        except Exception:
            weight = None

    if weight is None or not hasattr(weight, "shape") or len(weight.shape) < 1:
        return None

    student_out = int(weight.shape[0])
    if full_decoder_conv1_out <= 0:
        return None

    pruning_rate = 1.0 - (float(student_out) / float(full_decoder_conv1_out))
    pruning_rate = max(0.0, min(0.99, pruning_rate))
    return round(pruning_rate, 6)


def convert_to_channels_last_3d(module):
    for child in module.children():
        if isinstance(child, nn.Conv3d):
            child.weight.data = child.weight.data.to(memory_format=torch.channels_last_3d)
        else:
            convert_to_channels_last_3d(child)


class PrunableWanVAE(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        dim = max(1, int(round(dim * (1.0 - pruning_rate))))
        dec_dim = max(1, int(round(dec_dim * (1.0 - pruning_rate))))

        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        total_steps = 1 + (x.shape[2] - 1) // 4
        for step in range(total_steps):
            self._enc_conv_idx = [0]
            if step == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_chunk = self.encoder(
                    x[:, :, 1 + 4 * (step - 1):1 + 4 * step, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_chunk], 2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        total_steps = z.shape[2]
        x = self.conv2(z)
        for step in range(total_steps):
            self._conv_idx = [0]
            if step == 0:
                out = self.decoder(
                    x[:, :, step:step + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_chunk = self.decoder(
                    x[:, :, step:step + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_chunk], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        total_steps = z.shape[2]
        x = self.conv2(z)
        is_first = self._feat_map[0] is None
        for step in range(total_steps):
            self._conv_idx = [0]
            if step == 0 and is_first:
                out = self.decoder(
                    x[:, :, step:step + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            elif step == 0:
                out = self.decoder(
                    x[:, :, step:step + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_chunk = self.decoder(
                    x[:, :, step:step + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_chunk], 2)
        return unpatchify(out, patch_size=2)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _load_lightvae_model(pretrained_path=None, z_dim=48, dim=160, device="cpu", **kwargs):
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    with torch.device("meta"):
        model = PrunableWanVAE(**cfg)

    if pretrained_path is None or not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {pretrained_path}")

    logging.info("loading %s", pretrained_path)
    raw_state = torch.load(pretrained_path, map_location="cpu")
    state_dict = _normalize_vae_state_dict(raw_state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    logging.info(
        "LightVAE checkpoint loaded with strict=False (missing=%d, unexpected=%d)",
        len(missing),
        len(unexpected),
    )

    convert_to_channels_last_3d(model)
    return model


class LightVAE5BWrapper(nn.Module):
    def __init__(
        self,
        vae_path: str,
        pruning_rate: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pruning_rate is None:
            pruning_rate = infer_lightvae_pruning_rate_from_ckpt(vae_path)
            if pruning_rate is None:
                pruning_rate = 0.75
                logging.warning(
                    "Unable to infer LightVAE pruning rate from checkpoint; fallback to 0.75."
                )

        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
            0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.vae_path = os.path.abspath(vae_path)
        self.pruning_rate = pruning_rate
        self.device = torch.device(device)
        self.dtype = dtype

        self.model = _load_lightvae_model(
            pretrained_path=self.vae_path,
            pruning_rate=self.pruning_rate,
        ).eval().requires_grad_(False)
        self.to(device=self.device, dtype=self.dtype)

    def to(self, device=None, dtype=None):
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype
        self.model.to(device=device, dtype=dtype)
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.std = self.std.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        scale = [self.mean, 1.0 / self.std]
        decode_fn = self.model.cached_decode if use_cache else self.model.decode

        output = []
        for item in zs:
            output.append(
                decode_fn(item.unsqueeze(0).to(device=self.device, dtype=self.dtype), scale)
                .float()
                .clamp_(-1, 1)
                .squeeze(0)
            )
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)
