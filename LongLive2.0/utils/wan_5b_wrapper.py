import types
from typing import List, Optional
import os
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler

from wan_5b.modules.tokenizers import HuggingfaceTokenizer
from wan_5b.modules.model import WanModel
from wan_5b.modules.vae2_2 import _video_vae
from wan_5b.modules.t5 import umt5_xxl
from wan_5b.modules.causal_model import CausalWanModel


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        
        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        ).eval().requires_grad_(False)

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
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
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

    def decode_to_pixel_chunk(self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 1) -> torch.Tensor:
        """
        Decode latent frames to pixel space.
        
        Args:
            latent: Latent tensor with shape [batch_size, num_frames, num_channels, height, width]
            use_cache: Whether to use cached decoding (for streaming)
            chunk_size: Number of latent frames to decode at once (default 240 to avoid OOM)
        
        Returns:
            Decoded video tensor with shape [batch_size, num_frames, num_channels, height, width]
        """
        # latent shape: [batch_size, num_frames, num_channels, height, width]
        # zs shape after permute: [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:
                # Decode short clips in one pass.
                if use_cache:
                    # Start this segment from a clean cache.
                    self.model.clear_cache()
                decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                decoded = decoded.cpu()
                if use_cache:
                    # Clear after this segment so it cannot affect the next video.
                    self.model.clear_cache()
            else:
                # Decode longer clips in temporal chunks.
                decoded_chunks = []
                if use_cache:
                    # Clear once at the segment start; later chunks share the
                    # internal cache.
                    self.model.clear_cache()
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]  # [C, chunk_frames, H, W]
                    decoded_chunk = decode_function(chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                    decoded_chunks.append(decoded_chunk.cpu())

                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)
                if use_cache:
                    # Clear the cache after the full segment.
                    self.model.clear_cache()
            output.append(decoded)

        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            num_frame_per_block=1,
            t_scale=1.0,
            rope_method="linear",
            original_seq_len=None,
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size,
                num_frame_per_block=num_frame_per_block)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()
        self.model.t_scale = t_scale
        self.model.rope_method = rope_method
        self.model.original_seq_len = original_seq_len

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 28160  # [1, 32, 48, 44, 80]
    
        self.post_init()
        self._compiled_model_call = None

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

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
            name="WanDiffusionWrapper5B.model",
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
            suppress_errors=suppress_errors,
        )
        return self._compiled_model_call is not None

    def _call_model(self, *args, **kwargs):
        # iter-39 v2: publish kv_cache scalars BEFORE entering the compiled
        # graph. The earlier version (iter-39 v1) published them inside
        # `_forward_inference`, but that function IS compiled, so each
        # `.item()` triggered a graph break. Moving the reads to this eager
        # wrapper keeps the dict lookups in the compiled attention forward
        # free of `.item()` syncs without adding any graph break.
        kv_cache = kwargs.get("kv_cache", None)
        if kv_cache is not None and len(kv_cache) > 0:
            try:
                from wan_5b.modules.causal_model import _CURRENT_GRID_META
                first_block_cache = kv_cache[0]
                _CURRENT_GRID_META["global_end_index"] = int(
                    first_block_cache["global_end_index"].item()
                )
                _CURRENT_GRID_META["local_end_index"] = int(
                    first_block_cache["local_end_index"].item()
                )
                _ps = first_block_cache.get("pinned_start", None)
                if _ps is not None and hasattr(_ps, "item"):
                    _CURRENT_GRID_META["pinned_start"] = int(_ps.item())
                    _CURRENT_GRID_META["pinned_len"] = int(
                        first_block_cache["pinned_len"].item()
                    )
                else:
                    _CURRENT_GRID_META["pinned_start"] = -1
                    _CURRENT_GRID_META["pinned_len"] = 0
            except (KeyError, AttributeError, ImportError):
                pass
        defer_kv_updates = (
            os.environ.get("LLV2_DEFER_KV_UPDATES", "0") == "1"
            and kv_cache is not None
        )
        if defer_kv_updates:
            kwargs["defer_cache_updates"] = True

        if self._compiled_model_call is not None:
            # iter-25: signal cudagraph allocator that a new "step" starts.
            # Required for mode=reduce-overhead when modules cache state
            # (KV cache rolling buffers, fp4-quant scale tensors) so the
            # cudagraph pool knows it can safely reuse step-N memory now
            # that step-(N+1) is starting.
            mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if mark_step is not None:
                mark_step()
            result = self._compiled_model_call(*args, **kwargs)
        else:
            result = self.model(*args, **kwargs)

        if defer_kv_updates:
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    "LLV2_DEFER_KV_UPDATES expected model to return "
                    "(output, cache_update_infos)."
                )
            output, cache_update_infos = result
            if cache_update_infos:
                self.model._apply_cache_updates(kv_cache, cache_update_infos)
            return output
        return result

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
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

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        rope_temporal_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        rope_offset_was_set = (
            rope_temporal_offset is not None
            and hasattr(self.model, "rope_temporal_offset")
        )
        if rope_offset_was_set:
            prev_rope_temporal_offset = self.model.rope_temporal_offset
            self.model.rope_temporal_offset = rope_temporal_offset

        # X0 prediction
        if kv_cache is not None:
            flow_pred = self._call_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self._call_model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        if rope_offset_was_set:
            self.model.rope_temporal_offset = prev_rope_temporal_offset

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()


_MG_LIGHTVAE_DEFAULT_PATHS = {
    "mg_lightvae": os.path.join("wan_models", "Matrix-Game-3.0", "MG-LightVAE.pth"),
    "mg_lightvae_v2": os.path.join("wan_models", "Matrix-Game-3.0", "MG-LightVAE_v2.pth"),
}


def build_vae_5b(args):
    """Return the 5B VAE wrapper requested by args.vae_type."""
    vae_type = str(getattr(args, "vae_type", "wan")).lower().strip()

    if vae_type in ("wan", "wan2.2", ""):
        return WanVAEWrapper()

    if vae_type in _MG_LIGHTVAE_DEFAULT_PATHS:
        from utils.lightvae_5b_wrapper import LightVAE5BWrapper

        return LightVAE5BWrapper(vae_path=_MG_LIGHTVAE_DEFAULT_PATHS[vae_type])

    raise ValueError(
        f"Unknown vae_type '{vae_type}'. "
        "Expected one of: wan, mg_lightvae, mg_lightvae_v2."
    )
