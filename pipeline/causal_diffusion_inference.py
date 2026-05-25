# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

from tqdm import tqdm
from typing import List, Optional
import os
import statistics
import threading
import torch
import math

_LLV2_TIME = os.environ.get("LLV2_TIME") == "1"
_LLV2_DUMP_LATENT_DIR = os.environ.get("LLV2_DUMP_LATENT_DIR", "").strip()
# LLV2_PROFILE format: "<call_idx>:<wait>:<warmup>:<active>", e.g. "0:20:2:2"
_LLV2_PROFILE_SPEC = os.environ.get("LLV2_PROFILE", "").strip()
_LLV2_PROFILE_OUTPUT_DIR = os.environ.get("LLV2_PROFILE_OUTPUT_DIR", "").strip()
_LLV2_PROFILE_CALL_COUNTER = 0
from wan_5b.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, build_vae_5b
from utils.dataset import DEFAULT_SCENE_CUT_PREFIX
from utils.config import section_get, wan_default_config


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        model_name = getattr(args.model_kwargs, "model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported in this release, got {model_name}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = build_vae_5b(args) if vae is None else vae

        # iter-33: optionally compile the VAE decoder (cuda:2). The Python
        # `for step in range(total_steps)` wrapper in cached_decode keeps the
        # mutating feat_cache in eager; per-step self.decoder() call hits the
        # compiled version. Opt-in via LLV2_COMPILE_VAE=1; default off.
        if os.environ.get("LLV2_COMPILE_VAE", "0") == "1":
            try:
                inner = getattr(self.vae, "model", None)
                if inner is not None and hasattr(inner, "decoder"):
                    inner.decoder = torch.compile(
                        inner.decoder,
                        backend="inductor",
                        mode="max-autotune-no-cudagraphs",
                        fullgraph=False,
                        dynamic=False,
                    )
                    print("[torch.compile] VAE.decoder wrapped")
            except Exception as exc:
                print(f"[torch.compile][warn] VAE compile setup failed: {exc}")

        # Step 2: Initialize scheduler
        self.num_train_timesteps = getattr(args, "num_train_timestep", 1000)
        self.sampling_steps = section_get(args, "inference", "sampling_steps", 50)
        self.sample_solver = 'unipc'
        self.shift = getattr(args, "timestep_shift",
                             getattr(args.model_kwargs, "timestep_shift", 5.0))

        
        self.frame_seq_length = math.prod(args.image_or_video_shape[-2:]) // 4
        self.model_name = model_name
        self.num_transformer_blocks = wan_default_config[self.model_name]["num_transformer_blocks"]

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.quantize_kv = getattr(args, "kv_quant", False)
        self.kv_quant_scale_rule = getattr(args, "kv_quant_scale_rule", "mse")
        self.kv_quant_backend = getattr(args, "kv_quant_backend", "cuda")
        self.independent_first_frame = section_get(args, "inference", "independent_first_frame", False)
        self.local_attn_size = section_get(
            args, "inference", "local_attn_size", -1, aliases=("inference_local_attn_size",)
        )
        if self.local_attn_size == -1:
            self.local_attn_size = getattr(args, "model_kwargs", {}).get("local_attn_size", -1)
        self.sink_size = section_get(
            args, "inference", "sink_size", None, aliases=("inference_sink_size",)
        )
        if self.sink_size is None:
            _model_sink = getattr(args, "model_kwargs", {}).get("sink_size", None)
            if _model_sink is not None:
                self.sink_size = _model_sink
        if self.sink_size is None:
            self.sink_size = 0
        self.scene_cut_prefix = section_get(args, "inference", "scene_cut_prefix", DEFAULT_SCENE_CUT_PREFIX)
        self.multi_shot_sink = section_get(args, "inference", "multi_shot_sink", False)
        self.shot_clean_recache = section_get(args, "inference", "shot_clean_recache", False)
        self.global_sink_size = self.sink_size if self.multi_shot_sink else 0
        self.multi_shot_rope_offset = section_get(
            args,
            "inference",
            "multi_shot_rope_offset",
            0.0,
        )
        self.guidance_scale = section_get(args, "inference", "guidance_scale", getattr(args, "guidance_scale", 1.0))
        self.negative_prompt = section_get(args, "inference", "negative_prompt", getattr(args, "negative_prompt", ""))
        self.streaming_vae = section_get(args, "inference", "streaming_vae", getattr(args, "streaming_vae", False))
        self.async_vae = section_get(args, "inference", "async_vae", getattr(args, "async_vae", False))
        vae_device = section_get(args, "inference", "vae_device", getattr(args, "vae_device", None))
        self.vae_device = torch.device(vae_device) if vae_device else None

        if self.quantize_kv:
            from utils.quant import LongLiveQuantizationConfig

            self.kv_quant_config = LongLiveQuantizationConfig(
                scale_rule=self.kv_quant_scale_rule,
                backend=self.kv_quant_backend,
                type="kv",
            )
        else:
            self.kv_quant_config = None
        self._dit_model.kv_quant_config = self.kv_quant_config

        if self.streaming_vae and self.vae_device is not None:
            vae_mode = "streaming-pipeline"
        elif self.streaming_vae and self.async_vae:
            vae_mode = "streaming-async"
        elif self.streaming_vae:
            vae_mode = "streaming"
        else:
            vae_mode = "batch"
        print(
            f"KV inference with {self.num_frame_per_block} frames per block "
            f"(kv_quant={self.quantize_kv}, vae_decode={vae_mode})"
        )

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.inference_t_scale = getattr(args, "inference_t_scale", None)
        self.use_relative_rope = getattr(args, "use_relative_rope", False)
        self._rope_method_override = getattr(args, "rope_method", None)
        self._original_seq_len_override = getattr(args, "original_seq_len", None)

    @property
    def _dit_model(self):
        """Return the underlying CausalWanModel, unwrapping PeftModel if present.

        After LoRA wrapping, ``self.generator.model`` is a PeftModel whose
        structure is PeftModel -> LoraModel (.base_model) -> CausalWanModel
        (.model).  Direct attribute writes on PeftModel do NOT propagate to
        CausalWanModel, so any runtime overrides (t_scale, rope_method, …)
        must target the unwrapped model returned by this property.
        """
        model = self.generator.model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            return model.base_model.model
        return model

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts[0]
        )
        conditional_dict_list = [
            {"prompt_embeds": conditional_dict["prompt_embeds"][i:i+1]}
            for i in range(conditional_dict["prompt_embeds"].shape[0])
        ]
        use_cfg = self.guidance_scale != 1.0
        if use_cfg:
            unconditional_dict = self.text_encoder(
                text_prompts=[self.negative_prompt] * batch_size
            )
        else:
            unconditional_dict = None

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                if use_cfg:
                    self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["pinned_start"].fill_(-1)
                self.kv_cache_pos[block_index]["pinned_len"].zero_()
                if use_cfg:
                    self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["pinned_start"].fill_(-1)
                    self.kv_cache_neg[block_index]["pinned_len"].zero_()

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0

        # Save model state before overriding for inference.
        # Use _dit_model to reach the real CausalWanModel (PeftModel wrapping
        # intercepts attribute writes, so self.generator.model.xxx would land
        # on the wrapper instead of the model that reads them in forward()).
        dit = self._dit_model
        prev_local_attn_size = dit.local_attn_size
        prev_t_scale = getattr(dit, 't_scale', 1.0)
        prev_rope_method = getattr(dit, 'rope_method', 'linear')
        prev_original_seq_len = getattr(dit, 'original_seq_len', None)
        prev_use_relative_rope = getattr(dit, 'use_relative_rope', False)
        prev_rope_temporal_offset = getattr(dit, 'rope_temporal_offset', 0.0)
        prev_max_attention_sizes = {}
        prev_sink_sizes = {}
        prev_global_sink_sizes = {}
        for name, module in self.generator.model.named_modules():
            if hasattr(module, 'max_attention_size'):
                prev_max_attention_sizes[name] = module.max_attention_size
            if hasattr(module, 'sink_size'):
                prev_sink_sizes[name] = module.sink_size
            if hasattr(module, 'global_sink_size'):
                prev_global_sink_sizes[name] = module.global_sink_size

        dit.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {dit.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if self.sink_size is not None:
            self._set_all_modules_sink_size(self.sink_size)
            print(f"[inference] sink_size set to: {self.sink_size}"
                  f"{', multi_shot_sink enabled (pinned position)' if self.multi_shot_sink else ''}"
                  f"{', shot_clean_recache enabled' if self.shot_clean_recache else ''}")

        # Propagate the internally derived global sink length.
        self._set_all_modules_global_sink_size(self.global_sink_size)
        if self.global_sink_size and self.global_sink_size > 0:
            print(f"[inference] auto_global_sink_size set to: {self.global_sink_size} "
                  f"(first {self.global_sink_size} frames permanently anchored)")

        if self.inference_t_scale is not None:
            dit.t_scale = self.inference_t_scale
            print(f"[inference] t_scale overridden to: {dit.t_scale}")

        if self._rope_method_override is not None:
            dit.rope_method = self._rope_method_override
        if self._original_seq_len_override is not None:
            dit.original_seq_len = self._original_seq_len_override
        print(f"[inference] rope_method={dit.rope_method}, "
              f"original_seq_len={dit.original_seq_len}")

        dit.use_relative_rope = self.use_relative_rope
        if self.use_relative_rope:
            print(f"[inference] use_relative_rope enabled")

        dit.rope_temporal_offset = 0.0
        if self.multi_shot_rope_offset != 0.0:
            print(f"[inference] multi_shot_rope_offset={self.multi_shot_rope_offset} "
                  f"(multi-shot RoPE offset enabled)")

        try:
            raw_prompts = text_prompts[0] if isinstance(text_prompts[0], (list, tuple)) else text_prompts
            return self._inference_inner(
                noise=noise, batch_size=batch_size, num_frames=num_frames,
                num_channels=num_channels, height=height, width=width,
                num_blocks=num_blocks, num_input_frames=num_input_frames,
                num_output_frames=num_output_frames, output=output,
                conditional_dict=conditional_dict,
                conditional_dict_list=conditional_dict_list,
                unconditional_dict=unconditional_dict,
                use_cfg=use_cfg, initial_latent=initial_latent,
                return_latents=return_latents,
                current_start_frame=current_start_frame,
                cache_start_frame=cache_start_frame,
                raw_prompts=raw_prompts,
            )
        finally:
            dit.local_attn_size = prev_local_attn_size
            dit.t_scale = prev_t_scale
            dit.rope_method = prev_rope_method
            dit.original_seq_len = prev_original_seq_len
            dit.use_relative_rope = prev_use_relative_rope
            dit.rope_temporal_offset = prev_rope_temporal_offset
            for name, module in self.generator.model.named_modules():
                if name in prev_max_attention_sizes:
                    try:
                        module.max_attention_size = prev_max_attention_sizes[name]
                    except Exception:
                        pass
                if name in prev_sink_sizes:
                    try:
                        module.sink_size = prev_sink_sizes[name]
                    except Exception:
                        pass
                if name in prev_global_sink_sizes:
                    try:
                        module.global_sink_size = prev_global_sink_sizes[name]
                    except Exception:
                        pass

    def _inference_inner(
        self, noise, batch_size, num_frames, num_channels, height, width,
        num_blocks, num_input_frames, num_output_frames, output,
        conditional_dict, conditional_dict_list, unconditional_dict,
        use_cfg, initial_latent, return_latents,
        current_start_frame, cache_start_frame,
        raw_prompts=None,
    ):

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                if use_cfg:
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=unconditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length
                    )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                if use_cfg:
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=unconditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length
                    )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        # Multi-shot RoPE offset: track current shot index for phase offset.
        current_shot_index = 0
        phi = self.multi_shot_rope_offset
        self._dit_model.rope_temporal_offset = 0.0
        streaming_decode = self.streaming_vae and not return_latents
        pipeline_vae = streaming_decode and self.vae_device is not None
        async_vae = streaming_decode and self.async_vae and not pipeline_vae
        if streaming_decode:
            vae_dev = self.vae_device if pipeline_vae else noise.device
            vae_scale = [
                self.vae.mean.to(device=vae_dev, dtype=noise.dtype),
                1.0 / self.vae.std.to(device=vae_dev, dtype=noise.dtype),
            ]
            self.vae.model.clear_cache()
            video_chunks = []
            if async_vae:
                vae_stream = torch.cuda.Stream(device=noise.device)
                prev_vae_done = None
            if pipeline_vae:
                vae_thread_error = []
                vae_thread_chunks = []
                vae_work_queue = []
                vae_queue_lock = threading.Lock()
                vae_work_ready = threading.Event()
                vae_all_done = threading.Event()

                def _vae_thread_fn():
                    try:
                        while True:
                            vae_work_ready.wait()
                            vae_work_ready.clear()
                            while True:
                                with vae_queue_lock:
                                    if not vae_work_queue:
                                        break
                                    item = vae_work_queue.pop(0)
                                if item is None:
                                    vae_all_done.set()
                                    return
                                decoded = self.vae.model.cached_decode(
                                    item,
                                    vae_scale,
                                ).float().clamp_(-1, 1)
                                # Pinned-memory DtoH: pageable copy hits ~0.2 GB/s
                                # (1.5s per 313MB chunk → ~80s/prompt at end); pinned
                                # path runs at PCIe limit (~25 GB/s = ~12ms / chunk).
                                pinned = torch.empty(
                                    decoded.shape, dtype=decoded.dtype,
                                    device="cpu", pin_memory=True,
                                )
                                pinned.copy_(decoded, non_blocking=True)
                                torch.cuda.synchronize(decoded.device)
                                vae_thread_chunks.append(pinned)
                    except Exception as exc:
                        vae_thread_error.append(exc)
                        vae_all_done.set()

                vae_bg_thread = threading.Thread(target=_vae_thread_fn, daemon=True)
                vae_bg_thread.start()

        _block_events = [] if _LLV2_TIME else None
        global _LLV2_PROFILE_CALL_COUNTER
        _call_idx = _LLV2_PROFILE_CALL_COUNTER
        _LLV2_PROFILE_CALL_COUNTER += 1
        _prof = None
        _prof_trace_path = None
        if _LLV2_PROFILE_SPEC and _LLV2_PROFILE_OUTPUT_DIR:
            _parts = _LLV2_PROFILE_SPEC.split(":")
            _target_call = int(_parts[0]) if len(_parts) > 0 else 0
            _wait_n = int(_parts[1]) if len(_parts) > 1 else 20
            _warmup_n = int(_parts[2]) if len(_parts) > 2 else 2
            _active_n = int(_parts[3]) if len(_parts) > 3 else 2
            if _call_idx == _target_call:
                from torch.profiler import profile as _tp_profile, ProfilerActivity, schedule
                os.makedirs(_LLV2_PROFILE_OUTPUT_DIR, exist_ok=True)
                _prof_trace_path = os.path.join(
                    _LLV2_PROFILE_OUTPUT_DIR, f"trace_call{_call_idx}.json"
                )
                _prof = _tp_profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(
                        wait=_wait_n, warmup=_warmup_n, active=_active_n, repeat=1
                    ),
                    record_shapes=False,
                    with_stack=False,
                )
                _prof.start()
                print(
                    f"[LLV2_PROFILE] enabled for call={_call_idx} "
                    f"wait={_wait_n} warmup={_warmup_n} active={_active_n} "
                    f"-> {_prof_trace_path}",
                    flush=True,
                )
        for chunk_index, current_num_frames in enumerate(all_num_frames):
            if _LLV2_TIME:
                _ev_s = torch.cuda.Event(enable_timing=True)
                _ev_e = torch.cuda.Event(enable_timing=True)
                _ev_s.record()
            conditional_dict = conditional_dict_list[chunk_index]
            # Reset the cross-attention cache when each chunk uses a different
            # prompt; otherwise the model reuses the previous chunk's k/v and
            # ignores the current conditional_dict.
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False

            # Update RoPE phase offset on shot boundaries.
            is_shot_boundary = self._is_shot_boundary(raw_prompts, chunk_index)
            if is_shot_boundary and phi != 0.0:
                current_shot_index += 1
                self._dit_model.rope_temporal_offset = current_shot_index * phi
                print(f"[inference] multi-shot RoPE: shot_index={current_shot_index}, "
                      f"temporal_offset={self._dit_model.rope_temporal_offset:.4f}")

            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                if use_cfg:
                    flow_pred_uncond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=unconditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length
                    )
                    flow_pred = flow_pred_uncond + self.guidance_scale * (
                        flow_pred_cond - flow_pred_uncond)
                else:
                    flow_pred = flow_pred_cond

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                # iter-34: removed per-step debug print of kv_cache scalar
                # tensors (was forcing GPU→CPU sync every sampling step ×
                # 4 steps × 48 chunks = 192 stalls per prompt). Re-enable
                # behind LLV2_DEBUG_KV=1 if needed for debugging.
                if os.environ.get("LLV2_DEBUG_KV", "0") == "1":
                    print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                    print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            is_scene_cut = self._is_scene_cut(raw_prompts, chunk_index)

            if is_scene_cut and self.shot_clean_recache:
                print(f"[inference] Scene cut at chunk {chunk_index}, zeroing KV before recache")
                current_start_tokens = current_start_frame * self.frame_seq_length
                self._zero_kv_data(self.kv_cache_pos, current_start_tokens)
                if use_cfg:
                    self._zero_kv_data(self.kv_cache_neg, current_start_tokens)

            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            if use_cfg:
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

            # Step 3.3b: pin the current chunk for multi-shot sink on scene cut.
            if is_scene_cut:
                print(f"[inference] Scene cut at chunk {chunk_index}, pinning chunk as shot-sink")
                self._pin_current_chunk(self.kv_cache_pos, current_num_frames)
                if use_cfg:
                    self._pin_current_chunk(self.kv_cache_neg, current_num_frames)

            if streaming_decode:
                if async_vae:
                    diffusion_done = torch.cuda.Event()
                    diffusion_done.record()
                    if prev_vae_done is not None:
                        prev_vae_done.synchronize()
                    with torch.cuda.stream(vae_stream):
                        vae_stream.wait_event(diffusion_done)
                        chunk_bcthw = latents.permute(0, 2, 1, 3, 4).contiguous()
                        decoded_chunk = self.vae.model.cached_decode(
                            chunk_bcthw,
                            vae_scale,
                        ).float().clamp_(-1, 1)
                        video_chunks.append(decoded_chunk)
                    prev_vae_done = torch.cuda.Event()
                    prev_vae_done.record(vae_stream)
                elif pipeline_vae:
                    latent_on_vae = latents.permute(0, 2, 1, 3, 4).contiguous().to(vae_dev)
                    with vae_queue_lock:
                        vae_work_queue.append(latent_on_vae)
                    vae_work_ready.set()
                else:
                    chunk_bcthw = latents.permute(0, 2, 1, 3, 4).contiguous()
                    decoded_chunk = self.vae.model.cached_decode(
                        chunk_bcthw,
                        vae_scale,
                    ).float().clamp_(-1, 1)
                    video_chunks.append(decoded_chunk.cpu())
                    del decoded_chunk, chunk_bcthw
                    torch.cuda.empty_cache()

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

            if _LLV2_TIME:
                _ev_e.record()
                _block_events.append((_ev_s, _ev_e))
            if _prof is not None:
                _prof.step()

        if _prof is not None:
            _prof.stop()
            _prof.export_chrome_trace(_prof_trace_path)
            print(f"[LLV2_PROFILE] saved trace -> {_prof_trace_path}", flush=True)

        if _LLV2_TIME and _block_events:
            torch.cuda.synchronize()
            _times = [_s.elapsed_time(_e) for _s, _e in _block_events]
            _sorted = sorted(_times)
            _n = len(_sorted)
            _p10 = _sorted[max(0, _n // 10 - 1)]
            _p50 = statistics.median(_sorted)
            _p90 = _sorted[max(0, _n - _n // 10 - 1)]
            _mean = sum(_times) / _n
            _total = sum(_times)
            print(
                f"[LLV2_TIME] blocks={_n} mean_ms={_mean:.2f} p10_ms={_p10:.2f} "
                f"median_ms={_p50:.2f} p90_ms={_p90:.2f} total_ms={_total:.2f}",
                flush=True,
            )

        if _LLV2_DUMP_LATENT_DIR:
            os.makedirs(_LLV2_DUMP_LATENT_DIR, exist_ok=True)
            _existing = sum(
                1 for _f in os.listdir(_LLV2_DUMP_LATENT_DIR)
                if _f.startswith("latent_") and _f.endswith(".pt")
            )
            _path = os.path.join(_LLV2_DUMP_LATENT_DIR, f"latent_{_existing:04d}.pt")
            torch.save(output.detach().cpu(), _path)
            print(f"[LLV2_DUMP] saved latent {tuple(output.shape)} -> {_path}", flush=True)

        # Step 4: Decode the output


        if return_latents:
            return output
        elif streaming_decode:
            if async_vae:
                vae_stream.synchronize()
            elif pipeline_vae:
                with vae_queue_lock:
                    vae_work_queue.append(None)
                vae_work_ready.set()
                vae_all_done.wait()
                vae_bg_thread.join()
                if vae_thread_error:
                    raise RuntimeError(
                        f"[pipeline_vae] VAE decode failed: {vae_thread_error[0]}"
                    ) from vae_thread_error[0]
                video_chunks = vae_thread_chunks
            video_bcthw = torch.cat(video_chunks, dim=2)
            video = video_bcthw.permute(0, 2, 1, 3, 4)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            self.vae.model.clear_cache()
            return video
        else:
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 3 * self.num_frame_per_block * self.frame_seq_length

        block_token_size = self.num_frame_per_block * self.frame_seq_length
        max_blocks = kv_cache_size // block_token_size

        if self.quantize_kv:
            from utils.quant import clone_quantized_tensor, quantize_to_fp4

            print(
                f"[KV Cache] Quantized (nvfp4): block_token_size={block_token_size}, "
                f"max_blocks={max_blocks}, num_heads={num_heads}, layers={self.num_transformer_blocks}"
            )
            zero_block = torch.zeros(
                [block_token_size * num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            zero_qt = quantize_to_fp4(zero_block, self.kv_quant_config)

        for _ in range(self.num_transformer_blocks):
            if self.quantize_kv:
                kv_cache_pos.append({
                    "k": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "v": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "quantized": True,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                })
                kv_cache_neg.append({
                    "k": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "v": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                    "quantized": True,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                })
            else:
                kv_cache_pos.append({
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "quantized": False,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                })
                kv_cache_neg.append({
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "quantized": False,
                    "block_token_size": block_token_size,
                    "max_blocks": max_blocks,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def clear_cache(self):
        """
        Explicitly release large KV / cross-attention caches to free GPU memory.
        Safe to call between independent inference calls; caches will be
        re-created on demand by _initialize_kv_cache/_initialize_crossattn_cache.
        """
        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass

    def _set_all_modules_sink_size(self, sink_size_value: int):
        """
        Override sink_size on all submodules that define it.
        """
        if hasattr(self.generator.model, "sink_size"):
            setattr(self.generator.model, "sink_size", sink_size_value)

        for name, module in self.generator.model.named_modules():
            if hasattr(module, "sink_size"):
                try:
                    setattr(module, "sink_size", sink_size_value)
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

    def _is_shot_boundary(self, raw_prompts, chunk_index):
        """Return True when *chunk_index* starts a new shot (prompt-based detection).

        Pure prompt check — no dependency on sink config so that Narrative
        RoPE and other shot-aware features can reuse it independently.
        """
        if chunk_index == 0:
            return False
        if not isinstance(raw_prompts, (list, tuple)):
            return False
        if chunk_index >= len(raw_prompts):
            return False
        prompt = raw_prompts[chunk_index]
        return isinstance(prompt, str) and prompt.startswith(self.scene_cut_prefix)

    def _is_scene_cut(self, raw_prompts, chunk_index):
        """Return True when *chunk_index* is the first chunk of a new scene
        AND multi-shot sink is enabled."""
        if not self.multi_shot_sink:
            return False
        if not self.sink_size or self.sink_size == 0:
            return False
        return self._is_shot_boundary(raw_prompts, chunk_index)

    def _update_sink_for_scene_cut(self, kv_cache, current_num_frames):
        """Legacy copy-to-front sink relocation (used by training pipeline)."""
        global_sink_tokens = self.global_sink_size * self.frame_seq_length
        shot_sink_tokens = self.sink_size * self.frame_seq_length
        chunk_tokens = current_num_frames * self.frame_seq_length
        copy_len = min(shot_sink_tokens, chunk_tokens)

        # iter-38: local_end_index is in lockstep across all blocks (set by
        # _apply_cache_updates with the same value). Read once → 60 syncs to 1.
        local_end = int(kv_cache[0]["local_end_index"].item())
        chunk_start = local_end - chunk_tokens
        dst_start = global_sink_tokens
        src_slice = slice(chunk_start, chunk_start + copy_len)
        dst_slice = slice(dst_start, dst_start + copy_len)

        for block_cache in kv_cache:
            block_cache["k"][:, dst_slice] = block_cache["k"][:, src_slice].clone()
            block_cache["v"][:, dst_slice] = block_cache["v"][:, src_slice].clone()

    def _pin_current_chunk(self, kv_cache, current_num_frames):
        """Mark the current chunk's buffer position as pinned for multi-shot sink.

        The pinned region REPLACES the original sink on the next rolling event.
        No data is copied here — relocation happens inside the attention layer
        during rolling, ensuring zero duplication.
        """
        chunk_tokens = current_num_frames * self.frame_seq_length
        pin_len = min(self.sink_size * self.frame_seq_length, chunk_tokens)

        # iter-38: local_end_index is in lockstep across all blocks. Read once.
        local_end = int(kv_cache[0]["local_end_index"].item())
        chunk_start = local_end - chunk_tokens

        for block_cache in kv_cache:
            block_cache["pinned_start"].fill_(chunk_start)
            block_cache["pinned_len"].fill_(pin_len)

    def _zero_kv_data(self, kv_cache, current_start_tokens):
        """Reset KV cache for clean recache, preserving global sink."""
        global_sink_tokens = self.global_sink_size * self.frame_seq_length
        for block_cache in kv_cache:
            block_cache["local_end_index"].fill_(global_sink_tokens)
            block_cache["global_end_index"].fill_(current_start_tokens)
            block_cache["pinned_start"].fill_(-1)
            block_cache["pinned_len"].zero_()
