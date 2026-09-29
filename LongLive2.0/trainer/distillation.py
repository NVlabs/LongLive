# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging

from utils.dataset import cycle
from utils.dataset import MultiVideoConcatDataset, MultiTextConcatDataset, multi_video_collate_fn, eval_collate_fn, DEFAULT_SCENE_CUT_PREFIX
from utils.config import section_get, wan_default_config
from utils.distributed import EMA_FSDP, fsdp_wrap, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
from utils.sampler import build_training_sampler
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video

# LoRA related imports
import peft
from peft import get_peft_model_state_dict

from pipeline import (
    CausalDiffusionInferencePipeline
)
import time

class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = getattr(config, "causal", getattr(config, "all_causal", True))
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            if getattr(config, "wandb_key", None):
                wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                id=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
                resume="allow"
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError(f"Unsupported distribution matching loss: {config.distribution_loss}")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # ================================= NVFP4 Quantized Training / Inference =================================
        # `generator_quant` is the preferred student flag; `model_quant` is kept
        # as a legacy alias used by earlier Sage configs.
        self.generator_quant = getattr(config, "generator_quant", getattr(config, "model_quant", False))
        self.real_score_quant = getattr(config, "real_score_quant", False)
        self.fake_score_quant = getattr(config, "fake_score_quant", False)
        self.real_score_quant_materialize = getattr(config, "real_score_quant_materialize", True)

        if self.generator_quant or self.real_score_quant or self.fake_score_quant:
            from utils.quant import ModelQuantizationConfig, quantize_model_with_filter

            fallback_sr = getattr(config, "model_quant_scale_rule", "static_6")
            fallback_asr = getattr(config, "model_quant_activation_scale_rule", "static_6")
            fallback_wsr = getattr(config, "model_quant_weight_scale_rule", None)
            fallback_gsr = getattr(config, "model_quant_gradient_scale_rule", None)

            if self.generator_quant:
                gen_cfg = ModelQuantizationConfig(
                    scale_rule=getattr(config, "generator_quant_scale_rule", fallback_sr),
                    activation_scale_rule=getattr(config, "generator_quant_activation_scale_rule", fallback_asr),
                    weight_scale_rule=getattr(config, "generator_quant_weight_scale_rule", fallback_wsr),
                    gradient_scale_rule=getattr(config, "generator_quant_gradient_scale_rule", fallback_gsr),
                    keep_master_weights=True,
                    weight_scale_2d=True,
                )
                self.model.generator.model, gen_matched = quantize_model_with_filter(
                    self.model.generator.model,
                    quant_config=gen_cfg,
                    filtered_modules=getattr(config, "generator_quant_filtered_modules", None),
                    filter_profile="student",
                    use_default_filtered_modules=getattr(
                        config, "generator_quant_use_default_filtered_modules", True
                    ),
                    cast_model_to_bf16=False,
                    materialize_for_inference=False,
                    verbose=self.is_main_process,
                )
                if self.is_main_process:
                    print(
                        "[NVFP4] Generator (student) quantized training enabled, "
                        f"scale_rule={gen_cfg.scale_rule}, {len(gen_matched)} modules excluded"
                    )

            if self.real_score_quant:
                real_cfg = ModelQuantizationConfig(
                    scale_rule=getattr(config, "real_score_quant_scale_rule", fallback_sr),
                    activation_scale_rule=getattr(config, "real_score_quant_activation_scale_rule", fallback_asr),
                    weight_scale_rule=getattr(config, "real_score_quant_weight_scale_rule", fallback_wsr),
                    gradient_scale_rule=None,
                    keep_master_weights=True,
                    weight_scale_2d=True,
                )
                self.model.real_score.model, real_matched = quantize_model_with_filter(
                    self.model.real_score.model,
                    quant_config=real_cfg,
                    filtered_modules=getattr(config, "real_score_quant_filtered_modules", None),
                    filter_profile="teacher",
                    use_default_filtered_modules=getattr(
                        config, "real_score_quant_use_default_filtered_modules", True
                    ),
                    cast_model_to_bf16=False,
                    materialize_for_inference=False,
                    verbose=self.is_main_process,
                )
                if self.is_main_process:
                    real_score_plan = (
                        "auto materialize before FSDP wrapping"
                        if self.real_score_quant_materialize
                        else "keep master weights after checkpoint load"
                    )
                    print(
                        "[NVFP4] Real_score (teacher) quantized inference enabled, "
                        f"scale_rule={real_cfg.scale_rule}, {len(real_matched)} modules excluded, "
                        f"{real_score_plan}"
                    )

            if self.fake_score_quant:
                fake_cfg = ModelQuantizationConfig(
                    scale_rule=getattr(config, "fake_score_quant_scale_rule", fallback_sr),
                    activation_scale_rule=getattr(config, "fake_score_quant_activation_scale_rule", fallback_asr),
                    weight_scale_rule=getattr(config, "fake_score_quant_weight_scale_rule", fallback_wsr),
                    gradient_scale_rule=getattr(config, "fake_score_quant_gradient_scale_rule", fallback_gsr),
                    keep_master_weights=True,
                    weight_scale_2d=True,
                )
                self.model.fake_score.model, fake_matched = quantize_model_with_filter(
                    self.model.fake_score.model,
                    quant_config=fake_cfg,
                    filtered_modules=getattr(config, "fake_score_quant_filtered_modules", None),
                    filter_profile="critic",
                    use_default_filtered_modules=getattr(
                        config, "fake_score_quant_use_default_filtered_modules", True
                    ),
                    cast_model_to_bf16=False,
                    materialize_for_inference=False,
                    verbose=self.is_main_process,
                )
                if self.is_main_process:
                    print(
                        "[NVFP4] Fake_score (critic) quantized training enabled, "
                        f"scale_rule={fake_cfg.scale_rule}, {len(fake_matched)} modules excluded"
                    )

        # Auto resume configuration (needed for LoRA checkpoint loading)
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        # ================================= LoRA Configuration =================================
        self.is_lora_enabled = False
        self.lora_config = None

        if hasattr(config, 'adapter') and config.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            
            if self.is_main_process:
                print(f"LoRA enabled with config: {self.lora_config}")
                print("Loading base model and applying LoRA before FSDP wrapping...")
            
            # 1. Load base model first (config.generator_ckpt) before applying LoRA.
            generator_checkpoint_path = getattr(config, "generator_ckpt", None)
            if generator_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading base model from {generator_checkpoint_path} (before applying LoRA)")
                generator_checkpoint = torch.load(generator_checkpoint_path, map_location="cpu")
                
                # Load generator (directly; no key alignment needed since LoRA not applied yet)
                if isinstance(generator_checkpoint, dict) and "generator" in generator_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {generator_checkpoint_path}")
                    self.model.generator.load_state_dict(generator_checkpoint["generator"], strict=True)
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                elif isinstance(generator_checkpoint, dict) and "model" in generator_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {generator_checkpoint_path}")
                    self.model.generator.load_state_dict(generator_checkpoint["model"], strict=True)
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                else:
                    self.model.generator.load_state_dict(generator_checkpoint, strict=True)
                    if self.is_main_process:
                        print("Loading base model as raw state_dict")
                
                # Load critic from full/base checkpoints when available.
                if isinstance(generator_checkpoint, dict) and "critic" in generator_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {generator_checkpoint_path}")
                    self.model.fake_score.load_state_dict(generator_checkpoint["critic"], strict=True)
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                # Load training step from checkpoint metadata.
                if isinstance(generator_checkpoint, dict) and "step" in generator_checkpoint:
                    self.step = generator_checkpoint["step"]
                    if self.is_main_process:
                        print(f"base_checkpoint step: {self.step}")
                else:
                    if self.is_main_process:
                        print("Warning: Step not found in checkpoint, starting from step 0.")
                del generator_checkpoint
                gc.collect()
            else:
                if self.is_main_process:
                    print("No base model checkpoint specified, skipping base weight loading for LoRA training.")

        # Load real_score from a separate checkpoint (independent of LoRA / auto_resume)
        real_score_ckpt = getattr(config, "real_score_ckpt", None)
        if real_score_ckpt:
            if self.is_main_process:
                print(f"Loading real_score from {real_score_ckpt}")
            real_ckpt = torch.load(real_score_ckpt, map_location="cpu")
            if "generator" in real_ckpt:
                self.model.real_score.load_state_dict(real_ckpt["generator"], strict=True)
            elif "critic" in real_ckpt:
                self.model.real_score.load_state_dict(real_ckpt["critic"], strict=True)
            elif "model" in real_ckpt:
                self.model.real_score.load_state_dict(real_ckpt["model"], strict=True)
            else:
                if self.is_main_process:
                    print(f"No recognized key in {real_score_ckpt}, treating as raw state_dict")
                self.model.real_score.load_state_dict(real_ckpt, strict=True)
            del real_ckpt
            gc.collect()
            if self.is_main_process:
                print(f"Successfully loaded real_score from {real_score_ckpt}")

        # Apply LoRA wrapping if enabled (after all base weights are loaded, before FSDP)
        if self.is_lora_enabled:
            # 2. Apply LoRA wrapping now (after loading base model, before FSDP wrapping)
            if self.is_main_process:
                print("Applying LoRA to models...")
            self.model.generator.model = self._configure_lora_for_model(self.model.generator.model, "generator")

            # Configure LoRA for fake_score if needed
            if getattr(self.lora_config, 'apply_to_critic', True):
                self.model.fake_score.model = self._configure_lora_for_model(self.model.fake_score.model, "fake_score")
                if self.is_main_process:
                    print("LoRA applied to both generator and critic")
            else:
                if self.is_main_process:
                    print("LoRA applied to generator only")

            # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available).
            # Priority: auto_resume -> legacy lora_ckpt -> initialized adapters.
            lora_checkpoint_path = None
            lora_checkpoint = None
            if auto_resume and self.output_path:
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    lora_checkpoint_path = latest_checkpoint
                    if self.is_main_process:
                        print(f"Auto resume: Found LoRA checkpoint at {lora_checkpoint_path}")
                else:
                    if self.is_main_process:
                        print("Auto resume: No LoRA checkpoint found in logdir")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified for LoRA")
            else:
                if self.is_main_process:
                    print("Auto resume disabled for LoRA")

            if lora_checkpoint_path is not None:
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
            elif getattr(config, "lora_ckpt", None):
                lora_checkpoint_path = config.lora_ckpt
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
                if self.is_main_process:
                    print(f"Using legacy lora_ckpt: {lora_checkpoint_path}")
            elif self.is_main_process:
                print("No LoRA checkpoint specified, starting LoRA training from scratch")

            # Load LoRA checkpoint (before FSDP wrapping)
            if lora_checkpoint is not None:
                if self.is_main_process:
                    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} (before FSDP wrapping)")

                if "generator_lora" not in lora_checkpoint:
                    raise ValueError(f"LoRA checkpoint {lora_checkpoint_path} is not a valid LoRA checkpoint. "
                                     f"Found keys: {list(lora_checkpoint.keys())}")

                if self.is_main_process:
                    print(f"Loading LoRA generator weights: {len(lora_checkpoint['generator_lora'])} keys in checkpoint")
                peft.set_peft_model_state_dict(self.model.generator.model, lora_checkpoint["generator_lora"])
                del lora_checkpoint["generator_lora"]

                if getattr(self.lora_config, 'apply_to_critic', True):
                    if "critic_lora" not in lora_checkpoint:
                        raise ValueError(f"LoRA checkpoint {lora_checkpoint_path} is missing critic_lora.")
                    if self.is_main_process:
                        print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")
                    peft.set_peft_model_state_dict(self.model.fake_score.model, lora_checkpoint["critic_lora"])
                    del lora_checkpoint["critic_lora"]
                gc.collect()

                if "step" in lora_checkpoint:
                    self.step = lora_checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming LoRA training from step {self.step}")
            else:
                if self.is_main_process:
                    print("No LoRA checkpoint to load, starting from scratch")

        # Materialize quantized inference-only weights before FSDP can expose
        # sharded 1D parameter views. The student/critic are materialized only
        # in LoRA mode because their base weights are frozen after adapters.
        if self.generator_quant and self.is_lora_enabled:
            self._materialize_quantized_model_before_fsdp(
                self.model.generator.model,
                "Generator",
                cache_transposed_weights=True,
            )

        apply_lora_to_critic = getattr(self.lora_config, "apply_to_critic", True) if self.lora_config else False
        if self.fake_score_quant and self.is_lora_enabled and apply_lora_to_critic:
            self._materialize_quantized_model_before_fsdp(
                self.model.fake_score.model,
                "Fake_score",
                cache_transposed_weights=True,
            )

        if self.real_score_quant and self.real_score_quant_materialize:
            self._materialize_quantized_model_before_fsdp(
                self.model.real_score.model,
                "Real_score",
                cache_transposed_weights=False,
            )

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.is_lora_enabled:
                if self.is_main_process:
                    print(f"EMA disabled in LoRA mode (LoRA provides efficient parameter updates without EMA)")
                self.generator_ema = None
            else:
                print(f"Setting up EMA with weight {ema_weight}")
                self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        # Step 4: Initialize the optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 5: Initialize the dataloader
        self.use_backward_simulation = getattr(config, "backward_simulation", True)

        model_name = config.model_kwargs.model_name
        frame_raw_height = list(config.image_or_video_shape)[3] * wan_default_config[model_name]["spatial_compression_ratio"]
        frame_raw_width = list(config.image_or_video_shape)[4] * wan_default_config[model_name]["spatial_compression_ratio"]
        num_frame_per_block = getattr(config, "num_frame_per_block", 1)
        self.fps = wan_default_config[model_name].get("fps", 16)

        latent_frames_for_dataset = list(config.image_or_video_shape)[1]
        num_training_frames = getattr(config, "num_training_frames", latent_frames_for_dataset)
        assert latent_frames_for_dataset >= num_training_frames, (
            f"image_or_video_shape[1] ({latent_frames_for_dataset}) must be >= "
            f"num_training_frames ({num_training_frames}), otherwise the dataset "
            f"will not provide enough prompts for the rollout."
        )
        total_frames = (latent_frames_for_dataset - 1) * wan_default_config[model_name]["temporal_compression_ratio"] + 1
        if dist.get_rank() == 0:
            print(f"[Dataset] latent_frames_for_dataset={latent_frames_for_dataset}, total_frames={total_frames}")

        temporal_compression_ratio = wan_default_config[model_name]["temporal_compression_ratio"]
        first_chunk_frames = 1 + (num_frame_per_block - 1) * temporal_compression_ratio
        subsequent_chunk_frames = num_frame_per_block * temporal_compression_ratio
        num_blocks = 1 + (total_frames - first_chunk_frames) // subsequent_chunk_frames
        if not getattr(config, "generator_is_causal", True):
            num_blocks = 1

        chunks_per_shot = getattr(config, "chunks_per_shot", 0)
        scene_cut_prefix = getattr(config, "scene_cut_prefix", DEFAULT_SCENE_CUT_PREFIX)
        single_video_only = getattr(config, "uniform_prompt", False)
        allow_padding = getattr(config, "allow_padding", False)
        min_latent_frames = getattr(config, "min_latent_frames", 0)
        dataset_sample_warning_seconds = getattr(config, "dataset_sample_warning_seconds", 60.0)
        dataset_sample_warning_interval_seconds = getattr(
            config, "dataset_sample_warning_interval_seconds", 60.0
        )

        if self.use_backward_simulation and not self.config.i2v:
            dataset = MultiTextConcatDataset(
                data_path=config.data_path,
                num_blocks=num_blocks,
                chunks_per_shot=chunks_per_shot,
                scene_cut_prefix=scene_cut_prefix,
            )
            collate_fn = eval_collate_fn
            if dist.get_rank() == 0:
                print(f"[backward_simulation] Using MultiTextConcatDataset: "
                      f"data_path={config.data_path}, num_blocks={num_blocks}, "
                      f"chunks_per_shot={chunks_per_shot}")
        else:
            dataset = MultiVideoConcatDataset(
                data_dir=config.data_path,
                video_size=(frame_raw_height, frame_raw_width),
                total_frames=total_frames,
                deterministic=False,
                num_frame_per_block=num_frame_per_block,
                temporal_compression_ratio=temporal_compression_ratio,
                target_fps=self.fps,
                allow_padding=allow_padding,
                min_latent_frames=min_latent_frames,
                single_video_only=single_video_only,
                independent_first_frame=getattr(config, "independent_first_frame", False),
                return_image=getattr(config, "i2v", False),
                max_chunks_per_shot=chunks_per_shot,
                scene_cut_prefix=scene_cut_prefix,
                sample_warning_seconds=dataset_sample_warning_seconds,
                sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
            )
            collate_fn = multi_video_collate_fn
            if dist.get_rank() == 0 and single_video_only:
                print(f"[uniform_prompt] single_video_only enabled: each sample uses one video only")
        sampler = build_training_sampler(dataset, seed=config.seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=2, prefetch_factor=1, pin_memory=False,
            persistent_workers=False, collate_fn=collate_fn,
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = section_get(config, "evaluation", "interval", getattr(config, "vis_interval", -1))
        configured_vis_lengths = section_get(config, "evaluation", "num_frames", getattr(config, "vis_video_lengths", []))
        self.save_vis_latents_only = section_get(
            config,
            "evaluation",
            "save_latents_only",
            getattr(config, "return_latents", True),
            aliases=("return_latents", "save_latent_only"),
        )
        if isinstance(configured_vis_lengths, int):
            configured_vis_lengths = [configured_vis_lengths]
        if self.vis_interval > 0 and len(configured_vis_lengths) > 0:
            # Determine validation data path
            val_data_path = (
                getattr(config, "eval_data_path", None)
                or getattr(config, "val_data_path", None)
                or config.data_path
            )

            if self.config.i2v:
                val_dataset = MultiVideoConcatDataset(
                    data_dir=val_data_path,
                    video_size=(frame_raw_height, frame_raw_width),
                    total_frames=total_frames,
                    deterministic=True,
                    num_frame_per_block=num_frame_per_block,
                    temporal_compression_ratio=temporal_compression_ratio,
                    target_fps=self.fps,
                    allow_padding=allow_padding,
                    min_latent_frames=min_latent_frames,
                    single_video_only=single_video_only,
                    independent_first_frame=getattr(config, "independent_first_frame", False),
                    return_image=True,
                    max_chunks_per_shot=chunks_per_shot,
                    scene_cut_prefix=scene_cut_prefix,
                    sample_warning_seconds=dataset_sample_warning_seconds,
                    sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
                )
                val_collate_fn = multi_video_collate_fn
            else:
                val_dataset = MultiTextConcatDataset(
                    data_path=val_data_path,
                    num_blocks=num_blocks,
                    chunks_per_shot=chunks_per_shot,
                    scene_cut_prefix=scene_cut_prefix,
                    deterministic=True,
                )
                val_collate_fn = eval_collate_fn

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=section_get(config, "evaluation", "val_batch_size", getattr(config, "val_batch_size", 1)),
                sampler=sampler,
                num_workers=0,
                collate_fn=val_collate_fn,
            )

            # Take the first batch as fixed visualization batch
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
            except StopIteration:
                self.fixed_vis_batch = None
            
            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings
            # ----------------------------------------------------------------------------------------------------------
            # List of video lengths to visualize, e.g. [8, 16, 32]
            self.vis_video_lengths = configured_vis_lengths
            for _vl in self.vis_video_lengths:
                assert _vl <= latent_frames_for_dataset, (
                    f"vis_video_lengths entry {_vl} exceeds "
                    f"image_or_video_shape[1] ({latent_frames_for_dataset}), "
                    f"the dataset will not provide enough prompts for visualization."
                )

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()
            
        if not self.is_lora_enabled:
            # ================================= Standard (non-LoRA) model logic =================================
            checkpoint_path = None
            
            if auto_resume and self.output_path:
                # Auto resume: find latest checkpoint in logdir
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint
                    if self.is_main_process:
                        print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")
                else:
                    if self.is_main_process:
                        print("Auto resume: No checkpoint found in logdir, starting from scratch")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified, starting from scratch")
            else:
                if self.is_main_process:
                    print("Auto resume disabled, starting from scratch")
            
            if checkpoint_path is None:
                if getattr(config, "generator_ckpt", False):
                    # Explicit checkpoint path provided
                    checkpoint_path = config.generator_ckpt
                    if self.is_main_process:
                        print(f"Using explicit checkpoint: {checkpoint_path}")

            # Pre-load fake_score from a separate checkpoint if specified.
            # This will be overwritten if checkpoint_path also contains "critic".
            fake_score_ckpt = getattr(config, "fake_score_ckpt", None)
            if fake_score_ckpt:
                if self.is_main_process:
                    print(f"Loading fake_score from {fake_score_ckpt}")
                fake_ckpt = torch.load(fake_score_ckpt, map_location="cpu")
                if "critic" in fake_ckpt:
                    self.model.fake_score.load_state_dict(fake_ckpt["critic"], strict=True)
                elif "fake_score" in fake_ckpt:
                    self.model.fake_score.load_state_dict(fake_ckpt["fake_score"], strict=True)
                elif "model" in fake_ckpt:
                    self.model.fake_score.load_state_dict(fake_ckpt["model"], strict=True)
                else:
                    if self.is_main_process:
                        print(f"No recognized key in {fake_score_ckpt}, treating as raw state_dict")
                    self.model.fake_score.load_state_dict(fake_ckpt, strict=True)
                del fake_ckpt
                gc.collect()
                if self.is_main_process:
                    print(f"Successfully loaded fake_score from {fake_score_ckpt}")

            if checkpoint_path:
                if self.is_main_process:
                    print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Load generator
                if "generator" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
                    del checkpoint["generator"]
                elif "model" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["model"], strict=True)
                    del checkpoint["model"]
                else:
                    if self.is_main_process:
                        print(f"No 'generator'/'model' key found in {checkpoint_path}, treating as raw state_dict")
                    self.model.generator.load_state_dict(checkpoint, strict=True)
                
                # Load critic
                if "critic" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {checkpoint_path}")
                    self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
                    del checkpoint["critic"]
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found.")
                
                # Load EMA
                if "generator_ema" in checkpoint and self.generator_ema is not None:
                    if self.is_main_process:
                        print(f"Loading pretrained EMA from {checkpoint_path}")
                    self.generator_ema.load_state_dict(checkpoint["generator_ema"])
                    del checkpoint["generator_ema"]
                else:
                    if self.is_main_process:
                        print("Warning: EMA checkpoint not found or EMA not initialized.")

                gc.collect()
                
                # For auto resume, always resume full training state
                # Load optimizers
                if "generator_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming generator optimizer...")
                    gen_osd = FSDP.optim_state_dict_to_load(
                        self.model.generator,              # FSDP root module
                        self.generator_optimizer,          # newly created optimizer
                        checkpoint["generator_optimizer"]  # optimizer state dict at save time
                    )
                    del checkpoint["generator_optimizer"]
                    self.generator_optimizer.load_state_dict(gen_osd)
                    del gen_osd
                else:
                    if self.is_main_process:
                        print("Warning: Generator optimizer checkpoint not found.")
                
                if "critic_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming critic optimizer...")
                    crit_osd = FSDP.optim_state_dict_to_load(
                        self.model.fake_score,
                        self.critic_optimizer,
                        checkpoint["critic_optimizer"]
                    )
                    del checkpoint["critic_optimizer"]
                    self.critic_optimizer.load_state_dict(crit_osd)
                    del crit_osd
                else:
                    if self.is_main_process:
                        print("Warning: Critic optimizer checkpoint not found.")
                
                # Load training step
                if "step" in checkpoint:
                    self.step = checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming from step {self.step}")
                else:
                    if self.is_main_process:
                        print("Warning: Step not found in checkpoint, starting from step 0.")
                del checkpoint
                gc.collect()

            # Load real_score from a separate checkpoint (independent of auto_resume)
            real_score_ckpt = getattr(config, "real_score_ckpt", None)
            if real_score_ckpt and not (self.real_score_quant and self.real_score_quant_materialize):
                if self.is_main_process:
                    print(f"Loading real_score from {real_score_ckpt}")
                real_ckpt = torch.load(real_score_ckpt, map_location="cpu")
                if "generator" in real_ckpt:
                    self.model.real_score.load_state_dict(real_ckpt["generator"], strict=True)
                elif "critic" in real_ckpt:
                    self.model.real_score.load_state_dict(real_ckpt["critic"], strict=True)
                elif "model" in real_ckpt:
                    self.model.real_score.load_state_dict(real_ckpt["model"], strict=True)
                else:
                    if self.is_main_process:
                        print(f"No recognized key in {real_score_ckpt}, treating as raw state_dict")
                    self.model.real_score.load_state_dict(real_ckpt, strict=True)
                del real_ckpt
                gc.collect()
                if self.is_main_process:
                    print(f"Successfully loaded real_score from {real_score_ckpt}")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # Note: This should be done after potential resume to avoid accidentally deleting resumed EMA
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")

    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _materialize_quantized_model_before_fsdp(
        self,
        model,
        model_label,
        cache_transposed_weights=False,
    ):
        """Materialize NVFP4 weights before FSDP wraps quantized modules."""
        from utils.quant import _materialize_quantized_weights_for_inference

        current_rank = dist.get_rank()
        target_device = torch.device("cuda", torch.cuda.current_device())
        if self.is_main_process:
            print(f"[NVFP4] Materializing {model_label} sequentially before FSDP")

        for materialize_rank in range(self.world_size):
            if current_rank == materialize_rank:
                first_param = next(model.parameters(), None)
                model_device = first_param.device if first_param is not None else target_device
                if model_device != target_device:
                    model.to(target_device)
                mat_modules, master_bytes, quant_bytes = _materialize_quantized_weights_for_inference(
                    model,
                    target_device=target_device,
                    cache_transposed_weights=cache_transposed_weights,
                )
                if self.is_main_process:
                    print(
                        f"[NVFP4] {model_label} materialized: {len(mat_modules)} modules, "
                        f"master_weight={master_bytes / (1024**3):.3f} GiB freed, "
                        f"quantized_weight={quant_bytes / (1024**3):.3f} GiB"
                    )
                gc.collect()
            dist.barrier()

    def find_latest_checkpoint(self, logdir):
        """Find the latest checkpoint in the logdir."""
        if not os.path.exists(logdir):
            return None
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_path = os.path.join(logdir, item, "model.pt")
                    if os.path.exists(checkpoint_path):
                        checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Sort by step number and return the latest one
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_step, latest_path = checkpoint_dirs[-1]
        return latest_path

    def get_all_checkpoints(self, logdir):
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil
            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")

    def save(self):
        print("Start gathering distributed model states...")

        if self.is_lora_enabled:
            gen_lora_sd = self._gather_lora_state_dict(
                self.model.generator.model)
            crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)

            state_dict = {
                "generator_lora": gen_lora_sd,
                "critic_lora": crit_lora_sd,
                "step": self.step,
            }
        else:
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                generator_state_dict  = self.model.generator.state_dict()
                generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                                self.generator_optimizer)

            if dist.is_initialized():
                dist.barrier()

            with FSDP.state_dict_type(
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                critic_state_dict  = self.model.fake_score.state_dict()
                critic_opim_state_dict = FSDP.optim_state_dict(self.model.fake_score,
                                                self.critic_optimizer)

            if self.config.ema_start_step < self.step and self.generator_ema is not None:
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_ema": self.generator_ema.state_dict(),
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }
            else:
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }

        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)
            
            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        # Keep all ranks in sync so non-rank0 workers don't kick off the next
        # training iteration (and trigger NCCL watchdog timeouts) while rank0
        # is still writing the checkpoint to disk.
        if dist.is_initialized():
            dist.barrier()

        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        if getattr(self.config, "uniform_prompt", False):
            text_prompts = [[sample[0]] * len(sample) for sample in text_prompts]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 1.5: Prepare clean_latent and initial_latent for off-policy mode
        clean_latent = None
        initial_latent = None
        if not self.use_backward_simulation:
            if not getattr(self.config, "load_raw_video", False):
                clean_latent = batch["ode_latent"][:, -1].to(
                    device=self.device, dtype=self.dtype)
            else:
                frames = batch["frames"].to(
                    device=self.device, dtype=self.dtype)
                with torch.no_grad():
                    clean_latent = self.model.vae.encode_to_latent(frames).to(
                        device=self.device, dtype=self.dtype)
            if getattr(self.config, "i2v", False):
                initial_latent = clean_latent[:, 0:1]
        elif getattr(self.config, "i2v", False):
            image = batch.get("image", None)
            if image is None:
                raise ValueError("DMD i2v backward-simulation requires batch['image'].")
            image = image.to(device=self.device, dtype=self.dtype)
            if image.ndim == 4:
                image = image.unsqueeze(2)
            elif image.ndim != 5:
                raise ValueError(
                    f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}"
                )
            with torch.no_grad():
                initial_latent = self.model.vae.encode_to_latent(image).to(
                    device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # MultiVideoConcatDataset returns List[List[str]], flatten to List[str]
            text_prompts_flat = [p for sublist in text_prompts for p in sublist]
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts_flat)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        use_scene_cut_mask = (
            section_get(self.config, "inference", "multi_shot_sink", False)
            or section_get(
                self.config,
                "inference",
                "multi_shot_rope_offset",
                0.0,
            ) != 0.0
        )
        if use_scene_cut_mask:
            _prefix = getattr(self.config, "scene_cut_prefix", DEFAULT_SCENE_CUT_PREFIX)
            conditional_dict["scene_cut_mask"] = [
                p.startswith(_prefix) for p in text_prompts[0]
            ]

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})

            return generator_log_dict
        else:
            generator_log_dict = {}

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=initial_latent
        )

        # Scale loss for gradient accumulation and backward
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})

        return critic_log_dict

    def generate_video(self, pipeline, num_frames, prompts, image=None, latents_only=False):
        batch_size = len(prompts)
        if image is not None:
            image = image.to(device=self.device, dtype=self.dtype)
            if image.ndim == 4:
                image = image.unsqueeze(2)
            elif image.ndim != 5:
                raise ValueError(f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}")

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=self.device, dtype=self.dtype)
            if initial_latent.shape[0] != batch_size:
                initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            num_noise_frames = num_frames
            if num_noise_frames <= initial_latent.shape[1]:
                raise ValueError(
                    f"num_frames must exceed the i2v conditioning frames; "
                    f"got {num_frames} and {initial_latent.shape[1]}"
                )
            sampled_noise = torch.randn(
                [batch_size, num_noise_frames, self.config.image_or_video_shape[2], self.config.image_or_video_shape[3], self.config.image_or_video_shape[4]],
                device=self.device,
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, self.config.image_or_video_shape[2], self.config.image_or_video_shape[3], self.config.image_or_video_shape[4]],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            kwargs = dict(noise=sampled_noise, text_prompts=prompts)
            if initial_latent is not None:
                kwargs["initial_latent"] = initial_latent
            kwargs["return_latents"] = latents_only
            result = pipeline.inference(**kwargs)
            if latents_only:
                return result
            video = result
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'model') and hasattr(pipeline.vae.model, 'clear_cache'):
            pipeline.vae.model.clear_cache()
        return current_video
    
    def train(self):
        start_step = self.step
        try:
            while True:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

                if TRAIN_GENERATOR:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                # Whole-cycle gradient accumulation loop
                accumulated_generator_logs = []
                accumulated_critic_logs = []

                for accumulation_step in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)

                    # Train generator (if needed)
                    if TRAIN_GENERATOR:
                        extra_gen = self.fwdbwd_one_step(batch, True)
                        accumulated_generator_logs.append(extra_gen)

                    # Train critic
                    extra_crit = self.fwdbwd_one_step(batch, False)
                    accumulated_critic_logs.append(extra_crit)

                # Compute grad norm and update parameters
                if TRAIN_GENERATOR:
                    generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                    generator_log_dict = merge_dict_list(accumulated_generator_logs)
                    generator_log_dict["generator_grad_norm"] = generator_grad_norm

                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)
                else:
                    generator_log_dict = {}

                critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                critic_log_dict = merge_dict_list(accumulated_critic_logs)
                critic_log_dict["critic_grad_norm"] = critic_grad_norm

                self.critic_optimizer.step()

                # Increment the step since we finished gradient update
                self.step += 1

                # Create EMA params (if not already created)
                if (self.step >= self.config.ema_start_step) and \
                        (self.generator_ema is None) and (self.config.ema_weight > 0):
                    if not self.is_lora_enabled:
                        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                        if self.is_main_process:
                            print(f"EMA created at step {self.step} with weight {self.config.ema_weight}")
                    else:
                        if self.is_main_process:
                            print(f"EMA creation skipped at step {self.step} (disabled in LoRA mode)")

                # Save the model
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
                        wandb_loss_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                                "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                            }
                        )


                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )
                    if not self.disable_wandb:
                        wandb.log(wandb_loss_dict, step=self.step)

                if self.step % self.config.gc_interval == 0:
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main_process:
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")
                    else:
                        print(f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")

                # ---------------------------------------- Visualization ---------------------------------------------------

                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):
                    self._visualize()
                
                if self.step >= self.config.max_iters:
                    break
        
        except Exception as e:
            print(f"[ERROR] [Rank {dist.get_rank()}] Training crashed at step {self.step} with exception: {e}")
            print(f"[ERROR] [Rank {dist.get_rank()}] Exception traceback:", flush=True)
            import traceback
            traceback.print_exc()

    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # Define the specific modules we want to apply LoRA to
        all_causal = getattr(self.config, 'all_causal', False)
        generator_is_causal = getattr(self.config, 'generator_is_causal', True)
        if model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock'] if generator_is_causal else ['WanAttentionBlock']
        elif model_name == 'fake_score':
            adapter_target_modules = ['CausalWanAttentionBlock'] if all_causal else ['WanAttentionBlock']
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)
        
        target_linear_modules = list(target_linear_modules)
        
        if self.is_main_process:
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
            if getattr(self.lora_config, 'verbose', False):
                for module_name in sorted(target_linear_modules):
                    print(f"  - {module_name}")
        
        # Create LoRA config
        adapter_type = self.lora_config.get('type', 'lora')
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=self.lora_config.get('rank', 16),
                lora_alpha=self.lora_config.get('alpha', None) or self.lora_config.get('rank', 16),
                lora_dropout=self.lora_config.get('dropout', 0.0),
                target_modules=target_linear_modules,
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        
        # Apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        if self.is_main_process:
            print('peft_config', peft_config)
            lora_model.print_trainable_parameters()

        return lora_model


    def _gather_lora_state_dict(self, lora_model):
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        with FSDP.state_dict_type(
            lora_model,                       # lora_model contains nested FSDP submodules
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = lora_model.state_dict()
        return get_peft_model_state_dict(lora_model, state_dict=full)
    
    # --------------------------------------------------------------------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------------------------------------------------------------------

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""

        generator_is_causal = getattr(self.config, "generator_is_causal", True)

        if not generator_is_causal:
            # Bidirectional generator: no pipeline object needed,
            # _visualize_bidirectional handles the loop directly.
            self.vis_pipeline = "bidirectional"
        else:
            from copy import deepcopy
            vis_config = deepcopy(self.config)
            if "guidance_scale" not in getattr(vis_config, "inference", {}):
                vis_config.guidance_scale = 1.0
            if section_get(self.config, "inference", "sampling_steps", None) is None:
                vis_config.sampling_steps = 50
            self.vis_pipeline = CausalDiffusionInferencePipeline(
                args=vis_config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)

        # Visualization output directory (default: <logdir>/vis)
        self.vis_output_dir = os.path.join(self.output_path, "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if section_get(self.config, "evaluation", "use_ema", getattr(self.config, "vis_ema", False)):
            raise NotImplementedError("Visualization with EMA is not implemented")

    @torch.no_grad()
    def _generate_bidirectional(self, num_frames, prompts):
        """Full-sequence bidirectional multi-step denoising for visualization."""
        from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        batch_size = len(prompts)
        # Flatten prompts (List[List[str]] → List[str], take first per sample)
        text_prompts_flat = [p[0] if isinstance(p, list) else p for p in prompts]

        conditional_dict = self.model.text_encoder(text_prompts=text_prompts_flat)

        noise = torch.randn(
            [batch_size, num_frames,
             self.config.image_or_video_shape[2],
             self.config.image_or_video_shape[3],
             self.config.image_or_video_shape[4]],
            device=self.device, dtype=self.dtype)

        sampling_steps = section_get(self.config, "inference", "sampling_steps", getattr(self.config, "sampling_steps", 50))
        scheduler = self.model.generator.get_scheduler()
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=scheduler.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(sampling_steps, device=self.device,
                                       shift=scheduler.shift)

        latents = noise
        for t in sample_scheduler.timesteps:
            timestep = t * torch.ones(
                [batch_size, num_frames], device=self.device, dtype=torch.float32)
            flow_pred, _ = self.model.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
            )
            latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

        return latents

    def _visualize(self):
        """Generate validation samples to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return False

        # FSDP forward includes communication, so every rank must enter
        # visualization together; running rank 0 alone would hang.

        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return False

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        prompts = batch["prompts"]

        image = None
        if self.config.i2v and ("image" in batch):
            image = batch["image"]

        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating latents in LoRA mode (step {self.step})")

        for vid_len in self.vis_video_lengths:
            print(f"Generating validation samples of length {vid_len}")
            if self.vis_pipeline == "bidirectional":
                samples = self._generate_bidirectional(vid_len, prompts)
                if not self.save_vis_latents_only:
                    samples = self.model.vae.decode_to_pixel(samples)
                    samples = (samples * 0.5 + 0.5).clamp(0, 1)
                    samples = samples.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
            else:
                samples = self.generate_video(
                    self.vis_pipeline,
                    vid_len,
                    prompts,
                    image=image,
                    latents_only=self.save_vis_latents_only,
                )

            for idx in range(samples.shape[0]):
                if self.save_vis_latents_only:
                    sample_name = f"latents_step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.pt"
                    out_path = os.path.join(step_vis_dir, sample_name)
                    torch.save(samples[idx].cpu(), out_path)
                else:
                    sample_name = f"video_step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                    out_path = os.path.join(step_vis_dir, sample_name)
                    write_video(out_path, torch.as_tensor(samples[idx]).to(torch.uint8), fps=24)

            del samples
            torch.cuda.empty_cache()

        # Save prompts for reference
        prompt_path = os.path.join(
            step_vis_dir,
            f"prompts_rank_{dist.get_rank()}.txt",
        )
        with open(prompt_path, "w") as f:
            for i, p in enumerate(prompts):
                f.write(f"[sample {i}] {p}\n")

        # Release KV / cross-attention caches allocated during inference to prevent OOM
        # when training resumes. These caches can consume ~20+ GB of GPU memory.
        if hasattr(self.vis_pipeline, 'clear_cache'):
            self.vis_pipeline.clear_cache()

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Synchronize all ranks so that a crashed rank is detected immediately
        # rather than causing a 10-minute NCCL timeout on the next training collective.
        dist.barrier()

        return True
