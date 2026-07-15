# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import types

from model import CausalDiffusion
from wan_5b.distributed.sp_training import SequenceParallelHelper
from utils.dataset import MultiVideoConcatDataset, MultiTextConcatDataset, cycle, multi_video_collate_fn, eval_collate_fn
from utils.config import section_get, wan_default_config
from utils.misc import set_seed
from utils.sampler import build_training_sampler
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from torchvision.io import write_video
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, launch_distributed_job, FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)

def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    """
    Save prompts for one generated video to a txt file.
    Consecutive identical prompts are merged, e.g.:
        [0] a, [1] a, [2] b  =>  [0,1] a\n[2] b\n
    """
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return

            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                p = prompts_for_sample[seg_idx]
                if p == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    indices_str = ",".join(str(i) for i in current_indices)
                    f.write(f"[{indices_str}] {current_prompt}\n")
                    current_prompt = p
                    current_indices = [seg_idx]
            # flush the last run
            indices_str = ",".join(str(i) for i in current_indices)
            f.write(f"[{indices_str}] {current_prompt}\n")
    except Exception as e:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {e}")


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            if getattr(config, "wandb_key", None):
                wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        auto_resume = getattr(config, "auto_resume", True)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

        # Sequence Parallel is supported only for the 5B model; world_size must
        # equal sp_size * dp_size.
        self.sequence_parallel_size = getattr(config, "sequence_parallel_size", 1)
        world_size = dist.get_world_size()
        self.data_parallel_size = world_size // self.sequence_parallel_size if self.sequence_parallel_size > 1 else world_size
        self.sp_group = None
        self.dp_group = None

        if self.is_main_process and self.gradient_accumulation_steps > 1:
            eff_batch = config.batch_size * self.gradient_accumulation_steps * self.data_parallel_size
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}, effective batch size: {eff_batch}")

        if self.sequence_parallel_size > 1:
            assert config.model_kwargs.model_name == "Wan2.2-TI2V-5B", (
                f"sequence_parallel_size is only supported for Wan2.2-TI2V-5B model, but got {config.model_kwargs.model_name}"
            )
            assert world_size % self.sequence_parallel_size == 0, (
                f"world_size ({world_size}) must be divisible by sequence_parallel_size ({self.sequence_parallel_size})"
            )
            from wan_5b.distributed.sp_training import (
                validate_sequence_parallel_training_config,
            )
            validate_sequence_parallel_training_config(
                config,
                self.sequence_parallel_size,
                config.num_frame_per_block,
            )
            # Create SP process groups: each DP group contains sp_size ranks,
            # and all_to_all runs only within that group.
            from wan_5b.distributed.sp_training import (
                set_data_parallel_group,
                set_sequence_parallel_group,
            )
            sp_size = self.sequence_parallel_size
            dp_size = self.data_parallel_size
            sp_groups = []
            for g in range(dp_size):
                ranks_g = list(range(g * sp_size, (g + 1) * sp_size))
                sp_groups.append(dist.new_group(ranks=ranks_g))
            self.sp_group = sp_groups[global_rank // sp_size]
            set_sequence_parallel_group(self.sp_group)

            # Also create DP groups: ranks with the same SP rank across DP
            # replicas own the same sequence chunk. For sp_rank=k, the DP group
            # is [k, sp+k, 2*sp+k, ..., (dp-1)*sp+k]. This lets warmup gather
            # different batches of errors for the same block efficiently.
            dp_groups = []
            for k in range(sp_size):
                ranks_k = [g * sp_size + k for g in range(dp_size)]
                dp_groups.append(dist.new_group(ranks=ranks_k))
            self.dp_group = dp_groups[global_rank % sp_size]
            set_data_parallel_group(self.dp_group)
            if self.is_main_process:
                print(f"[SP] Sequence Parallel enabled, sp_size={sp_size}, dp_size={dp_size}, world_size={world_size}")

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.sp_helper = SequenceParallelHelper(self)

        # 2D mode only: print which GLOBAL block-position slice this rank is
        # responsible for. The LAST SP rank carries the most error-accumulated
        # tail blocks, useful when debugging position-bucketed error recycling.
        if self.model.error_buffer is not None and self.model.er_num_blocks > 0:
            lo = self.model.er_block_offset
            hi = lo + self.model.er_num_blocks
            global_rank_id = dist.get_rank()
            sp_rk = global_rank_id % max(self.sequence_parallel_size, 1)
            print(
                f"[ErrorBuffer] rank={global_rank_id} sp_rank={sp_rk} "
                f"covers GLOBAL blocks [{lo},{hi}) ({self.model.er_num_blocks} local blocks)"
            )

        # Bind the SP forward path before FSDP wrapping.
        model_name = getattr(getattr(config, "model_kwargs", None), "model_name", "") or ""
        if self.sequence_parallel_size > 1 and "Wan2.2-TI2V-5B" in model_name:
            from wan_5b.distributed.sequence_parallel import (
                sp_dit_causal_forward_train,
                sp_causal_attn_forward,
            )
            model = self.model.generator.model
            # Use the SP forward implementation in the training path.
            model._forward_train = types.MethodType(sp_dit_causal_forward_train, model)

            # Keep the original self_attn.forward so inference can temporarily
            # disable SP.
            self._sp_attn_blocks = []
            for block in model.blocks:
                sa = block.self_attn
                if not hasattr(sa, "_orig_forward"):
                    sa._orig_forward = sa.forward
                sa.forward = types.MethodType(sp_causal_attn_forward, sa)
                self._sp_attn_blocks.append(sa)

            if self.is_main_process:
                print("[SP] sp_dit_causal_forward_train and sp_causal_attn_forward are enabled")
                print("[SP] natural TF layout is the default training layout")
                if getattr(config, "load_raw_video", False):
                    print(f"[SP-VAE] chunk-halo VAE enabled, halo_latents={self.sp_helper.vae_halo_latents}")

        # ================================= NVFP4 Quantized Training =================================
        self.model_quant = getattr(config, "model_quant", False)
        if self.model_quant:
            from utils.quant import ModelQuantizationConfig, quantize_model_with_filter

            quant_cfg = ModelQuantizationConfig(
                scale_rule=getattr(config, "model_quant_scale_rule", "static_6"),
                activation_scale_rule=getattr(config, "model_quant_activation_scale_rule", "static_6"),
                weight_scale_rule=getattr(config, "model_quant_weight_scale_rule", None),
                gradient_scale_rule=getattr(config, "model_quant_gradient_scale_rule", None),
                keep_master_weights=True,
                weight_scale_2d=True,
            )
            self.model.generator.model, matched_modules = quantize_model_with_filter(
                self.model.generator.model,
                quant_config=quant_cfg,
                filtered_modules=getattr(config, "model_quant_filtered_modules", None),
                use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
                cast_model_to_bf16=False,
                materialize_for_inference=False,
                verbose=self.is_main_process,
            )
            if self.is_main_process:
                from fouroversix.matmul.cutlass.backend import CUTLASSMatmulBackend

                print(f"[NVFP4] CUTLASS available: {CUTLASSMatmulBackend.is_available()}")
                print(
                    "[NVFP4] Quantized AR training enabled "
                    "(keep_master_weights=True, weight_scale_2d=True)"
                )
                print(f"[NVFP4] {len(matched_modules)} modules excluded from quantization")

        # ================================= Load model weights (before FSDP) =================================
        # Load model weights before FSDP wrapping, while keys still match the
        # raw nn.Module. Optimizer, EMA, and step state are restored after FSDP
        # and the related objects are created, so keep raw_state.
        #
        # Priority: auto_resume from logdir > generator_ckpt for a
        # cold start > random initialization. This allows configs to keep
        # generator_ckpt set while interrupted training still resumes from the
        # latest step. The style mirrors trainer/distillation.py.
        raw_state = None

        checkpoint_path = None

        if auto_resume and self.output_path:
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

        if checkpoint_path is None and getattr(config, "generator_ckpt", False):
            checkpoint_path = config.generator_ckpt
            if self.is_main_process:
                print(f"Using explicit checkpoint: {checkpoint_path}")

        if checkpoint_path:
            if self.is_main_process:
                print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

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

            gc.collect()

            raw_state = checkpoint
            if "step" in raw_state:
                self.step = raw_state["step"]
                if self.is_main_process:
                    print(f"Resuming from step {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")

        # ================================= FSDP Wrap =================================
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

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

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        frame_raw_height = list(config.image_or_video_shape)[3] * wan_default_config[config.model_kwargs.model_name]["spatial_compression_ratio"]
        frame_raw_width = list(config.image_or_video_shape)[4] * wan_default_config[config.model_kwargs.model_name]["spatial_compression_ratio"]
        total_frames = (list(config.image_or_video_shape)[1] - 1) * wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"] + 1
        num_frame_per_block = config.num_frame_per_block
        self.fps = wan_default_config[config.model_kwargs.model_name].get("fps", 16)

        allow_padding = getattr(config, "allow_padding", False)
        min_latent_frames = getattr(config, "min_latent_frames", 0)
        single_video_only = getattr(config, "uniform_prompt", False)
        max_chunks_per_shot = getattr(config, "max_chunks_per_shot", 0)
        dataset_sample_warning_seconds = getattr(config, "dataset_sample_warning_seconds", 60.0)
        dataset_sample_warning_interval_seconds = getattr(
            config, "dataset_sample_warning_interval_seconds", 60.0
        )
        dataset = MultiVideoConcatDataset(
            data_dir=config.data_path,
            video_size=(frame_raw_height, frame_raw_width),
            total_frames=total_frames,
            deterministic=False,
            num_frame_per_block=num_frame_per_block,
            temporal_compression_ratio=wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"],
            target_fps=self.fps,
            allow_padding=allow_padding,
            min_latent_frames=min_latent_frames,
            single_video_only=single_video_only,
            independent_first_frame=getattr(config, "independent_first_frame", False),
            return_image=getattr(config, "i2v", False),
            max_chunks_per_shot=max_chunks_per_shot,
            sample_warning_seconds=dataset_sample_warning_seconds,
            sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
        )
        if allow_padding and self.is_main_process:
            print(f"[Padding] Variable-length training enabled: short videos will be padded with loss masking"
                  f" (min_latent_frames={min_latent_frames})")
        if single_video_only and self.is_main_process:
            print(f"[uniform_prompt] single_video_only enabled: each sample uses one video only")
        # SP ranks in the same SP group need the same batch because they shard
        # the sequence dimension. Use dp_rank for data parallel sampling.
        if self.sequence_parallel_size > 1:
            dp_rank = global_rank // self.sequence_parallel_size
            sampler = build_training_sampler(
                dataset,
                seed=config.seed,
                rank=dp_rank, num_replicas=self.data_parallel_size,
            )
        else:
            sampler = build_training_sampler(
                dataset,
                seed=config.seed,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=2,
            prefetch_factor=1,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=multi_video_collate_fn,
        )

        # Eval dataloader: batch size defaults to 1 to keep validation memory predictable.
        eval_data_path = getattr(config, "eval_data_path", config.data_path)
        inference_num_frames = section_get(config, "evaluation", "num_frames", getattr(config, "inference_num_frames", 0))
        if isinstance(inference_num_frames, (list, tuple)):
            inference_num_frames = inference_num_frames[0] if len(inference_num_frames) > 0 else 0
        eval_total_frames = (
            (inference_num_frames - 1) * wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"] + 1
            if inference_num_frames > 0 else total_frames
        )
        temporal_compression_ratio = wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"]
        first_chunk_frames = 1 + (num_frame_per_block - 1) * temporal_compression_ratio
        subsequent_chunk_frames = num_frame_per_block * temporal_compression_ratio
        num_blocks = 1 + (eval_total_frames - first_chunk_frames) // subsequent_chunk_frames
        chunks_per_shot = getattr(config, "chunks_per_shot", 0)
        scene_cut_prefix = getattr(config, "scene_cut_prefix", "The scene transitions. ")
        if getattr(config, "i2v", False):
            eval_dataset = MultiVideoConcatDataset(
                data_dir=eval_data_path,
                video_size=(frame_raw_height, frame_raw_width),
                total_frames=eval_total_frames,
                deterministic=True,
                num_frame_per_block=num_frame_per_block,
                temporal_compression_ratio=temporal_compression_ratio,
                target_fps=self.fps,
                allow_padding=allow_padding,
                min_latent_frames=min_latent_frames,
                single_video_only=single_video_only,
                independent_first_frame=getattr(config, "independent_first_frame", False),
                return_image=True,
                max_chunks_per_shot=max_chunks_per_shot,
                scene_cut_prefix=scene_cut_prefix,
                sample_warning_seconds=dataset_sample_warning_seconds,
                sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
            )
            eval_collate = multi_video_collate_fn
        else:
            eval_dataset = MultiTextConcatDataset(
                data_path=eval_data_path,
                num_blocks=num_blocks,
                chunks_per_shot=chunks_per_shot,
                scene_cut_prefix=scene_cut_prefix,
                deterministic=True,
            )
            eval_collate = eval_collate_fn
        if dist.get_rank() == 0:
            print(f"Using {eval_dataset.__class__.__name__} for eval: {eval_data_path}, num_blocks={num_blocks}")
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=False
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=section_get(config, "evaluation", "val_batch_size", 1),
            sampler=eval_sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=eval_collate,
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            print("EVAL DATASET SIZE %d" % len(eval_dataset))

        self.dataloader = cycle(dataloader)
        self.eval_dataloader = eval_dataloader

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0) and (self.step >= config.ema_start_step):
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load optimizer and EMA from checkpoint
        #    Model weights were loaded before FSDP wrapping; restore only
        #    optimizer and EMA state that depend on FSDP here.

        if raw_state is not None:
            if "generator_ema" in raw_state and self.generator_ema is not None:
                self.generator_ema.load_state_dict(raw_state["generator_ema"])
                if self.is_main_process:
                    print("Resuming generator EMA...")
            else:
                if self.is_main_process:
                    print("Warning: Generator EMA checkpoint not found.")

            if "generator_optimizer" in raw_state:
                gen_osd = FSDP.optim_state_dict_to_load(
                    self.model.generator,
                    self.generator_optimizer,
                    raw_state["generator_optimizer"],
                )
                del raw_state["generator_optimizer"]
                self.generator_optimizer.load_state_dict(gen_osd)
                del gen_osd
                if self.is_main_process:
                    print("Resuming generator optimizer...")
            else:
                if self.is_main_process:
                    print("Warning: Generator optimizer checkpoint not found.")

            del raw_state
            gc.collect()

        ##############################################################################################################

        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

        # Resume error buffer from checkpoint.
        #   Try ``*_sp{sp_rank}.pt`` first, fall back to ``*.pt`` (legacy).
        if self.model.error_buffer is not None and auto_resume:
            ckpt_dir = self.find_latest_checkpoint(self.output_path)
            if ckpt_dir is not None:
                ckpt_root = os.path.dirname(ckpt_dir)
                sp_size_ = max(self.sequence_parallel_size, 1)
                global_rank = dist.get_rank() if dist.is_initialized() else 0
                sp_rank = global_rank % sp_size_

                def _resolve_buf_file(stem):
                    if sp_size_ > 1:
                        p = os.path.join(ckpt_root, f"{stem}_sp{sp_rank}.pt")
                        if os.path.exists(p):
                            return p
                    p = os.path.join(ckpt_root, f"{stem}.pt")
                    return p if os.path.exists(p) else None

                for stem, buffer in [("error_buffer", self.model.error_buffer),
                                     ("noise_error_buffer", self.model.noise_error_buffer)]:
                    if buffer is None:
                        continue
                    bf = _resolve_buf_file(stem)
                    if bf is not None:
                        bf_state = torch.load(bf, map_location="cpu")
                        buffer.load_state_dict(bf_state)
                        del bf_state
                        s = buffer.stats()
                        rng = s.get('global_block_range', '')
                        shard = s.get('shard', '')
                        print(f"[{stem}] rank={global_rank} Resumed from "
                              f"{os.path.basename(bf)}: {s['total_entries']} entries, "
                              f"{s['filled_buckets']} buckets, "
                              f"total_added={s['total_added']} {rng} {shard}".rstrip())
                    elif self.is_main_process:
                        print(f"[{stem}] No saved buffer found, starting fresh.")

    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

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

        # Release large inference caches before saving when possible.
        if hasattr(self.model, "inference_pipeline") and self.model.inference_pipeline is not None:
            clear_fn = getattr(self.model.inference_pipeline, "clear_cache", None)
            if clear_fn is not None:
                try:
                    clear_fn()
                except Exception as e:
                    print(f"Warning: failed to clear inference cache before save: {e}")
            # Drop the inference pipeline reference so GC / empty_cache can
            # reclaim memory.
            self.model.inference_pipeline = None
            torch.cuda.empty_cache()
        
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            generator_state_dict  = self.model.generator.state_dict()
            generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                            self.generator_optimizer)

        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
                "generator_optimizer": generator_opim_state_dict,
                "step": self.step,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "generator_optimizer": generator_opim_state_dict,
                "step": self.step,
            }

        checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)

        # Save error buffer — unified per-sp_rank pattern:
        #   Each SP rank owns a different t-bucket shard (and different
        #   positions in 2D mode).  The first DP rank in each SP group
        #   writes ``error_buffer_sp{sp_rank}.pt``.
        #   Fallback (sp_size<=1): main_process writes ``error_buffer.pt``.
        if self.model.error_buffer is not None:
            sp_size_ = max(self.sequence_parallel_size, 1)
            _global_rank = dist.get_rank() if dist.is_initialized() else 0
            _sp_rank = _global_rank % sp_size_
            _is_first_dp = (_global_rank // sp_size_) == 0

            if dist.is_initialized():
                dist.barrier()

            should_save = _is_first_dp if sp_size_ > 1 else self.is_main_process
            if should_save:
                for stem, buffer in [("error_buffer", self.model.error_buffer),
                                     ("noise_error_buffer", self.model.noise_error_buffer)]:
                    if buffer is None:
                        continue
                    fname = f"{stem}_sp{_sp_rank}.pt" if sp_size_ > 1 else f"{stem}.pt"
                    fpath = os.path.join(checkpoint_dir, fname)
                    torch.save(buffer.state_dict(), fpath)
                    s = buffer.stats()
                    rng = s.get('global_block_range', '')
                    shard = s.get('shard', '')
                    print(f"[rank={_global_rank}] {stem} saved to {fname} "
                          f"({s['total_entries']} entries, {s['filled_buckets']} buckets) "
                          f"{rng} {shard}".rstrip())

        if self.is_main_process:
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

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        accumulation_steps = accumulation_steps or getattr(self, "gradient_accumulation_steps", 1)
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        clean_latent_is_sp_sharded = False
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1]
        else:  # encode raw video to latent
            (
                clean_latent,
                image_latent,
                clean_latent_is_sp_sharded,
            ) = self.sp_helper.encode_raw_video_latents(
                batch,
                batch_size=batch_size,
            )

        loss_mask = self.sp_helper.build_loss_mask(
            batch, clean_latent, clean_latent_is_sp_sharded
        )
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # turn text prompts: List[List[str]] into List[str]
            text_prompts_flat = [prompt for sublist in text_prompts for prompt in sublist]

            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts_flat)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 2.5: Sequence Parallel partitions sequence-owned tensors.
        if self.sequence_parallel_size > 1:
            clean_latent, conditional_dict, image_or_video_shape = (
                self.sp_helper.partition_training_inputs(
                    image_or_video_shape=image_or_video_shape,
                    clean_latent=clean_latent,
                    conditional_dict=conditional_dict,
                    clean_latent_is_sharded=clean_latent_is_sp_sharded,
                )
            )
            image_latent = self.sp_helper.local_i2v_initial_latent(image_latent)
        loss_mask, loss_mask_global_valid_count = self.sp_helper.partition_loss_mask(
            loss_mask,
            already_sharded=clean_latent_is_sp_sharded,
        )

        # Step 3: Train the generator
        gen_kwargs = dict(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            loss_mask=loss_mask,
            loss_mask_global_valid_count=loss_mask_global_valid_count,
            global_step=self.step,
        )
        generator_loss, log_dict = self.model.generator_loss(**gen_kwargs)
        if accumulation_step == 0:
            self.generator_optimizer.zero_grad(set_to_none=True)
        scaled_loss = generator_loss / accumulation_steps
        scaled_loss.backward()
        if accumulation_step == accumulation_steps - 1:
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm)

            self.generator_optimizer.step()
            self.step += 1
        else:
            generator_grad_norm = torch.tensor(0.0, device=self.device)

        # Run the remaining logic only after a full gradient-accumulation cycle.
        if accumulation_step != accumulation_steps - 1:
            return

        # Step 4: Update EMA (if enabled and after start step)
        if (self.step >= self.config.ema_start_step) and \
                (self.generator_ema is None) and \
                (getattr(self.config, "ema_weight", None) is not None) and \
                (self.config.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        # Update EMA after optimizer step
        if self.generator_ema is not None and self.step >= self.config.ema_start_step:
            self.generator_ema.update(self.model.generator)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Error buffer stats
        er_log_str = ""
        if "er_total_added" in log_dict:
            wandb_loss_dict["er_total_entries"] = log_dict["er_total_entries"]
            wandb_loss_dict["er_total_added"] = log_dict["er_total_added"]
            wandb_loss_dict["er_injected"] = int(log_dict["er_injected"])
            wandb_loss_dict["er_latent_injected"] = int(log_dict["er_latent_injected"])
            wandb_loss_dict["er_noise_injected"] = int(log_dict.get("er_noise_injected", False))
            wandb_loss_dict["er_noise_total_entries"] = log_dict.get("er_noise_total_entries", 0)
            ctx_flag = 'Y' if log_dict['er_injected'] else 'N'
            lat_flag = 'Y' if log_dict['er_latent_injected'] else 'N'
            noise_flag = 'Y' if log_dict.get('er_noise_injected', False) else 'N'
            er_log_str = (
                f", er_buf={log_dict['er_total_entries']}|"
                f"{log_dict.get('er_noise_total_entries', 0)} "
                f"({log_dict['er_filled_buckets']} buckets), "
                f"ctx={ctx_flag} lat={lat_flag} noise={noise_flag}"
            )

        # Step 5: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            print(
                f"[step {self.step:07d}] "
                f"generator_loss={wandb_loss_dict['generator_loss']:.6f}, "
                f"generator_grad_norm={wandb_loss_dict['generator_grad_norm']:.6f}"
                f"{er_log_str}"
            )

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def _set_sp_attn(self, enabled: bool):
        """
        Toggle SP self-attention between training and inference.
        This only applies to 5B runs with SP enabled.
        """
        if not hasattr(self, "_sp_attn_blocks"):
            return
        if self.sequence_parallel_size <= 1:
            return

        # Lazy import to avoid failures under non-5B configurations.
        try:
            from wan_5b.distributed.sequence_parallel import sp_causal_attn_forward
        except Exception:
            return

        for sa in self._sp_attn_blocks:
            if not hasattr(sa, "_orig_forward"):
                continue
            if enabled:
                sa.forward = types.MethodType(sp_causal_attn_forward, sa)
            else:
                sa.forward = sa._orig_forward

    @torch.no_grad()
    def _swap_ema_weights(self):
        """
        Bidirectionally swap model weights with EMA shadow weights.
        Calling this twice restores both the model and EMA to their original state.
        """
        with FSDP.summon_full_params(self.model.generator, writeback=True):
            for n, p in self.model.generator.module.named_parameters():
                cleaned_name = EMA_FSDP._clean_param_name(n)
                if cleaned_name in self.generator_ema.shadow:
                    ema_val = self.generator_ema.shadow[cleaned_name]
                    tmp = p.data.clone().float().cpu()
                    p.data.copy_(ema_val.to(dtype=p.dtype, device=p.device))
                    self.generator_ema.shadow[cleaned_name] = tmp

    def _run_evaluation_inference(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if self.model.inference_pipeline is None:
            self.model._initialize_inference_pipeline()

        out_dir = os.path.join(self.output_path, f"generated_video_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(out_dir, exist_ok=True)
        barrier()

        rank = dist.get_rank()
        vis_ema = section_get(self.config, "evaluation", "use_ema", getattr(self.config, "vis_ema", False))
        vis_ema = vis_ema and self.generator_ema is not None

        for eval_batch in self.eval_dataloader:
            eval_prompts = eval_batch["prompts"]
            eval_idx = eval_batch["idx"]
            eval_images = eval_batch.get("image", None)

            batch_size_eval = len(eval_prompts)
            for b in range(batch_size_eval):
                prompts_for_sample = eval_prompts[b]

                if self.is_main_process:
                    print(f"prompts_for_sample: {prompts_for_sample}")
                    print(len(prompts_for_sample))
                    print(prompts_for_sample[0][:60])

                sample_idx = (
                    eval_idx[b].item()
                    if hasattr(eval_idx, "shape")
                    else int(eval_idx[b])
                )

                save_latents_only = section_get(
                    self.config,
                    "evaluation",
                    "save_latents_only",
                    self.config.get("return_latents", False),
                    aliases=("return_latents", "save_latent_only"),
                )

                run_modes = [("", False)]
                if vis_ema:
                    run_modes.append(("_ema", True))

                for suffix, use_ema in run_modes:
                    generated_video = self.generate_video(
                        self.model.inference_pipeline,
                        [prompts_for_sample],
                        eval_images[b:b + 1] if eval_images is not None else None,
                        use_ema=use_ema,
                    )

                    if not save_latents_only:
                        video_path = os.path.join(
                            out_dir,
                            f"video{suffix}_rank{rank:02d}_idx{sample_idx:06d}.mp4",
                        )
                        write_video(video_path, generated_video[0], fps=self.fps)
                    else:
                        video_path = os.path.join(
                            out_dir,
                            f"latents{suffix}_rank{rank:02d}_idx{sample_idx:06d}.pt",
                        )
                        torch.save(generated_video[0], video_path)

                    if (not self.disable_wandb) and self.is_main_process and not save_latents_only:
                        caption = prompts_for_sample[0] if len(prompts_for_sample) > 0 else ""
                        log_key = f"generated_video{suffix}"
                        wandb.log(
                            {
                                log_key: wandb.Video(
                                    generated_video[0].transpose(0, 3, 1, 2),
                                    caption=f"{caption}",
                                    fps=self.fps,
                                    format="mp4",
                                ),
                            },
                            step=self.step,
                        )

                    del generated_video

                prompt_txt_path = os.path.join(
                    out_dir,
                    f"prompt_rank{rank:02d}_idx{sample_idx:06d}.txt",
                )
                save_prompts_to_txt(
                    prompts_for_sample,
                    prompt_txt_path,
                    self.is_main_process,
                )
        barrier()

        if hasattr(self.model, "inference_pipeline") and self.model.inference_pipeline is not None:
            clear_fn = getattr(self.model.inference_pipeline, "clear_cache", None)
            if clear_fn is not None:
                clear_fn()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_video(self, pipeline, prompts, image=None, use_ema=False):
        # Temporarily disable SP self-attention during inference to avoid
        # interfering with KV-cache logic.
        self._set_sp_attn(False)
        ema_applied = use_ema and self.generator_ema is not None
        if ema_applied:
            self._swap_ema_weights()
        try:
            batch_size = len(prompts)
            noise_shape = list(self.config.image_or_video_shape[1:])
            inference_num_frames = section_get(
                self.config, "evaluation", "num_frames", getattr(self.config, "inference_num_frames", 0)
            )
            if isinstance(inference_num_frames, (list, tuple)):
                inference_num_frames = inference_num_frames[0] if len(inference_num_frames) > 0 else 0
            if inference_num_frames > 0:
                noise_shape[0] = inference_num_frames
            initial_latent = None
            if image is not None:
                image = image.to(device="cuda", dtype=self.dtype)
                if image.ndim == 4:
                    image = image.unsqueeze(2)
                elif image.ndim != 5:
                    raise ValueError(f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}")
                initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=self.dtype)
                if initial_latent.shape[0] != batch_size:
                    initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
                if noise_shape[0] <= initial_latent.shape[1]:
                    raise ValueError(
                        f"evaluation.num_frames must exceed the i2v conditioning frames; "
                        f"got {inference_num_frames} and {initial_latent.shape[1]}"
                    )
            sampled_noise = torch.randn(
                [batch_size] + noise_shape, device="cuda", dtype=self.dtype
            )

            save_latents_only = section_get(
                self.config,
                "evaluation",
                "save_latents_only",
                self.config.get("return_latents", False),
                aliases=("return_latents", "save_latent_only"),
            )
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                initial_latent=initial_latent,
                return_latents=save_latents_only
            )
            if not save_latents_only:
                current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
            else:
                current_video = video
        finally:
            if ema_applied:
                self._swap_ema_weights()
            # Restore SP self-attention for training.
            self._set_sp_attn(True)

        return current_video

    def _sync_batch_for_sequence_parallel(self, batch, accumulation_step: int = 0):
        return self.sp_helper.sync_batch(batch, step=self.step)

    def train(self):
        if getattr(self.config, "generate_before_train", False):
            if self.is_main_process:
                print("[generate_before_train] Running evaluation inference before training starts...")
            self._run_evaluation_inference()
            if self.is_main_process:
                print("[generate_before_train] Inference done. Exiting.")
            barrier()
            return

        acc_steps = getattr(self, "gradient_accumulation_steps", 1)
        while True:
            for acc in range(acc_steps):
                batch = next(self.dataloader)

                # Synchronize batch contents across ranks under Sequence Parallel.
                if self.sequence_parallel_size > 1:
                    batch = self._sync_batch_for_sequence_parallel(batch, accumulation_step=acc)

                self.train_one_step(batch, accumulation_step=acc, accumulation_steps=acc_steps)
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            evaluation_interval = section_get(self.config, "evaluation", "interval", getattr(self.config, "generate_interval", 0))
            if evaluation_interval > 0 and self.step % evaluation_interval == 0:
                self._run_evaluation_inference()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if self.step >= self.config.max_iters:
                break
