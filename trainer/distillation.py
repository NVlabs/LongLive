# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import gc
import logging
import random
import re
from pathlib import Path

from utils.dataset import TextDataset, TwoTextDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD, DMDSwitch
from model.streaming_training import StreamingTrainingModel
import torch
import wandb
import time
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video

# LoRA related imports
import peft
from peft import get_peft_model_state_dict
import safetensors.torch

from utils.memory import gpu, get_cuda_free_memory_gb, log_gpu_memory
from pipeline import (
    CausalInferencePipeline,
    SwitchCausalInferencePipeline
)
from utils.debug_option import DEBUG, LOG_GPU_MEMORY, DEBUG_GRADIENT
from one_logger_utils import OneLoggerUtils
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
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        self.use_one_logger = getattr(config, "use_one_logger", True)
        if self.is_main_process and not self.disable_wandb:
            wandb.login(
                # host=config.wandb_host,
                key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        app_start_time = time.time_ns() / 1_000_000 
        
        # ------------------------------------- One Logger Setup ----------------------------------------------
        if self.use_one_logger and dist.get_rank() == 0 and not self.disable_wandb:
            app_tag_run_name = f"dmd_{config.real_name[:6]}_local_attn_size_{config.model_kwargs.local_attn_size}_lr_{config.lr}"
            app_tag_run_version = "0.0.0"
            app_tag = f"{app_tag_run_name}_{app_tag_run_version}_{config.batch_size}_{dist.get_world_size()}"
            one_logger_config = {
                "enable_for_current_rank": True,
                "one_logger_async": True,
                "one_logger_project": getattr(config, "one_logger_project", "self-forcing"),
                "log_every_n_train_iterations": getattr(config, "log_iters", 10),
                "app_tag_run_version": app_tag_run_version,
                "summary_data_schema_version": "1.0.0",
                "app_run_type": "training",
                "app_tag": app_tag,
                "app_tag_run_name": app_tag_run_name,
                "one_logger_run_name": app_tag_run_name,
                "world_size": dist.get_world_size(),
                "global_batch_size": config.batch_size * getattr(config, "gradient_accumulation_steps", 1) * dist.get_world_size(),
                "batch_size": config.batch_size,
                "train_iterations_target": getattr(config, "max_iters", 0),
                "train_samples_target": (getattr(config, "max_iters", 0) * config.batch_size) if getattr(config, "max_iters", 0) else 0,
                "is_train_iterations_enabled": True,
                "is_baseline_run": False,
                "is_test_iterations_enabled": False,
                "is_validation_iterations_enabled": True,
                "is_save_checkpoint_enabled": True,
                "is_log_throughput_enabled": False,
                "micro_batch_size": config.batch_size,
                "seq_length": getattr(config, "image_or_video_shape")[1] * getattr(config, "image_or_video_shape")[3] * getattr(config, "image_or_video_shape")[4],
                "save_checkpoint_strategy": "sync",
            }
            self.one_logger = OneLoggerUtils(one_logger_config)
            self.one_logger.on_app_start(app_start_time = app_start_time)  
        else:
            self.one_logger = None

        # Step 2: Initialize the model
        if self.one_logger is not None:
            self.one_logger.on_model_init_start()

        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd_switch":
            self.model = DMDSwitch(config, device=self.device)
        elif config.distribution_loss == "dmd_window":
            self.model = DMDWindow(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

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
            
            # 1. Load base model first (config.generator_ckpt) - before applying LoRA and FSDP
            base_checkpoint_path = getattr(config, "generator_ckpt", None)
            if base_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading base model from {base_checkpoint_path} (before applying LoRA)")
                base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
                
                # Load generator (directly; no key alignment needed since LoRA not applied yet)
                if "generator" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["generator"], strict=True)
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                elif "model" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["model"], strict=True)
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found in base model.")
                
                # Load critic
                if "critic" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {base_checkpoint_path}")
                    result = self.model.fake_score.load_state_dict(base_checkpoint["critic"], strict=True)
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found in base model.")
            else:
                if self.is_main_process:
                    raise ValueError("No base model checkpoint specified for LoRA training.")
            
            # Load training step
            if "step" in base_checkpoint:
                self.step = base_checkpoint["step"]
                if self.is_main_process:
                    print(f"base_checkpoint step: {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")
            
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
            
            # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available)
            lora_checkpoint_path = None
            if auto_resume and self.output_path:
                # Find the latest checkpoint and verify it is a LoRA checkpoint
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    try:
                        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                        if "generator_lora" in checkpoint and "critic_lora" in checkpoint:
                            lora_checkpoint_path = latest_checkpoint
                            if self.is_main_process:
                                print(f"Auto resume: Found LoRA checkpoint at {lora_checkpoint_path}")
                        else:
                            raise ValueError(f"Checkpoint {latest_checkpoint} is not a LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error validating checkpoint: {e}")
                        raise e
                else:
                    if self.is_main_process:
                        print("Auto resume: No LoRA checkpoint found in logdir")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified for LoRA")
            else:
                if self.is_main_process:
                    print("Auto resume disabled for LoRA")
            
            # If no auto-resumed LoRA checkpoint found, try config.lora_ckpt
            if lora_checkpoint_path is None:
                lora_ckpt_path = getattr(config, "lora_ckpt", None)
                if lora_ckpt_path:
                    try:
                        checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                        if "generator_lora" in checkpoint and "critic_lora" in checkpoint:
                            lora_checkpoint_path = lora_ckpt_path
                            if self.is_main_process:
                                print(f"Using explicit LoRA checkpoint: {lora_checkpoint_path}")
                        else:
                            raise ValueError(f"Explicit LoRA checkpoint {lora_ckpt_path} is not a valid LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error loading explicit LoRA checkpoint: {e}")
                        raise e
                else:
                    if self.is_main_process:
                        print("No LoRA checkpoint specified, starting LoRA training from scratch")
            
            # Load LoRA checkpoint (before FSDP wrapping)
            if lora_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} (before FSDP wrapping)")
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
                
                # Load LoRA weights using PEFT's standard method
                if "generator_lora" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading LoRA generator weights: {len(lora_checkpoint['generator_lora'])} keys in checkpoint")
                    
                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names
                    peft.set_peft_model_state_dict(self.model.generator.model, lora_checkpoint["generator_lora"])
                
                if "critic_lora" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")
                    
                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names
                    peft.set_peft_model_state_dict(self.model.fake_score.model, lora_checkpoint["critic_lora"])

                # Load training step
                if "step" in lora_checkpoint:
                    self.step = lora_checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming LoRA training from step {self.step}")
            else:
                if self.is_main_process:
                    print("No LoRA checkpoint to load, starting from scratch")

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

        # if not config.no_visualize or config.load_raw_video:
        #     print("Moving vae to device 2, self.device: ", self.device)
        #     self.model.vae = self.model.vae.to(
        #         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

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

        
        if self.one_logger is not None:
            self.one_logger.on_model_init_end()
        
        # Step 4: Initialize the optimizer
        if self.one_logger is not None:
            self.one_logger.on_optimizer_init_start()

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

        if self.one_logger is not None:
            self.one_logger.on_optimizer_init_end() 

        # Step 5: Initialize the dataloader
        if self.one_logger is not None:
            self.one_logger.on_dataloader_init_start()
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        elif self.config.distribution_loss == "dmd_switch":
            dataset = TwoTextDataset(config.data_path, config.switch_prompt_path)
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = getattr(config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:
            # Determine validation data path
            val_data_path = getattr(config, "val_data_path", None) or config.data_path

            if self.config.i2v:
                val_dataset = ShardingLMDBDataset(val_data_path, max_pair=int(1e8))
            elif self.config.distribution_loss == "dmd_switch":
                val_dataset = TwoTextDataset(val_data_path, config.val_switch_prompt_path)
            else:
                val_dataset = TextDataset(val_data_path)

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False)
            # streaming sampling to keep prompts fixed
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=getattr(config, "val_batch_size", 1),
                sampler=sampler,
                num_workers=8,
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
            self.vis_video_lengths = getattr(config, "vis_video_lengths", [])

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()
            
        if self.one_logger is not None:
            self.one_logger.on_dataloader_init_end() 

        if self.one_logger is not None:
            self.one_logger.on_load_checkpoint_start()
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

            if checkpoint_path:
                if self.is_main_process:
                    print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Load generator
                if "generator" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
                elif "model" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["model"], strict=True)
                else:
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found.")
                
                # Load critic
                if "critic" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {checkpoint_path}")
                    self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found.")
                
                # Load EMA
                if "generator_ema" in checkpoint and self.generator_ema is not None:
                    if self.is_main_process:
                        print(f"Loading pretrained EMA from {checkpoint_path}")
                    self.generator_ema.load_state_dict(checkpoint["generator_ema"])
                else:
                    if self.is_main_process:
                        print("Warning: EMA checkpoint not found or EMA not initialized.")
                
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
                    self.generator_optimizer.load_state_dict(gen_osd)
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
                    self.critic_optimizer.load_state_dict(crit_osd)
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

        if self.one_logger is not None:
            self.one_logger.on_load_checkpoint_end()
        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # Note: This should be done after potential resume to avoid accidentally deleting resumed EMA
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None
        
        # streaming training configuration
        self.streaming_training = getattr(config, "streaming_training", False)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)
        
        # Create streaming training model if enabled
        if self.streaming_training:
            self.streaming_model = StreamingTrainingModel(self.model, config)
            if self.is_main_process:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
        else:
            self.streaming_model = None
        
        # streaming training state (simplified)
        self.streaming_active = False  # Whether we're currently in a sequence
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")
            if self.streaming_training:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
            if LOG_GPU_MEMORY:
                log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())

        
        if self.one_logger is not None:
            self.one_logger.on_train_start(train_iterations_start = self.step, train_samples_start = self.step * self.config.batch_size)
        
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

    def _get_switch_frame_index(self, max_length=None):
        if getattr(self.config, "switch_mode", "fixed") == "random":
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                
                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                dist.broadcast(switch_idx_tensor, src=0)
                switch_idx = switch_idx_tensor.item()
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx


    def save(self):
        print("Start gathering distributed model states...")
        if getattr(self, 'one_logger', None) is not None and self.is_main_process:
            self.one_logger.on_save_checkpoint_start(global_step=self.step)

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
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                generator_state_dict  = self.model.generator.state_dict()
                generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                                self.generator_optimizer)

            with FSDP.state_dict_type(
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
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

        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
        if self.one_logger is not None:
            self.one_logger.on_save_checkpoint_success(global_step=self.step)
            self.one_logger.on_save_checkpoint_end(global_step=self.step)

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=None,
                initial_latent=None
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
            # Return original loss for logging
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return generator_log_dict
        else:
            generator_log_dict = {}

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=None,
            initial_latent=None
        )

        # Scale loss for gradient accumulation and backward
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
        # Return original loss for logging
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

        return critic_log_dict

    def generate_video(self, pipeline, num_frames, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video
    

    def generate_video_with_switch(self, pipeline, num_frames, prompts, switch_prompts, switch_frame_index, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts_first=prompts,
                text_prompts_second=switch_prompts,
                switch_frame_index=switch_frame_index,
                return_latents=True
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video

    def start_new_sequence(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence called")
        
        if LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: Before start_new_sequence", device=self.device, rank=dist.get_rank())
        
        # Fetch a new batch
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence: fetch new batch")
        batch = next(self.dataloader)

        # Prepare conditional information
        text_prompts = batch["prompts"]
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Setting up sequence: batch_size={batch_size}, i2v={self.config.i2v}")
            print(f"[SeqTrain-Trainer] image_or_video_shape={image_or_video_shape}")
        
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Created and cached conditional_dict")
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] Created and cached unconditional_dict")
            else:
                unconditional_dict = self.unconditional_dict
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After text encoding", device=self.device, rank=dist.get_rank())
        
        if self.streaming_model.possible_max_length is not None:
            # Ensure all processes choose the same length
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
            
            temp_max_length = self.streaming_model.possible_max_length[selected_idx]
        else:
            temp_max_length = self.streaming_model.max_length
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Model] Selected temporary max length: {temp_max_length} (from {self.streaming_model.possible_max_length})")
        

        # Handle DMD Switch related information
        switch_conditional_dict = None
        switch_frame_index = None
        if isinstance(self.model, DMDSwitch) and "switch_prompts" in batch:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Processing DMDSwitch info")
                
            with torch.no_grad():
                switch_conditional_dict = self.model.text_encoder(
                    text_prompts=batch["switch_prompts"]
                )
            switch_frame_index = self._get_switch_frame_index(temp_max_length)
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] switch_frame_index={switch_frame_index}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"streaming Training: After switch text encoding", device=self.device, rank=dist.get_rank())
        
        # Set up the sequence
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Calling streaming_model.setup_sequence")
            
        self.streaming_model.setup_sequence(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=image_latent,
            switch_conditional_dict=switch_conditional_dict,
            switch_frame_index=switch_frame_index,
            temp_max_length=temp_max_length,
        )
        
        self.streaming_active = True
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] streaming training sequence setup completed")
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After sequence setup", device=self.device, rank=dist.get_rank())

    def fwdbwd_one_step_streaming(self, train_generator):
        """Forward/backward pass using the new StreamingTrainingModel for serialized training"""
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # If no active sequence, start a new one
        if not self.streaming_active:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] No active sequence, starting new one")
            self.start_new_sequence()
        
        # Check whether we can generate more chunks
        if not self.streaming_model.can_generate_more():
            # Current sequence is finished; start a new one
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Current sequence completed, starting new one")
            self.streaming_active = False
            self.start_new_sequence()
        
        self.kv_cache_before_generator_rollout = None
        self.kv_cache_after_generator_rollout = None
        self.kv_cache_after_generator_backward = None
        self.kv_cache_before_critic_rollout = None
        self.kv_cache_after_critic_rollout = None
        self.kv_cache_after_critic_backward = None
        
        if train_generator:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training generator: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            # Compute generator loss
            generator_loss, generator_log_dict = self.streaming_model.compute_generator_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[DEBUG] Scaled generator loss: {scaled_generator_loss.item()}")

            try:
                scaled_generator_loss.backward()
            except RuntimeError as e:
                raise

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return generator_log_dict
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training critic: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Generated chunk shape: {generated_chunk.shape}")
                print(f"[SeqTrain-Trainer] Generated chunk requires_grad: {generated_chunk.requires_grad}")
            
            if generated_chunk.requires_grad:
                generated_chunk = generated_chunk.detach()

            # Compute critic loss
            critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic loss: {critic_loss.item()}")
            
            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()
            
            critic_log_dict.update({
                "critic_loss": critic_loss,
                "critic_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return critic_log_dict

    def train(self):
        start_step = self.step
        try:
            while True:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                if LOG_GPU_MEMORY:
                    log_gpu_memory(f"Before training", device=self.device, rank=dist.get_rank())
                
                if dist.get_rank() == 0 and DEBUG:
                    print(f"[Debug] Step {self.step}: switch_mode={getattr(self.config,'switch_mode','fixed')}")

                if self.one_logger is not None:
                    self.one_logger.on_train_batch_start()

                if self.streaming_training:
                    # Zero-out all optimizer gradients
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Whole-cycle accumulation step {accumulation_step + 1}/{self.gradient_accumulation_steps}")
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: Before whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                                print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training generator")
                            extra_gen = self.fwdbwd_one_step_streaming(True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training critic")
                        extra_crit = self.fwdbwd_one_step_streaming(False)
                        accumulated_critic_logs.append(extra_crit)
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: After whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm
                        
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Generator training completed, grad_norm={generator_grad_norm.item()}")
                        
                        self.generator_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Critic training completed, grad_norm={critic_grad_norm.item()}")
                    
                    self.critic_optimizer.step()
                    
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: After optimizer steps", device=self.device, rank=dist.get_rank())
                    
                    # Increase step count
                    self.step += 1
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] streaming training step completed: step={self.step}")
                        if hasattr(self, 'streaming_model') and self.streaming_model is not None:
                            current_seq_length = self.streaming_model.state.get("current_length", 0)
                            print(f"[SeqTrain-Trainer] Current sequence length: {current_seq_length}/{self.streaming_model.max_length}")
                            
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: Training step completed", device=self.device, rank=dist.get_rank())
                else:
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

                if self.one_logger is not None:
                    self.one_logger.on_train_batch_end()

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
                    if self.one_logger is not None:
                        self.one_logger.on_validation_start()

                    try:
                        self._visualize()
                    except Exception as e:
                        print(f"[Warning] Visualization failed at step {self.step}: {e}")
                
                    if self.one_logger is not None:
                        self.one_logger.on_validation_end()
                
                if self.step > self.config.max_iters:
                    break

            if self.one_logger is not None:
                self.one_logger.on_train_end()
                self.one_logger.on_app_end()
        
        except Exception as e:
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)
                import traceback
                traceback.print_exc()
        finally:
            # Clean up resources
            if self.one_logger is not None:
                try:
                    self.one_logger.on_train_end()
                    self.one_logger.on_app_end()
                except Exception as cleanup_e:
                    if self.is_main_process:
                        print(f"[WARNING] Failed to clean up one_logger: {cleanup_e}")


    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # Define the specific modules we want to apply LoRA to
        if model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock']
        elif model_name == 'fake_score':
            adapter_target_modules = ['WanAttentionBlock']
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
                # task_type="FEATURE_EXTRACTION"        # Remove this; not needed for diffusion models
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

        # Choose pipeline class depending on causal flag
        if 'switch' in self.config.distribution_loss:
            self.vis_pipeline = SwitchCausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)
        else:
            self.vis_pipeline = CausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)

        # Visualization output directory (default: <logdir>/vis)
        self.vis_output_dir = os.path.join(os.path.dirname(self.output_path), "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if self.config.vis_ema:
            raise NotImplementedError("Visualization with EMA is not implemented")

    def _visualize(self):
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_start()

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
            prompts = batch["prompts"]
            switch_prompts = batch["switch_prompts"]
            switch_frame_index = self._get_switch_frame_index()
        else:
            prompts = batch["prompts"]

        image = None
        if self.config.i2v and ("image" in batch):
            image = batch["image"]

        # Prepare model mode info for filename
        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating videos in LoRA mode (step {self.step})")
        
        for vid_len in self.vis_video_lengths:
            print(f"Generating video of length {vid_len}")
            if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                videos = self.generate_video_with_switch(self.vis_pipeline, vid_len, prompts, switch_prompts, switch_frame_index, image=image)
            else:
                videos = self.generate_video(self.vis_pipeline, vid_len, prompts, image=image)

            # Save each sample
            for idx, video_np in enumerate(videos):
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}_switch_frame_{switch_frame_index}.mp4"
                else:
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                out_path = os.path.join(
                    step_vis_dir,
                    video_name,
                )
                video_tensor = torch.from_numpy(video_np.astype("uint8"))
                write_video(out_path, video_tensor, fps=16)

            # After saving current length videos, release related tensors to reduce peak memory
            del videos, video_np, video_tensor  # type: ignore
            torch.cuda.empty_cache()

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_end()

        torch.cuda.empty_cache()
        import gc
        gc.collect()
