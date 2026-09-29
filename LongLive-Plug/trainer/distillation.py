# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import random

from utils.dataset import TextPromptDataset, eval_collate_fn
from utils.config import section_get, wan_default_config
from utils.distributed import fsdp_wrap, launch_distributed_job
from utils.resumable_dataloader import ResumableDistributedDataLoader
from utils.misc import (
    set_seed,
    merge_dict_list
)
from utils.lora_utils import (
    assert_trainable_lora_parameters_synced,
    configure_lora_for_model,
    load_lora_checkpoint,
)
import torch.distributed as dist
import numpy as np
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import os
import re
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video

# LoRA related imports
from peft import get_peft_model_state_dict

import time


LORA_GENERATOR_OPTIMIZER_FILENAME = "generator_optimizer.pt"
LORA_CRITIC_OPTIMIZER_FILENAME = "critic_optimizer.pt"
LORA_TRAINING_STATE_FILENAME = "training_state.pt"
LORA_CHECKPOINT_SUCCESS_FILENAME = "_SUCCESS"


def canonical_lora_adapter_config(adapter_config):
    """Return the behavior-affecting LoRA settings stored in checkpoints."""
    rank = int(adapter_config.get("rank", 16))
    alpha = adapter_config.get("alpha", None)
    if alpha is None:
        alpha = rank
    expected_targets = adapter_config.get("expected_target_modules", None)
    return {
        "type": str(adapter_config.get("type", "lora")).lower(),
        "rank": rank,
        "alpha": float(alpha),
        "dropout": float(adapter_config.get("dropout", 0.0)),
        "apply_to_critic": bool(adapter_config.get("apply_to_critic", True)),
        "expected_target_modules": (
            None if expected_targets is None else int(expected_targets)
        ),
    }


def validate_lora_adapter_config(
    saved_adapter,
    current_adapter,
    *,
    checkpoint_path,
    require_metadata,
):
    """Fail closed when an exact resume would change LoRA behavior."""
    if saved_adapter is None:
        if require_metadata:
            raise RuntimeError(
                f"LoRA resume checkpoint {checkpoint_path} has no adapter "
                "metadata, so an exact resume cannot be verified."
            )
        return False

    saved = canonical_lora_adapter_config(saved_adapter)
    current = canonical_lora_adapter_config(current_adapter)
    if saved != current:
        changed = {
            key: {"saved": saved[key], "current": current[key]}
            for key in saved
            if saved[key] != current[key]
        }
        raise RuntimeError(
            f"LoRA adapter config changed since {checkpoint_path}: {changed}"
        )
    return True


def _active_peft_adapter_name(lora_model):
    adapter_name = getattr(lora_model, "active_adapter", None)
    if callable(adapter_name):
        adapter_name = adapter_name()
    return adapter_name or "default"


def _insert_lora_adapter_name(state_dict, adapter_name):
    peft_state_dict = {}
    parameter_prefix = "lora_"

    for key, value in state_dict.items():
        new_key = key
        if parameter_prefix in key:
            _, _, suffix = key.rpartition(parameter_prefix)
            suffix_parts = suffix.split(".")
            if len(suffix_parts) > 1 and suffix_parts[1] != adapter_name:
                suffix_to_replace = ".".join(suffix_parts[1:])
                new_key = re.sub(
                    re.escape(suffix_to_replace) + r"$",
                    f"{adapter_name}.{suffix_to_replace}",
                    key,
                )
            elif len(suffix_parts) == 1:
                new_key = f"{key}.{adapter_name}"
        peft_state_dict[new_key] = value

    return peft_state_dict


def _load_lora_state_dict_compat(lora_model, lora_state_dict):
    """Load LoRA weights without PEFT's distributed TP compatibility path."""
    adapter_name = _active_peft_adapter_name(lora_model)
    peft_state_dict = _insert_lora_adapter_name(lora_state_dict, adapter_name)
    load_result = lora_model.load_state_dict(peft_state_dict, strict=False)

    missing_lora = [key for key in load_result.missing_keys if ".lora_" in key]
    unexpected_lora = [key for key in load_result.unexpected_keys if ".lora_" in key]
    if missing_lora or unexpected_lora:
        raise RuntimeError(
            "Failed to load LoRA checkpoint cleanly: "
            f"missing_lora={missing_lora[:10]}, unexpected_lora={unexpected_lora[:10]}"
        )

    return load_result


class Trainer:

    def __init__(self, config):
        self.config = config
        self.step = 0
        self.loaded_lora_checkpoint_step = None
        self.sfp_training = bool(getattr(config, "sfp_training", False))
        self._resume_rank_training_state = None

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.data_parallel_size = self.world_size

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0 and getattr(config, "randomize_seed", True):
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

        if (config.distribution_loss == "cfg_guidance"
                and config.cfg_state_source == "teacher_trajectory_cache"
                and not config.cfg_verify_cache_hashes):
            from utils.cache_validation import verify_distributed_cache
            verify_distributed_cache(config.data_path)

        # Step 2: Initialize the model. CFG-only guidance distillation has no
        # learned fake score / critic, while DMD requires one.
        self.uses_critic = config.distribution_loss == "dmd"
        if self.sfp_training and not self.uses_critic:
            raise ValueError("sfp_training requires distribution_loss=dmd")


        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "cfg_guidance":
            # Keep this import local so existing DMD-only integrations that
            # provide a minimal ``model`` module remain backwards compatible.
            from model import CFGGuidanceDistillation

            self.model = CFGGuidanceDistillation(config, device=self.device)
            if self.model.fake_score is not None:
                raise ValueError(
                    "CFG-only distillation must not allocate a fake score model."
                )
        else:
            raise ValueError(f"Unsupported distribution matching loss: {config.distribution_loss}")


        # Auto resume configuration (needed for LoRA checkpoint loading)
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        # ================================= LoRA Configuration =================================
        self._pending_generator_optimizer_state = None
        self.lora_resume_checkpoint_dir = None

        self.is_lora_enabled = True
        self.lora_config = config.adapter

        if self.is_main_process:
            print(f"LoRA enabled with config: {self.lora_config}")
            print("Loading base model and applying LoRA before FSDP wrapping...")

        # Apply adapters to pretrained backbones before FSDP wrapping.
        self.apply_lora_to_critic = (
            self.is_lora_enabled
            and self.uses_critic
            and getattr(self.lora_config, "apply_to_critic", True)
        )

        if self.is_main_process:
            print("Applying LoRA to models...")

        # HYBRID_SHARD uses replicated FSDP groups and this repository's
        # wrapper intentionally sets sync_module_states=False. Construct
        # every adapter from one common RNG stream on every rank, in a
        # fixed order, then restore the normal rank-local training stream.
        set_seed(config.seed)
        try:
            self.model.generator.model = self._configure_lora_for_model(
                self.model.generator.model, "generator"
            )

            # Configure LoRA for fake_score if needed.
            if self.apply_lora_to_critic:
                self.model.fake_score.model = self._configure_lora_for_model(
                    self.model.fake_score.model, "fake_score"
                )
                if self.is_main_process:
                    print("LoRA applied to both generator and critic")
            elif self.is_main_process:
                print("LoRA applied to generator only")
        finally:
            set_seed(config.seed + global_rank)
        if self.model.real_score is not None:
            teacher_lora_params = [
                name for name, _ in self.model.real_score.model.named_parameters()
                if "lora_" in name
            ]
            teacher_trainable_params = [
                name for name, param in self.model.real_score.model.named_parameters()
                if param.requires_grad
            ]
            if teacher_lora_params or teacher_trainable_params:
                raise RuntimeError(
                    "The real-score teacher must remain frozen and adapter-free in "
                    "LoRA distillation: "
                    f"lora_params={teacher_lora_params[:5]}, "
                    f"trainable_params={teacher_trainable_params[:5]}"
                )
            if self.is_main_process:
                print("Real-score teacher verified: 0 LoRA params, 0 trainable params")

        # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available).
        # Priority: auto_resume -> legacy lora_ckpt -> initialized adapters.
        lora_checkpoint_path = None
        lora_checkpoint = None
        if auto_resume and self.output_path:
            # Auto-resume is an exact training resume for every LoRA mode.
            # Optimizer/RNG/data-cursor state is stored in atomic sidecars;
            # an explicit ``lora_ckpt`` remains the weights-only escape hatch.
            required_checkpoint_keys = (
                "generator_lora",
                "adapter_contract",
                "optimizer_state_files",
                "step",
            )
            latest_checkpoint = self.find_latest_checkpoint(
                self.output_path,
                required_keys=required_checkpoint_keys,
                require_lora_optimizer_state=True,
            )
            if latest_checkpoint:
                lora_checkpoint_path = latest_checkpoint
                self.lora_resume_checkpoint_dir = os.path.dirname(
                    latest_checkpoint
                )
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

            adapter_was_verified = validate_lora_adapter_config(
                lora_checkpoint.get(
                    "adapter_contract", lora_checkpoint.get("adapter")
                ),
                self.lora_config,
                checkpoint_path=lora_checkpoint_path,
                require_metadata=self.lora_resume_checkpoint_dir is not None,
            )
            if self.is_main_process:
                if adapter_was_verified:
                    print("LoRA adapter resume contract verified")
                else:
                    print(
                        "Warning: weights-only lora_ckpt has no adapter "
                        "metadata; alpha/dropout compatibility was not verified."
                    )

            if "generator_lora" not in lora_checkpoint:
                raise ValueError(f"LoRA checkpoint {lora_checkpoint_path} is not a valid LoRA checkpoint. "
                                 f"Found keys: {list(lora_checkpoint.keys())}")

            if self.is_main_process:
                print(f"Loading LoRA generator weights: {len(lora_checkpoint['generator_lora'])} keys in checkpoint")
            self._load_lora_state_dict_compat(
                self.model.generator.model,
                lora_checkpoint["generator_lora"],
                "generator",
            )
            del lora_checkpoint["generator_lora"]

            if self.apply_lora_to_critic:
                if "critic_lora" not in lora_checkpoint:
                    raise ValueError(f"LoRA checkpoint {lora_checkpoint_path} is missing critic_lora.")
                if self.is_main_process:
                    print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")
                self._load_lora_state_dict_compat(
                    self.model.fake_score.model,
                    lora_checkpoint["critic_lora"],
                    "critic",
                )
                del lora_checkpoint["critic_lora"]
            gc.collect()

            if "step" in lora_checkpoint:
                self.loaded_lora_checkpoint_step = int(
                    lora_checkpoint["step"]
                )

            legacy_embedded_optimizer = (
                self.lora_resume_checkpoint_dir is None
                and "generator_optimizer" in lora_checkpoint
            )
            if (
                "step" in lora_checkpoint
                and (
                    self.lora_resume_checkpoint_dir is not None
                    or legacy_embedded_optimizer
                )
            ):
                self.step = int(lora_checkpoint["step"])
                if self.is_main_process:
                    print(f"Resuming LoRA training from step {self.step}")
            elif self.lora_resume_checkpoint_dir is not None:
                raise RuntimeError(
                    f"Exact resume checkpoint {lora_checkpoint_path} has no step"
                )
            elif self.is_main_process and "step" in lora_checkpoint:
                print(
                    "Treating explicit lora_ckpt as weights-only initialization; "
                    f"ignoring saved step {lora_checkpoint['step']}."
                )

            if legacy_embedded_optimizer:
                self._pending_generator_optimizer_state = lora_checkpoint[
                    "generator_optimizer"
                ]
            del lora_checkpoint
            gc.collect()
        else:
            if self.is_main_process:
                print("No LoRA checkpoint to load, starting from scratch")

        lora_models = {"generator": self.model.generator.model}
        if self.apply_lora_to_critic:
            lora_models["critic"] = self.model.fake_score.model
        lora_summaries = assert_trainable_lora_parameters_synced(
            lora_models, torch.device("cuda", self.device)
        )
        expected_lora_sha256 = {
            "generator": os.environ.get(
                "DMD_EXPECTED_FRESH_GENERATOR_LORA_SHA256"
            ),
            "critic": os.environ.get(
                "DMD_EXPECTED_FRESH_CRITIC_LORA_SHA256"
            ),
        }
        for role, expected_sha256 in expected_lora_sha256.items():
            if expected_sha256 is None:
                continue
            actual_sha256 = lora_summaries.get(role, {}).get("sha256")
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Pre-FSDP {role} LoRA does not match the pinned "
                    f"initialization: expected={expected_sha256}, "
                    f"actual={actual_sha256}"
                )
        if self.is_main_process:
            for role, summary in lora_summaries.items():
                print(
                    f"Pre-FSDP {role} LoRA verified across {self.world_size} "
                    f"ranks: {summary['parameter_count']} tensors, "
                    f"{summary['element_count']} elements, "
                    f"sha256={summary['sha256']}"
                )

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        if self.model.real_score is not None:
            self.model.real_score = fsdp_wrap(
                self.model.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.real_score_fsdp_wrap_strategy
            )

        if self.uses_critic:
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
        vae_dtype_name = str(
            getattr(
                config,
                "vae_dtype",
                "bfloat16" if config.mixed_precision else "float32",
            )
        ).lower()
        if vae_dtype_name in ("float32", "fp32", "float"):
            vae_dtype = torch.float32
        elif vae_dtype_name in ("bfloat16", "bf16"):
            vae_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported vae_dtype '{vae_dtype_name}'. "
                "Expected float32/fp32 or bfloat16/bf16."
            )
        self.vae_dtype = vae_dtype
        if self.model.vae is not None:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=vae_dtype,
            )

        # Step 4: Initialize the optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        if self._pending_generator_optimizer_state is not None:
            generator_osd = FSDP.optim_state_dict_to_load(
                self.model.generator,
                self.generator_optimizer,
                self._pending_generator_optimizer_state,
            )
            self.generator_optimizer.load_state_dict(generator_osd)
            self._pending_generator_optimizer_state = None
            del generator_osd
            if self.is_main_process:
                print("Resumed generator optimizer state from LoRA checkpoint")

        if self.uses_critic:
            self.critic_optimizer = torch.optim.AdamW(
                [param for param in self.model.fake_score.parameters()
                 if param.requires_grad],
                lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
                betas=(config.beta1_critic, config.beta2_critic),
                weight_decay=config.weight_decay
            )
        else:
            self.critic_optimizer = None

        if self.is_lora_enabled and self.lora_resume_checkpoint_dir is not None:
            self._restore_lora_optimizer_states(self.lora_resume_checkpoint_dir)

        # Step 5: Initialize the dataloader
        self.cfg_state_source = str(getattr(config, "cfg_state_source", "data"))

        model_name = config.model_kwargs.model_name
        self.fps = wan_default_config[model_name].get("fps", 16)
        latent_frames_for_dataset = int(config.image_or_video_shape[1])

        if self.cfg_state_source == "teacher_trajectory_cache":
            from utils.dataset import (
                CFGTeacherTrajectoryDataset,
                cfg_teacher_trajectory_collate_fn,
            )

            dataset = CFGTeacherTrajectoryDataset(
                config.data_path,
                expected_latent_shape=list(config.image_or_video_shape)[1:],
                expected_model_name=config.model_kwargs.model_name,
                expected_guidance_scale=config.teacher_guidance_scale,
                expected_sampling_steps=config.teacher_sampling_steps,
                expected_timestep_shift=config.model_kwargs.timestep_shift,
                verify_shard_hashes=config.cfg_verify_cache_hashes,
                verified_manifest_sha256=os.environ.get(
                    "CFG_TEACHER_TRAJECTORY_VERIFIED_MANIFEST_SHA256"
                ),
            )
            collate_fn = cfg_teacher_trajectory_collate_fn
            if dist.get_rank() == 0:
                print(
                    "[cfg_state_source] Using cached teacher trajectory: "
                    f"manifest={config.data_path}, expanded_records={len(dataset)}"
                )
        else:
            dataset = TextPromptDataset(config.data_path)
            collate_fn = eval_collate_fn

        # DistributedSampler requires the same seed on every data-parallel
        # rank.  The rank is already used to select a disjoint slice.
        data_seed = int(getattr(config, "data_seed", config.seed)) % (2**31)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=self.uses_critic,
            seed=data_seed,
        )

        if len(sampler) == 0:
            raise ValueError(
                "Distributed training sampler is empty: "
                f"dataset_size={len(dataset)}, data_parallel_size={self.data_parallel_size}, "
                f"drop_last={self.uses_critic}."
            )

        data_num_workers = int(getattr(config, "data_num_workers", 2))
        loader_generator = torch.Generator()
        loader_generator.manual_seed(data_seed)
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=data_num_workers,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collate_fn,
            generator=loader_generator,
        )
        if data_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = int(
                getattr(config, "data_prefetch_factor", 1)
            )
        dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

        if dist.get_rank() == 0:
            print(
                "DATASET SIZE %d; data_seed=%d; batches_per_epoch=%d"
                % (len(dataset), data_seed, len(dataloader))
            )
        self.train_sampler = sampler
        self.dataloader = ResumableDistributedDataLoader(
            dataloader,
            sampler,
            sampler_seed=data_seed,
            data_id=os.path.realpath(config.data_path),
        )

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = (
            -1
            if getattr(config, "no_visualize", False)
            else section_get(
                config,
                "evaluation",
                "interval",
                getattr(config, "vis_interval", -1),
            )
        )
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

            val_dataset = TextPromptDataset(val_data_path)
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


        ##############################################################################################################

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = (
            getattr(config, "max_grad_norm_critic", 10.0)
            if self.uses_critic
            else None
        )
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None

        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.data_parallel_size}")
            if self.sfp_training:
                print(
                    "Integrated SFP training mode enabled: generator and critic "
                    "consume separate distributed batches."
                )

        # Restore RNG last: model/FSDP/optimizer/DataLoader construction above
        # may consume randomness and must not perturb the resumed trajectory.
        self._restore_rank_training_state()


    @staticmethod
    def _lora_optimizer_checkpoint_paths(checkpoint_dir):
        return {
            "generator": os.path.join(
                checkpoint_dir, LORA_GENERATOR_OPTIMIZER_FILENAME
            ),
            "critic": os.path.join(
                checkpoint_dir, LORA_CRITIC_OPTIMIZER_FILENAME
            ),
            "training": os.path.join(
                checkpoint_dir, LORA_TRAINING_STATE_FILENAME
            ),
            "success": os.path.join(
                checkpoint_dir, LORA_CHECKPOINT_SUCCESS_FILENAME
            ),
        }

    @staticmethod
    def _atomic_torch_save(value, path):
        temporary_path = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(value, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _gather_fsdp_optimizer_state(self, model, optimizer):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            return FSDP.optim_state_dict(model, optimizer)

    def _optimizer_step_range(self, optimizer):
        local_steps = []
        for state in optimizer.state.values():
            step = state.get("step")
            if step is None:
                continue
            if torch.is_tensor(step):
                step = step.item()
            local_steps.append(int(step))

        count = torch.tensor(len(local_steps), device=self.device, dtype=torch.long)
        minimum = torch.tensor(
            min(local_steps) if local_steps else torch.iinfo(torch.long).max,
            device=self.device,
            dtype=torch.long,
        )
        maximum = torch.tensor(
            max(local_steps) if local_steps else 0,
            device=self.device,
            dtype=torch.long,
        )
        if dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        return int(count.item()), int(minimum.item()), int(maximum.item())

    def _restore_fsdp_optimizer_state(
        self, label, model, optimizer, path, expected_step=None
    ):
        full_optimizer_state = None
        load_error = None
        if self.is_main_process:
            print(f"Loading LoRA {label} optimizer state from {path}")
            try:
                full_optimizer_state = torch.load(path, map_location="cpu")
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"

        # All ranks must make the same decision before entering the FSDP
        # scatter collective. Otherwise a rank-0 I/O error strands its peers.
        if dist.is_initialized():
            load_status = [load_error]
            dist.broadcast_object_list(load_status, src=0)
            load_error = load_status[0]
        if load_error is not None:
            raise RuntimeError(
                f"Failed to load LoRA {label} optimizer state {path}: "
                f"{load_error}"
            )

        sharded_optimizer_state = FSDP.scatter_full_optim_state_dict(
            full_optimizer_state,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded_optimizer_state)
        del sharded_optimizer_state
        del full_optimizer_state

        state_count, minimum_step, maximum_step = self._optimizer_step_range(optimizer)
        if state_count == 0:
            raise RuntimeError(
                f"Restored LoRA {label} optimizer checkpoint contains no Adam state"
            )
        if expected_step is not None and (
            minimum_step != int(expected_step) or maximum_step != int(expected_step)
        ):
            raise RuntimeError(
                f"Restored LoRA {label} Adam step range "
                f"[{minimum_step}, {maximum_step}] does not match "
                f"expected step {int(expected_step)}"
            )
        if self.is_main_process:
            print(
                f"Restored LoRA {label} optimizer: {state_count} distributed "
                f"parameter states, Adam step range [{minimum_step}, {maximum_step}]"
            )

    def _restore_lora_optimizer_states(self, checkpoint_dir):
        paths = self._lora_optimizer_checkpoint_paths(checkpoint_dir)
        required_paths = [
            paths["generator"],
            paths["training"],
            paths["success"],
        ]
        if self.uses_critic:
            required_paths.append(paths["critic"])
        missing = [path for path in required_paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(
                "LoRA auto-resume checkpoint is incomplete; missing "
                f"artifacts: {missing}"
            )

        self._restore_fsdp_optimizer_state(
            "generator",
            self.model.generator,
            self.generator_optimizer,
            paths["generator"],
            expected_step=(
                (
                    self.step + int(self.config.dfake_gen_update_ratio) - 1
                ) // int(self.config.dfake_gen_update_ratio)
                if self.uses_critic
                else self.step
            ),
        )
        if self.uses_critic:
            self._restore_fsdp_optimizer_state(
                "critic",
                self.model.fake_score,
                self.critic_optimizer,
                paths["critic"],
                expected_step=self.step,
            )

        training_checkpoint = torch.load(
            paths["training"], map_location="cpu", weights_only=False
        )
        if int(training_checkpoint.get("checkpoint_format_version", -1)) != 1:
            raise RuntimeError(
                "Unsupported LoRA training-state checkpoint version: "
                f"{training_checkpoint.get('checkpoint_format_version')}"
            )
        if int(training_checkpoint.get("step", -1)) != self.step:
            raise RuntimeError(
                f"Training-state step={training_checkpoint.get('step')} does "
                f"not match model step={self.step}"
            )
        if int(training_checkpoint.get("world_size", -1)) != self.world_size:
            raise RuntimeError(
                "Exact resume requires the same world size: "
                f"saved={training_checkpoint.get('world_size')}, "
                f"current={self.world_size}"
            )
        rank_states = training_checkpoint.get("rank_states", [])
        if len(rank_states) != self.world_size:
            raise RuntimeError(
                f"Expected {self.world_size} rank training states, "
                f"found {len(rank_states)}"
            )
        rank = dist.get_rank() if dist.is_initialized() else 0
        rank_state = rank_states[rank]
        if int(rank_state.get("rank", -1)) != rank:
            raise RuntimeError(
                f"Training-state rank mismatch: saved={rank_state.get('rank')}, "
                f"current={rank}"
            )
        self._resume_rank_training_state = rank_state
        del training_checkpoint
        gc.collect()

    def _training_contract(self):
        contract = {
            "distribution_loss": str(self.config.distribution_loss),
            "model_name": str(self.config.model_kwargs.model_name),
            "model_dir": os.path.realpath(
                str(self.config.model_kwargs.get("model_dir", ""))
            ),
            "timestep_shift": float(self.config.model_kwargs.timestep_shift),
            "image_or_video_shape": [
                int(value) for value in self.config.image_or_video_shape
            ],
            "sfp_training": self.sfp_training,
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "dfake_gen_update_ratio": (
                int(self.config.dfake_gen_update_ratio)
                if self.uses_critic
                else 1
            ),
            "batch_size": int(self.config.batch_size),
            "sequence_parallel_size": 1,
            "data_parallel_size": int(self.data_parallel_size),
            "generator_optimizer": {
                "type": "AdamW",
                "lr": float(self.config.lr),
                "beta1": float(self.config.beta1),
                "beta2": float(self.config.beta2),
                "weight_decay": float(self.config.weight_decay),
            },
            "adapter": canonical_lora_adapter_config(self.config.adapter),
        }
        if self.uses_critic:
            contract["critic_optimizer"] = {
                "type": "AdamW",
                "lr": float(getattr(self.config, "lr_critic", self.config.lr)),
                "beta1": float(self.config.beta1_critic),
                "beta2": float(self.config.beta2_critic),
                "weight_decay": float(self.config.weight_decay),
            }
        else:
            contract["cfg_guidance"] = {
                "state_source": str(self.config.cfg_state_source),
                "teacher_guidance_scale": float(
                    self.config.teacher_guidance_scale
                ),
                "teacher_sampling_steps": int(
                    self.config.teacher_sampling_steps
                ),
                "loss_weighting": str(
                    getattr(self.config, "cfg_distill_loss_weighting", "uniform")
                ),
                "relative_guidance_epsilon": float(
                    getattr(self.config, "cfg_relative_guidance_epsilon", 1.0e-6)
                ),
                "relative_guidance_max_weight": float(
                    self.config.cfg_relative_guidance_max_weight
                ),
            }
        return contract

    def _capture_rank_training_state(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        return {
            "rank": rank,
            "step": int(self.step),
            "training_contract": self._training_contract(),
            "data": self.dataloader.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
        }

    def _restore_rank_training_state(self):
        state = self._resume_rank_training_state
        if state is None:
            return
        if int(state.get("step", -1)) != self.step:
            raise RuntimeError(
                f"Rank training-state step={state.get('step')} does not match "
                f"model step={self.step}"
            )
        saved_contract = state.get("training_contract")
        current_contract = self._training_contract()
        if saved_contract != current_contract:
            raise RuntimeError(
                "Cannot exactly resume because the training contract changed: "
                f"saved={saved_contract}, current={current_contract}"
            )

        self.dataloader.load_state_dict(state["data"])
        # Recreate the sampler/worker position before restoring the model RNG.
        # This prevents DataLoader iterator construction or skipped samples
        # from perturbing the next model-side random draw.
        self.dataloader.prepare()
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"], device=self.device)
        if self.is_main_process:
            data_state = state["data"]
            print(
                "Exact training state restored: "
                f"step={self.step}, data_epoch={data_state['epoch']}, "
                f"batch_in_epoch={data_state['batch_in_epoch']}, "
                f"batches_consumed={data_state['batches_consumed']}"
            )
        self._resume_rank_training_state = None


    def find_latest_checkpoint(
        self,
        logdir,
        required_keys=None,
        require_lora_optimizer_state=False,
    ):
        """Find the newest complete checkpoint, falling back if validation fails.

        ``required_keys`` is used for the comparatively small LoRA checkpoints.
        Full-model checkpoints are not eagerly loaded here because validating a
        multi-billion-parameter state dict would double resume memory and I/O.
        """
        if not os.path.exists(logdir):
            return None

        checkpoint_dirs = []
        incomplete_resume_checkpoints = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir = os.path.join(logdir, item)
                    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
                    if not os.path.isfile(checkpoint_path):
                        if require_lora_optimizer_state:
                            incomplete_resume_checkpoints.append(
                                (step, checkpoint_dir, [checkpoint_path])
                            )
                        continue
                    if require_lora_optimizer_state:
                        paths = self._lora_optimizer_checkpoint_paths(
                            checkpoint_dir
                        )
                        required_paths = [
                            paths["generator"],
                            paths["training"],
                            paths["success"],
                        ]
                        if getattr(self, "uses_critic", True):
                            required_paths.append(paths["critic"])
                        missing_paths = [
                            path for path in required_paths if not os.path.isfile(path)
                        ]
                        if missing_paths:
                            incomplete_resume_checkpoints.append(
                                (step, checkpoint_dir, missing_paths)
                            )
                            continue
                    checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue

        if not checkpoint_dirs:
            if require_lora_optimizer_state and incomplete_resume_checkpoints:
                step, checkpoint_dir, missing_paths = max(
                    incomplete_resume_checkpoints, key=lambda item: item[0]
                )
                raise RuntimeError(
                    "Found LoRA checkpoints but none can be exactly resumed. "
                    f"Latest incomplete checkpoint is step {step} at "
                    f"{checkpoint_dir}; missing={missing_paths}"
                )
            return None

        # Sort newest first. When LoRA keys are requested, validate and fall
        # back to the previous committed checkpoint instead of failing resume
        # because the newest file is corrupt or from another training mode.
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        invalid_candidates = []
        for directory_step, checkpoint_path in checkpoint_dirs:
            if required_keys or require_lora_optimizer_state:
                try:
                    candidate = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    missing = [
                        key for key in (required_keys or ()) if key not in candidate
                    ]
                    if int(candidate.get("step", -1)) != directory_step:
                        raise ValueError(
                            f"model step {candidate.get('step')} does not match "
                            f"directory step {directory_step}"
                        )
                    if require_lora_optimizer_state:
                        if int(candidate.get("checkpoint_format_version", -1)) != 3:
                            raise ValueError("checkpoint_format_version must be 3")
                        expected_optimizer_files = {
                            "generator": LORA_GENERATOR_OPTIMIZER_FILENAME,
                            "training": LORA_TRAINING_STATE_FILENAME,
                        }
                        if getattr(self, "uses_critic", True):
                            expected_optimizer_files["critic"] = (
                                LORA_CRITIC_OPTIMIZER_FILENAME
                            )
                        if candidate.get("optimizer_state_files") != expected_optimizer_files:
                            raise ValueError(
                                "optimizer_state_files does not match the exact "
                                f"resume contract: {candidate.get('optimizer_state_files')}"
                            )
                    if require_lora_optimizer_state:
                        checkpoint_dir = os.path.dirname(checkpoint_path)
                        paths = self._lora_optimizer_checkpoint_paths(
                            checkpoint_dir
                        )
                        with open(
                            paths["success"], "r", encoding="ascii"
                        ) as marker_file:
                            marker = marker_file.read()
                        if marker != f"step={directory_step}\n":
                            raise ValueError(
                                f"invalid completion marker {marker!r}"
                            )
                        training_state = torch.load(
                            paths["training"],
                            map_location="cpu",
                            weights_only=False,
                        )
                        if (
                            int(training_state.get("checkpoint_format_version", -1))
                            != 1
                            or int(training_state.get("step", -1))
                            != directory_step
                            or int(training_state.get("world_size", -1))
                            != int(self.world_size)
                        ):
                            raise ValueError(
                                "training-state metadata does not match the "
                                "checkpoint directory/current world size"
                            )
                        del training_state
                    del candidate
                    if missing:
                        raise ValueError(f"missing keys {missing}")
                except Exception as exc:
                    invalid_candidates.append((checkpoint_path, str(exc)))
                    if getattr(self, "is_main_process", False):
                        print(
                            f"Warning: Skipping invalid checkpoint "
                            f"{checkpoint_path}: {exc}"
                        )
                    continue
            return checkpoint_path
        if (required_keys or require_lora_optimizer_state) and invalid_candidates:
            raise RuntimeError(
                "Found checkpoint artifacts but none passed exact-resume "
                f"validation: {invalid_candidates}"
            )
        return None

    def get_all_checkpoints(self, logdir, require_complete_lora=False):
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
                    if not os.path.exists(checkpoint_file_path):
                        continue
                    if require_complete_lora:
                        paths = self._lora_optimizer_checkpoint_paths(
                            checkpoint_dir_path
                        )
                        required_paths = [
                            paths["generator"],
                            paths["training"],
                            paths["success"],
                        ]
                        if self.uses_critic:
                            required_paths.append(paths["critic"])
                        if not all(os.path.isfile(path) for path in required_paths):
                            continue
                        with open(
                            paths["success"], "r", encoding="ascii"
                        ) as marker_file:
                            if marker_file.read() != f"step={step}\n":
                                continue
                    checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue

        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(
        self, logdir, max_checkpoints, require_complete_lora=False
    ):
        """Remove old checkpoints if the number exceeds max_checkpoints.

        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return

        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return

        checkpoints = self.get_all_checkpoints(
            logdir, require_complete_lora=require_complete_lora
        )
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

    def _save_lora_checkpoint(self):
        checkpoint_dir = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}"
        )
        paths = self._lora_optimizer_checkpoint_paths(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, "model.pt")

        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if os.path.exists(paths["success"]):
                os.remove(paths["success"])

        local_training_state = self._capture_rank_training_state()
        if dist.is_initialized():
            rank_training_states = [None] * self.world_size
            dist.all_gather_object(rank_training_states, local_training_state)
        else:
            rank_training_states = [local_training_state]
        del local_training_state

        generator_lora_state = self._gather_lora_state_dict(
            self.model.generator.model
        )
        critic_lora_state = (
            self._gather_lora_state_dict(self.model.fake_score.model)
            if self.uses_critic
            else None
        )
        if self.is_main_process:
            optimizer_state_files = {
                "generator": LORA_GENERATOR_OPTIMIZER_FILENAME,
                "training": LORA_TRAINING_STATE_FILENAME,
            }
            if self.uses_critic:
                optimizer_state_files["critic"] = (
                    LORA_CRITIC_OPTIMIZER_FILENAME
                )
            adapter_checkpoint = {
                "checkpoint_format_version": 3,
                "generator_lora": generator_lora_state,
                "adapter": OmegaConf.to_container(
                    self.config.adapter, resolve=True
                ),
                "adapter_contract": canonical_lora_adapter_config(
                    self.config.adapter
                ),
                "optimizer_state_files": optimizer_state_files,
                "step": self.step,
            }
            if self.uses_critic:
                adapter_checkpoint["critic_lora"] = critic_lora_state
            self._atomic_torch_save(adapter_checkpoint, checkpoint_file)
            print(f"LoRA weights saved to {checkpoint_file}")
            del adapter_checkpoint
        del generator_lora_state
        del critic_lora_state
        if dist.is_initialized():
            dist.barrier()

        generator_optimizer_state = self._gather_fsdp_optimizer_state(
            self.model.generator, self.generator_optimizer
        )
        if self.is_main_process:
            self._atomic_torch_save(
                generator_optimizer_state, paths["generator"]
            )
            print(f"Generator optimizer saved to {paths['generator']}")
        del generator_optimizer_state
        if dist.is_initialized():
            dist.barrier()

        critic_optimizer_state = None
        if self.uses_critic:
            critic_optimizer_state = self._gather_fsdp_optimizer_state(
                self.model.fake_score, self.critic_optimizer
            )
        if self.is_main_process:
            if self.uses_critic:
                self._atomic_torch_save(critic_optimizer_state, paths["critic"])
                print(f"Critic optimizer saved to {paths['critic']}")
                del critic_optimizer_state
            training_checkpoint = {
                "checkpoint_format_version": 1,
                "step": int(self.step),
                "world_size": int(self.world_size),
                "rank_states": rank_training_states,
            }
            self._atomic_torch_save(training_checkpoint, paths["training"])
            print(f"Exact training state saved to {paths['training']}")
            del training_checkpoint
            success_tmp = f"{paths['success']}.tmp.{os.getpid()}"
            try:
                with open(success_tmp, "w", encoding="ascii") as marker:
                    marker.write(f"step={self.step}\n")
                    marker.flush()
                    os.fsync(marker.fileno())
                os.replace(success_tmp, paths["success"])
            finally:
                if os.path.exists(success_tmp):
                    os.remove(success_tmp)
            print(f"Complete LoRA checkpoint saved to {checkpoint_dir}")
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(
                    self.output_path,
                    max_checkpoints,
                    require_complete_lora=True,
                )
        elif self.uses_critic:
            del critic_optimizer_state
        del rank_training_states

        if dist.is_initialized():
            dist.barrier()

    def save(self):
        self._save_lora_checkpoint()
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _detach_log_tensors(log_dict):
        detached = {}
        for key, value in log_dict.items():
            detached[key] = value.detach() if torch.is_tensor(value) else value
        return detached

    def fwdbwd_one_step(self, batch, train_generator):
        if not train_generator and not self.uses_critic:
            raise RuntimeError(
                "CFG-only distillation has no critic training step."
            )

        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]


        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        use_cached_trajectory = (
            self.cfg_state_source == "teacher_trajectory_cache"
        )
        cached_generator_kwargs = {}
        if use_cached_trajectory:
            cached_generator_kwargs = {
                "cached_noisy_latent": batch["cfg_noisy_latent"].to(
                    device=self.device, dtype=self.dtype
                ),
                "cached_guided_target": batch["cfg_guided_target"].to(
                    device=self.device, dtype=torch.float32
                ),
                "cached_timestep": batch["cfg_timestep"].to(
                    device=self.device, dtype=torch.float32
                ),
                "cached_guidance_energy": batch["cfg_guidance_energy"].to(
                    device=self.device, dtype=torch.float32
                ),
            }
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # Flatten one text prompt per sample for the frozen text encoder.
            text_prompts_flat = [p for sublist in text_prompts for p in sublist]
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts_flat)

            if use_cached_trajectory:
                # The cached target already contains the frozen teacher's
                # unconditional CFG branch. Training performs student CFG=1.
                unconditional_dict = None
            elif not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                **cached_generator_kwargs,
            )
            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})
            generator_log_dict = self._detach_log_tensors(generator_log_dict)

            return generator_log_dict
        else:
            generator_log_dict = {}

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )

        # Scale loss for gradient accumulation and backward
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})
        critic_log_dict = self._detach_log_tensors(critic_log_dict)

        return critic_log_dict


    def _run_sfp_optimization_step(self, train_generator):
        """Run one optimization step in the original SFP update order.

        The generator is updated before the critic rollout is generated.  This
        is observably different from accumulating both losses and stepping the
        two optimizers together because critic_loss samples from the generator.
        """
        if train_generator:
            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_logs = []
            for _ in range(self.gradient_accumulation_steps):
                generator_batch = next(self.dataloader)
                generator_logs.append(
                    self.fwdbwd_one_step(generator_batch, True)
                )
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )
            generator_log_dict = merge_dict_list(generator_logs)
            generator_log_dict["generator_grad_norm"] = generator_grad_norm
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
        else:
            generator_log_dict = {}

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_logs = []
        for _ in range(self.gradient_accumulation_steps):
            critic_batch = next(self.dataloader)
            critic_logs.append(self.fwdbwd_one_step(critic_batch, False))
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic
        )
        critic_log_dict = merge_dict_list(critic_logs)
        critic_log_dict["critic_grad_norm"] = critic_grad_norm
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        return generator_log_dict, critic_log_dict

    def train(self):
        start_step = self.step
        max_iters = self.config.max_iters
        if self.step >= max_iters:
            if self.is_main_process:
                print(
                    f"Training already reached max_iters={max_iters} "
                    f"at step {self.step}; exiting without another iteration."
                )
            return

        max_runtime_seconds = getattr(self.config, "max_runtime_seconds", None)
        env_max_runtime_seconds = os.environ.get("TRAIN_MAX_RUNTIME_SECONDS")
        if env_max_runtime_seconds:
            max_runtime_seconds = float(env_max_runtime_seconds)
        runtime_start_time = time.time()

        try:
            if section_get(self.config, "evaluation", "before_train", False):
                self._visualize()

            while self.step < self.config.max_iters:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = (
                    not self.uses_critic
                    or self.step % self.config.dfake_gen_update_ratio == 0
                )

                if getattr(self, "sfp_training", False):
                    generator_log_dict, critic_log_dict = (
                        self._run_sfp_optimization_step(TRAIN_GENERATOR)
                    )
                else:
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    if self.uses_critic:
                        self.critic_optimizer.zero_grad(set_to_none=True)

                    # LongLive's shared-batch gradient accumulation path.
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    for _ in range(self.gradient_accumulation_steps):
                        batch = next(self.dataloader)
                        if TRAIN_GENERATOR:
                            accumulated_generator_logs.append(
                                self.fwdbwd_one_step(batch, True)
                            )
                        if self.uses_critic:
                            accumulated_critic_logs.append(
                                self.fwdbwd_one_step(batch, False)
                            )

                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(
                            self.max_grad_norm_generator
                        )
                        generator_log_dict = merge_dict_list(
                            accumulated_generator_logs
                        )
                        generator_log_dict["generator_grad_norm"] = (
                            generator_grad_norm
                        )
                        self.generator_optimizer.step()
                    else:
                        generator_log_dict = {}

                    if self.uses_critic:
                        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                            self.max_grad_norm_critic
                        )
                        critic_log_dict = merge_dict_list(accumulated_critic_logs)
                        critic_log_dict["critic_grad_norm"] = critic_grad_norm
                        self.critic_optimizer.step()
                    else:
                        critic_log_dict = {}

                if getattr(self.config, "empty_cache_after_step", False):
                    gc.collect()
                    torch.cuda.empty_cache()

                # Increment the step since we finished gradient update
                self.step += 1

                # Create EMA params (if not already created)

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
                        for metric_name in (
                            "generator_loss",
                            "generator_grad_norm",
                            "dmdtrain_gradient_norm",
                            "cfg_distill_loss",
                            "teacher_guidance_delta_norm",
                            "student_teacher_rmse",
                            "timestep_mean",
                            "relative_guidance_median_floor",
                            "relative_guidance_weight_max",
                        ):
                            if metric_name in generator_log_dict:
                                wandb_loss_dict[metric_name] = self._metric_value(
                                    generator_log_dict[metric_name]
                                )
                        if "relative_guidance_weight_max" in generator_log_dict:
                            value = generator_log_dict[
                                "relative_guidance_weight_max"
                            ]
                            wandb_loss_dict[
                                "relative_guidance_weight_max"
                            ] = (
                                value.detach().float().max().item()
                                if torch.is_tensor(value)
                                else float(value)
                            )

                    if self.uses_critic:
                        wandb_loss_dict.update(
                            {
                                "critic_loss": self._metric_value(
                                    critic_log_dict["critic_loss"]
                                ),
                                "critic_grad_norm": self._metric_value(
                                    critic_log_dict["critic_grad_norm"]
                                ),
                            }
                        )
                    if not self.disable_wandb:
                        wandb.log(wandb_loss_dict, step=self.step)
                    metrics_path = os.environ.get("CFG_TRAIN_METRICS_PATH")
                    metrics_jsonl_path = os.environ.get(
                        "CFG_TRAIN_METRICS_JSONL_PATH"
                    )
                    if metrics_path or metrics_jsonl_path:
                        import json
                        from pathlib import Path

                        metrics_payload = {
                            "step": int(self.step),
                            "cfg_state_source": self.cfg_state_source,
                            **wandb_loss_dict,
                        }
                        if metrics_path:
                            path = Path(metrics_path)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            temporary = path.with_suffix(path.suffix + ".tmp")
                            temporary.write_text(
                                json.dumps(
                                    metrics_payload,
                                    allow_nan=False,
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            temporary.replace(path)
                        if metrics_jsonl_path:
                            jsonl_path = Path(metrics_jsonl_path)
                            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                            with jsonl_path.open("a", encoding="utf-8") as handle:
                                handle.write(
                                    json.dumps(
                                        metrics_payload,
                                        allow_nan=False,
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )

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
                    progress_parts = [
                        f"step {self.step}",
                        f"per iteration time {iteration_time}",
                    ]
                    if TRAIN_GENERATOR and generator_log_dict:
                        for metric_name in (
                            "generator_loss",
                            "generator_grad_norm",
                            "dmdtrain_gradient_norm",
                            "teacher_guidance_delta_norm",
                            "student_teacher_rmse",
                            "timestep_mean",
                            "relative_guidance_median_floor",
                            "relative_guidance_weight_max",
                        ):
                            if metric_name in generator_log_dict:
                                progress_parts.append(
                                    f"{metric_name} "
                                    f"{self._metric_value(generator_log_dict[metric_name])}"
                                )
                    if self.uses_critic:
                        progress_parts.extend(
                            [
                                f"critic_loss {self._metric_value(critic_log_dict['critic_loss'])}",
                                f"critic_grad_norm {self._metric_value(critic_log_dict['critic_grad_norm'])}",
                            ]
                        )
                    print(", ".join(progress_parts))
                    history_path = os.environ.get("TRAIN_METRICS_JSONL")
                    if history_path:
                        import json
                        history = {
                            "step": int(self.step), "time": current_time,
                            "iteration_seconds": iteration_time,
                            "generator_updated": bool(TRAIN_GENERATOR),
                            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                            **wandb_loss_dict,
                        }
                        with open(history_path, "a", encoding="utf-8") as stream:
                            stream.write(json.dumps(history, allow_nan=False) + "\n")

                # ---------------------------------------- Visualization ---------------------------------------------------

                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):
                    self._visualize()

                # Save after every RNG-consuming iteration side effect (most
                # notably visualization), so the persisted RNG state really
                # is the state immediately before the next training step.
                saved_this_step = False
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                    saved_this_step = True

                if self.step >= self.config.max_iters:
                    break

                if max_runtime_seconds is not None and max_runtime_seconds > 0:
                    elapsed_runtime = time.time() - runtime_start_time
                    if self._synchronized_runtime_stop_requested(
                        elapsed_runtime=elapsed_runtime,
                        max_runtime_seconds=max_runtime_seconds,
                    ):
                        if self.is_main_process:
                            print(
                                f"Reached TRAIN_MAX_RUNTIME_SECONDS={max_runtime_seconds} "
                                f"after {elapsed_runtime:.1f}s at step {self.step}; "
                                "saving checkpoint and stopping training."
                            )
                        if (
                            not getattr(self.config, "no_save", False)
                            and (self.step - start_step) > 0
                            and not saved_this_step
                        ):
                            torch.cuda.empty_cache()
                            self.save()
                            torch.cuda.empty_cache()
                        break

        except Exception as e:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[ERROR] [Rank {rank}] Training crashed at step {self.step} with exception: {e}")
            print(f"[ERROR] [Rank {rank}] Exception traceback:", flush=True)
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _metric_value(value):
        if torch.is_tensor(value):
            return value.detach().float().mean().item()
        return float(value)

    def _synchronized_runtime_stop_requested(
        self,
        *,
        elapsed_runtime,
        max_runtime_seconds,
    ):
        """Return rank 0's runtime decision on every distributed rank.

        Checkpoint gathering is collective, so one rank must never enter the
        time-slice save while its peers continue training. Rank 0 is the
        canonical clock and broadcasts one decision at the end of each full
        optimization step.
        """

        local_stop = elapsed_runtime >= max_runtime_seconds
        if not dist.is_initialized():
            return local_stop

        stop_flag = torch.tensor(
            1 if (self.is_main_process and local_stop) else 0,
            dtype=torch.int32,
            device=self.device,
        )
        dist.broadcast(stop_flag, src=0)
        return bool(stop_flag.item())

    def _configure_lora_for_model(self, transformer, model_name):
        return configure_lora_for_model(
            transformer, model_name, self.lora_config,
            is_main_process=self.is_main_process,
        )


    def _load_lora_state_dict_compat(self, lora_model, lora_state_dict, model_name):
        load_lora_checkpoint(
            lora_model,
            lora_state_dict,
            model_name,
            is_main_process=self.is_main_process,
        )


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
        """Prepare full-sequence validation output."""

        self.vis_pipeline = "bidirectional"

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
        guidance_scale = section_get(
            self.config,
            "inference",
            "guidance_scale",
            getattr(self.config, "guidance_scale", 1.0),
        )
        use_cfg = guidance_scale != 1.0
        if use_cfg:
            unconditional_dict = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size
            )
        else:
            unconditional_dict = None

        sample_dtype = torch.float32
        noise = torch.randn(
            [batch_size, num_frames,
             self.config.image_or_video_shape[2],
             self.config.image_or_video_shape[3],
             self.config.image_or_video_shape[4]],
            device=self.device, dtype=sample_dtype)

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
            device_is_cuda = isinstance(self.device, int) or torch.device(self.device).type == "cuda"
            autocast_enabled = (
                torch.cuda.is_available()
                and device_is_cuda
                and self.dtype != torch.float32
            )
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
                flow_pred_cond, _ = self.model.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                if use_cfg:
                    flow_pred_uncond, _ = self.model.generator(
                        noisy_image_or_video=latents,
                        conditional_dict=unconditional_dict,
                        timestep=timestep,
                    )
                    flow_pred = flow_pred_uncond + guidance_scale * (
                        flow_pred_cond - flow_pred_uncond
                    )
                else:
                    flow_pred = flow_pred_cond
            flow_pred = flow_pred.to(latents.dtype)
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

        mode_info = "_lora"
        if self.is_main_process:
            print(f"Generating latents in LoRA mode (step {self.step})")

        for vid_len in self.vis_video_lengths:
            print(f"Generating validation samples of length {vid_len}")
            samples = self._generate_bidirectional(vid_len, prompts)
            if not self.save_vis_latents_only:
                samples = self.model.vae.decode_to_pixel(samples)
                samples = (samples * 0.5 + 0.5).clamp(0, 1)
                samples = samples.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0

            for idx in range(samples.shape[0]):
                if self.save_vis_latents_only:
                    sample_name = f"latents_step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.pt"
                    out_path = os.path.join(step_vis_dir, sample_name)
                    torch.save(samples[idx].cpu(), out_path)
                else:
                    sample_name = f"video_step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                    out_path = os.path.join(step_vis_dir, sample_name)
                    write_video(out_path, torch.as_tensor(samples[idx]).to(torch.uint8), fps=self.fps)

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

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Synchronize all ranks so that a crashed rank is detected immediately
        # rather than causing a 10-minute NCCL timeout on the next training collective.
        dist.barrier()

        return True
