# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/trainer.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import hashlib
import json
import os
import re
import signal
from collections.abc import Mapping
from math import ceil, cos, pi
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from .utils.checkpoint_initialization import initialize_from_checkpoint
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    RESUME_CONTRACT_VERSION = 2
    DEFAULT_LR_WARMUP_RATIO = 0.05
    DEFAULT_COSINE_ETA_MIN_RATIO = 0.01
    WANDB_RESUME_MODES = frozenset({"allow", "auto", "must", "never"})

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self._adam_beta1_explicit = "adam_beta1" in cfg
        self._adam_beta2_explicit = "adam_beta2" in cfg
        self._adam_epsilon_explicit = "adam_epsilon" in cfg
        self.adam_beta1 = float(cfg.get("adam_beta1", 0.9))
        self.adam_beta2 = float(cfg.get("adam_beta2", 0.95))
        self.adam_epsilon = float(cfg.get("adam_epsilon", 1.0e-8))
        for name, value in (
            ("adam_beta1", self.adam_beta1),
            ("adam_beta2", self.adam_beta2),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"`{name}` must be in [0, 1), got {value}.")
        if not np.isfinite(self.adam_epsilon) or self.adam_epsilon <= 0.0:
            raise ValueError(
                f"`adam_epsilon` must be finite and positive, got {self.adam_epsilon}."
            )
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        prefetch_factor = cfg.get("prefetch_factor")
        self.prefetch_factor = None if prefetch_factor is None else int(prefetch_factor)
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1 when configured.")
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        raw_scheduler_max_steps = cfg.get("scheduler_max_steps")
        self._scheduler_max_steps_explicit = raw_scheduler_max_steps is not None
        self._configured_scheduler_max_steps = (
            None if raw_scheduler_max_steps is None else int(raw_scheduler_max_steps)
        )
        if (
            self._configured_scheduler_max_steps is not None
            and self._configured_scheduler_max_steps <= 0
        ):
            raise ValueError("`scheduler_max_steps` must be positive when configured.")
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        save_state_every = cfg.get("save_state_every")
        self.save_state_every = (
            self.save_every if save_state_every is None else int(save_state_every)
        )
        self.save_final_state = bool(cfg.get("save_final_state", True))
        self.save_final_weights = bool(cfg.get("save_final_weights", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        sampler_cfg = cfg.get("sampler") or {}
        if isinstance(sampler_cfg, DictConfig):
            sampler_cfg = OmegaConf.to_container(sampler_cfg, resolve=True)
        self.sampler_strategy = str(sampler_cfg.get("strategy", "frame_uniform"))
        self.sampler_domain_weights = sampler_cfg.get("domain_weights")
        self.sampler_domain_strategies = sampler_cfg.get("domain_strategies")
        self.sampler_samples_per_epoch = sampler_cfg.get("samples_per_epoch")
        self._sampler_epoch_offset_explicit = "epoch_offset" in sampler_cfg
        self.sampler_epoch_offset = int(sampler_cfg.get("epoch_offset", 0))
        if self.sampler_epoch_offset < 0:
            raise ValueError("`sampler.epoch_offset` must be non-negative.")

        self.resume = cfg.resume
        init_checkpoint = cfg.get("init_checkpoint")
        if isinstance(init_checkpoint, DictConfig):
            init_checkpoint = OmegaConf.to_container(init_checkpoint, resolve=True)
        if init_checkpoint is not None and not isinstance(init_checkpoint, Mapping):
            raise TypeError(
                "`init_checkpoint` must be null or a mapping with `path` and "
                f"`adapter`, got {type(init_checkpoint)}."
            )
        self.init_checkpoint = init_checkpoint
        if self.resume and self.init_checkpoint is not None:
            raise ValueError(
                "`resume` and `init_checkpoint` are mutually exclusive: resume "
                "restores an existing run, while init_checkpoint starts a new run."
            )
        self.checkpoint_strict = bool(cfg.get("checkpoint_strict", False))
        self.lr_scheduler_type = str(cfg.lr_scheduler_type).strip().lower()
        self.lr_warmup_ratio = float(cfg.get("lr_warmup_ratio", self.DEFAULT_LR_WARMUP_RATIO))
        if not 0.0 <= self.lr_warmup_ratio < 1.0:
            raise ValueError(f"`lr_warmup_ratio` must be in [0, 1), got {self.lr_warmup_ratio}.")
        self._lr_eta_min_ratio_explicit = "lr_eta_min_ratio" in cfg
        self.lr_eta_min_ratio = float(
            cfg.get("lr_eta_min_ratio", self.DEFAULT_COSINE_ETA_MIN_RATIO)
        )
        if not 0.0 <= self.lr_eta_min_ratio <= 1.0:
            raise ValueError(f"`lr_eta_min_ratio` must be in [0, 1], got {self.lr_eta_min_ratio}.")
        self._lr_warmup_start_factor_explicit = "lr_warmup_start_factor" in cfg
        raw_warmup_start_factor = cfg.get("lr_warmup_start_factor")
        self.lr_warmup_start_factor = (
            None if raw_warmup_start_factor is None else float(raw_warmup_start_factor)
        )
        if self.lr_warmup_start_factor is not None and not 0.0 < self.lr_warmup_start_factor <= 1.0:
            raise ValueError(
                f"`lr_warmup_start_factor` must be in (0, 1], got {self.lr_warmup_start_factor}."
            )
        self._configure_resume_contract()
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self._configure_wandb_resume_settings()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        deepspeed_config = deepspeed_plugin.deepspeed_config if deepspeed_plugin is not None else {}
        deepspeed_grad_clip = float(deepspeed_config.get("gradient_clipping", 0.0))
        if "DEEPSPEED" in str(self.accelerator.distributed_type).upper() and not np.isclose(
            deepspeed_grad_clip, self.max_grad_norm
        ):
            raise ValueError(
                "DeepSpeed performs its optimizer step inside accelerator.backward; "
                "its gradient_clipping setting must match cfg.max_grad_norm: "
                f"deepspeed={deepspeed_grad_clip}, cfg={self.max_grad_norm}."
            )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f ds_grad_clip=%.4f",
            self.accelerator.distributed_type,
            deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            deepspeed_grad_clip,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Initial weights must be loaded before Adam/DeepSpeed creates optimizer
        # master parameters. Loading them after `accelerator.prepare` updates the
        # bf16 model but leaves ZeRO's fp32 masters stale, so the first optimizer
        # step can silently restore the pre-checkpoint weights.
        self.initialization_report = None
        self._load_initialization_checkpoint_before_optimizer()
        self._weight_checkpoint_loaded = False
        self._load_weight_checkpoint_before_optimizer()

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
        )
        self._before_optimizer_prepare()

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        self.scheduler_max_steps = (
            total_train_steps
            if self._configured_scheduler_max_steps is None
            else self._configured_scheduler_max_steps
        )
        if self.scheduler_max_steps > total_train_steps:
            raise ValueError(
                "`scheduler_max_steps` cannot exceed the training horizon: "
                f"scheduler={self.scheduler_max_steps}, training={total_train_steps}."
            )
        warmup_steps = int(self.scheduler_max_steps * self.lr_warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=self.lr_scheduler_type,
            total_train_steps=self.scheduler_max_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._last_weights_checkpoint_step = -1
        self._last_state_checkpoint_step = -1
        self._termination_requested = False
        self._graceful_deadline_unix = self._read_graceful_deadline()

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._resume_or_load_checkpoint()
        self._init_wandb()

        val_size = (
            len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        )
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)
        if self._graceful_deadline_unix is not None:
            logger.info(
                "Graceful chunk deadline: unix=%d remaining_seconds=%d",
                self._graceful_deadline_unix,
                max(self._graceful_deadline_unix - int(time.time()), 0),
            )

    def _before_optimizer_prepare(self) -> None:
        """Optional fail-fast validation before wrappers replace parameter IDs."""

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        run_id, resume_mode = self._wandb_resume_settings
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            id=run_id,
            resume=resume_mode,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s id=%s resume=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
            run_id,
            resume_mode,
        )

    def _configure_wandb_resume_settings(self) -> None:
        self._wandb_resume_settings = (
            self._wandb_resume_settings_from_env() if self.wandb_enabled else (None, None)
        )

    @classmethod
    def _wandb_resume_settings_from_env(cls) -> tuple[str | None, str | None]:
        run_id = os.environ.get("WANDB_RUN_ID")
        resume_mode = os.environ.get("WANDB_RESUME")
        if run_id is not None:
            if not run_id or re.search(r"[\s/\\#?%:]", run_id):
                raise ValueError(
                    "WANDB_RUN_ID must be non-empty and cannot contain whitespace "
                    "or any of: / \\ # ? % :."
                )
        if resume_mode is not None:
            if resume_mode not in cls.WANDB_RESUME_MODES:
                raise ValueError(
                    "WANDB_RESUME must be one of "
                    f"{sorted(cls.WANDB_RESUME_MODES)}, got {resume_mode!r}."
                )
        if (run_id is None) != (resume_mode is None):
            raise ValueError(
                "WANDB_RUN_ID and WANDB_RESUME must either both be set or both "
                "be unset so chunked training has an unambiguous run identity."
            )
        return run_id, resume_mode

    def _configure_resume_contract(self) -> None:
        contract_cfg = self.cfg.get("resume_contract") or {}
        if OmegaConf.is_config(contract_cfg):
            contract_cfg = OmegaConf.to_container(contract_cfg, resolve=True)
        if not isinstance(contract_cfg, Mapping):
            raise TypeError("`resume_contract` must be null or a mapping.")
        unknown = sorted(
            set(contract_cfg).difference(
                {
                    "required",
                    "init_checkpoint_sha256",
                    "normalization_stats_sha256",
                }
            )
        )
        if unknown:
            raise ValueError(f"Unknown `resume_contract` options: {unknown}.")

        self.resume_contract_required = bool(contract_cfg.get("required", False))
        self.original_init_checkpoint_sha256 = self._resolve_contract_sha256(
            config_digest=contract_cfg.get("init_checkpoint_sha256"),
            environment_name="INIT_CHECKPOINT_SHA256",
            identity_name="initialization checkpoint",
        )
        self.normalization_stats_sha256 = self._resolve_contract_sha256(
            config_digest=contract_cfg.get("normalization_stats_sha256"),
            environment_name="NORMALIZATION_STATS_SHA256",
            identity_name="normalization stats",
        )
        if (
            self.resume_contract_required
            and (self.resume or self.init_checkpoint is not None)
            and self.original_init_checkpoint_sha256 is None
        ):
            raise ValueError(
                "A required resume contract needs the original initialization "
                "checkpoint SHA-256. Set INIT_CHECKPOINT_SHA256 or "
                "`resume_contract.init_checkpoint_sha256`."
            )
        if self.resume_contract_required and self.normalization_stats_sha256 is None:
            raise ValueError(
                "A required resume contract needs the normalization stats "
                "SHA-256. Set NORMALIZATION_STATS_SHA256 or "
                "`resume_contract.normalization_stats_sha256`."
            )

    @staticmethod
    def _resolve_contract_sha256(
        *,
        config_digest,
        environment_name: str,
        identity_name: str,
    ) -> str | None:
        environment_digest = os.environ.get(environment_name)
        if config_digest is not None:
            config_digest = str(config_digest)
        if environment_digest is not None and not environment_digest:
            raise ValueError(f"{environment_name} cannot be empty when exported.")
        if (
            config_digest is not None
            and environment_digest is not None
            and config_digest != environment_digest
        ):
            raise ValueError(
                f"{identity_name.capitalize()} SHA-256 disagrees between the "
                f"resume-contract config and {environment_name}."
            )

        digest = config_digest if config_digest is not None else environment_digest
        if digest is not None and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"The resume-contract {identity_name} identity must be a "
                f"lowercase SHA-256 digest, got {digest!r}."
            )
        return digest

    @staticmethod
    def _canonical_json(payload) -> str:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _json_sha256(cls, payload) -> str:
        return hashlib.sha256(cls._canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def _normalize_json(cls, payload):
        return json.loads(cls._canonical_json(payload))

    def _resolved_train_data_config(self) -> dict | None:
        data_cfg = self.cfg.get("data")
        if data_cfg is None:
            return None
        if OmegaConf.is_config(data_cfg):
            data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        if not isinstance(data_cfg, Mapping):
            return None
        train_cfg = data_cfg.get("train")
        if train_cfg is None:
            return None
        if OmegaConf.is_config(train_cfg):
            train_cfg = OmegaConf.to_container(train_cfg, resolve=True)
        if not isinstance(train_cfg, Mapping):
            return None
        return self._normalize_json(dict(train_cfg))

    def _data_resume_contract(self) -> dict:
        contract = {
            "normalization": {
                "stats_sha256": self.normalization_stats_sha256,
            }
        }
        sampling_signature = getattr(self.train_dataset, "sampling_signature", None)
        if sampling_signature is not None:
            contract["sampling_signature"] = str(sampling_signature)

        train_data_cfg = self._resolved_train_data_config()
        if train_data_cfg is not None:
            contract["train_config_sha256"] = self._json_sha256(train_data_cfg)
            episode_selection = train_data_cfg.get("human300_episode_selection")
            if episode_selection is not None:
                contract["official_episode_selection"] = episode_selection
                contract["official_episode_selection_config_sha256"] = self._json_sha256(
                    episode_selection
                )

        resolved_selection = getattr(self.train_dataset, "human300_episode_selection", None)
        resolved_signature = getattr(resolved_selection, "signature", None)
        if resolved_signature is not None:
            contract["official_episode_selection_signature"] = str(resolved_signature)
        return contract

    def _sampler_resume_contract(self) -> dict:
        sampler = self.train_sampler
        domain_parameters = {
            "names": list(getattr(sampler, "domain_names", ())),
            "weights": dict(getattr(sampler, "domain_weights", {})),
            "strategies": dict(getattr(sampler, "domain_strategies", {})),
            "sampling_signatures": dict(getattr(sampler, "domain_sampling_signatures", {})),
            "quotas": (
                None
                if getattr(sampler, "domain_quotas", None) is None
                else dict(sampler.domain_quotas)
            ),
            "epoch_quotas": dict(getattr(sampler, "domain_epoch_quotas", {})),
            "quota_ranges": {
                name: list(quota_range)
                for name, quota_range in getattr(sampler, "domain_quota_ranges", {}).items()
            },
        }
        contract = {
            "strategy": str(sampler.strategy),
            "group_signature": str(sampler.group_signature),
            "world_size": int(self.accelerator.num_processes),
            "seed": int(sampler.seed),
            "samples_per_epoch": int(sampler.samples_per_epoch),
            "domain_parameters": domain_parameters,
        }
        if getattr(self, "_sampler_epoch_offset_explicit", False):
            contract["epoch_offset"] = int(self.sampler_epoch_offset)
        return contract

    def _build_training_resume_contract(self) -> dict:
        scheduler_max_steps = int(getattr(self, "scheduler_max_steps", self.max_steps))
        warmup_steps = int(scheduler_max_steps * self.lr_warmup_ratio)
        scheduler_contract = {
            "type": self.lr_scheduler_type,
            "warmup_ratio": float(self.lr_warmup_ratio),
            "warmup_steps": warmup_steps,
        }
        if getattr(self, "_scheduler_max_steps_explicit", False):
            scheduler_contract["horizon_steps"] = scheduler_max_steps
        if getattr(self, "_lr_warmup_start_factor_explicit", False):
            scheduler_contract["warmup_start_factor"] = self.lr_warmup_start_factor
        if getattr(self, "_lr_eta_min_ratio_explicit", False):
            scheduler_contract["eta_min_ratio"] = float(self.lr_eta_min_ratio)

        optimizer_contract = {
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
        }
        if getattr(self, "_adam_beta1_explicit", False):
            optimizer_contract["beta1"] = float(self.adam_beta1)
        if getattr(self, "_adam_beta2_explicit", False):
            optimizer_contract["beta2"] = float(self.adam_beta2)
        if getattr(self, "_adam_epsilon_explicit", False):
            optimizer_contract["epsilon"] = float(self.adam_epsilon)

        payload = {
            "version": self.RESUME_CONTRACT_VERSION,
            "training": {
                "max_steps": int(self.max_steps),
                "optimizer": optimizer_contract,
                "scheduler": scheduler_contract,
                "batch": {
                    "per_process": int(self.batch_size),
                    "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
                    "effective_global": int(
                        self.batch_size
                        * self.gradient_accumulation_steps
                        * self.accelerator.num_processes
                    ),
                },
                "save_schedule": {
                    "weights_every": int(self.save_every),
                    "state_every": int(self.save_state_every),
                    "save_final_state": bool(self.save_final_state),
                },
            },
            "sampler": self._sampler_resume_contract(),
            "data": self._data_resume_contract(),
            "initialization": {
                "checkpoint_sha256": self.original_init_checkpoint_sha256,
            },
        }
        payload = self._normalize_json(payload)
        payload["signature"] = self._json_sha256(payload)
        return payload

    @classmethod
    def _contract_mismatches(cls, saved, expected, path: str = "") -> list[str]:
        if isinstance(saved, Mapping) and isinstance(expected, Mapping):
            mismatches = []
            for key in sorted(set(saved).union(expected)):
                child_path = f"{path}.{key}" if path else str(key)
                if key not in saved:
                    mismatches.append(f"{child_path}: saved=<missing>, current={expected[key]!r}")
                elif key not in expected:
                    mismatches.append(f"{child_path}: saved={saved[key]!r}, current=<missing>")
                else:
                    mismatches.extend(
                        cls._contract_mismatches(saved[key], expected[key], child_path)
                    )
            return mismatches
        if saved != expected:
            return [f"{path}: saved={saved!r}, current={expected!r}"]
        return []

    def _validate_training_resume_contract(
        self,
        payload: dict | None,
        state_file: Path,
    ) -> None:
        saved = None if payload is None else payload.get("resume_contract")
        if saved is None:
            message = f"State file `{state_file}` has no immutable training resume contract."
            if self.resume_contract_required:
                raise ValueError(
                    f"{message} `resume_contract.required=true` forbids this legacy resume."
                )
            logger.warning(
                "%s Continuing in legacy compatibility mode without full "
                "optimizer/scheduler/data identity validation.",
                message,
            )
            return
        if not isinstance(saved, Mapping):
            raise ValueError(f"Training resume contract in `{state_file}` must be a mapping.")

        saved_payload = dict(saved)
        saved_signature = saved_payload.pop("signature", None)
        try:
            computed_signature = self._json_sha256(saved_payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Training resume contract in `{state_file}` is not canonical JSON."
            ) from exc
        if not isinstance(saved_signature, str) or saved_signature != computed_signature:
            raise ValueError(
                "Training resume contract integrity mismatch in "
                f"`{state_file}`: saved_signature={saved_signature!r}, "
                f"computed_signature={computed_signature!r}."
            )

        expected = self._build_training_resume_contract()
        expected_payload = dict(expected)
        expected_payload.pop("signature")
        mismatches = self._contract_mismatches(saved_payload, expected_payload)
        if mismatches:
            raise ValueError(
                "Training resume contract mismatch in "
                f"`{state_file}`: {'; '.join(mismatches)}. Resume with the "
                "original immutable training configuration."
            )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def close(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _handle_sigterm(self, _signum, _frame):
        self._termination_requested = True

    @staticmethod
    def _read_graceful_deadline() -> int | None:
        raw_deadline = os.environ.get("LONGWAM_GRACEFUL_DEADLINE_UNIX")
        if raw_deadline in (None, ""):
            return None
        if re.fullmatch(r"[1-9][0-9]*", raw_deadline) is None:
            raise ValueError(
                "LONGWAM_GRACEFUL_DEADLINE_UNIX must be a positive Unix timestamp, "
                f"got {raw_deadline!r}."
            )
        return int(raw_deadline)

    def _termination_requested_across_ranks(self) -> bool:
        # Opt-in, allocation-specific request from a CPU supervisor. Do not
        # signal torchrun itself: it can terminate ranks before they save.
        stop_file = os.environ.get("LONGWAM_GRACEFUL_STOP_FILE")
        if stop_file and self.accelerator.is_main_process and os.path.isfile(stop_file):
            self._termination_requested = True
        if self._graceful_deadline_unix is not None and time.time() >= self._graceful_deadline_unix:
            self._termination_requested = True
        requested = torch.tensor(
            int(self._termination_requested),
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        requested = self.accelerator.reduce(requested, reduction="sum")
        self._termination_requested = bool(requested.item())
        return self._termination_requested

    def _checkpoint_if_termination_requested(self) -> bool:
        if not self._termination_requested_across_ranks():
            return False

        weights_path = os.path.join(
            self.weights_dir,
            f"step_{self.global_step:06d}.pt",
        )
        state_path = os.path.join(
            self.state_dir,
            f"step_{self.global_step:06d}",
        )
        need_weights = self._last_weights_checkpoint_step != self.global_step
        need_state = self._last_state_checkpoint_step != self.global_step
        if need_weights or need_state:
            checkpoint = self.save_checkpoint(
                save_weights=need_weights,
                save_state=need_state,
            )
            if checkpoint["weights_path"] is not None:
                weights_path = checkpoint["weights_path"]
            if checkpoint["state_path"] is not None:
                state_path = checkpoint["state_path"]
        if self.accelerator.is_main_process:
            logger.info(
                "[term] graceful termination requested; checkpoint is complete "
                "at step=%d weights=%s state=%s",
                self.global_step,
                weights_path,
                state_path,
            )
        return True

    def _checkpoint_if_chunk_must_stop(self) -> bool:
        """Let a terminal optimizer step finish its normal logging/save path."""
        if self.max_steps is not None and self.global_step >= self.max_steps:
            # Still synchronize the request so a terminal SIGTERM can suppress
            # an expensive scheduled evaluation on every rank.
            self._termination_requested_across_ranks()
            return False
        return self._checkpoint_if_termination_requested()

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            strategy=self.sampler_strategy,
            domain_weights=self.sampler_domain_weights,
            domain_strategies=self.sampler_domain_strategies,
            samples_per_epoch=self.sampler_samples_per_epoch,
        )
        self.train_sampler.set_epoch_offset(self.sampler_epoch_offset)
        logger.info(
            "Training sampler: strategy=%s groups=%d signature=%s",
            self.train_sampler.strategy,
            len(self.train_sampler.dataset_frame_ranges),
            self.train_sampler.group_signature,
        )
        if self.train_sampler.strategy == "hierarchical":
            logger.info(
                "Hierarchical sampler: samples_per_epoch=%d global_micro_batch=%d "
                "domain_quotas=%s domain_strategies=%s data_fingerprints=%s",
                len(self.train_sampler),
                self.train_sampler.global_micro_batch_size,
                (
                    dict(self.train_sampler.domain_quotas)
                    if self.train_sampler.domain_quotas is not None
                    else {
                        name: list(quota_range)
                        for name, quota_range in self.train_sampler.domain_quota_ranges.items()
                    }
                ),
                dict(self.train_sampler.domain_strategies),
                dict(self.train_sampler.domain_sampling_signatures),
            )
        dataloader_kwargs = {}
        if self.num_workers > 0 and self.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            **dataloader_kwargs,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(
                f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:"
            )
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_sampler) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        if scheduler_type == "cosine_hf":

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_train_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + cos(pi * progress)))

            return LambdaLR(self.optimizer, lr_lambda)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * self.lr_eta_min_ratio,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'cosine_hf', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=(
                1.0 / warmup_steps
                if self.lr_warmup_start_factor is None
                else self.lr_warmup_start_factor
            ),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _step_scheduler_after_optimizer(self) -> None:
        """Advance LR only inside its declared horizon.

        Most recipes use the same scheduler and training horizon. Continuations
        may explicitly keep a completed schedule at its final LR while running a
        short, predeclared data-coverage tail.
        """

        scheduler_max_steps = int(
            getattr(
                self,
                "scheduler_max_steps",
                getattr(self, "max_steps", 2**63 - 1),
            )
        )
        if self.global_step < scheduler_max_steps:
            self.scheduler.step()

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not self._weight_checkpoint_loaded:
            raise RuntimeError(
                "Weight-only checkpoint was not loaded before optimizer initialization: "
                f"{resume_path}"
            )
        logger.warning(
            "Loaded .pt weights before optimizer initialization; optimizer, scheduler, "
            "and global step start fresh."
        )

    def _load_weight_checkpoint_before_optimizer(self):
        """Load file checkpoints before optimizer/ZeRO master weights are created."""
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint before optimizer initialization: %s", resume)
        self.model.load_checkpoint(
            str(resume_path),
            optimizer=None,
            strict=self.checkpoint_strict,
        )
        self._weight_checkpoint_loaded = True

    def _load_initialization_checkpoint_before_optimizer(self):
        """Initialize a new run from a schema-adapted checkpoint."""
        if self.init_checkpoint is None:
            return

        allowed = {"path", "adapter", "gripper"}
        unknown = sorted(set(self.init_checkpoint) - allowed)
        if unknown:
            raise ValueError(f"Unknown `init_checkpoint` options: {unknown}.")
        missing = [key for key in ("path", "adapter") if not self.init_checkpoint.get(key)]
        if missing:
            raise ValueError(f"`init_checkpoint` is missing required options: {missing}.")

        logger.info(
            "Initializing a new training run before optimizer creation: %s",
            self.init_checkpoint["path"],
        )
        self.initialization_report = initialize_from_checkpoint(
            self.model,
            checkpoint_path=self.init_checkpoint["path"],
            adapter=str(self.init_checkpoint["adapter"]),
            gripper=self.init_checkpoint.get("gripper"),
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(
                f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}"
            )
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(
                f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(
                f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}"
            )

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"`sample['action']` must be a torch.Tensor, got {type(action)}")
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(
                    f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
                )
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(
                    f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}"
                )
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(
            self.global_step + self.accelerator.process_index
        )
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]  # Tensor [3, T, H, W] in (-1, 1)
        action = (
            sample["action"][0] if "action" in sample and sample["action"] is not None else None
        )
        proprio = (
            sample["proprio"][0, 0]
            if "proprio" in sample and sample["proprio"] is not None
            else None
        )  # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = (
            (video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
        ).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(
                        f"{action_name} action must be a torch.Tensor, got {type(raw_action)}"
                    )
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {
                        meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta
                    },
                    "state": {
                        meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta
                    },
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (
                stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
            ).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        temporary = ckpt_path + ".incomplete"
        model.save_checkpoint(temporary, optimizer=None, step=self.global_step)
        os.replace(temporary, ckpt_path)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "sampler": {
                "strategy": self.train_sampler.strategy,
                "group_signature": self.train_sampler.group_signature,
                "world_size": int(self.accelerator.num_processes),
            },
            "resume_contract": self._build_training_resume_contract(),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _validate_sampler_resume_contract(self, payload: dict, state_file: Path):
        saved = payload.get("sampler")
        if saved is None:
            logger.warning(
                "State file `%s` predates the sampler resume contract; "
                "continuing without sampler strategy/group/world-size validation.",
                state_file,
            )
            return

        expected = {
            "strategy": self.train_sampler.strategy,
            "group_signature": self.train_sampler.group_signature,
            "world_size": int(self.accelerator.num_processes),
        }
        mismatches = []
        for key, expected_value in expected.items():
            saved_value = saved.get(key)
            if saved_value != expected_value:
                mismatches.append(f"{key}: saved={saved_value!r}, current={expected_value!r}")
        if mismatches:
            raise ValueError(
                "Sampler resume contract mismatch in "
                f"`{state_file}`: {'; '.join(mismatches)}. "
                "Resume with the original sampler strategy, dataset groups, and world size."
            )

    def _restore_dataloader_progress(self, *, epoch: int, batch_in_epoch: int):
        """Restore the next unread batch, including an epoch-boundary save."""
        if epoch < 0 or batch_in_epoch < 0:
            raise ValueError(
                "Dataloader progress must be non-negative, "
                f"got epoch={epoch}, batch_in_epoch={batch_in_epoch}."
            )

        batches_per_epoch = len(self.train_loader)
        if batches_per_epoch <= 0:
            raise ValueError("Cannot resume from an empty training dataloader.")
        if batch_in_epoch > batches_per_epoch:
            raise ValueError(
                "Saved dataloader progress exceeds the prepared epoch length: "
                f"batch_in_epoch={batch_in_epoch}, batches_per_epoch={batches_per_epoch}."
            )

        saved_epoch = epoch
        saved_batch_in_epoch = batch_in_epoch
        if batch_in_epoch == batches_per_epoch:
            # Checkpoints are written after consuming a batch but before the loop
            # observes StopIteration. Accelerate's DataLoaderShard does not advance
            # its iteration counter when the resumed iterator is already empty, so
            # leaving this as an end-of-epoch offset would replay the saved epoch.
            epoch += 1
            batch_in_epoch = 0

        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        self.train_sampler.set_epoch_offset(getattr(self, "sampler_epoch_offset", 0) + self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        logger.info(
            "Restored dataloader progress: saved_epoch=%d "
            "saved_batch_in_epoch=%d epoch=%d batch_in_epoch=%d sample_offset=%d",
            saved_epoch,
            saved_batch_in_epoch,
            self.epoch,
            self.batch_in_epoch,
            self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
        )

    def save_checkpoint(self, *, save_weights: bool = True, save_state: bool = True):
        if not save_weights and not save_state:
            raise ValueError("At least one checkpoint artifact must be requested.")
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if save_weights and self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()
        if save_weights:
            self._last_weights_checkpoint_step = self.global_step

        state_path = None
        if save_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            if self.accelerator.is_main_process:
                Path(state_path, ".longwam-complete").unlink(missing_ok=True)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()
            self._last_state_checkpoint_step = self.global_step

        if save_weights and save_state and self.accelerator.is_main_process:
            Path(state_path, ".longwam-complete").touch()
            from .utils.checkpoint_retention import prune_checkpoint_pairs

            try:
                prune_checkpoint_pairs(
                    self.weights_dir, self.state_dir, self.cfg.get("checkpoint_keep_last"),
                    self.cfg.get("protected_checkpoint_steps", ()),
                )
            except Exception:
                # A cleanup failure must not strand other ranks in a collective.
                logger.exception("Checkpoint committed, but retention cleanup failed")
        self.accelerator.wait_for_everyone()
        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._validate_training_resume_contract(payload, state_file)
            self._validate_sampler_resume_contract(payload, state_file)
        else:
            self._validate_training_resume_contract(None, state_file)

        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self._restore_dataloader_progress(
                    epoch=int(payload["epoch"]),
                    batch_in_epoch=int(payload["batch_in_epoch"]),
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d",
            state_dir,
            self.global_step,
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        try:
            return self._train_loop()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

    @staticmethod
    def _require_finite_training_values(
        *,
        loss: torch.Tensor | None = None,
        components: Mapping[str, object] | None = None,
        grad_norm: object | None = None,
    ) -> None:
        if loss is not None:
            if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).item()):
                raise FloatingPointError("Training loss is not a finite scalar.")
        if components is not None:
            nonfinite = [
                str(name) for name, value in components.items() if not np.isfinite(float(value))
            ]
            if nonfinite:
                raise FloatingPointError(f"Training loss components are non-finite: {nonfinite}.")
        if grad_norm is not None and not np.isfinite(float(grad_norm)):
            raise FloatingPointError("Training gradient norm is non-finite.")

    def _train_loop(self):
        self._set_dit_only_train_mode()

        if self.max_steps is None:
            raise ValueError(
                "`max_steps` must be set before entering the while-step training loop."
            )

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        domain_names = tuple(getattr(self.train_dataset, "domain_names", ()))
        accumulated_domain_counts = (
            torch.zeros(
                len(domain_names),
                device=self.accelerator.device,
                dtype=torch.long,
            )
            if domain_names
            else None
        )

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            if accumulated_domain_counts is not None:
                domain_index = sample.get("domain_index")
                if domain_index is None:
                    raise KeyError(
                        "Mixed training samples must include `domain_index` for ratio auditing."
                    )
                accumulated_domain_counts += torch.bincount(
                    domain_index.reshape(-1).to(
                        device=self.accelerator.device,
                        dtype=torch.long,
                    ),
                    minlength=len(domain_names),
                )

            with self.accelerator.accumulate(self.model):
                train_model = (
                    self.model
                    if hasattr(self.model, "training_loss")
                    else self.accelerator.unwrap_model(self.model)
                )

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._require_finite_training_values(
                    loss=loss,
                    components=loss_dict,
                )
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self._require_finite_training_values(grad_norm=grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self._step_scheduler_after_optimizer()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self._checkpoint_if_chunk_must_stop():
                        return
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(
                            float(value), device=loss.device, dtype=torch.float32
                        ).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(
                        grad_norm, device=loss.device, dtype=torch.float32
                    )
                    global_grad_norm = float(
                        self.accelerator.gather(grad_norm_tensor).mean().item()
                    )
                    global_domain_fractions = {}
                    if accumulated_domain_counts is not None:
                        global_domain_counts = self.accelerator.reduce(
                            accumulated_domain_counts,
                            reduction="sum",
                        )
                        total_domain_samples = int(global_domain_counts.sum().item())
                        if total_domain_samples <= 0:
                            raise RuntimeError("Mixed training domain counter is empty.")
                        global_domain_fractions = {
                            name: float(global_domain_counts[index].item()) / total_domain_samples
                            for index, name in enumerate(domain_names)
                        }
                        accumulated_domain_counts.zero_()

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                        and self.accelerator.is_main_process
                    ):
                        eta_str, steps_per_sec = self._estimate_eta()
                        samples_per_sec = (
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps
                        )
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join(
                                [f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())]
                            )
                            description += detail_str + " "
                        if global_domain_fractions:
                            description += (
                                "domains="
                                + ",".join(
                                    f"{name}:{fraction:.3f}"
                                    for name, fraction in global_domain_fractions.items()
                                )
                                + " "
                            )
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            samples_per_sec,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        for name, fraction in global_domain_fractions.items():
                            wandb_payload[f"data/domain_fraction_{name}"] = fraction
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                        and not self._termination_requested
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = (
                                "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f"
                                % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    save_weights = self.save_every > 0 and self.global_step % self.save_every == 0
                    save_state = (
                        self.save_state_every > 0 and self.global_step % self.save_state_every == 0
                    )
                    if save_weights or save_state:
                        ckpt_info = self.save_checkpoint(
                            save_weights=save_weights,
                            save_state=save_state,
                        )
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        need_weights = getattr(self, "save_final_weights", True) and (
                            self._last_weights_checkpoint_step != self.global_step
                        )
                        need_state = (
                            self.save_final_state
                            and self._last_state_checkpoint_step != self.global_step
                        )
                        if need_weights or need_state:
                            ckpt_info = self.save_checkpoint(
                                save_weights=need_weights,
                                save_state=need_state,
                            )
                        else:
                            ckpt_info = {
                                "weights_path": (
                                    os.path.join(
                                        self.weights_dir,
                                        f"step_{self.global_step:06d}.pt",
                                    )
                                    if self._last_weights_checkpoint_step == self.global_step
                                    else None
                                ),
                                "state_path": (
                                    os.path.join(
                                        self.state_dir,
                                        f"step_{self.global_step:06d}",
                                    )
                                    if self._last_state_checkpoint_step == self.global_step
                                    else None
                                ),
                            }
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        save_final_weights = bool(getattr(self, "save_final_weights", True))
        if not save_final_weights and not self.save_final_state:
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d without terminal checkpoint",
                    self.global_step,
                )
            return
        ckpt_info = self.save_checkpoint(
            save_weights=save_final_weights,
            save_state=self.save_final_state,
        )
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
