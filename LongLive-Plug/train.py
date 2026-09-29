# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from omegaconf import OmegaConf
import torch.distributed as dist
import wandb

from trainer import ScoreDistillationTrainer
from utils.config import normalize_config, validate_cfg_only_config


def apply_model_dir_override(config, model_dir):
    """Apply a CLI model path without splitting a CFG student/teacher pair."""
    config.model_kwargs.model_dir = model_dir
    for role in ("real_model_kwargs", "fake_model_kwargs"):
        if config.get(role) is not None:
            config[role].model_dir = model_dir
    if config.get("distribution_loss") == "cfg_guidance":
        validate_cfg_only_config(config)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--generate-before-train", action="store_true", help="Run one evaluation inference before training starts")
    parser.add_argument("--data-path", type=str, default=None, help="Override the training/evaluation data path")
    parser.add_argument("--model-dir", type=str, default=None, help="Override model_kwargs.model_dir")

    args = parser.parse_args()

    config = normalize_config(OmegaConf.load(args.config_path))
    if config.get("distribution_loss") == "cfg_guidance" and config.cfg_state_source != "teacher_trajectory_cache":
        raise ValueError("CFG training requires a teacher trajectory cache")
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.generate_before_train = args.generate_before_train
    if args.data_path is not None:
        config.data_path = args.data_path
        config.eval_data_path = args.data_path
    if args.model_dir is not None:
        apply_model_dir_override(config, args.model_dir)

    try:
        if config.trainer == "score_distillation":
            trainer = ScoreDistillationTrainer(config)
        else:
            raise ValueError(f"Unsupported trainer: {config.trainer}")
        trainer.train()
    finally:
        wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
