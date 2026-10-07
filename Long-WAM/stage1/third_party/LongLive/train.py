# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: The Self-Forcing contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: train.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/train.py
# Source: https://github.com/guandeh17/Self-Forcing
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from torch import distributed as dist
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer, DiffusionTrainer
from utils.config import normalize_config


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
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help=(
            "Operationally stop this allocation after the named optimizer step. "
            "The configured max_iters remains unchanged, so a later same-stage "
            "operational resume preserves the experiment's global step schedule."
        ),
    )

    args = parser.parse_args()

    config = normalize_config(OmegaConf.load(args.config_path))
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.generate_before_train = args.generate_before_train
    if args.stop_after_step is not None:
        if args.stop_after_step < 1:
            parser.error("--stop-after-step must be at least 1")
        if args.stop_after_step > int(config.max_iters):
            parser.error(
                "--stop-after-step cannot exceed training.max_iters "
                f"({args.stop_after_step} > {int(config.max_iters)})"
            )
    config.stop_after_step = args.stop_after_step

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    trainer.train()

    wandb.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
