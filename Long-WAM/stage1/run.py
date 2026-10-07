#!/usr/bin/env python3
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot-video integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/run.py
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

"""LongLive 2.0 Robot S/M/L training and checkpoint validation."""
import argparse
from datetime import timedelta
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "third_party/LongLive"
STEPS = {"S": 28602, "M": 11155, "L": 3972}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "validation"])
    p.add_argument("--stage", choices=STEPS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    p.add_argument("--generator-sha256", default="unspecified", help="Optional preverified model digest.")
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--training-step", type=int, help="Checkpoint step to label validation outputs.")
    p.add_argument("--stop-step", type=int, help="Defaults to the stage's final optimizer step.")
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--plan", action="store_true", help="Resolve configuration only; no model or output writes.")
    return p.parse_args()


def prepare(a):
    from omegaconf import OmegaConf
    if not 0 <= a.resume_step < STEPS[a.stage]:
        raise ValueError("invalid same-stage resume step")
    for key in ("output", "generator", "assets_root"):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    os.environ[f"LONGWAM_VIDEO_STAGE_{a.stage}_GENERATOR_MODEL"] = str(a.generator)
    os.environ[f"LONGWAM_VIDEO_STAGE_{a.stage}_GENERATOR_SHA256"] = a.generator_sha256
    os.environ["LONGWAM_VIDEO_VALIDATION_CHECKPOINT_MODEL"] = str(a.generator)
    path = ROOT / "configs" / f"{a.mode}_{a.stage.lower()}.yaml"
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    if a.mode == "train":
        a.stop_step = STEPS[a.stage] if a.stop_step is None else a.stop_step
        if not a.resume_step < a.stop_step <= STEPS[a.stage]:
            raise ValueError("stop step must be after the resume step and within the stage")
        if int(cfg.training.max_iters) != STEPS[a.stage]:
            raise ValueError("stage stopping step differs from configuration")
    elif a.resume_step or a.stop_step is not None or a.training_step is None or not 1 <= a.training_step <= STEPS[a.stage]:
        raise ValueError("validation requires a real --training-step, with no optimizer resume or stop step")
    return cfg, path


def main():
    a = arguments()
    cfg, config_path = prepare(a)
    from omegaconf import OmegaConf
    expected_world = 128 if a.mode == "train" else 8
    plan = dict(schema="longwam-launch-v1", mode=a.mode, stage=a.stage,
                output=str(a.output), generator=str(a.generator),
                generator_sha256=a.generator_sha256, source_step=a.resume_step,
                training_step=a.training_step, stop_step=a.stop_step,
                expected_world_size=expected_world,
                config=OmegaConf.to_container(cfg, resolve=True))
    plan["receipt_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if a.plan:
        print(json.dumps(plan, indent=2))
        return
    if int(os.environ.get("WORLD_SIZE", "0")) != expected_world:
        raise ValueError(f"this configuration requires torchrun WORLD_SIZE={expected_world}")
    if not (a.assets_root / "wan_models/Wan2.2-TI2V-5B").is_dir():
        raise FileNotFoundError("assets-root must contain wan_models/Wan2.2-TI2V-5B")
    if not a.resume_step and not a.generator.is_file():
        raise FileNotFoundError(a.generator)
    # Rank zero owns output creation; other ranks must not race its new manifest.
    if a.mode == "validation" and int(os.environ.get("RANK", "0")) == 0:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("validation requires a new output directory")
    if a.mode == "train" and not a.resume_step and a.output.exists() and any(a.output.glob("checkpoint_model_*")):
        raise ValueError("fresh initialization refuses an existing checkpoint directory")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(CORE))
    import torch
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    # Large distributed checkpoint loads may precede group collectives.
    c10d.default_pg_timeout = timedelta(minutes=60)
    c10d.default_pg_nccl_timeout = timedelta(minutes=60)
    import train
    from trainer import DiffusionTrainer
    if a.resume_step:
        bound = a.output / f"checkpoint_model_{a.resume_step:06d}/model.pt"
        original_find = DiffusionTrainer.find_latest_checkpoint
        def exact(self, directory):
            selected = original_find(self, directory)
            if selected is None or Path(selected).resolve() != bound:
                raise ValueError("requested full-state resume checkpoint is not the latest complete checkpoint")
            return selected
        DiffusionTrainer.find_latest_checkpoint = exact
    if a.mode == "validation" and a.stage in {"M", "L"}:
        from runtime import head_sharding
        head_sharding.install()
        original_init = DiffusionTrainer.__init__
        @functools.wraps(original_init)
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            head_sharding.parity_check(torch.device("cuda", torch.cuda.current_device()), torch.bfloat16)
            torch.cuda.empty_cache()
        DiffusionTrainer.__init__ = initialize
    if int(os.environ.get("RANK", "0")) == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        with (a.output / f"launch_{a.mode}_{a.resume_step:06d}.json").open("x") as stream:
            json.dump(plan, stream, indent=2)
    os.chdir(a.assets_root)
    runtime_output = a.output / "runtime" if a.mode == "validation" else a.output
    sys.argv = ["train.py", "--config_path", str(config_path), "--logdir", str(runtime_output),
                "--wandb-save-dir", str(a.output / "wandb")]
    if a.mode == "validation":
        sys.argv += ["--no_save", "--disable-wandb", "--no-auto-resume", "--generate-before-train"]
    else:
        sys.argv += ["--stop-after-step", str(a.stop_step)]
        if a.resume_step == 0:
            sys.argv.append("--no-auto-resume")
        if a.disable_wandb:
            sys.argv.append("--disable-wandb")
    train.main()


if __name__ == "__main__":
    main()
