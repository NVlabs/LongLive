from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import click
import modal

from ..resources import (
    FOUROVERSIX_CACHE_PATH,
    FOUROVERSIX_INSTALL_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    wandb_secret,
)

img = get_image(
    dependencies=[Dependency.flash_attention, Dependency.flame, Dependency.fouroversix],
    extra_pip_dependencies=["datasets<4.6"],
)


def train(
    *,
    batch_size: int,
    checkpoint_interval: int,
    checkpoint_keep_latest_k: int,
    checkpoint_load_step: int,
    context_length: int,
    dataset: str,
    dataset_name: str,
    dataset_split: str,
    exp_folder: str,
    gradient_accumulation_steps: int,
    initial_load_path: str | None,
    job_config_file: str,
    lr: float,
    lr_decay_type: str,
    model_config: str,
    model_name: str,
    no_torch_compile: bool,
    seed: int,
    tokenizer: str,
    training_steps: int | None,
) -> None:
    import torch

    # Cache activations and gradients and set dump folder
    os.environ["CACHE_ACTIVATIONS"] = "1"
    os.environ["CACHE_GRADIENTS"] = "1"
    os.environ["DUMP_FOLDER"] = f"{exp_folder}/{model_name}"

    # Set MODEL_NAME for wandb
    os.environ["MODEL_NAME"] = model_name

    # Set NGPU for flame
    os.environ["NGPU"] = str(torch.cuda.device_count())

    if training_steps is None:
        if dataset_name == "sample-10BT":
            num_tokens = 10_000_000_000
        elif dataset_name == "sample-100BT":
            num_tokens = 100_000_000_000
        else:
            msg = (
                "You must provide the number of training steps if not using the "
                "sample-10BT or sample-100BT datasets"
            )
            raise ValueError(msg)

        training_steps = num_tokens // int(
            context_length * batch_size * torch.cuda.device_count(),
        )

    # Start training
    args = [
        "bash",
        "train.sh",
        "--job.config_file",
        job_config_file,
        "--job.dump_folder",
        f"{exp_folder}/{model_name}",
        "--model.config",
        model_config,
        "--model.tokenizer_path",
        tokenizer,
        "--optimizer.name",
        "AdamW",
        "--optimizer.lr",
        str(lr),
        "--lr_scheduler.warmup_steps",
        "0",
        "--lr_scheduler.decay_ratio",
        "0.15",
        "--lr_scheduler.decay_type",
        lr_decay_type,
        "--lr_scheduler.lr_min",
        "0.01",
        "--training.batch_size",
        "1",
        "--training.seq_len",
        str(int(context_length * batch_size)),
        "--training.context_len",
        str(context_length),
        "--training.varlen",
        "--training.gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--training.steps",
        str(training_steps),
        "--training.max_norm",
        "1.0",
        "--training.skip_nan_inf",
        "--training.dataset",
        dataset,
        "--training.dataset_name",
        dataset_name,
        "--training.dataset_split",
        dataset_split,
        "--training.num_workers",
        "32",
        "--training.prefetch_factor",
        "2",
        "--training.seed",
        str(seed),
        "--checkpoint.interval",
        str(checkpoint_interval),
        "--checkpoint.load_step",
        str(checkpoint_load_step),
        "--checkpoint.keep_latest_k",
        str(checkpoint_keep_latest_k),
        "--metrics.log_freq",
        "1",
    ]

    if not no_torch_compile:
        args.append("--training.compile")

    if initial_load_path is not None:
        args.extend(
            [
                "--checkpoint.initial_load_path",
                initial_load_path,
                "--checkpoint.no_initial_load_model_weights_only",
            ],
        )

    subprocess.run(args, check=True)


@app.cls(
    image=img,
    gpu="B200:8",
    timeout=24 * 60 * 60,
    cpu=64,
    memory=8 * 64 * 1024,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
    secrets=[wandb_secret],
)
class ModalTrainer:
    """Run training jobs on Modal."""

    @modal.method()
    def train(self, **kwargs: dict[str, Any]) -> None:
        """Start a training job on Modal."""
        os.chdir(FOUROVERSIX_INSTALL_PATH / "third_party" / "flame")
        train(**kwargs)


@click.command()
@click.option("--batch-size", type=float, default=16)
@click.option("--checkpoint-interval", type=int, default=1000)
@click.option("--checkpoint-keep-latest-k", type=int, default=0)
@click.option("--checkpoint-load-step", type=int, default=-1)
@click.option("--context-length", type=int, default=8192)
@click.option("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
@click.option("--dataset-name", type=str, default="sample-100BT")
@click.option("--dataset-split", type=str, default="train")
@click.option("--detach", is_flag=True)
@click.option("--exp-folder", type=str, default="exp")
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--initial-load-path", type=str)
@click.option("--job-config-file", type=str, default="flame/models/fla.toml")
@click.option("--lr", type=float, default=1.2e-3)
@click.option("--lr-decay-type", type=str, default="linear")
@click.option("--modal", is_flag=True)
@click.option("--modal-gpu", type=str, default="B200:8")
@click.option("--model-config", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--no-torch-compile", is_flag=True)
@click.option("--seed", type=int, default=42)
@click.option("--tokenizer", type=str, default="fla-hub/transformer-1.3B-100B")
@click.option("--training-steps", type=int, default=None)
@click.option("--wait-for-pid", type=int, default=None)
def cli(**kwargs: dict[str, Any]) -> None:
    # Options that are not passed to the train function
    detach = kwargs.pop("detach", False)
    modal_gpu = kwargs.pop("modal_gpu", "B200:8")
    use_modal = kwargs.pop("modal", False)
    wait_for_pid = kwargs.pop("wait_for_pid", None)

    # Wait for the previous training job to finish
    if wait_for_pid is not None:
        while (
            subprocess.run(["kill", "-0", str(wait_for_pid)], check=False).returncode
            == 0
        ):
            time.sleep(1)

        time.sleep(60)

    if not Path(kwargs["model_config"]).exists():
        kwargs["model_config"] = (
            Path(__file__).parent.parent.parent / kwargs["model_config"]
        )

    if not Path(kwargs["model_config"]).exists():
        msg = f"Model config file not found: {kwargs['model_config']}"
        raise FileNotFoundError(msg)

    # Set exp folder on Modal
    if use_modal:
        with modal.enable_output(), app.run(detach=detach):
            kwargs["exp_folder"] = (FOUROVERSIX_CACHE_PATH / "exp").as_posix()
            kwargs["model_config"] = (
                (FOUROVERSIX_INSTALL_PATH / kwargs["model_config"])
                .absolute()
                .as_posix()
            )

            ModalTrainer.with_options(gpu=modal_gpu)().train.remote(**kwargs)
    else:
        kwargs["model_config"] = Path(kwargs["model_config"]).absolute().as_posix()

        os.chdir(Path(__file__).parent.parent.parent / "third_party" / "flame")
        train(**kwargs)


if __name__ == "__main__":
    cli()
