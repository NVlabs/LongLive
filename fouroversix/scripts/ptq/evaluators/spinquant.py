from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fouroversix
import modal
from fouroversix import ModelQuantizationConfig

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from ..utils import get_model_size
from .evaluator import PTQEvaluator

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

spinquant_img = get_image(
    dependencies=[
        Dependency.fast_hadamard_transform,
        Dependency.fouroversix,
        Dependency.spinquant,
    ],
    extra_pip_dependencies=["transformers<5.0"],
)


MIN_MODEL_SIZE_FOR_8xB200 = 32
SPINQUANT_STEPS = 100

SPINQUANT_ARGS = [
    "--model_max_length",
    "8192",
    "--fp16",
    "False",
    "--bf16",
    "True",
    "--w_bits",
    "4",
    "--a_bits",
    "4",
    "--k_bits",
    "16",
    "--v_bits",
    "16",
]


@app.cls(
    image=spinquant_img,
    timeout=24 * 60 * 60,
    secrets=[hf_secret],
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SpinQuantOptimizer:
    """Optimize a model with SpinQuant."""

    def optimize(
        self,
        model_name: str,
        *,
        quantization_config: ModelQuantizationConfig,
        spinquant_save_path: str,
        spinquant_steps: int,
    ) -> None:
        """Optimize a model with SpinQuant."""

        subprocess.run(
            [
                "torchrun",
                "--nnodes=1",
                "--nproc_per_node=auto",
                (
                    Path(fouroversix.__file__).parent.parent.parent
                    / "third_party"
                    / "spinquant"
                    / "optimize_rotation.py"
                ).as_posix(),
                "--input_model",
                model_name,
                "--output_dir",
                spinquant_save_path,
                "--output_rotation_path",
                spinquant_save_path,
                "--log_on_each_node",
                "False",
                "--per_device_train_batch_size",
                "1",
                "--logging_steps",
                "1",
                "--learning_rate",
                "1.5",
                "--weight_decay",
                "0.",
                "--lr_scheduler_type",
                "cosine",
                "--gradient_checkpointing",
                "True",
                "--save_safetensors",
                "False",
                "--max_steps",
                str(spinquant_steps),
                "--activation_scale_rule",
                quantization_config.activation_scale_rule.value,
                "--weight_scale_rule",
                quantization_config.weight_scale_rule.value,
                *SPINQUANT_ARGS,
            ],
            check=True,
        )

        cache_volume.commit()

    @modal.method()
    def optimize_on_modal(
        self,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """Optimize a model with SpinQuant on Modal."""
        return self.optimize(*args, **kwargs)


@app.cls(
    image=spinquant_img,
    timeout=24 * 60 * 60,
    secrets=[hf_secret],
    gpu="B200",
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SpinQuantEvaluator(PTQEvaluator):
    """Evaluate a quantized model with SpinQuant."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        save_path: Path,
        quantization_config: ModelQuantizationConfig,
        trust_remote_code: bool,
    ) -> AutoModelForCausalLM:
        """Export a quantized model with SpinQuant."""

        import fouroversix

        sys.path.append(
            (
                Path(fouroversix.__file__).parent.parent.parent
                / "third_party"
                / "spinquant"
            ).as_posix(),
        )

        from eval_utils.main import ptq_model
        from transformers import AutoConfig, AutoModelForCausalLM
        from utils.process_args import process_args_ptq

        save_path = (
            save_path
            / "spinquant"
            / (
                f"{model_name}-{quantization_config.activation_scale_rule.value}"
                f"-{quantization_config.weight_scale_rule.value}"
            )
        )

        if not (save_path / "R.bin").exists():
            model_is_large = get_model_size(model_name) >= MIN_MODEL_SIZE_FOR_8xB200

            if model_is_large:
                msg = (
                    "Automatic SpinQuant optimization is not supported for large "
                    "models. Please optimize the model manually."
                )
                raise RuntimeError(msg)

            SpinQuantOptimizer().optimize(
                model_name,
                quantization_config=quantization_config,
                spinquant_save_path=save_path.as_posix(),
                spinquant_steps=SPINQUANT_STEPS,
            )

        sys.argv = [
            sys.argv[0],
            "--input_model",
            model_name,
            "--do_train",
            "False",
            "--do_eval",
            "True",
            "--per_device_eval_batch_size",
            "4",
            "--rotate",
            "--optimized_rotation_path",
            (save_path / "R.bin").as_posix(),
            "--activation_scale_rule",
            quantization_config.activation_scale_rule.value,
            "--weight_scale_rule",
            quantization_config.weight_scale_rule.value,
            *SPINQUANT_ARGS,
        ]

        config = AutoConfig.from_pretrained(model_name)

        # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings,
        # clone lm_head from embed_tokens
        process_word_embeddings = False
        if config.tie_word_embeddings:
            config.tie_word_embeddings = False
            process_word_embeddings = True

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

        if process_word_embeddings:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

        model.to(device)
        model_args, _, ptq_args = process_args_ptq()

        cache_volume.reload()

        model = ptq_model(ptq_args, model, model_args)
        model.to(device)

        return model
