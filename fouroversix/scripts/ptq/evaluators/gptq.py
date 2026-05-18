import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fouroversix
from fouroversix import ModelQuantizationConfig

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

CALIBRATION_DATASET = "wikitext"

gptq_img = get_image(
    dependencies=[
        Dependency.fast_hadamard_transform,
        Dependency.qutlass,
        Dependency.fp_quant,
        Dependency.fouroversix,
    ],
)


@app.cls(
    image=gptq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
)
class GPTQEvaluator(PTQEvaluator):
    """Evaluate a model after quantizing it with GPTQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        save_path: Path,
        quantization_config: ModelQuantizationConfig,
        trust_remote_code: bool,
    ) -> "AutoModelForCausalLM":
        """Quantize a model with GPTQ."""

        sys.path.extend(
            [
                (
                    Path(fouroversix.__file__).parent.parent.parent
                    / "third_party"
                    / "fp-quant"
                ).as_posix(),
            ],
        )

        from model_quant import main
        from transformers import AutoModelForCausalLM

        save_path = save_path / "gptq" / model_name / quantization_config.__hash__()

        if not save_path.exists():
            sys.argv = [
                sys.argv[0],
                "--model_name_or_path",
                model_name,
                "--dataset_name_or_path",
                CALIBRATION_DATASET,
                "--w_bits",
                "4",
                "--a_bits",
                "4",
                "--export_quantized_model",
                "realquant",
                "--format",
                "nvfp",
                "--gptq",
                "--save_path",
                save_path.as_posix(),
                "--a_scale_rule",
                quantization_config.activation_scale_rule.value,
                "--w_scale_rule",
                quantization_config.weight_scale_rule.value,
            ]

            main()

        return AutoModelForCausalLM.from_pretrained(
            save_path,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
