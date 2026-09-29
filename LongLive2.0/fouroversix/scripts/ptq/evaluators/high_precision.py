from pathlib import Path

from fouroversix import ModelQuantizationConfig

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

hp_img = get_image()

with hp_img.imports():
    from transformers import AutoModelForCausalLM


@app.cls(
    image=hp_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class HighPrecisionEvaluator(PTQEvaluator):
    """Evaluate a model while keeping it in high precision."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        save_path: Path,  # noqa: ARG002
        quantization_config: ModelQuantizationConfig,  # noqa: ARG002
        trust_remote_code: bool = False,
    ) -> "AutoModelForCausalLM":
        """Return a model without any quantization."""

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
