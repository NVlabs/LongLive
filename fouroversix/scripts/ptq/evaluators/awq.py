from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fouroversix import FourOverSixLinear, ModelQuantizationConfig
from fouroversix.model.quantize import QuantizedModule

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .rtn import RTNEvaluatorImpl

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

awq_img = get_image(dependencies=[Dependency.fouroversix, Dependency.awq])


class FourOverSixLinearForAWQ(FourOverSixLinear):
    """
    Drop-in replacement for `FourOverSixLinear` that quantizes the weights and
    activations during AWQ calibration.
    """

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.config.keep_master_weights = True
        self.high_precision = False

    def apply_ptq(self) -> None:
        """
        Override the parent method to do nothing, since we need the high-precision
        weight when calibrating with AWQ.
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that can optionally be run in high precision. This is used to
        calculate the high-precision output to compare to during the auto-scale process
        in AWQ calibration.
        """

        return (
            F.linear(input, self.weight, self.bias)
            if self.high_precision
            else super().forward(input)
        )


@app.cls(
    image=awq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class AWQEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using AWQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        save_path: Path,
        quantization_config: ModelQuantizationConfig,
        trust_remote_code: bool,
    ) -> AutoModelForCausalLM:
        """Quantize a model using AWQ."""

        import torch
        from awq.quantize.pre_quant import apply_awq, run_awq
        from fouroversix import quantize_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Replace FourOverSixLinear with FourOverSixLinearForAWQ
        QuantizedModule.register(
            nn.Linear,
            replace_existing_modules_in_registry=True,
        )(FourOverSixLinearForAWQ)

        save_path = save_path / "awq" / model_name / quantization_config.__hash__()

        if not save_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=trust_remote_code,
            ).eval()

            quantize_model(model, quantization_config)

            enc = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )

            awq_results = run_awq(
                model,
                enc,
                w_bit=16,
                q_config={"q_group_size": -1, "zero_point": False},
                n_samples=128,
                seqlen=512,
                calib_data="wikitext",
            )

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(awq_results, save_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

        # Apply AWQ
        awq_results = torch.load(save_path, map_location="cuda")
        apply_awq(model, awq_results)

        # Quantize the model
        quantize_model(model, quantization_config)

        return model.to(device)
