from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fouroversix import ModelQuantizationConfig, ScaleRule

from ...resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, hf_secret
from ..experiment import Experiment
from ..utils import PTQMethod
from .rtn import RTNEvaluatorImpl, rtn_img

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.orm import Session


with rtn_img.imports():
    import torch
    import torch.nn as nn
    from fouroversix import (
        FourOverSixLinear,
        QuantizedModule,
        fp4_matmul,
        quantize_model,
    )
    from transformers import AutoModelForCausalLM


ALPHA_CANDIDATES = [x / 10 for x in range(11)]
WIKITEXT_TRAIN = "wikitext_train"


class FourOverSixLinearWithSmoothing(FourOverSixLinear):
    """
    Drop-in replacement for `FourOverSixLinear` that implements SmoothQuant-style
    scaling.
    """

    def __init__(
        self,
        *args: list[Any],
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)
        self.smoothquant_alpha = smoothquant_alpha

    def apply_ptq(self) -> None:
        """
        Override the parent method to do nothing, since we need the high-precision
        weight when doing PTQ with SmoothQuant.
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with SmoothQuant-style scaling."""

        out = torch.empty(
            *input.shape[:-1],
            self.weight.shape[0],
            device=input.device,
            dtype=self.config.output_dtype.torch_dtype(),
        )

        fprop_activation_config = self.config.get_activation_config()
        fprop_weight_config = self.config.get_weight_config(
            block_scale_2d=self.config.weight_scale_2d,
        )

        for i in range(input.shape[0]):
            s = (input[i].abs().max(dim=0).values ** self.smoothquant_alpha) / (
                self.weight.abs().max(dim=0).values ** (1 - self.smoothquant_alpha)
            )

            out[i] = fp4_matmul(
                input[i] / s[None, :],
                self.weight * s[None, :],
                out_dtype=self.config.output_dtype,
                input_config=fprop_activation_config,
                other_config=fprop_weight_config,
            )

        if self.bias is not None:
            out = out + self.bias

        return out


@app.cls(
    image=rtn_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SmoothQuantEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using SmoothQuant."""

    @classmethod
    def get_calibration_tasks(
        cls,
        model_name: str,
        session: Session,
        **kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Get the kwargs for tasks that should be used to calibrate the given model for
        this PTQ method before running evaluation.
        """

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            kwargs.get("activation_scale_rule"),
            kwargs.get("weight_scale_rule"),
            session,
        )

        calibration_experiments = get_calibration_experiments(
            model_name,
            kwargs.get("activation_scale_rule"),
            kwargs.get("weight_scale_rule"),
            session,
        )

        if smoothquant_alpha is None:
            return [
                {
                    "smoothquant_alpha": candidate_alpha,
                    "tasks": [WIKITEXT_TRAIN],
                }
                for candidate_alpha in ALPHA_CANDIDATES
                if not any(
                    experiment.smoothquant_alpha == candidate_alpha
                    for experiment in calibration_experiments
                )
            ]

        return []

    @classmethod
    def get_calibrated_kwargs(
        cls,
        model_name: str,
        session: Session,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get the calibrated kwargs for the given model and scale rules. If this model
        has not yet been calibrated with these scale rules, an error will be raised.
        """

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            kwargs.get("activation_scale_rule"),
            kwargs.get("weight_scale_rule"),
            session,
        )

        if smoothquant_alpha is None:
            msg = (
                "SmoothQuant has not been calibrated for this combination of model and "
                "scale rules"
            )
            raise ValueError(msg)

        return {"smoothquant_alpha": smoothquant_alpha}

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        save_path: Path,  # noqa: ARG002
        smoothquant_alpha: float,
        quantization_config: ModelQuantizationConfig,
        trust_remote_code: bool,
    ) -> AutoModelForCausalLM:
        """Quantize a model using SmoothQuant."""

        # Replace FourOverSixLinear with FourOverSixLinearWithSmoothing
        QuantizedModule.register(
            nn.Linear,
            replace_existing_modules_in_registry=True,
        )(FourOverSixLinearWithSmoothing)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

        quantize_model(model, quantization_config, smoothquant_alpha=smoothquant_alpha)
        return model


def get_calibration_experiments(
    model_name: str,
    activation_scale_rule: ScaleRule,
    weight_scale_rule: ScaleRule,
    db_session: Session,
) -> list[Experiment]:
    return (
        db_session.query(Experiment)
        .filter(
            Experiment.ptq_method == PTQMethod.smoothquant.value,
            Experiment.task == WIKITEXT_TRAIN,
            Experiment.model_name == model_name,
            Experiment.activation_scale_rule == activation_scale_rule.value,
            Experiment.weight_scale_rule == weight_scale_rule.value,
            Experiment.smoothquant_alpha.isnot(None),
        )
        .all()
    )


def get_smoothquant_alpha(
    model_name: str,
    activation_scale_rule: ScaleRule,
    weight_scale_rule: ScaleRule,
    session: Session,
) -> float | None:
    calibration_experiments = get_calibration_experiments(
        model_name,
        activation_scale_rule,
        weight_scale_rule,
        session,
    )

    if not all(
        any(
            experiment.smoothquant_alpha == alpha
            for experiment in calibration_experiments
        )
        for alpha in ALPHA_CANDIDATES
    ):
        return None

    return min(calibration_experiments, key=lambda x: x.metric_value).smoothquant_alpha
