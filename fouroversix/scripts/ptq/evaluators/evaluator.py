from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal
import torch
from fouroversix import (
    DataType,
    MatmulBackend,
    ModelQuantizationConfig,
    QuantizeBackend,
    ScaleRule,
)

from ..utils import EvaluationFramework

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from transformers import AutoConfig, AutoModelForCausalLM


class PTQEvaluator(ABC):
    """Base class for post-training quantization evaluators."""

    @classmethod
    def get_calibration_tasks(
        cls,
        model_name: str,  # noqa: ARG003
        session: Session,  # noqa: ARG003
        **kwargs: dict[str, Any],  # noqa: ARG003
    ) -> list[dict[str, Any]]:
        """
        Get the kwargs for tasks that should be used to calibrate the given model for
        this PTQ method before running evaluation.
        """
        return []

    @classmethod
    def get_calibrated_kwargs(
        cls,
        model_name: str,  # noqa: ARG003
        session: Session,  # noqa: ARG003
        **kwargs: dict[str, Any],  # noqa: ARG003
    ) -> dict[str, Any]:
        """
        Get the calibrated kwargs for the given model and scale rules. If this model
        has not yet been calibrated with these scale rules, an error will be raised.
        """
        return {}

    @abstractmethod
    def quantize_model(self, **kwargs: dict[str, Any]) -> AutoModelForCausalLM:
        """Quantize a model."""

    def evaluate(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        eval_framework: EvaluationFramework,
        limit: int | None,
        max_length: int | None,
        tasks: list[str],
        trust_remote_code: bool = False,
        disable_inference_mode: bool = False,
        matmul_backend: MatmulBackend | None = None,
        quantize_backend: QuantizeBackend | None = None,
        weight_scale_2d: bool = False,
        activation_scale_rule: ScaleRule | None = None,
        weight_scale_rule: ScaleRule | None = None,
        save_path: Path | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        inference_context = (
            nullcontext() if disable_inference_mode else torch.inference_mode()
        )

        with inference_context:
            model_config = AutoConfig.from_pretrained(model_name)
            quantization_config = ModelQuantizationConfig(
                activation_scale_rule=activation_scale_rule,
                dtype=dtype,
                matmul_backend=matmul_backend,
                output_dtype=DataType(
                    (
                        str(model_config.dtype).replace("torch.", "")
                        if model_config.dtype is not None
                        else "bfloat16"
                    ),
                ),
                quantize_backend=quantize_backend,
                weight_scale_2d=weight_scale_2d,
                weight_scale_rule=weight_scale_rule,
            )

            model = self.quantize_model(
                model_name=model_name,
                device=device,
                save_path=save_path,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

            if eval_framework == EvaluationFramework.lm_eval:
                from lm_eval import evaluator
                from lm_eval.models.huggingface import HFLM
                from lm_eval.tasks import TaskManager

                full_results = evaluator.simple_evaluate(
                    model=HFLM(
                        pretrained=model,
                        device=device,
                        max_length=max_length,
                    ),
                    tasks=tasks,
                    device=device,
                    limit=limit,
                    task_manager=TaskManager(
                        include_path=(
                            Path(__file__).parent.parent / "tasks"
                        ).as_posix(),
                    ),
                )

                results = []

                for task in full_results["results"]:
                    result = full_results["results"][task]

                    if "acc_norm,none" in result:
                        metric_name = "acc_norm,none"
                    elif "acc,none" in result:
                        metric_name = "acc,none"
                    elif "word_perplexity,none" in result:
                        metric_name = "word_perplexity,none"
                    else:
                        metric_name = None

                    results.append(
                        (
                            task,
                            metric_name,
                            result.get(metric_name),
                            full_results["results"][task],
                        ),
                    )

            elif eval_framework == EvaluationFramework.inspect_ai:
                import inspect_ai
                from inspect_ai.model import Model
                from inspect_ai.model._generate_config import GenerateConfig

                from .utils import local_hf

                config = GenerateConfig()
                full_results = inspect_ai.eval(
                    tasks=tasks,
                    model=Model(local_hf(model_name, model, config), config, None),
                    limit=limit,
                    log_dir=(save_path / "inspect_ai_logs").as_posix(),
                    display="none",
                )

                results = []

                for log in full_results:
                    metrics = {
                        k: v.value
                        for score in log.results.scores
                        for k, v in score.metrics.items()
                    }

                    metric_name = "accuracy" if "accuracy" in metrics else None

                    results.append(
                        (
                            log.eval.task,
                            metric_name,
                            metrics.get(metric_name),
                            metrics,
                        ),
                    )

            del model
            torch.cuda.empty_cache()

        return results

    @modal.method()
    def evaluate_on_modal(
        self,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model on Modal."""
        return self.evaluate(*args, **kwargs)
