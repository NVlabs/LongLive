from __future__ import annotations

from enum import Enum


class EvaluationFramework(str, Enum):
    """Frameworks to use when evaluating models."""

    inspect_ai = "inspect_ai"
    lm_eval = "lm_eval"


class PTQMethod(str, Enum):
    """Methods of post-training quantization."""

    awq = "awq"
    high_precision = "high_precision"
    gptq = "gptq"
    rtn = "rtn"
    smoothquant = "smoothquant"
    spinquant = "spinquant"


def get_model_size(model_name: str | None) -> float:
    return float(model_name.split("-")[-1][:-1]) if model_name else 0.0
