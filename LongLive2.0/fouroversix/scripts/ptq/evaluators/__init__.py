from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import PTQMethod
from .awq import AWQEvaluator
from .gptq import GPTQEvaluator
from .high_precision import HighPrecisionEvaluator
from .rtn import RTNEvaluator
from .smoothquant import SmoothQuantEvaluator
from .spinquant import SpinQuantEvaluator

if TYPE_CHECKING:
    from .evaluator import PTQEvaluator


def get_evaluator(ptq_method: PTQMethod) -> type[PTQEvaluator]:
    """Get the evaluator class for the given PTQ method."""

    if ptq_method == PTQMethod.awq:
        return AWQEvaluator
    if ptq_method == PTQMethod.gptq:
        return GPTQEvaluator
    if ptq_method == PTQMethod.high_precision:
        return HighPrecisionEvaluator
    if ptq_method == PTQMethod.rtn:
        return RTNEvaluator
    if ptq_method == PTQMethod.smoothquant:
        return SmoothQuantEvaluator
    if ptq_method == PTQMethod.spinquant:
        return SpinQuantEvaluator

    msg = f"Unsupported PTQ method: {ptq_method}"
    raise ValueError(msg)
