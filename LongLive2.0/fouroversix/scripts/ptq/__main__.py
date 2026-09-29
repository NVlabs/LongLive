import warnings
from typing import Any

import click
import modal
from fouroversix.utils import DataType, MatmulBackend, QuantizeBackend, ScaleRule

from ..resources import app
from .coordinators import LocalEvaluationCoordinator, ModalEvaluationCoordinator
from .utils import EvaluationFramework, PTQMethod


@click.command()
@click.option(
    "--activation-scale-rule",
    "--a-scale-rule",
    type=ScaleRule,
    default=ScaleRule.mse,
)
@click.option("--detach", is_flag=True)
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=DataType, default=DataType.nvfp4)
@click.option(
    "--eval-framework",
    "-f",
    type=EvaluationFramework,
    default=EvaluationFramework.lm_eval,
)
@click.option("--group-name", type=str, default=None)
@click.option("--limit", type=int, default=None)
@click.option("--matmul-backend", type=MatmulBackend, default=None)
@click.option("--max-length", type=int, default=None)
@click.option("--modal", is_flag=True)
@click.option("--modal-gpu", type=str)
@click.option("--model-name", "-m", type=str, multiple=True, required=True)
@click.option("--ptq-method", "-p", type=PTQMethod, multiple=True, required=True)
@click.option("--quantize-backend", type=QuantizeBackend, default=None)
@click.option("--task", "-t", type=str, multiple=True, default=["wikitext"])
@click.option("--trust-remote-code", is_flag=True)
@click.option(
    "--weight-scale-rule",
    "--w-scale-rule",
    type=ScaleRule,
    default=ScaleRule.mse,
)
@click.option("--weight-scale-2d", "--w-scale-2d", is_flag=True)
def cli(
    *,
    detach: bool,
    group_name: str | None,
    modal_gpu: str,
    **kwargs: dict[str, Any],
) -> None:
    activation_scale_rule = kwargs.get("activation_scale_rule")
    dtype = kwargs.get("dtype")
    weight_scale_rule = kwargs.get("weight_scale_rule")

    model_names = kwargs.pop("model_name")
    ptq_methods = kwargs.pop("ptq_method")
    tasks = kwargs.pop("task")
    use_modal = kwargs.pop("modal")

    # Expand shortcuts
    if model_names[0] == "llamaqwen":
        model_names = [
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-70B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-32B",
        ]

    if isinstance(tasks, tuple):
        tasks = list(tasks)

    if dtype == DataType.mxfp4 and (
        not activation_scale_rule.is_static() or not weight_scale_rule.is_static()
    ):
        msg = (
            "MXFP4 quantization only supports static scale rules. Setting "
            "activation_scale_rule and weight_scale_rule to static_6..."
        )
        warnings.warn(msg, stacklevel=1)

        kwargs["activation_scale_rule"] = ScaleRule.static_6
        kwargs["weight_scale_rule"] = ScaleRule.static_6

    if use_modal:
        with modal.enable_output(), app.run(detach=detach):
            coordinator = ModalEvaluationCoordinator(group_name_str=group_name or "")
            coordinator.start.remote(
                model_names,
                ptq_methods,
                tasks,
                modal_gpu=modal_gpu,
                **kwargs,
            )
    else:
        coordinator = LocalEvaluationCoordinator(group_name)
        coordinator.start(model_names, ptq_methods, tasks, **kwargs)


if __name__ == "__main__":
    cli()
