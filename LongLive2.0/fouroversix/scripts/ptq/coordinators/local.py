import itertools
import multiprocessing
from pathlib import Path
from typing import Any

import torch

from ..evaluators import get_evaluator
from ..utils import PTQMethod
from .base import BaseEvaluationCoordinator

FOUROVERSIX_ROOT_DIR = Path(__file__).parent.parent.parent.parent


class LocalEvaluationCoordinator(BaseEvaluationCoordinator):
    """Evaluation coordinator for running PTQ experiments locally."""

    def __init__(self, group_name: str | None = None) -> None:
        self.database_path = FOUROVERSIX_ROOT_DIR / "results.db"
        self.group_name = group_name

    def evaluate(
        self,
        model_name: str,
        ptq_method: PTQMethod,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a model with a given PTQ method."""

        evaluator_cls = get_evaluator(ptq_method)

        return evaluator_cls().evaluate(
            model_name=model_name,
            save_path=FOUROVERSIX_ROOT_DIR / "ptq",
            **kwargs,
        )

    def run_calibration_tasks(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Run any tasks that should be used to calibrate models for a given PTQ method
        and set of parameters before running evaluation.
        """

        experiments = 0

        for model_name, ptq_method in itertools.product(model_names, ptq_methods):
            tasks_to_evaluate = self.get_tasks_to_evaluate(
                model_name,
                ptq_method,
                tasks,
            )

            if len(tasks_to_evaluate) == 0:
                continue

            evaluator_cls = get_evaluator(ptq_method)

            for calibration_task_kwargs in evaluator_cls.get_calibration_tasks(
                model_name,
                self.get_session(),
                **kwargs,
            ):
                task_queue.put(
                    (model_name, ptq_method, {**kwargs, **calibration_task_kwargs}),
                )
                experiments += 1

        for _ in range(experiments):
            self.save_results(*result_queue.get())

    def start(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        *,
        device: str,
        **kwargs: dict[str, Any],
    ) -> None:
        """Start the evaluation coordinator."""

        multiprocessing.set_start_method("spawn", force=True)

        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Start one worker per GPU
        num_workers = torch.cuda.device_count() if device == "cuda" else 1
        workers = []

        for gpu_id in range(num_workers):
            p = multiprocessing.Process(
                target=self.worker,
                args=(
                    f"cuda:{gpu_id}" if device == "cuda" else device,
                    task_queue,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)

        # Run calibration tasks if necessary for each model and PTQ method
        self.run_calibration_tasks(
            model_names,
            ptq_methods,
            tasks,
            task_queue,
            result_queue,
            **kwargs,
        )

        # Run evaluation tasks after models have been calibrated
        experiments = 0

        for model_name, ptq_method in itertools.product(model_names, ptq_methods):
            tasks_to_evaluate = self.get_tasks_to_evaluate(
                model_name,
                ptq_method,
                tasks,
            )

            if len(tasks_to_evaluate) == 0:
                continue

            evaluator_cls = get_evaluator(ptq_method)

            calibrated_kwargs = evaluator_cls.get_calibrated_kwargs(
                model_name,
                self.get_session(),
                **kwargs,
            )

            task_queue.put(
                (
                    model_name,
                    ptq_method,
                    {**kwargs, "tasks": tasks_to_evaluate, **calibrated_kwargs},
                ),
            )
            experiments += 1

        # Send shutdown signals (one per worker)
        for _ in range(num_workers):
            task_queue.put(None)

        # Collect results
        for _ in range(experiments):
            self.save_results(*result_queue.get())

        for p in workers:
            p.join()

    def worker(
        self,
        device: str,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> None:
        """Worker process for running PTQ experiments locally."""

        while True:
            worker_task = task_queue.get()

            if worker_task is None:
                break

            model_name, ptq_method, kwargs = worker_task

            results = self.evaluate(
                model_name,
                ptq_method,
                **{**kwargs, "device": device},
            )

            result_queue.put((model_name, ptq_method, kwargs, results))
