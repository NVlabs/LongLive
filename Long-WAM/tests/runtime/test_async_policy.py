# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/runtime/test_async_policy.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Test elapsed-prefix alignment, reset and actual spawned-worker failures."""

import os
import time

import numpy as np
import pytest

from longwam.runtime.execution import AsyncPolicy, Schedule
from longwam.runtime.worker import SpawnedInferenceProcess, RemoteInferenceError


def initializer(payload):
    def handle(operation, request):
        if operation == "fail":
            raise ValueError("owned worker failure")
        if operation == "sleep":
            time.sleep(0.1)
            return os.getpid()
        if operation == "infer":
            anchor = request["anchor"]
            return {**request, "actions": np.arange(anchor, anchor + 8)[:, None]}
        return None

    return handle, {}


@pytest.fixture
def worker():
    with SpawnedInferenceProcess(initializer, None, name="runtime-test", startup_timeout_s=20) as w:
        yield w


@pytest.mark.parametrize("execute,stride", [(6, 3), (6, 4), (7, 5)])
def test_handoffs_skip_elapsed_prefix_and_reset(worker, execute, stride):
    policy = AsyncPolicy(worker, Schedule(8, execute, stride))
    policy.reset("first")
    assert [policy.act({})[0] for _ in range(30)] == list(range(30))
    policy.reset("second")
    assert policy.act({})[0] == 0
    assert policy.episode == 2


def test_worker_timeout_can_be_retried_and_errors_are_forwarded(worker):
    pending = worker.submit("sleep", None)
    with pytest.raises(TimeoutError):
        pending.result(timeout=0.001)
    assert pending.result(timeout=2) == worker.pid != os.getpid()
    with pytest.raises(RemoteInferenceError, match="owned worker failure"):
        worker.call("fail", None)
    assert worker.call("reset", None) is None


@pytest.mark.parametrize("values", [(0, 0, 0), (8, 7, 3), (8, 4, 4), (8, 9, 5)])
def test_schedule_rejects_invalid_temporal_contract(values):
    with pytest.raises(ValueError):
        Schedule(*values)


class Future:
    def __init__(self, value):
        self.value = value

    def done(self):
        return False

    def result(self, **kwargs):
        return self.value


class WrongWorker:
    def notify(self, *args):
        pass

    def submit(self, operation, payload):
        return Future({"episode": -1, "anchor": 0, "actions": np.zeros((8, 1))})


def test_stale_episode_is_never_committed():
    policy = AsyncPolicy(WrongWorker(), Schedule(8, 6, 4))
    with pytest.raises(RuntimeError, match="Stale"):
        policy.act({})


def initialize_backlog(_):
    import time

    state = {}

    def handle(operation, payload):
        if operation == "delay":
            time.sleep(0.3)
        elif operation == "observe":
            state["value"] = payload.copy()
        elif operation == "read":
            return state["value"]

    return handle, {}


def test_observation_transport_is_buffered_and_owns_arrays():
    import numpy as np
    from longwam.runtime.worker import SpawnedInferenceProcess

    with SpawnedInferenceProcess(initialize_backlog, None, name="backlog-test") as worker:
        future = worker.submit("delay")
        observation = np.ones(2_000_000, dtype=np.uint8)
        worker.notify("observe", observation)
        observation.fill(2)
        # A direct pipe send would block behind the running inference here.
        assert not future.done()
        future.result(timeout=3)
        np.testing.assert_array_equal(worker.call("read"), np.ones_like(observation))


def test_sync_replans_at_boundary_and_uses_same_worker(worker):
    from longwam.runtime.execution import SyncPolicy, SyncSchedule

    policy = SyncPolicy(worker, SyncSchedule(8, 6))
    policy.reset("sync")
    assert [policy.act({})[0] for _ in range(30)] == list(range(30))
    assert policy.metrics["requests"] == 5
    assert policy.metrics["late_handoffs"] == 0
    policy.reset("second")
    assert policy.act({})[0] == 0


@pytest.mark.parametrize("execution", ["sync", "async"])
@pytest.mark.parametrize("execute", [24, 7])
def test_runtime_settings_share_model_horizon_and_replan_contract(execution, execute):
    from longwam.runtime.settings import read_config
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cfg = read_config(
        root / "configs/eval/libero.yaml",
        [f"execution={execution}", "action_horizon=32", f"execute_steps={execute}"],
    )
    assert cfg.replan_steps == cfg.execute_steps == execute
    assert cfg.trigger_stride == ((execute + 1) // 2 if execution == "async" else None)
    assert cfg.streaming_vae is False
    assert cfg.hardware == "reference"


@pytest.mark.parametrize(
    "overrides",
    [
        ["execution=threaded"],
        ["execution=async", "trigger_stride=1"],
        ["execution=sync", "trigger_stride=5"],
        ["hardware=rtx5090"],
        ["policy_options.use_action_ensembler=true"],
    ],
)
def test_runtime_rejects_inconsistent_execution_settings(overrides):
    from longwam.runtime.settings import read_config
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        read_config(root / "configs/eval/libero.yaml", overrides)


@pytest.mark.parametrize("execution", ["sync", "async"])
def test_robotwin_policy_derives_horizon_from_checkpoint_config(execution, monkeypatch):
    from pathlib import Path
    from omegaconf import OmegaConf
    from longwam import evaluation
    from longwam.runtime.policy import _create_execution
    from longwam.runtime.settings import read_config
    from longwam.runtime import worker as transport

    root = Path(__file__).resolve().parents[2]
    cfg = read_config(
        root / "configs/eval/robotwin2.yaml",
        [
            f"execution={execution}",
            "checkpoint=/unused/model.pt",
            "model_config=/unused/config.yaml",
            "stats=/unused/stats.json",
        ],
    )
    assert cfg.action_horizon is None
    monkeypatch.setattr(
        evaluation,
        "native_config",
        lambda settings: OmegaConf.create({"EVALUATION": {"action_horizon": 32}}),
    )

    class Worker:
        def __init__(self, initializer, settings, **kwargs):
            assert settings["action_horizon"] == 32
            assert settings["execution"] == execution

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(transport, "SpawnedInferenceProcess", Worker)
    with _create_execution(cfg) as policy:
        assert policy.schedule.horizon == 32
        assert policy.schedule.execute == 24
