# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from longwam.benchmarks.robocasa_gr1.contract import (
    ACTION_FIELDS,
    EGO_VIEW_KEY,
    OFFICIAL_EGO_SHAPE,
    PROMPT_KEY,
    STATE_FIELDS,
    TASK_SPECS,
)
from longwam.benchmarks.robocasa_gr1.rollout import (
    PolicyInference,
    evaluate_task,
    run_episode_group,
    success_from_raw_info,
    summarize,
    summarize_policy_e2e_latency,
)
from longwam.benchmarks.robocasa_gr1.temporal_contract import (
    TRAINING_PLAN_KEY,
    TemporalSpec,
)


def _observation(step: int = 0) -> dict:
    observation = {
        EGO_VIEW_KEY: np.full(OFFICIAL_EGO_SHAPE, step % 256, dtype=np.uint8),
        PROMPT_KEY: "unlocked_waist: place the object",
    }
    for key, width in STATE_FIELDS:
        observation[key] = np.full(width, step, dtype=np.float32)
    return observation


class _FakeEnv:
    def __init__(self, *, successes=(), done_at: int | None = None) -> None:
        self.successes = set(successes)
        self.done_at = done_at
        self.step_count = 0
        self.reset_seed = None
        self.reset_kwargs = None
        self.actions: list[dict[str, np.ndarray]] = []
        self.closed = False

    def reset(self, **kwargs):
        self.reset_kwargs = kwargs
        self.step_count = 0
        return _observation(), {"success": False}

    def step(self, action):
        assert tuple(action) == tuple(key for key, _ in ACTION_FIELDS)
        self.actions.append({key: value.copy() for key, value in action.items()})
        self.step_count += 1
        done = self.done_at == self.step_count
        return (
            _observation(self.step_count),
            0.0,
            done,
            False,
            {"success": self.step_count in self.successes},
        )

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.releases: list[str] = []
        self.infers: list[tuple[str, int]] = []
        self.observes: list[tuple[str, int]] = []
        base = np.arange(29, dtype=np.float32)
        self.chunk = np.stack([base + 1000 * index for index in range(16)])

    def reset(self, episode_id: str) -> None:
        self.resets.append(episode_id)

    def release(self, episode_id: str) -> None:
        self.releases.append(episode_id)

    def infer(self, *, episode_id, control_step, observation, prompt):
        assert prompt == "unlocked_waist: place the object"
        self.infers.append((episode_id, control_step))
        return PolicyInference(
            actions=self.chunk.copy(),
            input_image_sha256="0" * 64,
            policy_inference_seconds=0.1 + 0.01 * len(self.infers),
        )

    def observe(self, *, episode_id, control_step, observation):
        self.observes.append((episode_id, control_step))


class _AsyncFakeEnv(_FakeEnv):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__(done_at=1)
        self.name = name
        self.events = events
        self.pending_action = None

    def step_async(self, action):
        self.events.append(f"send:{self.name}")
        self.pending_action = action

    def step_wait(self):
        self.events.append(f"wait:{self.name}")
        action = self.pending_action
        self.pending_action = None
        return super().step(action)


def test_execute16_caches_every_raw_control_and_has_no_action_memory():
    env = _FakeEnv(successes={5, 32})
    client = _FakeClient()
    task = TASK_SPECS[0]
    result = run_episode_group(
        envs=[env],
        client=client,  # type: ignore[arg-type]
        task=task,
        episode_indices=[0],
        max_control_steps=32,
    )[0]

    assert result["control_steps"] == 32
    assert result["policy_calls"] == 2
    assert result["success"] is True
    assert [step for _, step in client.infers] == [0, 16]
    assert [step for _, step in client.observes] == [
        *range(1, 16),
        *range(17, 32),
    ]
    assert len(env.actions) == 32
    assert client.releases == client.resets
    # Every chunk executes rows 0..15 directly. There is no dummy block or
    # previous-action queue carried into the next policy call.
    np.testing.assert_array_equal(env.actions[0]["action.left_arm"], np.arange(7))
    np.testing.assert_array_equal(env.actions[15]["action.left_arm"], np.arange(7) + 15_000)
    np.testing.assert_array_equal(env.actions[16]["action.left_arm"], np.arange(7))


def test_success_uses_only_last_real_step_per_chunk_then_ors_chunks():
    task = TASK_SPECS[0]
    early_only = run_episode_group(
        envs=[_FakeEnv(successes={5})],
        client=_FakeClient(),  # type: ignore[arg-type]
        task=task,
        episode_indices=[0],
        max_control_steps=16,
    )[0]
    assert early_only["success"] is False

    first_chunk_last = run_episode_group(
        envs=[_FakeEnv(successes={16})],
        client=_FakeClient(),  # type: ignore[arg-type]
        task=task,
        episode_indices=[0],
        max_control_steps=32,
    )[0]
    assert first_chunk_last["success"] is True


def test_done_breaks_inside_chunk_and_uses_that_last_actual_step():
    env = _FakeEnv(successes={3}, done_at=3)
    client = _FakeClient()
    result = run_episode_group(
        envs=[env],
        client=client,  # type: ignore[arg-type]
        task=TASK_SPECS[0],
        episode_indices=[0],
        max_control_steps=720,
    )[0]
    assert result["control_steps"] == 3
    assert result["policy_calls"] == 1
    assert result["success"] is True
    assert result["terminated_early"] is True
    assert [step for _, step in client.observes] == [1, 2]
    assert len(env.actions) == 3


def test_raw_step_boundary_rejects_an_already_aggregated_info_history():
    assert success_from_raw_info({"success": np.bool_(True)}) is True
    with pytest.raises(ValueError, match=r"info\['success'\]\[0\]"):
        success_from_raw_info({"success": np.array([True, False])})


def test_official_n_envs_five_are_reset_as_one_group_and_closed():
    envs: list[_FakeEnv] = []

    def factory(_gym_id: str) -> _FakeEnv:
        env = _FakeEnv(done_at=1)
        envs.append(env)
        return env

    client = _FakeClient()
    episodes = evaluate_task(
        task=TASK_SPECS[0],
        client=client,  # type: ignore[arg-type]
        env_factory=factory,
        episode_count=10,
        n_envs=5,
        max_control_steps=720,
    )
    assert len(episodes) == 10
    assert len(envs) == 10 and all(env.closed for env in envs)
    assert len(set(client.resets[:5])) == 5
    assert [episode["episode_index"] for episode in episodes] == list(range(10))


def test_async_env_steps_are_issued_to_the_whole_group_before_waiting():
    events: list[str] = []
    run_episode_group(
        envs=[_AsyncFakeEnv("a", events), _AsyncFakeEnv("b", events)],
        client=_FakeClient(),  # type: ignore[arg-type]
        task=TASK_SPECS[0],
        episode_indices=[0, 1],
        max_control_steps=720,
    )
    assert events == ["send:a", "send:b", "wait:a", "wait:b"]


def test_official_resets_do_not_inject_a_seed():
    env = _FakeEnv(done_at=1)
    result = run_episode_group(
        envs=[env],
        client=_FakeClient(),  # type: ignore[arg-type]
        task=TASK_SPECS[0],
        episode_indices=[0],
        max_control_steps=16,
    )[0]
    assert env.reset_kwargs == {}
    assert result["reset_seed"] is None


def test_summary_is_unweighted_across_tasks_and_splits_six_vs_eighteen():
    episodes = []
    for spec in TASK_SPECS:
        episodes.append({"task": spec.dataset_suffix, "success": spec.articulated_close})
    summary = summarize(episodes)
    assert summary["task_count"] == 24
    assert summary["mean_task_success_rate"] == pytest.approx(6 / 24)
    assert summary["articulated_close"] == {"tasks": 6, "mean_task_success_rate": 1.0}
    assert summary["rearrangement"] == {"tasks": 18, "mean_task_success_rate": 0.0}


def test_policy_latency_discards_one_task_warmup_and_keeps_recomputable_raw():
    episodes = [
        {"policy_calls": 2, "policy_inference_seconds": [10.0, 1.0]},
        {"policy_calls": 2, "policy_inference_seconds": [2.0, 3.0]},
    ]
    latency = summarize_policy_e2e_latency(episodes)
    assert latency is not None
    assert latency["warmup_policy_calls_discarded"] == 1
    assert latency["warmup_seconds"] == [10.0]
    assert latency["raw_seconds"] == [1.0, 2.0, 3.0]
    assert latency["count"] == 3
    assert latency["mean_seconds"] == pytest.approx(2.0)
    assert latency["median_seconds"] == pytest.approx(2.0)
    assert latency["p95_seconds"] == pytest.approx(3.0)
    assert latency["max_seconds"] == pytest.approx(3.0)
