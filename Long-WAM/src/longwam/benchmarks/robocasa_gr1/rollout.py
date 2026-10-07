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

"""Reusable benchmark implementation, separated from historical cluster orchestration."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import socket
import statistics
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from longwam.benchmarks.robocasa_gr1.contract import (
    EXECUTE_ACTIONS,
    MAX_CONTROL_STEPS,
    OFFICIAL_EPISODES_PER_TASK,
    OFFICIAL_N_ENVS,
    PROMPT_KEY,
    TASK_BY_DATASET,
    TASK_BY_GYM_ID,
    TASK_SPECS,
    TaskSpec,
    action_row_to_env,
    array_sha256,
    decode_response,
    encode_request,
    ensure_unlocked_waist_prompt,
    extract_official_ego256,
    extract_raw_state,
    receive_packet,
    send_packet,
    validate_action_chunk,
)

LOGGER = logging.getLogger(__name__)

POLICY_E2E_LATENCY_SCHEMA = "longwam.policy-infer-joint-ar-latency/v1"


@dataclass(frozen=True)
class PolicyInference:
    actions: np.ndarray
    input_image_sha256: str
    policy_inference_seconds: float


class UnixPolicyClient:
    """Dependency-free client for the separate Long-WAM GPU process."""

    def __init__(self, socket_path: Path, *, timeout_seconds: float = 600.0) -> None:
        self.socket_path = Path(socket_path)
        self.timeout_seconds = float(timeout_seconds)
        self._connection: socket.socket | None = None

    def __enter__(self) -> "UnixPolicyClient":
        self.connect()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def connect(self) -> None:
        if self._connection is not None:
            raise RuntimeError("Policy client is already connected.")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(self.timeout_seconds)
                connection.connect(str(self.socket_path))
                self._connection = connection
                response = self._exchange(encode_request("ping"))
                if response["status"] != "ok":
                    raise RuntimeError("GR1 policy server did not become ready.")
                return
            except (FileNotFoundError, ConnectionRefusedError):
                connection.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Policy server did not become ready: {self.socket_path}")
                time.sleep(0.2)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def reset(self, episode_id: str) -> None:
        self._exchange(encode_request("reset", episode_id=episode_id))

    def release(self, episode_id: str) -> None:
        self._exchange(encode_request("release", episode_id=episode_id))

    def observe(
        self, *, episode_id: str, control_step: int, observation: Mapping[str, Any]
    ) -> None:
        self._exchange(
            encode_request(
                "observe",
                episode_id=episode_id,
                control_step=control_step,
                image=extract_official_ego256(observation),
            )
        )

    def infer(
        self,
        *,
        episode_id: str,
        control_step: int,
        observation: Mapping[str, Any],
        prompt: str,
    ) -> PolicyInference:
        image = extract_official_ego256(observation)
        response = self._exchange(
            encode_request(
                "infer",
                episode_id=episode_id,
                control_step=control_step,
                image=image,
                state=extract_raw_state(observation),
                prompt=prompt,
            )
        )
        if not {
            "actions",
            "input_image_sha256",
            "policy_inference_seconds",
        }.issubset(response):
            raise ValueError(
                "GR1 policy response is missing actions, input fingerprint, or timing."
            )
        if response["input_image_sha256"] != array_sha256(image):
            raise ValueError("GR1 policy response was produced from a different ego frame.")
        return PolicyInference(
            actions=validate_action_chunk(response["actions"]),
            input_image_sha256=response["input_image_sha256"],
            policy_inference_seconds=float(response["policy_inference_seconds"]),
        )

    def _exchange(self, payload: bytes) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Policy client is not connected.")
        send_packet(self._connection, payload)
        response_payload = receive_packet(self._connection)
        if response_payload is None:
            raise ConnectionError("Policy server closed without a response.")
        response = decode_response(response_payload)
        if response["status"] == "error":
            raise RuntimeError(f"Policy server error: {response['message']}")
        return response


class SpawnedGymEnv:
    """One isolated Gym environment using the official spawn-process pattern."""

    def __init__(self, gym_id: str) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_gym_env_worker,
            args=(child, gym_id),
            daemon=True,
        )
        self._pending_step = False
        self._closed = False
        self._process.start()
        child.close()
        try:
            self._receive("startup")
        except Exception:
            self._connection.close()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            self._closed = True
            raise

    def reset(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self._send("reset", None)
        return self._receive("reset")

    def step_async(self, action: Mapping[str, np.ndarray]) -> None:
        if self._pending_step:
            raise RuntimeError("A GR1 environment step is already pending.")
        self._send("step", dict(action))
        self._pending_step = True

    def step_wait(self) -> tuple[Any, ...]:
        if not self._pending_step:
            raise RuntimeError("No GR1 environment step is pending.")
        try:
            return self._receive("step")
        finally:
            self._pending_step = False

    def step(self, action: Mapping[str, np.ndarray]) -> tuple[Any, ...]:
        self.step_async(action)
        return self.step_wait()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._pending_step:
                self.step_wait()
            if self._process.is_alive():
                self._send("close", None)
                self._receive("close")
        except (BrokenPipeError, EOFError, OSError, RuntimeError):
            LOGGER.warning("GR1 environment process did not shut down cleanly.", exc_info=True)
        finally:
            self._connection.close()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            self._closed = True

    def _send(self, operation: str, payload: Any) -> None:
        if self._closed or not self._process.is_alive():
            raise RuntimeError("GR1 environment process is unavailable.")
        self._connection.send((operation, payload))

    def _receive(self, operation: str) -> Any:
        try:
            status, payload = self._connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"GR1 environment process exited during {operation}.") from exc
        if status != "ok":
            raise RuntimeError(f"GR1 environment {operation} failed:\n{payload}")
        return payload


def _gym_env_worker(connection: Any, gym_id: str) -> None:
    """Own one simulator instance and its process-local RNG."""
    env = None
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        env = gym.make(gym_id, enable_render=True)
        connection.send(("ok", None))
        while True:
            operation, payload = connection.recv()
            if operation == "reset":
                if payload is not None:
                    raise ValueError("The official GR1 evaluation reset must be unseeded.")
                result = env.reset()
            elif operation == "step":
                result = env.step(payload)
            elif operation == "close":
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"Unsupported GR1 environment operation: {operation!r}.")
            connection.send(("ok", result))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


@dataclass
class _RolloutState:
    env: Any
    task: TaskSpec
    episode_index: int
    episode_id: str
    reset_seed: None
    observation: Mapping[str, Any]
    prompt: str
    started: float
    control_steps: int = 0
    policy_calls: int = 0
    policy_inference_seconds: list[float] = field(default_factory=list)
    success: bool = False
    done: bool = False
    completed_at: float | None = None


def success_from_raw_info(info: Mapping[str, Any]) -> bool:
    """Read one raw step before emulating the wrapper's one-item info history."""
    if "success" not in info:
        raise KeyError("GR1 raw step info is missing 'success'.")
    value = np.asarray(info["success"])
    if value.shape != ():
        raise ValueError(
            "GR1 success must be a raw scalar. History/vector arrays such as "
            "info['success'][0] are prohibited."
        )
    return bool(value.item())


def run_episode_group(
    *,
    envs: Sequence[Any],
    client: UnixPolicyClient,
    task: TaskSpec,
    episode_indices: Sequence[int],
    max_control_steps: int = MAX_CONTROL_STEPS,
) -> list[dict[str, Any]]:
    """Interleave up to five envs while preserving raw 20-Hz history exactly."""
    if len(envs) != len(episode_indices) or not envs:
        raise ValueError("envs and episode_indices must have equal non-zero length.")
    if len(envs) > OFFICIAL_N_ENVS:
        raise ValueError(f"A GR1 group may contain at most {OFFICIAL_N_ENVS} envs.")
    if max_control_steps <= 0 or max_control_steps % EXECUTE_ACTIONS:
        raise ValueError(f"max_control_steps must be a positive multiple of {EXECUTE_ACTIONS}.")

    states: list[_RolloutState] = []
    for env, episode_index in zip(envs, episode_indices, strict=True):
        observation, _ = env.reset()
        prompt = ensure_unlocked_waist_prompt(observation.get(PROMPT_KEY, ""))
        episode_id = f"{task.dataset_suffix}:{episode_index}"
        client.reset(episode_id)
        states.append(
            _RolloutState(
                env=env,
                task=task,
                episode_index=int(episode_index),
                episode_id=episode_id,
                reset_seed=None,
                observation=observation,
                prompt=prompt,
                started=time.monotonic(),
            )
        )

    while any(not state.done for state in states):
        chunks: dict[str, np.ndarray] = {}
        active = [
            state for state in states if not state.done and state.control_steps < max_control_steps
        ]
        if not active:
            break
        for state in active:
            inference = client.infer(
                episode_id=state.episode_id,
                control_step=state.control_steps,
                observation=state.observation,
                prompt=state.prompt,
            )
            chunks[state.episode_id] = inference.actions
            if (
                not math.isfinite(inference.policy_inference_seconds)
                or inference.policy_inference_seconds <= 0
            ):
                raise ValueError("Policy returned a non-positive or non-finite latency.")
            state.policy_inference_seconds.append(inference.policy_inference_seconds)
            state.policy_calls += 1

        last_infos: dict[str, Mapping[str, Any]] = {}
        for action_index in range(EXECUTE_ACTIONS):
            step_states = [state for state in active if not state.done]
            synchronous_results: dict[str, tuple[Any, ...]] = {}
            asynchronous: list[_RolloutState] = []
            for state in step_states:
                action = chunks[state.episode_id][action_index]
                env_action = action_row_to_env(action)
                if callable(getattr(state.env, "step_async", None)) and callable(
                    getattr(state.env, "step_wait", None)
                ):
                    state.env.step_async(env_action)
                    asynchronous.append(state)
                else:
                    synchronous_results[state.episode_id] = state.env.step(env_action)
            for state in asynchronous:
                synchronous_results[state.episode_id] = state.env.step_wait()

            for state in step_states:
                observation, _, terminated, truncated, info = synchronous_results[state.episode_id]
                state.observation = observation
                state.control_steps += 1
                last_infos[state.episode_id] = info
                state.done = _scalar_done(terminated, "terminated") or _scalar_done(
                    truncated, "truncated"
                )
                if state.done:
                    state.completed_at = time.monotonic()
                    continue
                if action_index + 1 < EXECUTE_ACTIONS:
                    client.observe(
                        episode_id=state.episode_id,
                        control_step=state.control_steps,
                        observation=state.observation,
                    )

        for state in active:
            if state.episode_id not in last_infos:
                raise AssertionError("An active GR1 chunk executed no real action.")
            # With observation deltas [0], the pinned MultiStepWrapper retains
            # one info item. Its info["success"][0] is therefore this same
            # final real control, and the official client ORs it across chunks.
            state.success |= success_from_raw_info(last_infos[state.episode_id])
            if state.control_steps >= max_control_steps:
                state.done = True
                state.completed_at = time.monotonic()

    results = [
        {
            "task": state.task.dataset_suffix,
            "gym_id": state.task.gym_id,
            "episode_index": state.episode_index,
            "episode_id": state.episode_id,
            "reset_seed": state.reset_seed,
            "prompt": state.prompt,
            "control_steps": state.control_steps,
            "policy_calls": state.policy_calls,
            "policy_inference_seconds": state.policy_inference_seconds,
            "success": state.success,
            "terminated_early": state.control_steps < max_control_steps,
            "duration_seconds": (state.completed_at or time.monotonic()) - state.started,
        }
        for state in states
    ]
    for state in states:
        client.release(state.episode_id)
    return results


def evaluate_task(
    *,
    task: TaskSpec,
    client: UnixPolicyClient,
    env_factory: Callable[[str], Any],
    episode_count: int = OFFICIAL_EPISODES_PER_TASK,
    n_envs: int = OFFICIAL_N_ENVS,
    max_control_steps: int = MAX_CONTROL_STEPS,
    on_group_complete: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one task in official groups of five logical environments."""
    if episode_count <= 0 or n_envs <= 0 or n_envs > OFFICIAL_N_ENVS:
        raise ValueError("episode_count and n_envs must be positive; n_envs may not exceed five.")
    episodes: list[dict[str, Any]] = []
    for start in range(0, episode_count, n_envs):
        indices = list(range(start, min(start + n_envs, episode_count)))
        envs = []
        try:
            for _ in indices:
                envs.append(env_factory(task.gym_id))
            group = run_episode_group(
                envs=envs,
                client=client,
                task=task,
                episode_indices=indices,
                max_control_steps=max_control_steps,
            )
        finally:
            for env in envs:
                env.close()
        episodes.extend(group)
        if on_group_complete is not None:
            on_group_complete(group)
    return episodes


def _policy_latency_payload(
    *,
    warmup_seconds: Sequence[float],
    raw_seconds: Sequence[float],
) -> dict[str, Any]:
    warmup = [float(value) for value in warmup_seconds]
    raw = [float(value) for value in raw_seconds]
    if not raw or any(not math.isfinite(value) or value <= 0 for value in (*warmup, *raw)):
        raise ValueError("Policy latency samples must be finite, positive, and non-empty.")
    ordered = sorted(raw)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "schema": POLICY_E2E_LATENCY_SCHEMA,
        "measurement_scope": "model.infer_joint_ar_full_call",
        "cuda_synchronize": "before_and_after",
        "clock": "time.perf_counter",
        "unit": "seconds",
        "warmup_policy_calls_discarded": len(warmup),
        "warmup_seconds": warmup,
        "count": len(raw),
        "mean_seconds": float(statistics.fmean(raw)),
        "median_seconds": float(statistics.median(raw)),
        "p95_seconds": float(p95),
        "max_seconds": float(max(raw)),
        "raw_seconds": raw,
    }


def summarize_policy_e2e_latency(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Discard exactly the first chronological policy call for one task."""
    all_seconds: list[float] = []
    saw_timing = False
    for episode in episodes:
        samples = episode.get("policy_inference_seconds")
        if samples is None:
            if saw_timing:
                raise ValueError("Task mixes timed and untimed policy episodes.")
            continue
        if not isinstance(samples, list):
            raise ValueError("Episode policy_inference_seconds must be a list.")
        if not saw_timing and all_seconds:
            raise AssertionError("Unreachable mixed policy timing state.")
        saw_timing = True
        if int(episode.get("policy_calls", -1)) != len(samples):
            raise ValueError("Episode policy timing count disagrees with policy_calls.")
        all_seconds.extend(float(value) for value in samples)
    if not saw_timing:
        return None
    if any("policy_inference_seconds" not in episode for episode in episodes):
        raise ValueError("Task mixes timed and untimed policy episodes.")
    if len(all_seconds) < 2:
        raise ValueError("A task needs a warmup call plus at least one measured call.")
    return _policy_latency_payload(
        warmup_seconds=all_seconds[:1],
        raw_seconds=all_seconds[1:],
    )


def aggregate_policy_e2e_latency_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not summaries:
        return None
    warmup: list[float] = []
    raw: list[float] = []
    for summary in summaries:
        if summary.get("schema") != POLICY_E2E_LATENCY_SCHEMA:
            raise ValueError("Cannot aggregate a malformed policy latency summary.")
        warmup.extend(float(value) for value in summary.get("warmup_seconds", []))
        raw.extend(float(value) for value in summary.get("raw_seconds", []))
    return _policy_latency_payload(warmup_seconds=warmup, raw_seconds=raw)


def summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    for spec in TASK_SPECS:
        records = [record for record in episodes if record["task"] == spec.dataset_suffix]
        if not records:
            continue
        successes = sum(bool(record["success"]) for record in records)
        row = {
            "gym_id": spec.gym_id,
            "episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
            "articulated_close": spec.articulated_close,
        }
        latency = summarize_policy_e2e_latency(records)
        if latency is not None:
            row["policy_e2e_latency"] = latency
        per_task[spec.dataset_suffix] = row

    def family(articulated_close: bool) -> dict[str, Any]:
        rows = [row for row in per_task.values() if row["articulated_close"] is articulated_close]
        return {
            "tasks": len(rows),
            "mean_task_success_rate": (
                float(np.mean([row["success_rate"] for row in rows])) if rows else None
            ),
        }

    rates = [row["success_rate"] for row in per_task.values()]
    result = {
        "task_count": len(per_task),
        "episode_count": len(episodes),
        "mean_task_success_rate": float(np.mean(rates)) if rates else None,
        "articulated_close": family(True),
        "rearrangement": family(False),
        "per_task": per_task,
    }
    latency_summaries = [
        row["policy_e2e_latency"] for row in per_task.values() if "policy_e2e_latency" in row
    ]
    if latency_summaries:
        if len(latency_summaries) != len(per_task):
            raise ValueError("Evaluation mixes timed and untimed tasks.")
        result["policy_e2e_latency"] = aggregate_policy_e2e_latency_summaries(latency_summaries)
    return result


def resolve_tasks(values: Sequence[str]) -> tuple[TaskSpec, ...]:
    if not values or values == ["all"]:
        return TASK_SPECS
    resolved = []
    for value in values:
        if value in TASK_BY_GYM_ID:
            spec = TASK_BY_GYM_ID[value]
        elif value in TASK_BY_DATASET:
            spec = TASK_BY_DATASET[value]
        else:
            matches = [spec for spec in TASK_SPECS if spec.dataset_suffix == value]
            if len(matches) != 1:
                raise ValueError(f"Unknown GR1 task: {value!r}.")
            spec = matches[0]
        if spec not in resolved:
            resolved.append(spec)
    return tuple(resolved)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar_done(value: Any, label: str) -> bool:
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f"GR1 raw {label} must be scalar, got {value.shape}.")
    return bool(value.item())


def _load_env_factory() -> Callable[[str], Any]:
    return SpawnedGymEnv


def _append_group(
    output: Path,
    results: dict[str, Any],
    group: Sequence[Mapping[str, Any]],
) -> None:
    results["episodes"].extend(dict(record) for record in group)
    results["summary"] = summarize(results["episodes"])
    if "policy_e2e_latency" in results["summary"]:
        results["policy_e2e_latency"] = results["summary"]["policy_e2e_latency"]
    atomic_write_json(output, results)
