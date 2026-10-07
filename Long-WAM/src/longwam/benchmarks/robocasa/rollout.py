# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa/eval_robocasa.py
# Changes: Research evaluator split into a portable benchmark rollout module.
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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/robocasa/rollout.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Reusable benchmark implementation, separated from historical cluster orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from longwam.benchmarks.robocasa.contract import (
    ACTION_DIM,
    ACTION_HORIZON,
    CAMERA_KEYS,
    array_sha256,
    camera_bundle_sha256,
    dataset_action_to_canonical,
    decode_response,
    encode_request,
    extract_cameras,
    extract_dataset_state,
    receive_packet,
    sanitize_dataset_action,
    send_packet,
    validate_sha256,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TASK_SET = "target50"

VIDEO_FPS = 20

INITIAL_PROVENANCE_SCHEMA = "robocasa-initial-v2"

INITIAL_PROVENANCE_FIELDS = frozenset(
    {
        "schema",
        "camera_order",
        "camera_sha256",
        "camera_bundle_sha256",
        "state_sha256",
        "simulator_state_sha256",
        "prompt_sha256",
        "ep_meta_sha256",
        "model_mosaic_sha256",
        "layout_id",
        "style_id",
    }
)

TARGET_GROUPS = ("atomic_seen", "composite_seen", "composite_unseen")

SUPPORTED_TASK_SETS = (DEFAULT_TASK_SET, *TARGET_GROUPS)

TASK_DESCRIPTION_KEY = "annotation.human.task_description"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON result without exposing a partial file."""
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


def derive_episode_reset_seed(
    base_seed: int,
    split: str,
    task: str,
    episode_index: int,
) -> int:
    """Derive a shard-order-independent RoboCasa scenario seed."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("RoboCasa base seed must be a non-negative integer.")
    if not isinstance(split, str) or not split:
        raise ValueError("RoboCasa split must be a non-empty string.")
    if not isinstance(task, str) or not task:
        raise ValueError("RoboCasa task must be a non-empty string.")
    if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
        raise ValueError("RoboCasa episode index must be a non-negative integer.")
    payload = (f"robocasa-episode-reset-v1\0{base_seed}\0{split}\0{task}\0{episode_index}").encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("RoboCasa episode metadata keys must be strings.")
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"RoboCasa episode metadata contains a non-JSON value: {type(value).__name__}.")


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON value independent of mapping insertion order."""
    payload = json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _current_episode_metadata(env: Any) -> dict[str, Any]:
    """Read the metadata cached while the current wrapped observation was made."""
    wrapper = env.unwrapped
    metadata = wrapper.last_ep_meta
    if not isinstance(metadata, Mapping):
        raise TypeError("RoboCasa wrapper last_ep_meta must be a mapping.")
    return dict(metadata)


def _current_simulator_state(env: Any) -> np.ndarray:
    """Return the full MuJoCo reset state in a stable numeric representation."""
    wrapper = env.unwrapped
    simulator = getattr(wrapper, "env", None)
    sim = getattr(simulator, "sim", None)
    if sim is None or not hasattr(sim, "get_state"):
        raise TypeError("RoboCasa wrapper does not expose a MuJoCo simulator state.")
    state = np.asarray(sim.get_state().flatten(), dtype="<f8")
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("RoboCasa MuJoCo reset state must be finite and one-dimensional.")
    return np.ascontiguousarray(state)


def initial_observation_provenance(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    episode_metadata: Mapping[str, Any],
    simulator_state: np.ndarray,
) -> dict[str, Any]:
    """Fingerprint the exact reset observation before policy inference."""
    cameras = extract_cameras(observation)
    state = extract_dataset_state(observation).astype("<f4", copy=False)
    return {
        "schema": INITIAL_PROVENANCE_SCHEMA,
        "camera_order": list(CAMERA_KEYS),
        "camera_sha256": {key: array_sha256(cameras[key]) for key in CAMERA_KEYS},
        "camera_bundle_sha256": camera_bundle_sha256(cameras),
        "state_sha256": array_sha256(state),
        "simulator_state_sha256": array_sha256(simulator_state),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "ep_meta_sha256": canonical_json_sha256(episode_metadata),
        "model_mosaic_sha256": None,
        "layout_id": _json_compatible(episode_metadata.get("layout_id")),
        "style_id": _json_compatible(episode_metadata.get("style_id")),
    }


def validate_initial_observation_provenance(value: Any) -> dict[str, Any]:
    """Validate one complete reset-observation fingerprint record."""
    if not isinstance(value, Mapping) or set(value) != INITIAL_PROVENANCE_FIELDS:
        raise ValueError("RoboCasa initial provenance has an incomplete schema.")
    if value.get("schema") != INITIAL_PROVENANCE_SCHEMA:
        raise ValueError("RoboCasa initial provenance schema is unsupported.")
    if value.get("camera_order") != list(CAMERA_KEYS):
        raise ValueError("RoboCasa initial camera order changed.")
    camera_sha256 = value.get("camera_sha256")
    if not isinstance(camera_sha256, Mapping) or set(camera_sha256) != set(CAMERA_KEYS):
        raise ValueError("RoboCasa initial camera fingerprint inventory changed.")
    normalized_camera_sha256 = {
        key: validate_sha256(
            camera_sha256[key],
            field_name=f"initial_provenance.camera_sha256.{key}",
        )
        for key in CAMERA_KEYS
    }
    normalized = {
        "schema": INITIAL_PROVENANCE_SCHEMA,
        "camera_order": list(CAMERA_KEYS),
        "camera_sha256": normalized_camera_sha256,
        **{
            field: validate_sha256(
                value[field],
                field_name=f"initial_provenance.{field}",
            )
            for field in (
                "camera_bundle_sha256",
                "state_sha256",
                "simulator_state_sha256",
                "prompt_sha256",
                "ep_meta_sha256",
                "model_mosaic_sha256",
            )
        },
        "layout_id": _json_compatible(value.get("layout_id")),
        "style_id": _json_compatible(value.get("style_id")),
    }
    canonical_json_sha256(
        {
            "layout_id": normalized["layout_id"],
            "style_id": normalized["style_id"],
        }
    )
    return normalized


def resolve_tasks(
    task_registry: Mapping[str, Sequence[str]],
    task_set: str,
    requested_tasks: Sequence[str] | None = None,
) -> list[str]:
    """Resolve an ordered target task list, optionally selecting a subset."""
    if task_set not in SUPPORTED_TASK_SETS:
        raise ValueError(f"Unsupported task set {task_set!r}; choose from {SUPPORTED_TASK_SETS}.")
    if task_set not in task_registry:
        raise KeyError(f"RoboCasa registry is missing task set {task_set!r}.")

    registered = list(task_registry[task_set])
    if len(registered) != len(set(registered)):
        raise ValueError(f"RoboCasa task set {task_set!r} contains duplicates.")
    if requested_tasks is None:
        selected = registered
    else:
        selected = list(requested_tasks)
        if not selected:
            raise ValueError("--tasks must contain at least one task when provided.")
        if len(selected) != len(set(selected)):
            raise ValueError("--tasks contains duplicates.")
        unknown = sorted(set(selected) - set(registered))
        if unknown:
            raise ValueError(f"Tasks are not members of {task_set!r}: {unknown}.")
    if not selected:
        raise ValueError(f"RoboCasa task set {task_set!r} is empty.")
    return selected


def success_from_info(info: Mapping[str, Any]) -> bool:
    """Use RoboCasa's benchmark success field as the sole success criterion."""
    if "success" not in info:
        raise KeyError("RoboCasa step info is missing the required 'success' field.")
    return bool(info["success"])


@dataclass(frozen=True)
class PolicyInference:
    actions: np.ndarray
    camera_bundle_sha256: str
    model_mosaic_sha256: str


class UnixPolicyClient:
    """Synchronous, persistent client for one local Long-WAM policy server."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        response_timeout_seconds: float,
        startup_timeout_seconds: float,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.response_timeout_seconds = float(response_timeout_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self._connection: socket.socket | None = None

    def connect(self) -> None:
        if self._connection is not None:
            raise RuntimeError("Policy client is already connected.")
        if len(os.fsencode(self.socket_path)) > 103:
            raise ValueError(f"AF_UNIX socket path must be at most 103 bytes: {self.socket_path}.")

        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error: OSError | None = None
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.response_timeout_seconds)
            try:
                connection.connect(str(self.socket_path))
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                connection.close()
                last_error = exc
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Policy server did not become ready at {self.socket_path} "
                        f"within {self.startup_timeout_seconds:.1f}s."
                    ) from last_error
                time.sleep(0.25)
                continue
            except BaseException:
                connection.close()
                raise
            self._connection = connection
            return

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def _exchange(self, payload: bytes) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Policy client is not connected.")
        send_packet(self._connection, payload)
        response_payload = receive_packet(self._connection)
        if response_payload is None:
            raise ConnectionError("Policy server closed the socket without a response.")
        response = decode_response(response_payload)
        if response["status"] == "error":
            raise RuntimeError(f"Policy server error: {response['message']}")
        return response

    def ping(self) -> None:
        self._exchange(encode_request("ping"))

    def reset(self, episode_id: str) -> None:
        self._exchange(encode_request("reset", episode_id=episode_id))

    def observe(
        self,
        *,
        episode_id: str,
        control_step: int,
        observation: Mapping[str, Any],
    ) -> None:
        self._exchange(
            encode_request(
                "observe",
                episode_id=episode_id,
                control_step=control_step,
                cameras=extract_cameras(observation),
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
        cameras = extract_cameras(observation)
        response = self._exchange(
            encode_request(
                "infer",
                episode_id=episode_id,
                control_step=control_step,
                cameras=cameras,
                state=extract_dataset_state(observation),
                prompt=prompt,
            )
        )
        required = {
            "actions",
            "camera_bundle_sha256",
            "model_mosaic_sha256",
        }
        missing = required - response.keys()
        if missing:
            raise ValueError(f"Policy inference response is missing fields: {sorted(missing)}.")
        local_camera_sha256 = camera_bundle_sha256(cameras)
        remote_camera_sha256 = str(response["camera_bundle_sha256"])
        if remote_camera_sha256 != local_camera_sha256:
            raise ValueError("Policy server camera bundle differs from the simulator request.")
        return PolicyInference(
            actions=np.asarray(response["actions"], dtype=np.float32),
            camera_bundle_sha256=remote_camera_sha256,
            model_mosaic_sha256=str(response["model_mosaic_sha256"]),
        )

    def shutdown(self) -> None:
        self._exchange(encode_request("shutdown"))


def camera_mosaic(observation: Mapping[str, Any]) -> np.ndarray:
    """Tile the already-oriented left, right, and wrist views without flipping."""
    cameras = extract_cameras(observation)
    return np.ascontiguousarray(np.concatenate([cameras[key] for key in CAMERA_KEYS], axis=1))


class EpisodeVideoRecorder:
    """Best-effort incremental MP4 writer that cannot fail the evaluation."""

    def __init__(self, temporary_path: Path) -> None:
        self.temporary_path = temporary_path
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required for RoboCasa MP4 recording.")
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "768x256",
                "-framerate",
                str(VIDEO_FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(temporary_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._failed = False

    def append(self, observation: Mapping[str, Any]) -> None:
        if self._process is None:
            return
        try:
            assert self._process.stdin is not None
            frame = camera_mosaic(observation)
            self._process.stdin.write(frame.tobytes())
        except Exception:
            LOGGER.exception("Disabling video after an encoding error.")
            self.abort()

    def close(self) -> bool:
        if self._process is not None:
            try:
                assert self._process.stdin is not None
                self._process.stdin.close()
                return_code = self._process.wait(timeout=60)
                if return_code != 0:
                    LOGGER.error("ffmpeg exited with code %d.", return_code)
                    self._failed = True
            except Exception:
                LOGGER.exception("Could not finalize episode video.")
                self._failed = True
                self._process.kill()
                self._process.wait()
            finally:
                self._process = None
        if self._failed:
            self.temporary_path.unlink(missing_ok=True)
            return False
        return True

    def abort(self) -> None:
        self._failed = True
        self.close()


class VideoManager:
    """Retain only bounded numbers of success and failure episode videos."""

    def __init__(
        self,
        video_dir: Path | None,
        *,
        max_success_videos: int,
        max_failure_videos: int,
    ) -> None:
        self.video_dir = video_dir
        self.limits = {
            "success": int(max_success_videos),
            "failure": int(max_failure_videos),
        }
        self.saved = {"success": 0, "failure": 0}
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        task: str,
        episode_index: int,
    ) -> EpisodeVideoRecorder | None:
        if self.video_dir is None or all(
            self.saved[label] >= limit for label, limit in self.limits.items()
        ):
            return None
        temporary = self.video_dir / (
            f".{task}_episode{episode_index:03d}.{os.getpid()}.partial.mp4"
        )
        try:
            return EpisodeVideoRecorder(temporary)
        except Exception:
            LOGGER.exception("Could not start episode video; evaluation will continue.")
            temporary.unlink(missing_ok=True)
            return None

    def finish(
        self,
        recorder: EpisodeVideoRecorder | None,
        *,
        task: str,
        episode_index: int,
        success: bool,
    ) -> str | None:
        if recorder is None or not recorder.close():
            return None
        label = "success" if success else "failure"
        if self.saved[label] >= self.limits[label]:
            recorder.temporary_path.unlink(missing_ok=True)
            return None
        destination = self.video_dir / (f"{task}_episode{episode_index:03d}_{label}.mp4")
        try:
            os.replace(recorder.temporary_path, destination)
        except Exception:
            LOGGER.exception("Could not retain episode video; evaluation will continue.")
            recorder.temporary_path.unlink(missing_ok=True)
            return None
        self.saved[label] += 1
        return str(destination)


def _trace_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Trace values must be non-empty [T, D], got {values.shape}.")
    return {
        name: getattr(values, name)(axis=0).astype(float).tolist()
        for name in ("min", "max", "mean", "std")
    }


def _save_diagnostic_trace(
    path: Path,
    *,
    raw_actions: Sequence[np.ndarray],
    sanitized_actions: Sequence[np.ndarray],
    canonical_actions: Sequence[np.ndarray],
    states_before: Sequence[np.ndarray],
    states_after: Sequence[np.ndarray],
    inference_control_steps: Sequence[int],
    inference_chunks: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Atomically persist one bounded rollout trace and return compact diagnostics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "raw_dataset_actions": np.asarray(raw_actions, dtype=np.float32),
        "sanitized_dataset_actions": np.asarray(sanitized_actions, dtype=np.float32),
        "canonical_actions": np.asarray(canonical_actions, dtype=np.float32),
        "states_before": np.asarray(states_before, dtype=np.float32),
        "states_after": np.asarray(states_after, dtype=np.float32),
        "inference_control_steps": np.asarray(inference_control_steps, dtype=np.int64),
        "inference_chunks": np.asarray(inference_chunks, dtype=np.float32),
    }
    steps = arrays["raw_dataset_actions"].shape[0]
    expected_shapes = {
        "raw_dataset_actions": (steps, ACTION_DIM),
        "sanitized_dataset_actions": (steps, ACTION_DIM),
        "canonical_actions": (steps, ACTION_DIM),
        "states_before": (steps, 16),
        "states_after": (steps, 16),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"Diagnostic trace {name} must have shape {expected_shape}, "
                f"got {arrays[name].shape}."
            )
    if arrays["inference_chunks"].ndim != 3 or (
        arrays["inference_chunks"].shape[1:] != (ACTION_HORIZON, ACTION_DIM)
    ):
        raise ValueError(
            "Diagnostic inference chunks must have shape "
            f"[N, {ACTION_HORIZON}, {ACTION_DIM}], got "
            f"{arrays['inference_chunks'].shape}."
        )
    if arrays["inference_control_steps"].shape != (arrays["inference_chunks"].shape[0],):
        raise ValueError("Inference control-step and chunk counts differ.")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Diagnostic trace contains non-finite values.")

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    raw = arrays["raw_dataset_actions"]
    sanitized = arrays["sanitized_dataset_actions"]
    states = np.concatenate((arrays["states_before"][:1], arrays["states_after"]), axis=0)
    return {
        "trace_path": str(path),
        "dataset_action_order": [
            "base_x",
            "base_y",
            "base_yaw",
            "base_torso",
            "mode",
            "eef_x",
            "eef_y",
            "eef_z",
            "eef_rx",
            "eef_ry",
            "eef_rz",
            "gripper",
        ],
        "raw_action": {
            **_trace_stats(raw),
            "outside_unit_range_fraction": (np.abs(raw) > 1.0).mean(axis=0).astype(float).tolist(),
        },
        "sanitized_action": {
            **_trace_stats(sanitized),
            "at_limit_fraction": (np.abs(sanitized) >= 0.99).mean(axis=0).astype(float).tolist(),
        },
        "mode_positive_fraction": float((sanitized[:, 4] > 0).mean()),
        "mode_switches": int(np.count_nonzero(np.diff(sanitized[:, 4]))),
        "gripper_positive_fraction": float((sanitized[:, 11] > 0).mean()),
        "gripper_switches": int(np.count_nonzero(np.diff(sanitized[:, 11]))),
        "state": {
            **_trace_stats(states),
            "start": states[0].astype(float).tolist(),
            "end": states[-1].astype(float).tolist(),
        },
        "base_position_displacement": float(np.linalg.norm(states[-1, :3] - states[0, :3])),
        "eef_position_displacement": float(np.linalg.norm(states[-1, 7:10] - states[0, 7:10])),
    }


def run_episode(
    *,
    env: Any,
    client: UnixPolicyClient,
    task: str,
    episode_index: int,
    env_seed: int,
    split: str,
    horizon: int,
    replan_steps: int,
    convert_action: Callable[[np.ndarray], Any],
    video_recorder: EpisodeVideoRecorder | None = None,
    diagnostic_trace_path: Path | None = None,
) -> dict[str, Any]:
    """Run one episode while forwarding every raw-step observation to the server."""
    if horizon <= 0:
        raise ValueError(f"RoboCasa horizon must be positive, got {horizon}.")
    if not 1 <= replan_steps <= ACTION_HORIZON:
        raise ValueError(f"replan_steps must be in [1, {ACTION_HORIZON}], got {replan_steps}.")

    episode_id = f"{task}:{env_seed}:{episode_index}"
    reset_seed = derive_episode_reset_seed(
        env_seed,
        split,
        task,
        episode_index,
    )
    random.seed(reset_seed)
    np.random.seed(reset_seed & 0xFFFFFFFF)
    observation, _ = env.reset(seed=reset_seed)
    prompt = str(observation.get(TASK_DESCRIPTION_KEY, "")).strip()
    if not prompt:
        raise ValueError(f"RoboCasa observation is missing a non-empty {TASK_DESCRIPTION_KEY!r}.")
    episode_metadata = _current_episode_metadata(env)
    metadata_prompt = str(episode_metadata.get("lang", "")).strip()
    if metadata_prompt != prompt:
        raise ValueError(
            "RoboCasa cached episode metadata language differs from the observation prompt."
        )
    initial_provenance = initial_observation_provenance(
        observation,
        prompt=prompt,
        episode_metadata=episode_metadata,
        simulator_state=_current_simulator_state(env),
    )
    client.reset(episode_id)
    if video_recorder is not None:
        video_recorder.append(observation)

    inference_calls = 0
    started = time.monotonic()
    succeeded = False
    executed_steps = 0
    raw_actions: list[np.ndarray] = []
    sanitized_actions: list[np.ndarray] = []
    canonical_actions: list[np.ndarray] = []
    states_before: list[np.ndarray] = []
    states_after: list[np.ndarray] = []
    inference_control_steps: list[int] = []
    inference_chunks: list[np.ndarray] = []

    from longwam.runtime.execution import SyncPolicy, SyncSchedule
    from longwam.runtime.worker import _InlineInferenceWorker
    current_observation = None

    def handle(operation, payload):
        nonlocal current_observation, inference_calls
        if operation == "reset":
            return None
        if operation == "observe":
            current_observation = payload["observation"]
            if payload["step"] % replan_steps:
                client.observe(episode_id=episode_id, control_step=payload["step"],
                               observation=current_observation)
            return None
        control_step = payload["anchor"]
        observation = current_observation
        inference = client.infer(
            episode_id=episode_id,
            control_step=control_step,
            observation=observation,
            prompt=prompt,
        )
        action_chunk = inference.actions
        if control_step == 0:
            if inference.camera_bundle_sha256 != initial_provenance["camera_bundle_sha256"]:
                raise ValueError(
                    "Initial policy camera fingerprint differs from the "
                    "simulator reset observation."
                )
            initial_provenance["model_mosaic_sha256"] = inference.model_mosaic_sha256
        if action_chunk.ndim != 2 or action_chunk.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Policy actions must have shape [T, {ACTION_DIM}], got {action_chunk.shape}."
            )
        if action_chunk.shape[0] < replan_steps:
            raise ValueError(
                f"Policy returned {action_chunk.shape[0]} actions, but "
                f"replan_steps={replan_steps}."
            )
        if diagnostic_trace_path is not None:
            inference_control_steps.append(control_step)
            inference_chunks.append(action_chunk.copy())
        inference_calls += 1
        return {"episode": payload["episode"], "anchor": control_step,
                "actions": np.asarray(action_chunk[:replan_steps], dtype=np.float32)}

    executor = SyncPolicy(_InlineInferenceWorker(handle), SyncSchedule(replan_steps, replan_steps))
    executor.reset(prompt)
    for control_step in range(horizon):
        dataset_action = executor.act(observation)
        sanitized_action = sanitize_dataset_action(dataset_action)
        canonical_action = dataset_action_to_canonical(sanitized_action)
        if diagnostic_trace_path is not None:
            states_before.append(extract_dataset_state(observation))
        observation, _, _, _, info = env.step(convert_action(canonical_action))
        if diagnostic_trace_path is not None:
            raw_actions.append(dataset_action.copy())
            sanitized_actions.append(sanitized_action)
            canonical_actions.append(canonical_action)
            states_after.append(extract_dataset_state(observation))
        if video_recorder is not None:
            video_recorder.append(observation)
        executed_steps += 1
        if success_from_info(info):
            succeeded = True
            break

    initial_provenance = validate_initial_observation_provenance(initial_provenance)
    episode = {
        "episode_id": episode_id,
        "episode_index": int(episode_index),
        "reset_seed": reset_seed,
        "task": task,
        "prompt": prompt,
        "horizon": int(horizon),
        "steps": executed_steps,
        "inference_calls": inference_calls,
        "success": succeeded,
        "duration_seconds": time.monotonic() - started,
        "initial_provenance": initial_provenance,
    }
    if diagnostic_trace_path is not None:
        episode["diagnostics"] = _save_diagnostic_trace(
            diagnostic_trace_path,
            raw_actions=raw_actions,
            sanitized_actions=sanitized_actions,
            canonical_actions=canonical_actions,
            states_before=states_before,
            states_after=states_after,
            inference_control_steps=inference_control_steps,
            inference_chunks=inference_chunks,
        )
    return episode


def summarize_results(
    episodes: Sequence[Mapping[str, Any]],
    task_registry: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    def rate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(record["success"]) for record in records)
        count = len(records)
        return {
            "episodes": count,
            "successes": successes,
            "success_rate": successes / count if count else None,
        }

    per_task: dict[str, dict[str, Any]] = {}
    for task in dict.fromkeys(str(record["task"]) for record in episodes):
        per_task[task] = rate([record for record in episodes if record["task"] == task])

    per_group: dict[str, dict[str, Any]] = {}
    for group in TARGET_GROUPS:
        members = set(task_registry[group])
        per_group[group] = rate([record for record in episodes if record["task"] in members])

    return {
        "overall": rate(episodes),
        "per_group": per_group,
        "per_task": per_task,
    }


def _load_robocasa_runtime() -> tuple[
    Mapping[str, Sequence[str]],
    Callable[[str], int],
    Callable[..., Any],
    Callable[[np.ndarray], Any],
]:
    # Imports remain lazy so pure client logic is testable in the policy environment.
    # Prime lxml's RPATH-bound libxml2 before simulator imports can preload an
    # incompatible system libxml2 with the same SONAME.
    import gymnasium as gym
    import lxml.etree  # noqa: F401
    import robocasa  # noqa: F401  # Registers robocasa/* Gym environments.
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    return TASK_SET_REGISTRY, get_task_horizon, gym.make, convert_action
