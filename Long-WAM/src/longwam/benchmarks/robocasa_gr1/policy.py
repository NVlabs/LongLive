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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/robocasa_gr1/policy.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Reusable benchmark implementation, separated from historical cluster orchestration."""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import stat
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from longwam.benchmarks.robocasa_gr1.contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_VIDEO_FREQ_RATIO,
    MODEL_EGO_SHAPE,
    MODEL_STATE_DIM,
    RAW_STATE_DIM,
    array_sha256,
    decode_request,
    encode_response,
    receive_packet,
    send_packet,
    validate_action_chunk,
)
from longwam.benchmarks.robocasa_gr1.temporal_contract import (  # noqa: E402
    TRAINING_PLAN_KEY,
    VAE_TEMPORAL_FACTOR,
    TemporalSpec,
)

LOGGER = logging.getLogger("robocasa-gr1-policy-server")

DEFAULT_NUM_INFERENCE_STEPS = 10

VIDEO_RANGE_TOLERANCE = 8 * torch.finfo(torch.float32).eps

EXPECTED_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"


class RawFrameHistory:
    """Bounded 20-Hz history sampled according to one sealed temporal spec."""

    def __init__(
        self,
        frame_shape: tuple[int, int, int] = (3, *MODEL_EGO_SHAPE[:2]),
        *,
        temporal_spec: TemporalSpec | None = None,
    ) -> None:
        self.temporal_spec = temporal_spec or TemporalSpec.from_past_obs_size(48)
        self.frame_shape = tuple(int(value) for value in frame_shape)
        if self.frame_shape != (3, 224, 224):
            raise ValueError(
                f"GR1 model frame shape must be (3, 224, 224), got {self.frame_shape}."
            )
        self.raw_history_frames = self.temporal_spec.P + 1
        self._frames: deque[torch.Tensor] = deque(maxlen=self.raw_history_frames)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def snapshot(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._frames)

    def restore(self, frames: Sequence[torch.Tensor]) -> None:
        if len(frames) > self.raw_history_frames:
            raise ValueError(f"History snapshot exceeds {self.raw_history_frames} frames.")
        self._frames = deque(frames, maxlen=self.raw_history_frames)

    def append(self, frame: torch.Tensor) -> None:
        frame = torch.as_tensor(frame)
        if tuple(frame.shape) != self.frame_shape:
            raise ValueError(
                f"GR1 frame must have shape {self.frame_shape}, got {tuple(frame.shape)}."
            )
        if not frame.is_floating_point() or not bool(torch.isfinite(frame).all().item()):
            raise ValueError("GR1 model frame must contain finite floating-point values.")
        self._frames.append(
            frame.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )

    @property
    def current(self) -> torch.Tensor:
        if not self._frames:
            raise RuntimeError("No GR1 observation has been cached.")
        return self._frames[-1]

    def sampled_window(self) -> torch.Tensor:
        if not self._frames:
            raise RuntimeError("No GR1 observation has been cached.")
        from longwam.runtime.policy import build_observation_window

        return build_observation_window(list(self._frames), self.temporal_spec.P, ACTION_VIDEO_FREQ_RATIO)


class RequestSequences:
    """Track monotonic raw control observations for multiple logical envs."""

    def __init__(self) -> None:
        self._next_steps: dict[str, int] = {}

    def reset(self, episode_id: str) -> None:
        episode_id = str(episode_id)
        if not episode_id:
            raise ValueError("Reset requires a non-empty episode_id.")
        self._next_steps[episode_id] = 0

    def validate(self, episode_id: str, control_step: int) -> None:
        if episode_id not in self._next_steps:
            raise RuntimeError(f"Episode {episode_id!r} must be reset before observations.")
        expected = self._next_steps[episode_id]
        if int(control_step) != expected:
            raise ValueError(
                f"Episode {episode_id!r} expected control_step={expected}, got {control_step}."
            )

    def release(self, episode_id: str) -> None:
        if self._next_steps.pop(str(episode_id), None) is None:
            raise RuntimeError(f"Episode {episode_id!r} has not been reset.")

    def commit(self, episode_id: str, control_step: int) -> None:
        self.validate(episode_id, control_step)
        self._next_steps[episode_id] += 1


@dataclass(frozen=True)
class PolicyInference:
    actions: np.ndarray
    input_image_sha256: str
    policy_inference_seconds: float


class CachedTextContext:
    """Load the exact prompt embeddings sealed by the GR1 training plan."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        context_len: int,
        file_identities: Sequence[Mapping[str, Any]],
    ) -> None:
        cache_dir = cache_dir.expanduser()
        if cache_dir.is_symlink() or not cache_dir.is_dir():
            raise ValueError(f"GR1 text cache must be a non-symlink directory: {cache_dir}")
        self.cache_dir = cache_dir.resolve(strict=True)
        self.context_len = int(context_len)
        if self.context_len != 128:
            raise ValueError(f"GR1 text context length must be 128, got {self.context_len}.")
        required = {"path", "sha256", "bytes", "device", "inode", "mtime_ns"}
        identities: dict[str, dict[str, Any]] = {}
        for identity in file_identities:
            if set(identity) != required:
                raise ValueError("Malformed GR1 text-cache file identity.")
            path = Path(str(identity["path"]))
            if path.parent.resolve(strict=True) != self.cache_dir:
                raise ValueError(f"GR1 text-cache file is outside the sealed directory: {path}")
            if path.name in identities:
                raise ValueError(f"Duplicate GR1 text-cache filename: {path.name}")
            identities[path.name] = dict(identity)
        if len(identities) != 186:
            raise ValueError(
                f"GR1 text cache must contain 186 sealed prompts, got {len(identities)}."
            )
        self._identities = identities
        self._loaded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        from longwam.datasets.lerobot.robot_video_dataset import get_text_cache_path

        prompt = str(prompt)
        if prompt in self._loaded:
            return self._loaded[prompt]
        raw_path = Path(get_text_cache_path(self.cache_dir, prompt, self.context_len))
        identity = self._identities.get(raw_path.name)
        if identity is None:
            raise KeyError(f"Prompt is absent from the sealed GR1 text cache: {prompt!r}")
        if (
            raw_path.is_symlink()
            or not raw_path.is_file()
            or not stat.S_ISREG(raw_path.stat().st_mode)
        ):
            raise ValueError(f"Invalid GR1 text-cache file: {raw_path}")
        metadata = raw_path.stat()
        observed = {
            "bytes": metadata.st_size,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mtime_ns": metadata.st_mtime_ns,
        }
        if observed != {key: identity[key] for key in observed}:
            raise ValueError(f"GR1 text-cache identity changed: {raw_path}")
        digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if digest != identity["sha256"]:
            raise ValueError(f"GR1 text-cache SHA-256 changed: {raw_path}")
        payload = torch.load(raw_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) != {"context", "mask"}:
            raise ValueError(f"Malformed GR1 text-cache payload: {raw_path}")
        context = torch.as_tensor(payload["context"])
        mask = torch.as_tensor(payload["mask"], dtype=torch.bool)
        if tuple(context.shape) != (self.context_len, 4096):
            raise ValueError(
                f"GR1 text context has wrong shape at {raw_path}: {tuple(context.shape)}"
            )
        if tuple(mask.shape) != (self.context_len,) or not bool(torch.isfinite(context).all()):
            raise ValueError(f"GR1 text context/mask is invalid: {raw_path}")
        # Match RobotVideoDataset exactly: zero padding embeddings, then expose
        # the full fixed-length sequence to Wan2.2 attention.
        context = context.clone()
        context[~mask] = 0
        result = (context, torch.ones_like(mask))
        self._loaded[prompt] = result
        return result


class RoboCasaGR1LongWAMPolicy:
    """Exact eval transforms, sealed per-env history, and H16 Long-WAM inference."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        video_transform: Callable[[torch.Tensor], torch.Tensor],
        state_transform: Callable[[np.ndarray], Any],
        action_normalizer: Any,
        prompt_template: str,
        text_context: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
        policy_seed: int | None,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        temporal_spec: TemporalSpec | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        self.model = model
        self.video_transform = video_transform
        self.state_transform = state_transform
        self.action_normalizer = action_normalizer
        self.prompt_template = str(prompt_template)
        self.text_context = text_context
        self.policy_seed = policy_seed
        self.num_inference_steps = int(num_inference_steps)
        self.temporal_spec = temporal_spec or TemporalSpec.from_past_obs_size(48)
        self.histories: dict[str, RawFrameHistory] = {}

    def reset(self, episode_id: str) -> None:
        self.histories[str(episode_id)] = RawFrameHistory(temporal_spec=self.temporal_spec)

    def release(self, episode_id: str) -> None:
        if self.histories.pop(str(episode_id), None) is None:
            raise RuntimeError(f"Episode {episode_id!r} has not been reset.")

    def observe(self, *, episode_id: str, image: np.ndarray) -> str:
        history = self._history(episode_id)
        history.append(self._transform_image(image))
        return array_sha256(image)

    def infer(
        self,
        *,
        episode_id: str,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> PolicyInference:
        history = self._history(episode_id)
        snapshot = history.snapshot()
        try:
            input_image_sha256 = self.observe(episode_id=episode_id, image=image)
            current = history.current.unsqueeze(0).to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
            obs_window = history.sampled_window().to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
            proprio = self._transform_state(state)
            prompt = self.prompt_template.format(task=instruction)
            context, context_mask = self.text_context(prompt)
            with torch.inference_mode():
                prediction, policy_inference_seconds = self._timed_infer_joint_ar(
                    prompt=None,
                    input_image=current,
                    obs_window=obs_window,
                    action_horizon=ACTION_HORIZON,
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=None,
                    text_cfg_scale=1.0,
                    negative_prompt="",
                    rand_device="cpu",
                    tiled=False,
                    seed=self.policy_seed,
                )
            if "action" not in prediction:
                raise KeyError("infer_joint_ar response is missing 'action'.")
            actions = torch.as_tensor(prediction["action"])
            if actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            if tuple(actions.shape) != (ACTION_HORIZON, ACTION_DIM):
                raise ValueError(
                    "Model action must have shape "
                    f"{(ACTION_HORIZON, ACTION_DIM)}, got {tuple(actions.shape)}."
                )
            denormalized = self.action_normalizer.denormalize(
                actions.detach().to(device="cpu", dtype=torch.float32)
            )
            return PolicyInference(
                actions=validate_action_chunk(denormalized),
                input_image_sha256=input_image_sha256,
                policy_inference_seconds=policy_inference_seconds,
            )
        except Exception:
            history.restore(snapshot)
            raise

    def _history(self, episode_id: str) -> RawFrameHistory:
        try:
            return self.histories[str(episode_id)]
        except KeyError as exc:
            raise RuntimeError(f"Episode {episode_id!r} has not been reset.") from exc

    def _timed_infer_joint_ar(self, **kwargs: Any) -> tuple[Mapping[str, Any], float]:
        """Synchronously time exactly one full model policy call."""
        device = torch.device(self.model.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            prediction = self.model.infer_joint_ar(**kwargs)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        if not isinstance(prediction, Mapping):
            raise TypeError("infer_joint_ar must return a mapping.")
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"Invalid synchronized policy latency: {elapsed!r}.")
        return prediction, float(elapsed)

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        image = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.uint8)
        transformed = torch.as_tensor(self.video_transform(image), dtype=torch.float32)
        if tuple(transformed.shape) != (1, 3, 224, 224):
            raise ValueError(
                "GR1 eval video transform must return (1, 3, 224, 224), "
                f"got {tuple(transformed.shape)}."
            )
        if not bool(torch.isfinite(transformed).all().item()):
            raise ValueError("GR1 eval video transform returned non-finite values.")
        minimum = float(transformed.min().item())
        maximum = float(transformed.max().item())
        if minimum < -VIDEO_RANGE_TOLERANCE or maximum > 1.0 + VIDEO_RANGE_TOLERANCE:
            raise ValueError(
                "GR1 eval video transform must remain within floating-point tolerance of [0,1] "
                "before "
                f"Long-WAM normalization, got [{minimum}, {maximum}]."
            )
        # Preserve the transform output exactly as training does. Bilinear
        # antialiasing may overshoot an endpoint by one float32 ULP; clipping
        # here would create an evaluation-only preprocessing difference.
        return transformed[0].mul(2.0).sub(1.0).contiguous()

    def _transform_state(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (RAW_STATE_DIM,):
            raise ValueError(
                f"GR1 raw state must have shape {(RAW_STATE_DIM,)}, got {state.shape}."
            )
        transformed = torch.as_tensor(
            self.state_transform(torch.from_numpy(state)),
            dtype=torch.float32,
        )
        if tuple(transformed.shape) != (MODEL_STATE_DIM,):
            raise ValueError(
                f"GR1 field-wise sine/cosine state must be {(MODEL_STATE_DIM,)}, "
                f"got {tuple(transformed.shape)}."
            )
        if not bool(torch.isfinite(transformed).all().item()):
            raise ValueError("GR1 transformed state contains non-finite values.")
        return transformed.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)


class PolicyRequestDispatcher:
    """Small transaction boundary shared by socket and CPU contract tests."""

    def __init__(self, policy: RoboCasaGR1LongWAMPolicy) -> None:
        self.policy = policy
        self.sequences = RequestSequences()

    def dispatch(self, request: Mapping[str, Any]) -> tuple[bytes, bool]:
        operation = str(request["operation"])
        try:
            if operation == "ping":
                return encode_response(message="ready"), False
            if operation == "reset":
                episode_id = str(request["episode_id"])
                self.sequences.reset(episode_id)
                self.policy.reset(episode_id)
                return encode_response(message=f"reset:{episode_id}"), False
            if operation == "release":
                episode_id = str(request["episode_id"])
                self.sequences.release(episode_id)
                self.policy.release(episode_id)
                return encode_response(message=f"released:{episode_id}"), False
            if operation in {"observe", "infer"}:
                episode_id = str(request["episode_id"])
                control_step = int(request["control_step"])
                self.sequences.validate(episode_id, control_step)
                if operation == "observe":
                    self.policy.observe(episode_id=episode_id, image=request["image"])
                    response = encode_response()
                else:
                    inference = self.policy.infer(
                        episode_id=episode_id,
                        image=request["image"],
                        state=request["state"],
                        instruction=str(request["prompt"]),
                    )
                    response = encode_response(
                        actions=inference.actions,
                        input_image_sha256=inference.input_image_sha256,
                        policy_inference_seconds=inference.policy_inference_seconds,
                    )
                self.sequences.commit(episode_id, control_step)
                return response, False
            if operation == "shutdown":
                return encode_response(message="shutting down"), True
            raise ValueError(f"Unsupported operation: {operation!r}.")
        except Exception as exc:
            LOGGER.exception("GR1 policy request failed: operation=%s", operation)
            return _error_response(exc), False


class UnixPolicyServer:
    """Single-model AF_UNIX server with multiple logical episode histories."""

    def __init__(self, socket_path: Path, dispatcher: PolicyRequestDispatcher) -> None:
        self.socket_path = Path(socket_path)
        self.dispatcher = dispatcher
        self._socket_identity: tuple[int, int] | None = None

    def serve_forever(self) -> None:
        self._prepare_socket_path()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)  # Local simulator and policy run as the same user.
            socket_stat = self.socket_path.stat()
            self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
            listener.listen(1)
            LOGGER.info("READY socket=%s", self.socket_path)
            should_stop = False
            while not should_stop:
                connection, _ = listener.accept()
                with connection:
                    should_stop = self._serve_connection(connection)
        finally:
            listener.close()
            self._remove_owned_socket()

    def _serve_connection(self, connection: socket.socket) -> bool:
        while True:
            payload = receive_packet(connection)
            if payload is None:
                return False
            try:
                response, should_stop = self.dispatcher.dispatch(decode_request(payload))
            except Exception as exc:
                LOGGER.exception("Invalid GR1 policy request")
                response, should_stop = _error_response(exc), False
            send_packet(connection, response)
            if should_stop:
                return True

    def _prepare_socket_path(self) -> None:
        if len(os.fsencode(self.socket_path)) > 103:
            raise ValueError(f"AF_UNIX socket path is too long: {self.socket_path}")
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.socket_path.exists():
            return
        if not stat.S_ISSOCK(self.socket_path.stat().st_mode):
            raise FileExistsError(f"Refusing to remove non-socket path: {self.socket_path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(self.socket_path))
        except (ConnectionRefusedError, FileNotFoundError):
            self.socket_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Another policy server owns {self.socket_path}.")
        finally:
            probe.close()

    def _remove_owned_socket(self) -> None:
        if self._socket_identity is None:
            return
        try:
            socket_stat = self.socket_path.stat()
        except FileNotFoundError:
            return
        if (socket_stat.st_dev, socket_stat.st_ino) == self._socket_identity:
            self.socket_path.unlink()


def _validate_run_config(
    config: Any,
    training_plan: Mapping[str, Any] | None = None,
) -> TemporalSpec:
    train = config.data.train
    processor = train.processor
    temporal_spec = TemporalSpec.from_past_obs_size(int(train.past_obs_size))
    if training_plan is not None:
        sealed = training_plan.get(TRAINING_PLAN_KEY)
        if sealed is None:
            # Read compatibility for the one completed P48 run predating the
            # temporal contract.  Every other P is required to be explicitly
            # present in the signed training plan.
            if temporal_spec.P != 48:
                raise ValueError(
                    "Signed training plan has no temporal_spec; only legacy P48 plans may omit it."
                )
        elif not isinstance(sealed, Mapping):
            raise ValueError("Signed training plan temporal_spec must be an object.")
        else:
            sealed_spec = TemporalSpec.from_mapping(sealed)
            if sealed_spec != temporal_spec:
                raise ValueError(
                    "Signed training plan and saved config disagree on temporal_spec: "
                    f"plan P={sealed_spec.P}, config P={temporal_spec.P}."
                )
    expected = {
        "data.train.num_frames": (int(train.num_frames), temporal_spec.N),
        "data.train.past_obs_size": (int(train.past_obs_size), temporal_spec.P),
        "data.train.action_chunk": (int(train.action_chunk), ACTION_HORIZON),
        "data.train.context_len": (int(train.context_len), 128),
        "data.train.action_video_freq_ratio": (
            int(train.action_video_freq_ratio),
            ACTION_VIDEO_FREQ_RATIO,
        ),
        "processor.num_image_steps": (int(processor.num_image_steps), temporal_spec.I),
        "processor.num_obs_steps": (int(processor.num_obs_steps), temporal_spec.N),
        "processor.num_output_cameras": (int(processor.num_output_cameras), 1),
        "processor.action_output_dim": (int(processor.action_output_dim), ACTION_DIM),
        "processor.proprio_output_dim": (int(processor.proprio_output_dim), MODEL_STATE_DIM),
        "model.proprio_dim": (int(config.model.proprio_dim), MODEL_STATE_DIM),
        "model.longwam_past_obs_size": (
            int(config.model.longwam_past_obs_size),
            temporal_spec.P,
        ),
        "model.longwam_num_imagine_frames": (
            int(config.model.longwam_num_imagine_frames),
            temporal_spec.k,
        ),
        "model.video_action_dim": (int(config.model.video_dit_config.action_dim), ACTION_DIM),
        "model.action_dim": (int(config.model.action_dit_config.action_dim), ACTION_DIM),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]

    def shape(value: Any) -> int | tuple[int, ...]:
        if isinstance(value, int):
            return int(value)
        return tuple(int(item) for item in value)

    def metadata(name: str) -> tuple[tuple[str, int | tuple[int, ...], int | tuple[int, ...]], ...]:
        return tuple(
            (str(item.key), shape(item.raw_shape), shape(item.shape))
            for item in getattr(train.shape_meta, name)
        )

    expected_metadata = {
        "images": (("ego_view", (3, 256, 256), (3, 224, 224)),),
        "action": (("default", 44, ACTION_DIM),),
        "state": (("default", 44, MODEL_STATE_DIM),),
    }
    for name, wanted in expected_metadata.items():
        actual = metadata(name)
        if actual != wanted:
            mismatches.append(f"data.train.shape_meta.{name}={actual!r} (expected {wanted!r})")
    if str(train.concat_multi_camera).strip().lower() not in {"none", "single", "null"}:
        mismatches.append(
            f"data.train.concat_multi_camera={train.concat_multi_camera!r} (expected single ego view)"
        )
    transforms = tuple(processor.action_state_transforms)
    if len(transforms) != 1 or not str(transforms[0].get("_target_", "")).endswith(
        ".GR1StateActionTransform"
    ):
        mismatches.append(
            "processor.action_state_transforms must contain only GR1StateActionTransform"
        )
    val_transforms = tuple(processor.val_transforms)
    if (
        len(val_transforms) != 1
        or not str(val_transforms[0].get("_target_", "")).endswith(".GR1OfficialVideoTransform")
        or bool(val_transforms[0].get("training", True))
    ):
        mismatches.append(
            "processor.val_transforms must be deterministic GR1OfficialVideoTransform"
        )
    if bool(processor.use_stepwise_action_norm):
        mismatches.append("processor.use_stepwise_action_norm must be false")
    if processor.norm_exception_mode is not None:
        mismatches.append("processor.norm_exception_mode must be null (no generic clamp)")
    if str(config.model.video_dit_config.video_attention_mask_mode) != "per_frame_causal":
        mismatches.append(
            "model.video_dit_config.video_attention_mask_mode must be per_frame_causal"
        )
    if bool(config.model.load_text_encoder):
        mismatches.append(
            "model.load_text_encoder must remain false (use sealed training contexts)"
        )
    if str(config.model.model_id) != EXPECTED_MODEL_ID:
        mismatches.append(f"model.model_id must remain {EXPECTED_MODEL_ID}")
    if bool(config.model.redirect_common_files):
        mismatches.append("model.redirect_common_files must remain false (use the sealed Wan VAE)")
    if mismatches:
        raise ValueError(
            "Saved run violates the GR1 temporal/H16 single-view contract:\n- "
            + "\n- ".join(mismatches)
        )
    return temporal_spec


def _validate_loaded_model(
    model: torch.nn.Module,
    temporal_spec: TemporalSpec | None = None,
) -> None:
    temporal_spec = temporal_spec or TemporalSpec.from_past_obs_size(48)
    temporal_factor = int(model.vae.temporal_downsample_factor)
    expected_clean_latents = (len(temporal_spec.history_offsets) - 1) // temporal_factor + 1
    expected = {
        "vae.temporal_downsample_factor": (temporal_factor, VAE_TEMPORAL_FACTOR),
        "num_clean_frames": (int(model.num_clean_frames), temporal_spec.M),
        "history-derived clean latents": (expected_clean_latents, temporal_spec.M),
        "num_imagine_frames": (int(model.num_imagine_frames), temporal_spec.k),
        "action_dim": (int(model.action_expert.action_dim), ACTION_DIM),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if model.text_encoder is not None or model.tokenizer is not None:
        mismatches.append("text encoder/tokenizer must not be loaded for cached-text evaluation")
    if mismatches:
        raise ValueError(
            f"Loaded model violates the GR1 M{temporal_spec.M}/k2/H16 contract:\n- "
            + "\n- ".join(mismatches)
        )


def _model_dtype(mixed_precision: str) -> torch.dtype:
    try:
        return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            mixed_precision.strip().lower()
        ]
    except KeyError as exc:
        raise ValueError(f"Unsupported mixed_precision={mixed_precision!r}.") from exc


def _error_response(exc: Exception) -> bytes:
    return encode_response(status="error", message=f"{type(exc).__name__}: {exc}")
