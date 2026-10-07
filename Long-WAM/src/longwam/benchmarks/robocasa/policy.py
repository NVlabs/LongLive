# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa/policy_server.py
# Changes: Research policy server split into a portable benchmark policy module.
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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/benchmarks/robocasa/policy.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Reusable benchmark implementation, separated from historical cluster orchestration."""

from __future__ import annotations

import logging
import os
import socket
import stat
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from longwam.benchmarks.robocasa.contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_VIDEO_FREQ_RATIO,
    CAMERA_KEYS,
    PAST_OBS_SIZE,
    SAMPLED_OBS_FRAMES,
    STATE_DIM,
    array_sha256,
    camera_bundle_sha256,
    decode_request,
    encode_response,
    receive_packet,
    send_packet,
)
from longwam.datasets.lerobot.robocasa_contract import (  # noqa: E402
    ROBOCASA_DELTA_ACTION_DIM_MASK,
    ROBOCASA_LATENT_SLOT_LAYOUT,
    ROBOCASA_LEFT_MAIN_LAYOUT,
    ROBOCASA_MOSAIC_LAYOUT,
    ROBOCASA_RGB_FRAME_SHAPE,
    ROBOCASA_RGB_LAYOUTS,
)

LOGGER = logging.getLogger("robocasa-policy-server")

MODEL_FRAME_SHAPE = ROBOCASA_RGB_FRAME_SHAPE

LATENT_SLOT_FRAME_SHAPE = (3, 3, 256, 256)

MODEL_FRAME_SHAPES = {
    ROBOCASA_MOSAIC_LAYOUT: MODEL_FRAME_SHAPE,
    ROBOCASA_LEFT_MAIN_LAYOUT: MODEL_FRAME_SHAPE,
    ROBOCASA_LATENT_SLOT_LAYOUT: LATENT_SLOT_FRAME_SHAPE,
}

RAW_HISTORY_FRAMES = PAST_OBS_SIZE + 1

EXPECTED_NUM_FRAMES = 81

EXPECTED_NUM_CLEAN_LATENTS = 4

EXPECTED_NUM_IMAGINE_FRAMES = 2

DEFAULT_NUM_INFERENCE_STEPS = 10

EXPECTED_CONTEXT_LEN = 128

EXPECTED_NUM_IMAGE_STEPS = 21

EXPECTED_NUM_OUTPUT_CAMERAS = 3


class RawFrameHistory:
    """Bounded raw-step history with training-identical oldest-frame padding."""

    def __init__(
        self,
        *,
        frame_shape: tuple[int, ...] = MODEL_FRAME_SHAPE,
    ) -> None:
        self.frame_shape = tuple(int(value) for value in frame_shape)
        if len(self.frame_shape) not in (3, 4):
            raise ValueError(
                f"A model frame must have shape [C,H,W] or [V,C,H,W], got {self.frame_shape}."
            )
        self._frames: deque[torch.Tensor] = deque(maxlen=RAW_HISTORY_FRAMES)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def snapshot(self) -> tuple[torch.Tensor, ...]:
        """Return a cheap immutable checkpoint of the stored tensor references."""
        return tuple(self._frames)

    def restore(self, frames: Sequence[torch.Tensor]) -> None:
        """Restore a checkpoint, including any frame evicted at max capacity."""
        if len(frames) > RAW_HISTORY_FRAMES:
            raise ValueError(
                f"History checkpoint contains {len(frames)} frames, limit is {RAW_HISTORY_FRAMES}."
            )
        self._frames = deque(frames, maxlen=RAW_HISTORY_FRAMES)

    def append(self, frame: torch.Tensor) -> None:
        frame = torch.as_tensor(frame)
        if tuple(frame.shape) != self.frame_shape:
            raise ValueError(
                f"Model frame must have shape {self.frame_shape}, got {tuple(frame.shape)}."
            )
        if not frame.is_floating_point():
            raise TypeError(f"Model frame must be floating point, got {frame.dtype}.")
        if not bool(torch.isfinite(frame).all().item()):
            raise ValueError("Model frame contains non-finite values.")
        self._frames.append(
            frame.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )

    @property
    def current(self) -> torch.Tensor:
        if not self._frames:
            raise RuntimeError("Cannot access the current frame before an observation.")
        return self._frames[-1]

    def sampled_window(self) -> torch.Tensor:
        """Insert time before H/W and sample raw offsets 0,4,...,48.

        Mosaic frames produce ``[1,C,13,H,W]``. Per-view frames produce
        ``[1,V,C,13,H,W]`` without changing camera order or spatial content.
        """
        if not self._frames:
            raise RuntimeError("Cannot build an observation window before an observation.")

        from longwam.runtime.policy import build_observation_window

        return build_observation_window(list(self._frames), PAST_OBS_SIZE, ACTION_VIDEO_FREQ_RATIO)


class RequestSequence:
    """Enforce one observation for every monotonically increasing control step."""

    def __init__(self) -> None:
        self.episode_id: str | None = None
        self.next_control_step = 0

    def reset(self, episode_id: str) -> None:
        episode_id = str(episode_id)
        if not episode_id:
            raise ValueError("Reset requires a non-empty episode_id.")
        self.episode_id = episode_id
        self.next_control_step = 0

    def validate_observation(self, *, episode_id: str, control_step: int) -> None:
        if self.episode_id is None:
            raise RuntimeError("An episode reset is required before observations.")
        if episode_id != self.episode_id:
            raise ValueError(
                f"Observation episode_id {episode_id!r} does not match active "
                f"episode {self.episode_id!r}."
            )
        if int(control_step) != self.next_control_step:
            raise ValueError(f"Expected control_step={self.next_control_step}, got {control_step}.")

    def commit_observation(self, *, episode_id: str, control_step: int) -> None:
        self.validate_observation(
            episode_id=episode_id,
            control_step=control_step,
        )
        self.next_control_step += 1


def cameras_to_model_frame(
    cameras: Mapping[str, np.ndarray],
    *,
    concat_multi_camera: str = ROBOCASA_MOSAIC_LAYOUT,
) -> torch.Tensor:
    """Map raw cameras to the selected training-identical model representation."""
    from longwam.benchmarks.robocasa.contract import extract_cameras
    from longwam.datasets.lerobot.robocasa_contract import (
        build_robocasa_camera_layout,
        validate_robocasa_latent_slot_cameras,
    )

    cameras = extract_cameras(cameras)
    camera_batch = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(cameras[key]))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
            for key in CAMERA_KEYS
        ],
        dim=0,
    )
    if concat_multi_camera in ROBOCASA_RGB_LAYOUTS:
        frame = build_robocasa_camera_layout(
            camera_batch,
            layout=concat_multi_camera,
        )
    elif concat_multi_camera == ROBOCASA_LATENT_SLOT_LAYOUT:
        validate_robocasa_latent_slot_cameras(camera_batch.unsqueeze(1))
        frame = camera_batch
    else:
        raise ValueError(
            "Unsupported RoboCasa camera representation "
            f"{concat_multi_camera!r}; expected one of "
            f"{tuple(MODEL_FRAME_SHAPES)}."
        )

    expected_shape = MODEL_FRAME_SHAPES[concat_multi_camera]
    if tuple(frame.shape) != expected_shape:
        raise AssertionError(
            f"RoboCasa model frame has shape {tuple(frame.shape)}, "
            f"expected {expected_shape} for {concat_multi_camera!r}."
        )
    return frame.mul_(2.0).sub_(1.0).contiguous()


@dataclass(frozen=True)
class PolicyInference:
    """One action chunk and the exact model mosaic used to produce it."""

    actions: np.ndarray
    model_mosaic_sha256: str


class RoboCasaLongWAMPolicy:
    """State normalization, P4 history, inference, and action denormalization."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        prompt_template: str,
        policy_seed: int | None,
        concat_multi_camera: str = ROBOCASA_MOSAIC_LAYOUT,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    ) -> None:
        self.model = model
        self.processor = processor
        self.prompt_template = str(prompt_template)
        self.policy_seed = policy_seed
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.num_inference_steps = int(num_inference_steps)
        if concat_multi_camera not in MODEL_FRAME_SHAPES:
            raise ValueError(f"Unsupported RoboCasa camera representation {concat_multi_camera!r}.")
        self.concat_multi_camera = concat_multi_camera
        self.history = RawFrameHistory(frame_shape=MODEL_FRAME_SHAPES[concat_multi_camera])

    def reset(self) -> None:
        self.history.clear()

    def observe(self, cameras: Mapping[str, np.ndarray]) -> str:
        self.history.append(
            cameras_to_model_frame(
                cameras,
                concat_multi_camera=self.concat_multi_camera,
            )
        )
        return array_sha256(self.history.current)

    def infer(
        self,
        *,
        cameras: Mapping[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> PolicyInference:
        history_snapshot = self.history.snapshot()
        try:
            model_mosaic_sha256 = self.observe(cameras)
            input_image = self.history.current.unsqueeze(0).to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
            obs_window = self.history.sampled_window().to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
            proprio = self._normalize_state(state)
            prompt = self.prompt_template.format(task=instruction)

            with torch.inference_mode():
                prediction = self.model.infer_joint_ar(
                    prompt=prompt,
                    input_image=input_image,
                    obs_window=obs_window,
                    action_horizon=ACTION_HORIZON,
                    proprio=proprio,
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
            return PolicyInference(
                actions=self._denormalize_action(prediction["action"]),
                model_mosaic_sha256=model_mosaic_sha256,
            )
        except Exception:
            # The sequence is not committed on failure. Keep the history
            # transactional so this control step can be retried exactly.
            self.history.restore(history_snapshot)
            raise

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"RoboCasa state must have shape {(STATE_DIM,)}, got {state.shape}.")
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state field.")
        state_key = state_meta[0]["key"]
        batch = {
            "state": {
                state_key: torch.from_numpy(state).unsqueeze(0),
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        normalized = batch["state"][state_key]
        if tuple(normalized.shape) != (1, STATE_DIM):
            raise ValueError(
                f"Normalized state must have shape {(1, STATE_DIM)}, got {tuple(normalized.shape)}."
            )
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError("Normalized state contains non-finite values.")
        return normalized

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        action = torch.as_tensor(action)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if tuple(action.shape) != (1, ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                "Model action must have shape "
                f"{(1, ACTION_HORIZON, ACTION_DIM)}, got {tuple(action.shape)}."
            )
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action field.")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denormalized = normalizer.backward(action.to(device="cpu", dtype=torch.float32))[0].numpy()
        if not np.isfinite(denormalized).all():
            raise ValueError("Denormalized action contains non-finite values.")
        return np.ascontiguousarray(denormalized, dtype=np.float32)


class PolicyRequestDispatcher:
    """Decode-independent request dispatcher, kept small for CPU contract tests."""

    def __init__(self, policy: RoboCasaLongWAMPolicy) -> None:
        self.policy = policy
        self.sequence = RequestSequence()

    def dispatch(self, request: Mapping[str, Any]) -> tuple[bytes, bool]:
        operation = str(request["operation"])
        try:
            if operation == "ping":
                return encode_response(message="ready"), False
            if operation == "reset":
                episode_id = str(request["episode_id"])
                if not episode_id:
                    raise ValueError("Reset requires a non-empty episode_id.")
                self.sequence.reset(episode_id)
                self.policy.reset()
                return encode_response(message=f"reset:{episode_id}"), False
            if operation in {"observe", "infer"}:
                episode_id = str(request["episode_id"])
                control_step = int(request["control_step"])
                self.sequence.validate_observation(
                    episode_id=episode_id,
                    control_step=control_step,
                )
                if operation == "observe":
                    self.policy.observe(request["cameras"])
                    response = encode_response()
                else:
                    camera_sha256 = camera_bundle_sha256(request["cameras"])
                    inference = self.policy.infer(
                        cameras=request["cameras"],
                        state=request["state"],
                        instruction=str(request["prompt"]),
                    )
                    response = encode_response(
                        actions=inference.actions,
                        camera_bundle_sha256=camera_sha256,
                        model_mosaic_sha256=inference.model_mosaic_sha256,
                    )
                self.sequence.commit_observation(
                    episode_id=episode_id,
                    control_step=control_step,
                )
                return response, False
            if operation == "shutdown":
                return encode_response(message="shutting down"), True
            raise ValueError(f"Unsupported operation: {operation!r}.")
        except Exception as exc:
            LOGGER.exception("Policy request failed: operation=%s", operation)
            return _error_response(exc), False


class UnixPolicyServer:
    """Single-policy AF_UNIX server supporting reconnecting simulator clients."""

    def __init__(self, socket_path: Path, dispatcher: PolicyRequestDispatcher) -> None:
        self.socket_path = Path(socket_path)
        self.dispatcher = dispatcher
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None

    def serve_forever(self) -> None:
        self._prepare_socket_path()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener = listener
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
                    try:
                        should_stop = self._serve_connection(connection)
                    except (BrokenPipeError, ConnectionError, ValueError) as exc:
                        LOGGER.warning("Simulator connection closed: %s", exc)
        finally:
            listener.close()
            self._listener = None
            self._remove_owned_socket()

    def _serve_connection(self, connection: socket.socket) -> bool:
        while True:
            payload = receive_packet(connection)
            if payload is None:
                return False
            try:
                request = decode_request(payload)
                response, should_stop = self.dispatcher.dispatch(request)
            except Exception as exc:
                LOGGER.exception("Invalid policy request")
                response = _error_response(exc)
                should_stop = False
            send_packet(connection, response)
            if should_stop:
                return True

    def _prepare_socket_path(self) -> None:
        encoded_length = len(os.fsencode(self.socket_path))
        if encoded_length > 103:
            raise ValueError(
                f"AF_UNIX socket path is too long ({encoded_length} bytes): {self.socket_path}"
            )
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.socket_path.exists():
            return
        mode = self.socket_path.stat().st_mode
        if not stat.S_ISSOCK(mode):
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


def _validate_run_config(config: Any) -> str:
    train = config.data.train
    processor = train.processor
    expected = {
        "data.train.num_frames": (int(train.num_frames), EXPECTED_NUM_FRAMES),
        "data.train.past_obs_size": (int(train.past_obs_size), PAST_OBS_SIZE),
        "data.train.action_chunk": (int(train.action_chunk), ACTION_HORIZON),
        "data.train.context_len": (int(train.context_len), EXPECTED_CONTEXT_LEN),
        "data.train.action_video_freq_ratio": (
            int(train.action_video_freq_ratio),
            ACTION_VIDEO_FREQ_RATIO,
        ),
        "data.train.processor.num_obs_steps": (
            int(processor.num_obs_steps),
            EXPECTED_NUM_FRAMES,
        ),
        "data.train.processor.num_image_steps": (
            int(processor.num_image_steps),
            EXPECTED_NUM_IMAGE_STEPS,
        ),
        "data.train.processor.num_output_cameras": (
            int(processor.num_output_cameras),
            EXPECTED_NUM_OUTPUT_CAMERAS,
        ),
        "data.train.processor.action_output_dim": (
            int(processor.action_output_dim),
            ACTION_DIM,
        ),
        "data.train.processor.proprio_output_dim": (
            int(processor.proprio_output_dim),
            STATE_DIM,
        ),
        "data.train.processor.norm_default_mode": (
            str(processor.norm_default_mode),
            "min/max",
        ),
        "model.proprio_dim": (int(config.model.proprio_dim), STATE_DIM),
        "model.longwam_past_obs_size": (
            int(config.model.longwam_past_obs_size),
            PAST_OBS_SIZE,
        ),
        "model.longwam_num_imagine_frames": (
            int(config.model.longwam_num_imagine_frames),
            EXPECTED_NUM_IMAGINE_FRAMES,
        ),
        "model.video_dit_config.action_dim": (
            int(config.model.video_dit_config.action_dim),
            ACTION_DIM,
        ),
        "model.action_dit_config.action_dim": (
            int(config.model.action_dit_config.action_dim),
            ACTION_DIM,
        ),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    for name, actual in (
        (
            "data.train.processor.use_stepwise_action_norm",
            processor.use_stepwise_action_norm,
        ),
        (
            "data.train.processor.action_state_transforms",
            processor.action_state_transforms,
        ),
        (
            "data.train.processor.norm_exception_mode",
            processor.norm_exception_mode,
        ),
    ):
        expected_value = False if name.endswith("use_stepwise_action_norm") else None
        if actual is not expected_value:
            mismatches.append(f"{name}={actual!r} (expected {expected_value!r})")

    delta_mask = tuple(processor.delta_action_dim_mask.default)
    if delta_mask != ROBOCASA_DELTA_ACTION_DIM_MASK or any(
        type(value) is not bool for value in delta_mask
    ):
        mismatches.append(
            "data.train.processor.delta_action_dim_mask.default="
            f"{delta_mask!r} (expected {ROBOCASA_DELTA_ACTION_DIM_MASK!r})"
        )

    def vector_metadata(name: str) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (str(item.key), int(item.raw_shape), int(item.shape))
            for item in getattr(train.shape_meta, name)
        )

    action_metadata = vector_metadata("action")
    state_metadata = vector_metadata("state")
    expected_action_metadata = (("default", ACTION_DIM, ACTION_DIM),)
    expected_state_metadata = (("default", STATE_DIM, STATE_DIM),)
    if action_metadata != expected_action_metadata:
        mismatches.append(
            "data.train.shape_meta.action="
            f"{action_metadata!r} (expected {expected_action_metadata!r})"
        )
    if state_metadata != expected_state_metadata:
        mismatches.append(
            f"data.train.shape_meta.state={state_metadata!r} (expected {expected_state_metadata!r})"
        )
    concat_multi_camera = str(train.concat_multi_camera)
    expected_frame_shape = MODEL_FRAME_SHAPES.get(concat_multi_camera)
    if expected_frame_shape is None:
        mismatches.append(
            "data.train.concat_multi_camera="
            f"{concat_multi_camera!r} (expected one of "
            f"{tuple(MODEL_FRAME_SHAPES)!r})"
        )
    else:
        video_size = tuple(int(value) for value in train.video_size)
        expected_video_size = expected_frame_shape[-2:]
        if video_size != expected_video_size:
            mismatches.append(
                f"data.train.video_size={video_size!r} "
                f"(expected {expected_video_size!r} for "
                f"{concat_multi_camera!r})"
            )
    if concat_multi_camera == ROBOCASA_LATENT_SLOT_LAYOUT:
        camera_metadata = tuple(
            (
                str(item.key),
                tuple(int(value) for value in item.raw_shape),
                tuple(int(value) for value in item.shape),
            )
            for item in train.shape_meta.images
        )
        expected_metadata = (
            ("robot0_agentview_left", (3, 256, 256), (3, 256, 256)),
            ("robot0_agentview_right", (3, 256, 256), (3, 256, 256)),
            ("robot0_eye_in_hand", (3, 256, 256), (3, 256, 256)),
        )
        if camera_metadata != expected_metadata:
            mismatches.append(
                "data.train.shape_meta.images does not preserve the fixed "
                "left/right/wrist 3x256x256 latent-slot contract"
            )
    if str(config.model.video_dit_config.video_attention_mask_mode) != ("per_frame_causal"):
        mismatches.append(
            "model.video_dit_config.video_attention_mask_mode must be 'per_frame_causal'"
        )
    if mismatches:
        raise ValueError(
            "Saved run config does not match the RoboCasa M4/k2/action32 "
            "evaluation contract:\n- " + "\n- ".join(mismatches)
        )
    return concat_multi_camera


def _validate_loaded_model(model: torch.nn.Module) -> None:
    temporal_factor = int(model.vae.temporal_downsample_factor)
    expected_clean_latents = (SAMPLED_OBS_FRAMES - 1) // temporal_factor + 1
    expected = {
        "num_clean_frames": (
            int(model.num_clean_frames),
            EXPECTED_NUM_CLEAN_LATENTS,
        ),
        "history-derived clean latents": (
            expected_clean_latents,
            EXPECTED_NUM_CLEAN_LATENTS,
        ),
        "num_imagine_frames": (
            int(model.num_imagine_frames),
            EXPECTED_NUM_IMAGINE_FRAMES,
        ),
        "action_dim": (int(model.action_expert.action_dim), ACTION_DIM),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "Loaded model does not match M4/k2/action32:\n- " + "\n- ".join(mismatches)
        )


def _model_dtype(mixed_precision: str) -> torch.dtype:
    precision = mixed_precision.strip().lower()
    mapping = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported mixed_precision={mixed_precision!r}.") from exc


def _error_response(exc: Exception) -> bytes:
    message = f"{type(exc).__name__}: {exc}"
    return encode_response(status="error", message=message)
