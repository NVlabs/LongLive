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

"""Frozen RoboCasa GR1 Tabletop evaluation contract.

This module deliberately depends only on NumPy and the Python standard library
so the simulator and Long-WAM policy can run in separate environments.  It
freezes the N1.5 task registry, the arms-and-waist modalities, and the exact
single-view evaluation preprocessing described by the benchmark repository.
"""

from __future__ import annotations

import hashlib
import io
import math
import socket
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from longwam.benchmarks.robocasa_gr1.temporal_contract import (
    ACTION_HORIZON as TEMPORAL_ACTION_HORIZON,
    ACTION_VIDEO_FREQ_RATIO as TEMPORAL_ACTION_VIDEO_FREQ_RATIO,
    TemporalSpec,
)


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 4 * 1024 * 1024

DATASET_PREFIX = "gr1_arms_waist."
EGO_VIEW_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
MODEL_EGO_VIEW_KEY = "video.ego_view"
PROMPT_KEY = "annotation.human.coarse_action"
UNLOCKED_WAIST_PREFIX = "unlocked_waist: "

MUJOCO_EGO_SHAPE = (800, 1280, 3)
OFFICIAL_EGO_SHAPE = (256, 256, 3)
MODEL_EGO_SHAPE = (224, 224, 3)
RAW_EVAL_CROP = (310, 770, 110, 1130)
CENTER_CROP_SIZE = 243

STATE_FIELDS = (
    ("state.left_arm", 7),
    ("state.right_arm", 7),
    ("state.left_hand", 6),
    ("state.right_hand", 6),
    ("state.waist", 3),
)
ACTION_FIELDS = tuple((name.replace("state.", "action."), width) for name, width in STATE_FIELDS)
RAW_STATE_DIM = sum(width for _, width in STATE_FIELDS)
MODEL_STATE_DIM = 2 * RAW_STATE_DIM
ACTION_DIM = sum(width for _, width in ACTION_FIELDS)

ACTION_HORIZON = TEMPORAL_ACTION_HORIZON
EXECUTE_ACTIONS = 16
# These exports intentionally remain the legacy P48 values.  New runs carry a
# TemporalSpec in their signed plan and must not infer their horizon from these
# compatibility constants.
PAST_OBS_SIZE = 48
ACTION_VIDEO_FREQ_RATIO = TEMPORAL_ACTION_VIDEO_FREQ_RATIO
LEGACY_TEMPORAL_SPEC = TemporalSpec.from_past_obs_size(PAST_OBS_SIZE)
HISTORY_OFFSETS = LEGACY_TEMPORAL_SPEC.history_offsets
SAMPLED_OBS_FRAMES = len(HISTORY_OFFSETS)
MAX_CONTROL_STEPS = 720
OFFICIAL_N_ENVS = 5
OFFICIAL_EPISODES_PER_TASK = 10

_HEADER = struct.Struct("!Q")
_ARRAY_HASH_DOMAIN = b"long-wam-gr1-array-sha256-v1\0"


@dataclass(frozen=True)
class TaskSpec:
    """One immutable official dataset-to-environment mapping."""

    dataset_suffix: str
    gym_id: str
    articulated_close: bool = False

    @property
    def dataset_name(self) -> str:
        return f"{DATASET_PREFIX}{self.dataset_suffix}"


TASK_SPECS = (
    TaskSpec(
        "CupToDrawer",
        "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
        True,
    ),
    TaskSpec(
        "PotatoToMicrowave",
        "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
        True,
    ),
    TaskSpec(
        "PlaceMilkToMicrowave",
        "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
        True,
    ),
    TaskSpec(
        "PlaceBottleToCabinet",
        "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
        True,
    ),
    TaskSpec(
        "WineToCabinet",
        "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
        True,
    ),
    TaskSpec(
        "CanToDrawer",
        "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
        True,
    ),
    TaskSpec(
        "CuttingboardToBasket",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "CuttingboardToCardboardBox",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "CuttingboardToPan",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "CuttingboardToPot",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "CuttingboardToTieredBasket",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlacematToBasket",
        "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlacematToBowl",
        "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlacematToPlate",
        "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlacematToTieredShelf",
        "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlateToBowl",
        "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlateToCardboardBox",
        "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlateToPan",
        "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "PlateToPlate",
        "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "TrayToCardboardBox",
        "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "TrayToPlate",
        "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "TrayToPot",
        "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "TrayToTieredBasket",
        "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    TaskSpec(
        "TrayToTieredShelf",
        "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
)

TASK_BY_DATASET = {spec.dataset_name: spec for spec in TASK_SPECS}
TASK_BY_GYM_ID = {spec.gym_id: spec for spec in TASK_SPECS}
if len(TASK_SPECS) != 24 or len(TASK_BY_DATASET) != 24 or len(TASK_BY_GYM_ID) != 24:
    raise RuntimeError("The frozen GR1 task registry must contain 24 unique mappings.")
if sum(spec.articulated_close for spec in TASK_SPECS) != 6:
    raise RuntimeError("The frozen GR1 registry must contain six articulated close tasks.")


def array_sha256(array: Any) -> str:
    """Hash exact contiguous values together with dtype and shape."""
    if all(hasattr(array, name) for name in ("detach", "cpu", "contiguous", "numpy")):
        array = array.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(np.asarray(array))
    if value.dtype.hasobject:
        raise TypeError("Object arrays cannot be fingerprinted.")
    digest = hashlib.sha256()
    digest.update(_ARRAY_HASH_DOMAIN)
    _update_length_prefixed(digest, value.dtype.str.encode("ascii"))
    digest.update(struct.pack("!I", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("!Q", int(dimension)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def extract_raw_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Concatenate the five official current-state fields into ordered 29D."""
    values = []
    for key, width in STATE_FIELDS:
        if key not in observation:
            raise KeyError(f"GR1 observation is missing {key!r}.")
        value = np.asarray(observation[key], dtype=np.float32)
        if value.shape != (width,):
            raise ValueError(f"{key} must have shape {(width,)}, got {value.shape}.")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values.")
        values.append(value)
    return np.ascontiguousarray(np.concatenate(values), dtype=np.float32)


def action_row_to_env(action: Any) -> dict[str, np.ndarray]:
    """Split one denormalized 29D action into the official five Gym fields."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"GR1 action must have shape {(ACTION_DIM,)}, got {action.shape}.")
    if not np.isfinite(action).all():
        raise ValueError("GR1 action contains non-finite values.")
    result: dict[str, np.ndarray] = {}
    offset = 0
    for key, width in ACTION_FIELDS:
        result[key] = np.ascontiguousarray(action[offset : offset + width])
        offset += width
    if offset != ACTION_DIM:
        raise AssertionError("Internal GR1 action field contract error.")
    return result


def validate_action_chunk(actions: Any) -> np.ndarray:
    """Require one complete H16 x 29 real-action chunk."""
    actions = np.asarray(actions, dtype=np.float32)
    expected = (ACTION_HORIZON, ACTION_DIM)
    if actions.shape != expected:
        raise ValueError(f"GR1 action chunk must have shape {expected}, got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError("GR1 action chunk contains non-finite values.")
    return np.ascontiguousarray(actions)


def ensure_unlocked_waist_prompt(value: Any) -> str:
    """Validate the exact coarse prompt emitted by the official Gym wrapper."""
    if not isinstance(value, str):
        raise TypeError(f"GR1 coarse-action prompt must be a string, got {type(value).__name__}.")
    text = value
    if text != text.strip():
        raise ValueError("GR1 coarse-action prompt must not have surrounding whitespace.")
    if text.startswith("locked_waist:"):
        raise ValueError("GR1 arms-and-waist evaluation cannot use a locked-waist prompt.")
    if text == UNLOCKED_WAIST_PREFIX.rstrip():
        raise ValueError("GR1 coarse-action prompt has an empty instruction.")
    if not text.startswith(UNLOCKED_WAIST_PREFIX):
        raise ValueError(
            f"GR1 prompt must start with the official prefix {UNLOCKED_WAIST_PREFIX!r}."
        )
    if text.count(UNLOCKED_WAIST_PREFIX) != 1:
        raise ValueError("GR1 prompt must contain the unlocked-waist prefix exactly once.")
    instruction = text[len(UNLOCKED_WAIST_PREFIX) :]
    if not instruction:
        raise ValueError("GR1 coarse-action prompt has an empty instruction.")
    if instruction != instruction.strip():
        raise ValueError("GR1 coarse-action instruction must not have surrounding whitespace.")
    return text


def extract_official_ego256(observation: Mapping[str, Any]) -> np.ndarray:
    """Read the official already-flipped/cropped/padded 256-square RGB field."""
    if EGO_VIEW_KEY not in observation:
        raise KeyError(f"GR1 observation is missing {EGO_VIEW_KEY!r}.")
    image = np.asarray(observation[EGO_VIEW_KEY])
    _validate_rgb(image, OFFICIAL_EGO_SHAPE, EGO_VIEW_KEY)
    return np.ascontiguousarray(image)


ResizeRGB = Callable[[np.ndarray, tuple[int, int], str], np.ndarray]


def mujoco_rgb_to_official_ego256(
    raw_rgb: Any,
    *,
    resize: ResizeRGB | None = None,
) -> np.ndarray:
    """Apply basic vertical flip followed by the official raw-camera hotfix."""
    raw_rgb = np.asarray(raw_rgb)
    _validate_rgb(raw_rgb, MUJOCO_EGO_SHAPE, "raw MuJoCo ego RGB")
    vertically_corrected = np.ascontiguousarray(raw_rgb[::-1, :, :])
    return vertically_corrected_rgb_to_official_ego256(
        vertically_corrected,
        resize=resize,
    )


def vertically_corrected_rgb_to_official_ego256(
    image: Any,
    *,
    resize: ResizeRGB | None = None,
) -> np.ndarray:
    """Crop 800x1280 RGB to 720x480, pad to 720-square, then resize to 256."""
    image = np.asarray(image)
    _validate_rgb(image, MUJOCO_EGO_SHAPE, "vertically corrected ego RGB")
    resize = resize or _opencv_resize_rgb
    top, bottom, left, right = RAW_EVAL_CROP
    cropped = np.ascontiguousarray(image[top:bottom, left:right, :])
    resized = resize(cropped, (480, 720), "area")
    _validate_rgb(resized, (480, 720, 3), "720x480 ego resize")
    padded = np.pad(
        resized,
        ((120, 120), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    output = resize(np.ascontiguousarray(padded), (256, 256), "area")
    _validate_rgb(output, OFFICIAL_EGO_SHAPE, "official ego256")
    return np.ascontiguousarray(output)


def official_ego256_to_model224(
    image: Any,
    *,
    resize: ResizeRGB | None = None,
) -> np.ndarray:
    """Apply CenterCrop243 and linear resize for simulator-side diagnostics.

    Formal policy inference uses the shared training implementation
    ``GR1OfficialVideoTransform(training=False)`` so its antialiasing and
    floating-point behavior are byte-identical to the dataset path.
    """
    image = np.asarray(image)
    _validate_rgb(image, OFFICIAL_EGO_SHAPE, "official ego256")
    resize = resize or _opencv_resize_rgb
    start = (OFFICIAL_EGO_SHAPE[0] - CENTER_CROP_SIZE) // 2
    cropped = np.ascontiguousarray(
        image[start : start + CENTER_CROP_SIZE, start : start + CENTER_CROP_SIZE, :]
    )
    output = resize(cropped, MODEL_EGO_SHAPE[:2], "linear")
    _validate_rgb(output, MODEL_EGO_SHAPE, "model ego224")
    return np.ascontiguousarray(output)


def encode_request(
    operation: str,
    *,
    episode_id: str = "",
    control_step: int = -1,
    image: np.ndarray | None = None,
    state: np.ndarray | None = None,
    prompt: str = "",
) -> bytes:
    """Encode one non-pickle simulator-to-policy request."""
    if operation not in {"ping", "reset", "release", "observe", "infer", "shutdown"}:
        raise ValueError(f"Unsupported operation: {operation!r}.")
    arrays: dict[str, np.ndarray] = {
        "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        "operation": np.asarray(operation),
        "episode_id": np.asarray(str(episode_id)),
        "control_step": np.asarray(int(control_step), dtype=np.int64),
        "prompt": np.asarray(str(prompt)),
    }
    if image is not None:
        image = np.asarray(image)
        _validate_rgb(image, OFFICIAL_EGO_SHAPE, "request ego256")
        arrays["ego256"] = np.ascontiguousarray(image)
    if state is not None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (RAW_STATE_DIM,):
            raise ValueError(
                f"Request state must have shape {(RAW_STATE_DIM,)}, got {state.shape}."
            )
        arrays["state"] = np.ascontiguousarray(state)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return _bounded_payload(buffer.getvalue(), "request")


def decode_request(payload: bytes) -> dict[str, Any]:
    """Decode and strictly validate a simulator-to-policy request."""
    _bounded_payload(payload, "request")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"protocol_version", "operation", "episode_id", "control_step", "prompt"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Request is missing fields: {sorted(missing)}.")
    version = _scalar(arrays["protocol_version"], "protocol_version", int)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: {version} != {PROTOCOL_VERSION}.")
    operation = _scalar(arrays["operation"], "operation", str)
    if operation not in {"ping", "reset", "release", "observe", "infer", "shutdown"}:
        raise ValueError(f"Unsupported operation: {operation!r}.")
    result: dict[str, Any] = {
        "operation": operation,
        "episode_id": _scalar(arrays["episode_id"], "episode_id", str),
        "control_step": _scalar(arrays["control_step"], "control_step", int),
        "prompt": _scalar(arrays["prompt"], "prompt", str),
    }
    if operation in {"observe", "infer"}:
        if "ego256" not in arrays:
            raise ValueError("Observation request is missing ego256.")
        image = np.asarray(arrays["ego256"])
        _validate_rgb(image, OFFICIAL_EGO_SHAPE, "request ego256")
        result["image"] = np.ascontiguousarray(image)
    if operation == "infer":
        if "state" not in arrays:
            raise ValueError("Inference request is missing state.")
        state = np.asarray(arrays["state"], dtype=np.float32)
        if state.shape != (RAW_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(
                f"Inference state must be finite {(RAW_STATE_DIM,)}, got {state.shape}."
            )
        result["state"] = np.ascontiguousarray(state)
        result["prompt"] = ensure_unlocked_waist_prompt(result["prompt"])
    return result


def encode_response(
    *,
    status: str = "ok",
    actions: np.ndarray | None = None,
    input_image_sha256: str = "",
    policy_inference_seconds: float | None = None,
    message: str = "",
) -> bytes:
    """Encode one policy response, requiring a complete H16 action chunk."""
    if status not in {"ok", "error"}:
        raise ValueError(f"Unsupported response status: {status!r}.")
    if status == "error" and (
        not message or actions is not None or policy_inference_seconds is not None
    ):
        raise ValueError("An error response needs a message and cannot contain actions.")
    arrays: dict[str, np.ndarray] = {
        "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        "status": np.asarray(status),
        "message": np.asarray(str(message)),
    }
    if actions is not None:
        arrays["actions"] = validate_action_chunk(actions)
        if not _is_sha256(input_image_sha256):
            raise ValueError("Action response requires a lowercase input_image_sha256.")
        arrays["input_image_sha256"] = np.asarray(input_image_sha256)
        if policy_inference_seconds is not None and (
            not math.isfinite(float(policy_inference_seconds))
            or float(policy_inference_seconds) <= 0
        ):
            raise ValueError("policy_inference_seconds must be finite and positive.")
        if policy_inference_seconds is not None:
            arrays["policy_inference_seconds"] = np.asarray(
                float(policy_inference_seconds), dtype=np.float64
            )
    elif policy_inference_seconds is not None:
        raise ValueError("Response without actions cannot contain policy timing.")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return _bounded_payload(buffer.getvalue(), "response")


def decode_response(payload: bytes) -> dict[str, Any]:
    """Decode and strictly validate one policy response."""
    _bounded_payload(payload, "response")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"protocol_version", "status", "message"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Response is missing fields: {sorted(missing)}.")
    version = _scalar(arrays["protocol_version"], "protocol_version", int)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: {version} != {PROTOCOL_VERSION}.")
    status = _scalar(arrays["status"], "status", str)
    message = _scalar(arrays["message"], "message", str)
    if status not in {"ok", "error"}:
        raise ValueError(f"Unsupported response status: {status!r}.")
    if status == "error":
        if not message or "actions" in arrays or "policy_inference_seconds" in arrays:
            raise ValueError("Malformed error response.")
        return {"status": status, "message": message}
    result: dict[str, Any] = {"status": status, "message": message}
    if "actions" in arrays:
        result["actions"] = validate_action_chunk(arrays["actions"])
        if "input_image_sha256" not in arrays:
            raise ValueError("Action response is missing input_image_sha256.")
        digest = _scalar(arrays["input_image_sha256"], "input_image_sha256", str)
        if not _is_sha256(digest):
            raise ValueError("Action response input_image_sha256 is malformed.")
        result["input_image_sha256"] = digest
        if "policy_inference_seconds" in arrays:
            seconds = _scalar(
                arrays["policy_inference_seconds"],
                "policy_inference_seconds",
                float,
            )
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Action response policy timing is malformed.")
            result["policy_inference_seconds"] = seconds
    elif "input_image_sha256" in arrays or "policy_inference_seconds" in arrays:
        raise ValueError("Response without actions cannot contain action metadata.")
    return result


def send_packet(connection: socket.socket, payload: bytes) -> None:
    payload = _bounded_payload(payload, "packet")
    connection.sendall(_HEADER.pack(len(payload)))
    connection.sendall(payload)


def receive_packet(connection: socket.socket) -> bytes | None:
    header = _receive_exact(connection, _HEADER.size, allow_eof=True)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size > MAX_PACKET_BYTES:
        raise ValueError(f"Packet is {size} bytes, limit is {MAX_PACKET_BYTES}.")
    return _receive_exact(connection, size, allow_eof=False)


def _opencv_resize_rgb(
    image: np.ndarray, output_hw: tuple[int, int], interpolation: str
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in simulator runtime
        raise RuntimeError("Exact GR1 evaluation resizing requires opencv-python.") from exc
    modes = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}
    if interpolation not in modes:
        raise ValueError(f"Unsupported resize interpolation: {interpolation!r}.")
    height, width = output_hw
    return cv2.resize(image, (width, height), interpolation=modes[interpolation])


def _validate_rgb(image: np.ndarray, shape: tuple[int, int, int], label: str) -> None:
    if image.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {image.shape}.")
    if image.dtype != np.uint8:
        raise TypeError(f"{label} must be uint8 RGB, got {image.dtype}.")


def _bounded_payload(payload: bytes, label: str) -> bytes:
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Encoded {label} is {len(payload)} bytes, limit is {MAX_PACKET_BYTES}.")
    return payload


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64 and value == value.lower() and all(c in "0123456789abcdef" for c in value)
    )


def _receive_exact(connection: socket.socket, size: int, *, allow_eof: bool) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            if allow_eof and not chunks:
                return None
            raise ConnectionError("Socket closed before the full packet arrived.")
        chunks.extend(chunk)
    return bytes(chunks)


def _scalar(array: np.ndarray, name: str, converter: type) -> Any:
    array = np.asarray(array)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}.")
    return converter(array.item())


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("!Q", len(value)))
    digest.update(value)
