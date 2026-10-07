# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa/contract.py
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

"""Pure NumPy contracts shared by the RoboCasa simulator and policy processes."""

from __future__ import annotations

import hashlib
import io
import re
import socket
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np


PROTOCOL_VERSION = 2
MAX_PACKET_BYTES = 16 * 1024 * 1024

CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
CAMERA_SHAPE = (256, 256, 3)

STATE_FIELDS = (
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.gripper_qpos", 2),
)
STATE_DIM = 16

ACTION_DIM = 12
ACTION_HORIZON = 32
PAST_OBS_SIZE = 48
ACTION_VIDEO_FREQ_RATIO = 4
SAMPLED_OBS_FRAMES = PAST_OBS_SIZE // ACTION_VIDEO_FREQ_RATIO + 1
RESPONSE_STATUSES = frozenset({"ok", "error"})

_HEADER = struct.Struct("!Q")
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_HASH_DOMAIN = b"long-wam-array-sha256-v1\0"
_CAMERA_HASH_DOMAIN = b"long-wam-camera-bundle-sha256-v1\0"


def validate_sha256(value: str, *, field_name: str = "sha256") -> str:
    """Return a canonical SHA-256 digest or reject non-lowercase encodings."""
    if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters.")
    return value


def array_sha256(array: Any) -> str:
    """Hash an array's exact contiguous CPU bytes together with dtype and shape.

    NumPy arrays and torch tensors are supported without importing torch into the
    simulator environment. Tensors are detached and copied to CPU before hashing.
    """
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


def camera_bundle_sha256(cameras: Mapping[str, np.ndarray]) -> str:
    """Hash raw HWC uint8 cameras in the semantic order defined by CAMERA_KEYS."""
    validated = extract_cameras(cameras)
    digest = hashlib.sha256()
    digest.update(_CAMERA_HASH_DOMAIN)
    for key in CAMERA_KEYS:
        _update_length_prefixed(digest, key.encode("utf-8"))
        _update_length_prefixed(digest, bytes.fromhex(array_sha256(validated[key])))
    return digest.hexdigest()


def extract_dataset_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Return the exact base-first 16D state stored in RoboCasa365 LeRobot."""
    fields = []
    for key, width in STATE_FIELDS:
        if key not in observation:
            raise KeyError(f"RoboCasa observation is missing {key!r}.")
        value = np.asarray(observation[key], dtype=np.float32)
        if value.shape != (width,):
            raise ValueError(
                f"RoboCasa observation {key!r} must have shape {(width,)}, got {value.shape}."
            )
        fields.append(value)
    state = np.concatenate(fields)
    if state.shape != (STATE_DIM,):
        raise AssertionError(f"Internal state contract error: {state.shape}.")
    return state


def extract_cameras(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy the three already-oriented uint8 Gym-wrapper camera observations."""
    cameras = {}
    for key in CAMERA_KEYS:
        if key not in observation:
            raise KeyError(f"RoboCasa observation is missing {key!r}.")
        image = np.asarray(observation[key])
        if image.shape != CAMERA_SHAPE:
            raise ValueError(
                f"RoboCasa camera {key!r} must have shape {CAMERA_SHAPE}, got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TypeError(f"RoboCasa camera {key!r} must be uint8, got {image.dtype}.")
        cameras[key] = np.ascontiguousarray(image)
    return cameras


def sanitize_dataset_action(action: np.ndarray) -> np.ndarray:
    """Clip continuous controls and quantize RoboCasa's two ±1 controls."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(
            f"RoboCasa dataset action must have shape {(ACTION_DIM,)}, got {action.shape}."
        )
    action = np.clip(action, -1.0, 1.0)
    action[4] = 1.0 if action[4] >= 0.0 else -1.0
    action[11] = 1.0 if action[11] >= 0.0 else -1.0
    return action


def dataset_action_to_canonical(action: np.ndarray) -> np.ndarray:
    """Map LeRobot order to the canonical order accepted by ``convert_action``.

    Dataset:
      ``[base(4), mode(1), eef_pos(3), eef_rot(3), gripper(1)]``

    Canonical:
      ``[eef_pos(3), eef_rot(3), gripper(1), base(4), mode(1)]``
    """
    action = sanitize_dataset_action(action)
    canonical = np.concatenate((action[5:8], action[8:11], action[11:12], action[0:4], action[4:5]))
    if canonical.shape != (ACTION_DIM,):
        raise AssertionError(f"Internal action contract error: {canonical.shape}.")
    return canonical


def encode_request(
    operation: str,
    *,
    episode_id: str = "",
    control_step: int = -1,
    cameras: Mapping[str, np.ndarray] | None = None,
    state: np.ndarray | None = None,
    prompt: str = "",
) -> bytes:
    """Encode one dependency-free, non-pickle AF_UNIX request."""
    if operation not in {"ping", "reset", "observe", "infer", "shutdown"}:
        raise ValueError(f"Unsupported operation: {operation!r}.")

    arrays: dict[str, np.ndarray] = {
        "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        "operation": np.asarray(operation),
        "episode_id": np.asarray(str(episode_id)),
        "control_step": np.asarray(int(control_step), dtype=np.int64),
        "prompt": np.asarray(str(prompt)),
    }
    if cameras is not None:
        validated = extract_cameras(cameras)
        arrays.update(
            left=validated[CAMERA_KEYS[0]],
            right=validated[CAMERA_KEYS[1]],
            wrist=validated[CAMERA_KEYS[2]],
        )
    if state is not None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"RoboCasa state must have shape {(STATE_DIM,)}, got {state.shape}.")
        arrays["state"] = state

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Encoded request is {len(payload)} bytes, limit is {MAX_PACKET_BYTES}.")
    return payload


def decode_request(payload: bytes) -> dict[str, Any]:
    """Decode and strictly validate one request produced by :func:`encode_request`."""
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Request is {len(payload)} bytes, limit is {MAX_PACKET_BYTES}.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        request = {key: archive[key] for key in archive.files}

    required = {
        "protocol_version",
        "operation",
        "episode_id",
        "control_step",
        "prompt",
    }
    missing = required - request.keys()
    if missing:
        raise ValueError(f"Request is missing fields: {sorted(missing)}.")

    version = _scalar(request["protocol_version"], "protocol_version", int)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: got {version}, expected {PROTOCOL_VERSION}.")
    operation = _scalar(request["operation"], "operation", str)
    if operation not in {"ping", "reset", "observe", "infer", "shutdown"}:
        raise ValueError(f"Unsupported operation: {operation!r}.")
    decoded: dict[str, Any] = {
        "operation": operation,
        "episode_id": _scalar(request["episode_id"], "episode_id", str),
        "control_step": _scalar(request["control_step"], "control_step", int),
        "prompt": _scalar(request["prompt"], "prompt", str),
    }

    if operation in {"observe", "infer"}:
        missing = {"left", "right", "wrist"} - request.keys()
        if missing:
            raise ValueError(f"Observation request is missing arrays: {sorted(missing)}.")
        cameras = {
            CAMERA_KEYS[0]: request["left"],
            CAMERA_KEYS[1]: request["right"],
            CAMERA_KEYS[2]: request["wrist"],
        }
        decoded["cameras"] = extract_cameras(cameras)
    if operation == "infer":
        if "state" not in request:
            raise ValueError("Inference request is missing state.")
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Inference state must have shape {(STATE_DIM,)}, got {state.shape}.")
        if not decoded["prompt"]:
            raise ValueError("Inference request prompt must be non-empty.")
        decoded["state"] = state
    return decoded


def encode_response(
    *,
    status: str = "ok",
    actions: np.ndarray | None = None,
    camera_bundle_sha256: str | None = None,
    model_mosaic_sha256: str | None = None,
    message: str = "",
) -> bytes:
    """Encode a small non-pickle policy response."""
    if status not in RESPONSE_STATUSES:
        raise ValueError(f"Unsupported response status: {status!r}.")
    if status == "error" and not message:
        raise ValueError("An error response must include a message.")
    if status == "error" and actions is not None:
        raise ValueError("An error response cannot include actions.")

    arrays: dict[str, np.ndarray] = {
        "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        "status": np.asarray(status),
        "message": np.asarray(str(message)),
    }
    if actions is not None:
        if camera_bundle_sha256 is None or model_mosaic_sha256 is None:
            raise ValueError(
                "An action response must include camera_bundle_sha256 and model_mosaic_sha256."
            )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"RoboCasa response actions must have shape [T, {ACTION_DIM}], got {actions.shape}."
            )
        if not 1 <= actions.shape[0] <= ACTION_HORIZON:
            raise ValueError(
                f"RoboCasa response must contain 1..{ACTION_HORIZON} actions, "
                f"got {actions.shape[0]}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("RoboCasa response actions contain non-finite values.")
        arrays["actions"] = np.ascontiguousarray(actions)
        arrays["camera_bundle_sha256"] = np.asarray(
            validate_sha256(
                camera_bundle_sha256,
                field_name="camera_bundle_sha256",
            )
        )
        arrays["model_mosaic_sha256"] = np.asarray(
            validate_sha256(
                model_mosaic_sha256,
                field_name="model_mosaic_sha256",
            )
        )
    elif camera_bundle_sha256 is not None or model_mosaic_sha256 is not None:
        raise ValueError("A response without actions cannot include input fingerprints.")

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Encoded response is {len(payload)} bytes, limit is {MAX_PACKET_BYTES}.")
    return payload


def decode_response(payload: bytes) -> dict[str, Any]:
    """Decode and strictly validate one response produced by :func:`encode_response`."""
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Response is {len(payload)} bytes, limit is {MAX_PACKET_BYTES}.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        response = {key: archive[key] for key in archive.files}

    required = {"protocol_version", "status", "message"}
    missing = required - response.keys()
    if missing:
        raise ValueError(f"Response is missing fields: {sorted(missing)}.")
    version = _scalar(response["protocol_version"], "protocol_version", int)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: got {version}, expected {PROTOCOL_VERSION}.")
    status = _scalar(response["status"], "status", str)
    if status not in RESPONSE_STATUSES:
        raise ValueError(f"Unsupported response status: {status!r}.")
    message = _scalar(response["message"], "message", str)
    if status == "error" and not message:
        raise ValueError("An error response must include a message.")

    decoded: dict[str, Any] = {"status": status, "message": message}
    has_actions = "actions" in response
    fingerprint_fields = {
        "camera_bundle_sha256",
        "model_mosaic_sha256",
    }
    present_fingerprints = fingerprint_fields & response.keys()
    if has_actions and present_fingerprints != fingerprint_fields:
        missing = sorted(fingerprint_fields - present_fingerprints)
        raise ValueError(f"Action response is missing input fingerprints: {missing}.")
    if not has_actions and present_fingerprints:
        raise ValueError("A response without actions cannot include input fingerprints.")
    if has_actions:
        if status != "ok":
            raise ValueError("An error response cannot include actions.")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"RoboCasa response actions must have shape [T, {ACTION_DIM}], got {actions.shape}."
            )
        if not 1 <= actions.shape[0] <= ACTION_HORIZON:
            raise ValueError(
                f"RoboCasa response must contain 1..{ACTION_HORIZON} actions, "
                f"got {actions.shape[0]}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("RoboCasa response actions contain non-finite values.")
        decoded["actions"] = np.ascontiguousarray(actions)
        for field_name in sorted(fingerprint_fields):
            decoded[field_name] = validate_sha256(
                _scalar(response[field_name], field_name, str),
                field_name=field_name,
            )
    return decoded


def send_packet(connection: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Packet exceeds {MAX_PACKET_BYTES} bytes.")
    connection.sendall(_HEADER.pack(len(payload)))
    connection.sendall(payload)


def receive_packet(connection: socket.socket) -> bytes | None:
    """Receive one framed packet, returning ``None`` only on a clean EOF."""
    header = _receive_exact(connection, _HEADER.size, allow_eof=True)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size > MAX_PACKET_BYTES:
        raise ValueError(f"Incoming packet declares {size} bytes.")
    payload = _receive_exact(connection, size, allow_eof=False)
    assert payload is not None
    return payload


def _receive_exact(
    connection: socket.socket,
    size: int,
    *,
    allow_eof: bool,
) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            if allow_eof and not chunks:
                return None
            raise ConnectionError(f"Socket closed after {len(chunks)}/{size} expected bytes.")
        chunks.extend(chunk)
    return bytes(chunks)


def _scalar(array: np.ndarray, name: str, converter: type) -> Any:
    array = np.asarray(array)
    if array.shape != ():
        raise ValueError(f"{name!r} must be a scalar, got shape {array.shape}.")
    try:
        return converter(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid scalar field {name!r}.") from exc


def _update_length_prefixed(
    digest: Any,
    value: bytes,
) -> None:
    digest.update(struct.pack("!Q", len(value)))
    digest.update(value)
