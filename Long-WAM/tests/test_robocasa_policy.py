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

import numpy as np
import pytest
import torch

from longwam.benchmarks.robocasa.contract import (
    ACTION_DIM,
    CAMERA_KEYS,
    CAMERA_SHAPE,
    array_sha256,
    camera_bundle_sha256,
    decode_response,
)
from longwam.benchmarks.robocasa.policy import (
    PolicyInference,
    PolicyRequestDispatcher,
    RawFrameHistory,
    RequestSequence,
    RoboCasaLongWAMPolicy,
    cameras_to_model_frame,
)
from longwam.datasets.lerobot.robocasa_contract import (
    ROBOCASA_LATENT_SLOT_LAYOUT,
)


def _cameras() -> dict[str, np.ndarray]:
    return {
        key: np.full(CAMERA_SHAPE, value, dtype=np.uint8) for value, key in enumerate(CAMERA_KEYS)
    }


def test_history_samples_raw_offsets_zero_through_48_by_four():
    history = RawFrameHistory(frame_shape=(3, 2, 2))
    for value in range(49):
        history.append(torch.full((3, 2, 2), float(value)))

    window = history.sampled_window()
    assert window.shape == (1, 3, 13, 2, 2)
    assert window[0, 0, :, 0, 0].tolist() == list(range(0, 49, 4))


def test_history_repeats_oldest_frame_at_episode_start_and_stays_bounded():
    history = RawFrameHistory(frame_shape=(3, 1, 1))
    history.append(torch.full((3, 1, 1), 7.0))
    history.append(torch.full((3, 1, 1), 8.0))
    sampled = history.sampled_window()[0, 0, :, 0, 0].tolist()
    assert sampled == [7.0] * 12 + [8.0]

    for value in range(100):
        history.append(torch.full((3, 1, 1), float(value)))
    assert len(history) == 49
    assert history.sampled_window()[0, 0, :, 0, 0].tolist() == list(map(float, range(51, 100, 4)))


def test_latent_slot_history_inserts_time_after_view_and_channel_axes():
    history = RawFrameHistory(frame_shape=(3, 3, 2, 2))
    for value in range(49):
        history.append(torch.full((3, 3, 2, 2), float(value)))

    window = history.sampled_window()

    assert window.shape == (1, 3, 3, 13, 2, 2)
    assert window[0, 2, 1, :, 0, 0].tolist() == list(range(0, 49, 4))


def test_request_sequence_rejects_missing_duplicate_skipped_and_wrong_episode():
    sequence = RequestSequence()
    with pytest.raises(RuntimeError, match="reset"):
        sequence.validate_observation(episode_id="ep-1", control_step=0)

    sequence.reset("ep-1")
    sequence.commit_observation(episode_id="ep-1", control_step=0)
    with pytest.raises(ValueError, match="Expected control_step=1"):
        sequence.validate_observation(episode_id="ep-1", control_step=0)
    with pytest.raises(ValueError, match="Expected control_step=1"):
        sequence.validate_observation(episode_id="ep-1", control_step=2)
    with pytest.raises(ValueError, match="does not match"):
        sequence.validate_observation(episode_id="ep-2", control_step=1)

    sequence.commit_observation(episode_id="ep-1", control_step=1)
    sequence.reset("ep-2")
    assert sequence.episode_id == "ep-2"
    assert sequence.next_control_step == 0


class _FakePolicy:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def reset(self) -> None:
        self.events.append(("reset", None))

    def observe(self, cameras) -> str:
        self.events.append(("observe", cameras))
        return "3" * 64

    def infer(self, *, cameras, state, instruction) -> PolicyInference:
        self.events.append(("infer", instruction))
        return PolicyInference(
            actions=np.zeros((32, ACTION_DIM), dtype=np.float32),
            model_mosaic_sha256="4" * 64,
        )


def test_dispatcher_commits_only_valid_monotonic_observations():
    policy = _FakePolicy()
    dispatcher = PolicyRequestDispatcher(policy)  # type: ignore[arg-type]

    response, stop = dispatcher.dispatch({"operation": "reset", "episode_id": "ep-1"})
    assert not stop
    assert decode_response(response)["status"] == "ok"

    base = {
        "episode_id": "ep-1",
        "cameras": _cameras(),
    }
    response, _ = dispatcher.dispatch({**base, "operation": "observe", "control_step": 0})
    assert decode_response(response)["status"] == "ok"
    response, _ = dispatcher.dispatch(
        {
            **base,
            "operation": "infer",
            "control_step": 1,
            "state": np.zeros(16, dtype=np.float32),
            "prompt": "close the cabinet",
        }
    )
    decoded = decode_response(response)
    assert decoded["status"] == "ok"
    assert decoded["actions"].shape == (32, ACTION_DIM)
    assert decoded["camera_bundle_sha256"] == camera_bundle_sha256(base["cameras"])
    assert decoded["model_mosaic_sha256"] == "4" * 64

    response, _ = dispatcher.dispatch({**base, "operation": "observe", "control_step": 1})
    assert decode_response(response)["status"] == "error"
    assert dispatcher.sequence.next_control_step == 2
    assert [event[0] for event in policy.events] == ["reset", "observe", "infer"]


class _ActionNormalizer:
    def backward(self, action: torch.Tensor) -> torch.Tensor:
        return action + 3.0


class _Normalizer:
    def __init__(self) -> None:
        self.normalizers = {"action": {"default": _ActionNormalizer()}}

    def forward(self, batch):
        batch["state"]["default"] *= 2.0
        return batch


class _Processor:
    shape_meta = {
        "state": [{"key": "default"}],
        "action": [{"key": "default"}],
    }

    def __init__(self) -> None:
        self.normalizer = _Normalizer()

    def action_state_transform(self, batch):
        return batch


class _Model:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def infer_joint_ar(self, **kwargs):
        self.kwargs = kwargs
        return {"action": torch.zeros((32, ACTION_DIM))}


class _FailingModel(_Model):
    def infer_joint_ar(self, **kwargs):
        raise RuntimeError("synthetic inference failure")


def test_policy_uses_exact_p4_infer_and_normalization_contract():
    model = _Model()
    policy = RoboCasaLongWAMPolicy(
        model=model,  # type: ignore[arg-type]
        processor=_Processor(),
        prompt_template="Instruction: {task}",
        policy_seed=17,
    )
    inference = policy.infer(
        cameras=_cameras(),
        state=np.ones(16, dtype=np.float32),
        instruction="close the cabinet",
    )

    assert inference.actions.shape == (32, ACTION_DIM)
    np.testing.assert_array_equal(
        inference.actions,
        np.full_like(inference.actions, 3.0),
    )
    assert inference.model_mosaic_sha256 == array_sha256(policy.history.current)
    assert model.kwargs["prompt"] == "Instruction: close the cabinet"
    assert model.kwargs["input_image"].shape == (1, 3, 384, 320)
    assert model.kwargs["obs_window"].shape == (1, 3, 13, 384, 320)
    assert model.kwargs["action_horizon"] == 32
    assert model.kwargs["num_inference_steps"] == 10
    assert model.kwargs["rand_device"] == "cpu"
    assert model.kwargs["seed"] == 17
    np.testing.assert_array_equal(
        model.kwargs["proprio"].numpy(),
        np.full((1, 16), 2.0, dtype=np.float32),
    )


def test_policy_latent_slots_use_native_three_view_current_and_history():
    model = _Model()
    policy = RoboCasaLongWAMPolicy(
        model=model,  # type: ignore[arg-type]
        processor=_Processor(),
        prompt_template="Instruction: {task}",
        policy_seed=17,
        concat_multi_camera=ROBOCASA_LATENT_SLOT_LAYOUT,
    )

    policy.infer(
        cameras=_cameras(),
        state=np.ones(16, dtype=np.float32),
        instruction="close the cabinet",
    )

    assert model.kwargs["input_image"].shape == (1, 3, 3, 256, 256)
    assert model.kwargs["obs_window"].shape == (1, 3, 3, 13, 256, 256)
    assert policy.history.current.shape == (3, 3, 256, 256)


def test_failed_inference_rolls_back_uncommitted_history():
    policy = RoboCasaLongWAMPolicy(
        model=_FailingModel(),  # type: ignore[arg-type]
        processor=_Processor(),
        prompt_template="Instruction: {task}",
        policy_seed=17,
    )
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        policy.infer(
            cameras=_cameras(),
            state=np.ones(16, dtype=np.float32),
            instruction="close the cabinet",
        )
    assert len(policy.history) == 0


def test_observe_hashes_the_exact_contiguous_float32_history_frame():
    policy = RoboCasaLongWAMPolicy(
        model=_Model(),  # type: ignore[arg-type]
        processor=_Processor(),
        prompt_template="Instruction: {task}",
        policy_seed=17,
    )
    cameras = _cameras()
    digest = policy.observe(cameras)

    assert policy.history.current.dtype == torch.float32
    assert policy.history.current.device.type == "cpu"
    assert policy.history.current.is_contiguous()
    assert digest == array_sha256(policy.history.current)
    assert digest == array_sha256(policy.history.current.numpy())
    assert digest == array_sha256(cameras_to_model_frame(cameras))


def test_model_mosaic_hash_changes_when_left_and_right_are_swapped():
    cameras = _cameras()
    swapped = dict(cameras)
    swapped[CAMERA_KEYS[0]], swapped[CAMERA_KEYS[1]] = (
        swapped[CAMERA_KEYS[1]],
        swapped[CAMERA_KEYS[0]],
    )
    assert array_sha256(cameras_to_model_frame(cameras)) != array_sha256(
        cameras_to_model_frame(swapped)
    )


def test_latent_slot_frame_preserves_every_raw_pixel_in_fixed_camera_order():
    row, column = np.indices(CAMERA_SHAPE[:2], dtype=np.uint16)
    cameras = {
        key: np.stack(
            (
                (row + 17 * index) % 256,
                (column + 31 * index) % 256,
                (row + column + 47 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        for index, key in enumerate(CAMERA_KEYS)
    }

    actual = cameras_to_model_frame(
        cameras,
        concat_multi_camera=ROBOCASA_LATENT_SLOT_LAYOUT,
    )
    expected = torch.stack(
        [
            torch.from_numpy(cameras[key])
            .permute(2, 0, 1)
            .to(torch.float32)
            .div(255.0)
            .mul(2.0)
            .sub(1.0)
            for key in CAMERA_KEYS
        ]
    )

    assert actual.shape == (3, 3, 256, 256)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        cameras_to_model_frame(cameras),
        cameras_to_model_frame(
            cameras,
            concat_multi_camera="robocasa",
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_latent_slot_frame_fails_fast_on_missing_or_missized_camera():
    cameras = _cameras()
    missing = dict(cameras)
    del missing[CAMERA_KEYS[1]]
    with pytest.raises(KeyError, match=CAMERA_KEYS[1]):
        cameras_to_model_frame(
            missing,
            concat_multi_camera=ROBOCASA_LATENT_SLOT_LAYOUT,
        )

    missized = dict(cameras)
    missized[CAMERA_KEYS[2]] = np.zeros((255, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have shape"):
        cameras_to_model_frame(
            missized,
            concat_multi_camera=ROBOCASA_LATENT_SLOT_LAYOUT,
        )


def test_failed_inference_restores_a_full_history_including_evicted_frame():
    policy = RoboCasaLongWAMPolicy(
        model=_FailingModel(),  # type: ignore[arg-type]
        processor=_Processor(),
        prompt_template="Instruction: {task}",
        policy_seed=17,
    )
    for value in range(49):
        policy.history.append(torch.full((3, 384, 320), float(value)))
    before = policy.history.snapshot()

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        policy.infer(
            cameras=_cameras(),
            state=np.ones(16, dtype=np.float32),
            instruction="close the cabinet",
        )

    after = policy.history.snapshot()
    assert len(after) == len(before) == 49
    assert all(actual is expected for actual, expected in zip(after, before))
