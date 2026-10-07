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

import io
import socket

import numpy as np
import pytest

from longwam.benchmarks.robocasa_gr1.contract import (
    ACTION_DIM,
    ACTION_FIELDS,
    ACTION_HORIZON,
    EGO_VIEW_KEY,
    HISTORY_OFFSETS,
    MODEL_EGO_SHAPE,
    MUJOCO_EGO_SHAPE,
    OFFICIAL_EGO_SHAPE,
    PROTOCOL_VERSION,
    STATE_FIELDS,
    TASK_SPECS,
    action_row_to_env,
    array_sha256,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    ensure_unlocked_waist_prompt,
    extract_official_ego256,
    extract_raw_state,
    mujoco_rgb_to_official_ego256,
    official_ego256_to_model224,
    receive_packet,
    send_packet,
)


EXPECTED_MAPPING = (
    ("CupToDrawer", "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PotatoToMicrowave", "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PlaceMilkToMicrowave", "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"),
    ("PlaceBottleToCabinet", "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"),
    ("WineToCabinet", "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"),
    ("CanToDrawer", "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"),
    (
        "CuttingboardToBasket",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "CuttingboardToCardboardBox",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "CuttingboardToPan",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "CuttingboardToPot",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "CuttingboardToTieredBasket",
        "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlacematToBasket",
        "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlacematToBowl",
        "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlacematToPlate",
        "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlacematToTieredShelf",
        "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlateToBowl",
        "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlateToCardboardBox",
        "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlateToPan",
        "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "PlateToPlate",
        "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "TrayToCardboardBox",
        "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "TrayToPlate",
        "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "TrayToPot",
        "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "TrayToTieredBasket",
        "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
    (
        "TrayToTieredShelf",
        "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    ),
)


def _observation() -> dict[str, np.ndarray]:
    observation = {}
    offset = 0
    for key, width in STATE_FIELDS:
        observation[key] = np.arange(offset, offset + width, dtype=np.float32)
        offset += width
    observation[EGO_VIEW_KEY] = np.zeros(OFFICIAL_EGO_SHAPE, dtype=np.uint8)
    return observation


def test_task_registry_freezes_all_24_exact_official_mappings():
    assert tuple((spec.dataset_suffix, spec.gym_id) for spec in TASK_SPECS) == EXPECTED_MAPPING
    assert all(spec.dataset_name == f"gr1_arms_waist.{spec.dataset_suffix}" for spec in TASK_SPECS)
    assert sum(spec.articulated_close for spec in TASK_SPECS) == 6


def test_state_and_action_field_orders_are_exact():
    state = extract_raw_state(_observation())
    np.testing.assert_array_equal(state, np.arange(29, dtype=np.float32))

    action = action_row_to_env(np.arange(ACTION_DIM, dtype=np.float32))
    assert tuple(action) == tuple(key for key, _ in ACTION_FIELDS)
    offset = 0
    for key, width in ACTION_FIELDS:
        np.testing.assert_array_equal(action[key], np.arange(offset, offset + width))
        offset += width


def test_prompt_has_exactly_one_unlocked_waist_prefix():
    expected = "unlocked_waist: put the cup in the drawer"
    assert ensure_unlocked_waist_prompt(expected) == expected
    with pytest.raises(ValueError, match="must start"):
        ensure_unlocked_waist_prompt("put the cup in the drawer")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        ensure_unlocked_waist_prompt("unlocked_waist:   put the cup in the drawer")
    with pytest.raises(ValueError, match="exactly once"):
        ensure_unlocked_waist_prompt(f"unlocked_waist: {expected}")
    with pytest.raises(ValueError, match="locked-waist"):
        ensure_unlocked_waist_prompt("locked_waist: put the cup away")
    with pytest.raises(ValueError, match="empty"):
        ensure_unlocked_waist_prompt("unlocked_waist:")


def test_raw_rgb_pipeline_flips_then_crops_pads_and_never_swaps_channels():
    rows, cols = np.indices(MUJOCO_EGO_SHAPE[:2], dtype=np.uint16)
    raw = np.stack((rows % 251, cols % 253, (rows + 2 * cols) % 255), axis=-1).astype(np.uint8)
    calls: list[tuple[np.ndarray, tuple[int, int], str]] = []

    def resize(image: np.ndarray, output_hw: tuple[int, int], interpolation: str) -> np.ndarray:
        calls.append((image.copy(), output_hw, interpolation))
        return np.full((*output_hw, 3), 17 + len(calls), dtype=np.uint8)

    output = mujoco_rgb_to_official_ego256(raw, resize=resize)

    assert output.shape == OFFICIAL_EGO_SHAPE
    np.testing.assert_array_equal(calls[0][0], raw[::-1, :, :][310:770, 110:1130, :])
    assert calls[0][1:] == ((480, 720), "area")
    padded = calls[1][0]
    assert padded.shape == (720, 720, 3)
    assert not padded[:120].any() and not padded[-120:].any()
    assert np.all(padded[120:600] == 18)
    assert calls[1][1:] == ((256, 256), "area")


def test_default_raw_hotfix_is_pixel_identical_to_official_opencv_sequence():
    cv2 = pytest.importorskip("cv2")
    rows, cols = np.indices(MUJOCO_EGO_SHAPE[:2], dtype=np.uint16)
    raw = np.stack((rows % 251, cols % 253, (rows + cols) % 255), axis=-1).astype(np.uint8)
    corrected = raw[::-1, :, :]
    expected = corrected[310:770, 110:1130, :]
    expected = cv2.resize(expected, (720, 480), interpolation=cv2.INTER_AREA)
    expected = np.pad(expected, ((120, 120), (0, 0), (0, 0)), mode="constant")
    expected = cv2.resize(expected, (256, 256), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(mujoco_rgb_to_official_ego256(raw), expected)


def test_model_transform_center_crops_243_then_linearly_resizes_224():
    image = np.arange(np.prod(OFFICIAL_EGO_SHAPE), dtype=np.uint32).reshape(OFFICIAL_EGO_SHAPE)
    image = (image % 256).astype(np.uint8)
    captured = {}

    def resize(crop: np.ndarray, output_hw: tuple[int, int], interpolation: str) -> np.ndarray:
        captured["crop"] = crop.copy()
        captured["output_hw"] = output_hw
        captured["interpolation"] = interpolation
        return np.zeros((*output_hw, 3), dtype=np.uint8)

    output = official_ego256_to_model224(image, resize=resize)
    np.testing.assert_array_equal(captured["crop"], image[6:249, 6:249, :])
    assert captured["output_hw"] == (224, 224)
    assert captured["interpolation"] == "linear"
    assert output.shape == MODEL_EGO_SHAPE
    np.testing.assert_array_equal(extract_official_ego256({EGO_VIEW_KEY: image}), image)


def test_protocol_round_trip_requires_one_complete_h16_chunk():
    observation = _observation()
    image = observation[EGO_VIEW_KEY]
    request = decode_request(
        encode_request(
            "infer",
            episode_id="CupToDrawer:0",
            control_step=16,
            image=image,
            state=extract_raw_state(observation),
            prompt="unlocked_waist: put the cup away",
        )
    )
    assert request["prompt"] == "unlocked_waist: put the cup away"
    assert request["control_step"] == 16
    np.testing.assert_array_equal(request["state"], np.arange(29, dtype=np.float32))

    actions = np.arange(ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(
        ACTION_HORIZON, ACTION_DIM
    )
    response = decode_response(
        encode_response(
            actions=actions,
            input_image_sha256=array_sha256(image),
            policy_inference_seconds=0.125,
        )
    )
    np.testing.assert_array_equal(response["actions"], actions)
    assert response["policy_inference_seconds"] == pytest.approx(0.125)
    with pytest.raises(ValueError, match="must have shape"):
        encode_response(actions=actions[:-1], input_image_sha256=array_sha256(image))
    with pytest.raises(ValueError, match="finite and positive"):
        encode_response(
            actions=actions,
            input_image_sha256=array_sha256(image),
            policy_inference_seconds=float("nan"),
        )


def test_protocol_rejects_version_mismatch_and_transport_round_trips():
    payload = encode_request("ping")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["protocol_version"] = np.asarray(PROTOCOL_VERSION + 1)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with pytest.raises(ValueError, match="Protocol version mismatch"):
        decode_request(buffer.getvalue())

    sender, receiver = socket.socketpair()
    try:
        send_packet(sender, payload)
        assert receive_packet(receiver) == payload
    finally:
        sender.close()
        receiver.close()


def test_history_offsets_are_exactly_minus48_through_zero_by_four():
    assert HISTORY_OFFSETS == tuple(range(-48, 1, 4))
