# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/tests/test_robocasa_data_contract.py
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

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import longwam.datasets.lerobot.robocasa_video_dataset as robocasa_video_dataset
from longwam.benchmarks.robocasa.contract import (
    CAMERA_KEYS as EVAL_CAMERA_KEYS,
    decode_request,
    encode_request,
)
from longwam.benchmarks.robocasa.policy import cameras_to_model_frame
from longwam.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from longwam.datasets.lerobot.episode_selection import (
    ResolvedPerDatasetEpisodeSelection,
)
from longwam.datasets.lerobot.processors.longwam_processor import LongWAMProcessor
from longwam.datasets.lerobot.robocasa_contract import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_CAMERA_KEYS,
    ROBOCASA_DELTA_ACTION_DIM_MASK,
    ROBOCASA_FPS,
    ROBOCASA_STATE_DIM,
    build_robocasa_camera_mosaic,
    discover_human300_dataset_dirs,
    validate_human300_dataset_inventory,
    validate_robocasa_lerobot_contract,
)
from longwam.datasets.lerobot.robocasa_video_dataset import (
    RoboCasaHuman300VideoDataset,
)
from longwam.datasets.lerobot.robot_video_dataset import (
    DEFAULT_PROMPT,
    RobotVideoDataset,
)
from longwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from longwam.utils.config_resolvers import register_default_resolvers


ROOT = Path(__file__).resolve().parents[1]


def _metadata() -> dict:
    features = {
        f"observation.images.{key}": {"shape": [256, 256, 3]} for key in ROBOCASA_CAMERA_KEYS
    }
    features.update(
        {
            "observation.state": {"shape": [ROBOCASA_STATE_DIM]},
            "action": {"shape": [ROBOCASA_ACTION_DIM]},
        }
    )
    return {"fps": ROBOCASA_FPS, "features": features}


def _write_dataset(root: Path, group: str, task: str) -> Path:
    dataset = root / group / task / "20260101" / "lerobot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    return dataset


def test_human300_discovery_is_sorted_and_rejects_partial_data(tmp_path):
    composite = _write_dataset(tmp_path, "composite", "TaskB")
    atomic = _write_dataset(tmp_path, "atomic", "TaskA")
    paths = discover_human300_dataset_dirs(
        tmp_path, expected_group_counts={"atomic": 1, "composite": 1}
    )
    assert paths == (atomic.resolve(), composite.resolve())

    with pytest.raises(ValueError, match="expected group counts"):
        discover_human300_dataset_dirs(
            tmp_path, expected_group_counts={"atomic": 2, "composite": 1}
        )


def test_human300_inventory_rejects_duplicate_paths(tmp_path):
    dataset = _write_dataset(tmp_path, "atomic", "TaskA")
    with pytest.raises(ValueError, match="duplicates"):
        validate_human300_dataset_inventory([dataset, dataset])


def test_robocasa_metadata_contract_checks_frequency_and_shapes(tmp_path):
    dataset = _write_dataset(tmp_path, "atomic", "TaskA")
    validate_robocasa_lerobot_contract([dataset])

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["action"]["shape"] = [11]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="action shape"):
        validate_robocasa_lerobot_contract([dataset])


def test_robocasa_mosaic_layout_and_equal_source_area():
    cameras = torch.stack([torch.full((2, 3, 8, 8), value) for value in (0.1, 0.5, 0.9)])
    mosaic = build_robocasa_camera_mosaic(cameras)

    assert mosaic.shape == (2, 3, 384, 320)
    assert torch.allclose(mosaic[..., :256, :160], torch.tensor(0.1))
    assert torch.allclose(mosaic[..., :256, 160:], torch.tensor(0.5))
    assert torch.allclose(mosaic[..., 256:, :], torch.tensor(0.9))
    assert 256 * 160 == 256 * 160 == 128 * 320

    with pytest.raises(ValueError, match="requires 3 cameras"):
        build_robocasa_camera_mosaic(cameras[:2])


class _IdentityNormalizer:
    @staticmethod
    def forward(batch):
        return batch


def test_padded_actions_zero_only_continuous_delta_dimensions():
    shape_meta = OmegaConf.create(
        {
            "images": [{"key": "image", "shape": [3, 2, 2]}],
            "action": [{"key": "default", "raw_shape": 12, "shape": 12}],
            "state": [{"key": "default", "raw_shape": 16, "shape": 16}],
        }
    )
    processor = LongWAMProcessor(
        shape_meta=shape_meta,
        num_obs_steps=2,
        num_image_steps=2,
        num_output_cameras=1,
        action_output_dim=12,
        proprio_output_dim=16,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
        delta_action_dim_mask={"default": list(ROBOCASA_DELTA_ACTION_DIM_MASK)},
    )
    processor.train()
    processor._normalizer = _IdentityNormalizer()
    data = {
        "task": "test task",
        "images": {"image": torch.zeros(2, 3, 2, 2)},
        "image_is_pad": torch.tensor([False, False]),
        "action": {"default": torch.tensor([[1.0] * 12, [7.0] * 12])},
        "action_is_pad": torch.tensor([False, True]),
        "state": {"default": torch.zeros(2, 16)},
        "state_is_pad": torch.tensor([False, False]),
        "idx": 0,
    }

    sample = processor.preprocess(data)
    expected_padded = torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0])
    assert torch.equal(sample["action"][0], torch.ones(12))
    assert torch.equal(sample["action"][1], expected_padded)


class _FakeProcessedDataset:
    def __init__(self, sample):
        self.sample = sample

    def __getitem__(self, _index):
        return self.sample

    def __len__(self):
        return 1


class _Identity:
    @staticmethod
    def __call__(value):
        return value


def _directional_camera(camera_index: int) -> np.ndarray:
    """Non-symmetric RGB image that exposes swaps and either image flip."""
    row, column = np.indices((256, 256), dtype=np.uint16)
    return np.stack(
        (
            (row + 37 * camera_index) % 256,
            (3 * column + 53 * camera_index) % 256,
            (2 * row + column + 71 * camera_index) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_robot_video_dataset_robocasa_sample_contract():
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    pixel_values = torch.stack([torch.full((21, 3, 8, 8), value) for value in (0.1, 0.5, 0.9)])
    dataset.lerobot_dataset = _FakeProcessedDataset(
        {
            "pixel_values": pixel_values,
            "image_is_pad": torch.zeros(21, dtype=torch.bool),
            "action": torch.zeros(32, 12),
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "proprio": torch.zeros(81, 16),
            "proprio_is_pad": torch.zeros(81, dtype=torch.bool),
            "instruction": "test task",
        }
    )
    dataset.num_frames = 81
    dataset.past_obs_size = 48
    dataset.action_chunk = 32
    dataset.video_sample_indices = list(range(0, 81, 4))
    dataset.decode_only_sampled_video_frames = True
    dataset.skip_padding_as_possible = False
    dataset.max_padding_retry = 0
    dataset.concat_multi_camera = "robocasa"
    dataset.resize_transform = _Identity()
    dataset.crop_transform = _Identity()
    dataset.normalize_transform = _Identity()
    dataset.override_instruction = None
    dataset._get_cached_text_context = lambda _prompt: (
        torch.zeros(128, 4),
        torch.ones(128, dtype=torch.bool),
    )

    sample = dataset._get(0)
    assert sample["video"].shape == (3, 21, 384, 320)
    assert sample["action"].shape == (32, 12)
    assert sample["proprio"].shape == (80, 16)
    assert torch.allclose(sample["video"][..., :256, :160], torch.tensor(0.1))
    assert torch.allclose(sample["video"][..., :256, 160:], torch.tensor(0.5))
    assert torch.allclose(sample["video"][..., 256:, :], torch.tensor(0.9))


def test_robocasa_train_eval_pixels_camera_order_and_prompt_are_equivalent():
    raw_cameras = {
        eval_key: _directional_camera(index) for index, eval_key in enumerate(EVAL_CAMERA_KEYS)
    }
    raw_prompt = "Close the lid blender by securely placing the lid on top."
    decoded = decode_request(
        encode_request(
            "infer",
            episode_id="pixel-contract",
            control_step=0,
            cameras=raw_cameras,
            state=np.zeros(ROBOCASA_STATE_DIM, dtype=np.float32),
            prompt=raw_prompt,
        )
    )
    eval_frame = cameras_to_model_frame(decoded["cameras"])

    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    camera_video = torch.stack(
        [
            torch.from_numpy(raw_cameras[eval_key])
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div(255.0)
            .unsqueeze(0)
            .expand(21, -1, -1, -1)
            for eval_key in EVAL_CAMERA_KEYS
        ]
    )
    dataset.lerobot_dataset = _FakeProcessedDataset(
        {
            "pixel_values": camera_video,
            "image_is_pad": torch.zeros(21, dtype=torch.bool),
            "action": torch.zeros(32, ROBOCASA_ACTION_DIM),
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "proprio": torch.zeros(81, ROBOCASA_STATE_DIM),
            "proprio_is_pad": torch.zeros(81, dtype=torch.bool),
            "instruction": raw_prompt,
        }
    )
    dataset.num_frames = 81
    dataset.past_obs_size = 48
    dataset.action_chunk = 32
    dataset.video_sample_indices = list(range(0, 81, 4))
    dataset.decode_only_sampled_video_frames = True
    dataset.skip_padding_as_possible = False
    dataset.max_padding_retry = 0
    dataset.concat_multi_camera = "robocasa"
    dataset.resize_transform = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
    dataset.crop_transform = CenterCrop(args={"img_w": 320, "img_h": 384})
    dataset.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
    dataset.override_instruction = None
    dataset._get_cached_text_context = lambda _prompt: (
        torch.zeros(128, 4),
        torch.ones(128, dtype=torch.bool),
    )

    train_sample = dataset._get(0)

    torch.testing.assert_close(
        train_sample["video"][:, 0],
        eval_frame,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert ROBOCASA_CAMERA_KEYS == tuple(key.removeprefix("video.") for key in EVAL_CAMERA_KEYS)
    assert decoded["prompt"] == raw_prompt
    assert train_sample["prompt"] == DEFAULT_PROMPT.format(task=decoded["prompt"])


def test_robocasa_task_config_resolves_exact_human300_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("LONGWAM_DATA_ROOT", str(tmp_path))
    register_default_resolvers()
    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=["task=robocasa365"],
        )

    data = cfg.data.train
    assert len(data.dataset_dirs) == 300
    assert len(set(data.dataset_dirs)) == 300
    assert all(str(path).startswith(str(tmp_path)) for path in data.dataset_dirs)
    assert all(Path(path).name == "lerobot" for path in data.dataset_dirs)
    assert sum("/atomic/" in path for path in data.dataset_dirs) == 65
    assert sum("/composite/" in path for path in data.dataset_dirs) == 235
    assert data.expected_fps == 20
    assert data.human300_episode_selection.strategy == "official_random_subset"
    assert data.human300_episode_selection.demos_per_task == 100
    assert data.human300_episode_selection.seed == 0
    assert data.human300_episode_selection.expected_task_count == 300
    assert data.human300_episode_selection.expected_episode_count == 30_000
    assert data.num_frames == 81
    assert data.past_obs_size == 48
    assert data.action_chunk == 32
    assert data.action_video_freq_ratio == 4
    assert data.processor.num_image_steps == 21
    assert data.concat_multi_camera == "robocasa_left_main"
    assert tuple(item.key for item in data.shape_meta.images) == ROBOCASA_CAMERA_KEYS
    assert data.shape_meta.action[0].shape == 12
    assert data.shape_meta.state[0].shape == 16
    assert tuple(data.processor.delta_action_dim_mask.default) == (ROBOCASA_DELTA_ACTION_DIM_MASK)
    assert cfg.model.proprio_dim == 16
    assert cfg.model.action_dit_config.action_dim == 12


def test_human300_dataset_fails_before_loading_when_path_count_is_wrong():
    with pytest.raises(ValueError, match="requires 300 dataset paths"):
        RoboCasaHuman300VideoDataset(dataset_dirs=[])


def test_human300_adapter_preserves_loader_path_spelling(monkeypatch):
    configured_paths = tuple(f"/loader-alias/task-{index}" for index in range(300))
    validated_paths = tuple(Path(f"/canonical/task-{index}") for index in range(300))
    captured = {}

    monkeypatch.setattr(
        robocasa_video_dataset,
        "validate_human300_dataset_inventory",
        lambda paths: validated_paths,
    )
    monkeypatch.setattr(
        robocasa_video_dataset,
        "validate_robocasa_lerobot_contract",
        lambda paths, expected_fps: captured.setdefault("validated", (tuple(paths), expected_fps)),
    )
    monkeypatch.setattr(
        RobotVideoDataset,
        "__init__",
        lambda self, dataset_dirs, **kwargs: captured.setdefault("loaded", tuple(dataset_dirs)),
    )

    RoboCasaHuman300VideoDataset(dataset_dirs=configured_paths)

    assert captured["validated"] == (validated_paths, ROBOCASA_FPS)
    assert captured["loaded"] == configured_paths


def test_human300_adapter_passes_exact_official_episode_allowlists(monkeypatch):
    configured_paths = tuple(f"/loader-alias/task-{index}" for index in range(300))
    selected = ResolvedPerDatasetEpisodeSelection(
        episode_indices=tuple(tuple(range(100)) for _ in range(300)),
        dataset_frame_counts=(100,) * 300,
        total_episodes=30_000,
        signature="selection-signature",
    )
    captured = {}

    def resolve_selection(roots, **kwargs):
        captured["resolved"] = (tuple(roots), kwargs)
        return selected

    monkeypatch.setattr(
        robocasa_video_dataset,
        "resolve_per_dataset_episode_selection",
        resolve_selection,
    )
    monkeypatch.setattr(
        RobotVideoDataset,
        "__init__",
        lambda self, dataset_dirs, **kwargs: captured.setdefault(
            "loaded", (tuple(dataset_dirs), kwargs)
        ),
    )

    dataset = RoboCasaHuman300VideoDataset(
        dataset_dirs=configured_paths,
        validate_dataset_contract=False,
        human300_episode_selection={
            "strategy": "official_random_subset",
            "demos_per_task": 100,
            "seed": 0,
            "expected_task_count": 300,
            "expected_episode_count": 30_000,
        },
    )

    assert captured["resolved"] == (
        configured_paths,
        {
            "episodes_per_dataset": 100,
            "seed": 0,
            "expected_dataset_count": 300,
            "expected_total_episodes": 30_000,
        },
    )
    assert captured["loaded"][0] == configured_paths
    assert captured["loaded"][1]["per_dataset_episode_selection"] is selected
    assert dataset.human300_episode_selection is selected
