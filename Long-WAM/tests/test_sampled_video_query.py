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

from types import SimpleNamespace

import pytest
import torch

import longwam.datasets.lerobot.base_lerobot_dataset as base_dataset_module
from longwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from longwam.datasets.lerobot.lerobot.datasets.utils import get_delta_indices
from longwam.datasets.lerobot.processors.longwam_processor import LongWAMProcessor
from longwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def test_base_dataset_queries_selected_images_but_full_state(monkeypatch, tmp_path):
    captured = {}

    class FakeMetadata:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = root
            self.fps = 50
            self.total_episodes = 1

    class FakeMultiDataset:
        def __init__(self, *, delta_timestamps, **_kwargs):
            captured.update(delta_timestamps)
            self.dataset_frame_ranges = ((0, 100),)
            self._datasets = [
                SimpleNamespace(
                    episode_data_index={
                        "from": torch.tensor([0]),
                        "to": torch.tensor([100]),
                    }
                )
            ]

        def __len__(self):
            return 100

    monkeypatch.setattr(base_dataset_module, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(base_dataset_module, "MultiLeRobotDataset", FakeMultiDataset)

    BaseLerobotDataset(
        dataset_dirs=[str(tmp_path / "task")],
        shape_meta={
            "images": [{"key": "cam_high", "raw_shape": [3, 480, 640]}],
            "state": [{"key": "default", "raw_shape": 14}],
            "action": [{"key": "default", "raw_shape": 14}],
        },
        obs_size=81,
        past_obs_size=48,
        action_size=32,
        val_set_proportion=0,
        is_training_set=True,
        image_sample_indices=range(0, 81, 4),
    )

    offsets = get_delta_indices(captured, fps=50)
    assert offsets["observation.images.cam_high"] == list(range(-48, 33, 4))
    assert offsets["observation.state"] == list(range(-48, 33))
    assert offsets["action"] == list(range(32))


class _FakeProcessedDataset:
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return self.sample


def _robot_video_shell(sample, *, sampled_query):
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = _FakeProcessedDataset(sample)
    dataset.skip_padding_as_possible = False
    dataset.max_padding_retry = 0
    dataset.num_frames = 81
    dataset.past_obs_size = 48
    dataset.action_chunk = 32
    dataset.action_video_freq_ratio = 4
    dataset.video_sample_indices = list(range(0, 81, 4))
    dataset.decode_only_sampled_video_frames = sampled_query
    dataset.concat_multi_camera = "horizontal"
    dataset.resize_transform = lambda value: value
    dataset.crop_transform = lambda value: value
    dataset.normalize_transform = lambda value: value
    dataset.override_instruction = None
    dataset.context_len = 4
    dataset._get_cached_text_context = lambda _prompt: (
        torch.ones(4, 2),
        torch.ones(4, dtype=torch.bool),
    )
    return dataset


def _processed_sample(pixel_values, image_is_pad):
    return {
        "pixel_values": pixel_values,
        "image_is_pad": image_is_pad,
        "action": torch.arange(32 * 14, dtype=torch.float32).view(32, 14),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "proprio": torch.arange(81 * 14, dtype=torch.float32).view(81, 14),
        "proprio_is_pad": torch.zeros(81, dtype=torch.bool),
        "instruction": "move the object",
    }


def test_sampled_video_query_is_output_equivalent_and_not_subsampled_twice():
    full_video = torch.arange(3 * 81 * 3 * 2 * 2, dtype=torch.float32).view(3, 81, 3, 2, 2)
    full_padding = torch.tensor([(index % 7) == 0 for index in range(81)])
    sampled_video = full_video[:, ::4].clone()
    sampled_padding = full_padding[::4].clone()

    legacy = _robot_video_shell(
        _processed_sample(full_video, full_padding), sampled_query=False
    )._get(0)
    optimized = _robot_video_shell(
        _processed_sample(sampled_video, sampled_padding), sampled_query=True
    )._get(0)

    for key in (
        "video",
        "image_is_pad",
        "action",
        "action_is_pad",
        "proprio",
        "proprio_is_pad",
        "context",
        "context_mask",
    ):
        assert torch.equal(optimized[key], legacy[key]), key
    assert optimized["video"].shape == (3, 21, 2, 6)
    assert optimized["proprio"].shape == (80, 14)
    assert optimized["action"].shape == (32, 14)
    assert optimized["prompt"] == legacy["prompt"]


class _FakeMerger:
    def set_shape_meta(self, _shape_meta):
        pass


def _processor(*, num_image_steps=None):
    return LongWAMProcessor(
        shape_meta={"images": [], "action": [], "state": []},
        num_obs_steps=81,
        num_image_steps=num_image_steps,
        num_output_cameras=3,
        action_output_dim=14,
        proprio_output_dim=14,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=_FakeMerger(),
        train_transforms=[],
        val_transforms=[],
    )


def test_processor_image_steps_are_independent_and_backward_compatible():
    assert _processor().num_image_steps == 81
    assert _processor(num_image_steps=21).num_image_steps == 21
    with pytest.raises(ValueError, match="must be positive"):
        _processor(num_image_steps=0)
