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

"""Official RoboCasa GR1 arms-and-waist data transforms.

This module intentionally depends only on PyTorch and torchvision so the same
field ordering, action normalization, and stored-view image transform can be
reused by training and evaluation without importing the simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms import functional as vision_functional
from torchvision.transforms.functional import InterpolationMode

from ..gr1_tabletop_recipe import OLD_GR1_SOURCE

GR1_DATASET_REVISION = OLD_GR1_SOURCE.revision
GR1_WAIST_PREFIX = "unlocked_waist: "


@dataclass(frozen=True)
class GR1FieldSpec:
    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


# The source vectors are 44D, but the official fourier_gr1_arms_waist recipe
# selects these fields in this order. Legs and neck are intentionally absent.
GR1_FIELD_SPECS = (
    GR1FieldSpec("left_arm", 0, 7),
    GR1FieldSpec("right_arm", 22, 29),
    GR1FieldSpec("left_hand", 7, 13),
    GR1FieldSpec("right_hand", 29, 35),
    GR1FieldSpec("waist", 41, 44),
)
GR1_FIELD_ORDER = tuple(field.name for field in GR1_FIELD_SPECS)
GR1_FIELD_DIMS = tuple(field.size for field in GR1_FIELD_SPECS)
GR1_RAW_DIM = 44
GR1_ACTION_DIM = sum(GR1_FIELD_DIMS)
GR1_STATE_DIM = 2 * GR1_ACTION_DIM
GR1_SELECTED_INDICES = tuple(
    index for field in GR1_FIELD_SPECS for index in range(field.start, field.end)
)


def _require_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if not value.is_floating_point():
        value = value.to(torch.float32)
    return value


def split_gr1_fields(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the five official fields from either a raw-44 or ordered-29 tensor."""

    value = _require_tensor(value, name="GR1 state/action")
    if value.shape[-1] == GR1_RAW_DIM:
        return tuple(value[..., field.start : field.end] for field in GR1_FIELD_SPECS)
    if value.shape[-1] == GR1_ACTION_DIM:
        fields = []
        offset = 0
        for size in GR1_FIELD_DIMS:
            fields.append(value[..., offset : offset + size])
            offset += size
        return tuple(fields)
    raise ValueError(
        "GR1 state/action must have raw dimension 44 or selected dimension 29, "
        f"got shape {tuple(value.shape)}."
    )


def select_gr1_fields(value: torch.Tensor) -> torch.Tensor:
    """Map raw 44D state/action to official ordered 29D arms/hands/waist."""

    return torch.cat(split_gr1_fields(value), dim=-1)


def fieldwise_sincos_state(value: torch.Tensor) -> torch.Tensor:
    """Apply official per-field ``[sin(field), cos(field)]`` encoding (58D)."""

    fields = split_gr1_fields(value)
    return torch.cat(
        [encoded for field in fields for encoded in (torch.sin(field), torch.cos(field))],
        dim=-1,
    )


def invert_fieldwise_sincos_state(value: torch.Tensor) -> torch.Tensor:
    """Recover the ordered 29D angles from a field-wise 58D representation."""

    value = _require_tensor(value, name="GR1 sine/cosine state")
    if value.shape[-1] != GR1_STATE_DIM:
        raise ValueError(f"GR1 sine/cosine state must be 58D, got shape {tuple(value.shape)}.")
    fields = []
    offset = 0
    for size in GR1_FIELD_DIMS:
        sine = value[..., offset : offset + size]
        cosine = value[..., offset + size : offset + 2 * size]
        fields.append(torch.atan2(sine, cosine))
        offset += 2 * size
    return torch.cat(fields, dim=-1)


def validate_coarse_prompt(prompt: str) -> str:
    """Validate and return an official non-empty, exactly-once waist prompt."""

    if not isinstance(prompt, str):
        raise TypeError(f"GR1 coarse prompt must be a string, got {type(prompt).__name__}.")
    if prompt != prompt.strip():
        raise ValueError("GR1 coarse prompt must not contain leading or trailing whitespace.")
    if not prompt.startswith(GR1_WAIST_PREFIX):
        raise ValueError(f"GR1 coarse prompt must start with {GR1_WAIST_PREFIX!r}, got {prompt!r}.")
    if prompt.count(GR1_WAIST_PREFIX) != 1:
        raise ValueError("GR1 coarse prompt must contain the waist prefix exactly once.")
    if not prompt[len(GR1_WAIST_PREFIX) :].strip():
        raise ValueError("GR1 coarse prompt is empty after the waist prefix.")
    return prompt


def _as_finite_vector(value: Any, *, name: str, size: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 1 or tensor.numel() != size:
        raise ValueError(f"{name} must contain {size} values, got shape {tuple(tensor.shape)}.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


@dataclass(frozen=True)
class GR1ActionNormalizer:
    """Official per-coordinate min/max normalizer with no output clamp."""

    minimum: torch.Tensor
    maximum: torch.Tensor

    def __post_init__(self) -> None:
        minimum = _as_finite_vector(self.minimum, name="action minimum", size=GR1_ACTION_DIM)
        maximum = _as_finite_vector(self.maximum, name="action maximum", size=GR1_ACTION_DIM)
        if bool((maximum < minimum).any().item()):
            raise ValueError(
                "Every GR1 action maximum must be greater than or equal to its minimum."
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_dataset_revision: str = GR1_DATASET_REVISION,
    ) -> "GR1ActionNormalizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls.from_payload(payload, expected_dataset_revision=expected_dataset_revision)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_dataset_revision: str = GR1_DATASET_REVISION,
    ) -> "GR1ActionNormalizer":
        """Load the frozen derived-stats schema and reject approximate variants."""

        if not isinstance(payload, Mapping):
            raise TypeError("GR1 statistics payload must be a mapping.")
        expected_header = {
            "schema_version": 1,
            "benchmark": "robocasa_gr1_tabletop",
            "dataset_revision": expected_dataset_revision,
            "dataset_count": 24,
        }
        for key, expected in expected_header.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"GR1 statistics {key!r} must be {expected!r}, got {payload.get(key)!r}."
                )
        if tuple(payload.get("field_order", ())) != GR1_FIELD_ORDER:
            raise ValueError(
                f"GR1 statistics field_order must be {GR1_FIELD_ORDER}, "
                f"got {payload.get('field_order')!r}."
            )
        action_fields = payload.get("action_fields")
        if not isinstance(action_fields, Mapping):
            raise ValueError("GR1 statistics must contain an action_fields mapping.")
        if set(action_fields) != set(GR1_FIELD_ORDER):
            raise ValueError(
                "GR1 action_fields must contain exactly the five official fields; "
                f"got {tuple(action_fields)}."
            )

        minima = []
        maxima = []
        for field in GR1_FIELD_SPECS:
            stats = action_fields[field.name]
            if not isinstance(stats, Mapping) or set(stats) != {"min", "max"}:
                raise ValueError(f"GR1 action field {field.name!r} must contain only min and max.")
            minimum = _as_finite_vector(
                stats["min"], name=f"action_fields.{field.name}.min", size=field.size
            )
            maximum = _as_finite_vector(
                stats["max"], name=f"action_fields.{field.name}.max", size=field.size
            )
            minima.append(minimum)
            maxima.append(maximum)
        return cls(torch.cat(minima), torch.cat(maxima))

    def _stats_like(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        minimum = self.minimum.to(device=value.device, dtype=value.dtype)
        maximum = self.maximum.to(device=value.device, dtype=value.dtype)
        return minimum, maximum

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize raw-44 or ordered-29 actions to ordered 29D, without clamp."""

        selected = select_gr1_fields(value)
        minimum, maximum = self._stats_like(selected)
        span = maximum - minimum
        varying = span != 0
        normalized = torch.zeros_like(selected)
        normalized[..., varying] = (
            2 * (selected[..., varying] - minimum[varying]) / span[varying] - 1
        )
        # The N1.5 reference maps constant coordinates to zero.
        normalized[..., ~varying] = 0
        return normalized

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        """Invert normalized ordered-29 actions; constant coordinates recover min=max."""

        value = _require_tensor(value, name="normalized GR1 action")
        if value.shape[-1] != GR1_ACTION_DIM:
            raise ValueError(f"Normalized GR1 action must be 29D, got shape {tuple(value.shape)}.")
        minimum, maximum = self._stats_like(value)
        return (value + 1) / 2 * (maximum - minimum) + minimum


class GR1StateActionTransform:
    """LongWAMBase processor transform for raw default-key GR1 state/action."""

    def __init__(
        self,
        action_stats_path: str | Path,
        expected_dataset_revision: str = GR1_DATASET_REVISION,
    ):
        self.action_normalizer = GR1ActionNormalizer.from_json(
            action_stats_path,
            expected_dataset_revision=expected_dataset_revision,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "state" not in batch or "default" not in batch["state"]:
            raise KeyError("GR1 batch is missing state.default.")
        batch["state"]["default"] = fieldwise_sincos_state(batch["state"]["default"])
        if "action" in batch:
            if "default" not in batch["action"]:
                raise KeyError("GR1 batch is missing action.default.")
            batch["action"]["default"] = self.action_normalizer.normalize(
                batch["action"]["default"]
            )
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "state" in batch and "default" in batch["state"]:
            batch["state"]["default"] = invert_fieldwise_sincos_state(batch["state"]["default"])
        if "action" in batch and "default" in batch["action"]:
            batch["action"]["default"] = self.action_normalizer.denormalize(
                batch["action"]["default"]
            )
        return batch


def _canonicalize_video(video: torch.Tensor) -> torch.Tensor:
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"GR1 video must be a torch.Tensor, got {type(video).__name__}.")
    if video.ndim == 3 and video.shape[-1] == 3:  # HWC
        video = video.permute(2, 0, 1).unsqueeze(0)
    elif video.ndim == 3 and video.shape[0] == 3:  # CHW
        video = video.unsqueeze(0)
    elif video.ndim == 4 and video.shape[1] == 3:  # TCHW
        pass
    else:
        raise ValueError(
            "GR1 video must be HWC, CHW, or TCHW with three RGB channels, "
            f"got shape {tuple(video.shape)}."
        )
    if tuple(video.shape[-2:]) != (256, 256):
        raise ValueError(
            "GR1 stored/evaluation-adapted ego frames must be 256x256 before the "
            f"official crop, got {tuple(video.shape[-2:])}."
        )
    if video.dtype == torch.uint8:
        return video.to(torch.float32) / 255.0
    if not video.is_floating_point():
        raise TypeError(f"GR1 video dtype must be uint8 or floating point, got {video.dtype}.")
    if not bool(torch.isfinite(video).all().item()):
        raise ValueError("GR1 video contains non-finite values.")
    if bool(((video < 0) | (video > 1)).any().item()):
        raise ValueError("Floating-point GR1 video must already be in [0, 1].")
    return video.to(torch.float32)


class GR1OfficialVideoTransform(nn.Module):
    """Official stored-view crop/resize/jitter with shared temporal randomness.

    Input is RGB uint8 (or [0,1] floating point) in HWC, CHW, or TCHW form.
    Output is always float32 ``[T, 3, 224, 224]``. Training samples one crop and
    one color-jitter parameter set for the entire temporal clip; evaluation
    uses a deterministic center crop and does not jitter.
    """

    crop_size = 243  # int(256 * 0.95), matching the N1.5 reference.
    output_size = (224, 224)

    def __init__(self, training: bool):
        super().__init__()
        self.training_transform = bool(training)
        self.brightness = (0.7, 1.3)
        self.contrast = (0.6, 1.4)
        self.saturation = (0.5, 1.5)
        self.hue = (-0.08, 0.08)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = _canonicalize_video(video)
        if self.training_transform:
            top, left, height, width = RandomCrop.get_params(
                video, output_size=(self.crop_size, self.crop_size)
            )
        else:
            height = width = self.crop_size
            top = (video.shape[-2] - height) // 2
            left = (video.shape[-1] - width) // 2
        video = vision_functional.crop(video, top, left, height, width)
        video = vision_functional.resize(
            video,
            self.output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        if self.training_transform:
            order, brightness, contrast, saturation, hue = ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            for transform_id in order.tolist():
                if transform_id == 0 and brightness is not None:
                    video = vision_functional.adjust_brightness(video, brightness)
                elif transform_id == 1 and contrast is not None:
                    video = vision_functional.adjust_contrast(video, contrast)
                elif transform_id == 2 and saturation is not None:
                    video = vision_functional.adjust_saturation(video, saturation)
                elif transform_id == 3 and hue is not None:
                    video = vision_functional.adjust_hue(video, hue)
        return video


class NoClampIdentityNormalizer:
    """Processor adapter: GR1 actions are already officially normalized."""

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def get_stats(self) -> dict[str, Any]:
        return {}
