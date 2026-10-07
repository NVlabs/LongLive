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

"""Fail-closed temporal specifications for the GR1 memory-length ablation.

This module intentionally uses only the Python standard library.  Training,
evaluation, and lifecycle code can therefore sign and validate the same
temporal contract without importing Torch, Hydra, or the simulator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final


TEMPORAL_SPEC_SCHEMA: Final = "longwam.robocasa-gr1-tabletop-temporal-spec/v1"
TEMPORAL_SPEC_VERSION: Final = 1
TRAINING_PLAN_KEY: Final = "temporal_spec"

ALLOWED_PAST_OBS_SIZES: Final = (0, 48, 96, 192, 384, 768)
CONTROL_HZ: Final = 20
ACTION_HORIZON: Final = 16
VIDEO_SAMPLE_STRIDE: Final = 4
# Compatibility name used by the frozen benchmark contract.  Both names are
# intentionally the same primitive and are signed through ``as_dict``.
ACTION_VIDEO_FREQ_RATIO: Final = VIDEO_SAMPLE_STRIDE
VAE_TEMPORAL_FACTOR: Final = 4
RAW_FRAMES_PER_CLEAN_LATENT: Final = VIDEO_SAMPLE_STRIDE * VAE_TEMPORAL_FACTOR
IMAGINE_FRAMES: Final = 2
FUTURE_RAW_STEPS: Final = IMAGINE_FRAMES * RAW_FRAMES_PER_CLEAN_LATENT


def _require_plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a plain int, got {type(value).__name__}.")
    return value


@dataclass(frozen=True, slots=True)
class TemporalSpec:
    """One immutable past-memory length with the fixed GR1 H16/k2 contract.

    ``past_obs_size`` counts raw 20-Hz control steps strictly *before* the
    current observation.  Consequently ``P=0`` means current-frame-only
    visual conditioning: it removes past memory but never removes the current
    observation, the H16 action output, or the k2 imagined future.
    """

    past_obs_size: int

    def __post_init__(self) -> None:
        past_obs_size = _require_plain_int(self.past_obs_size, "past_obs_size")
        if past_obs_size not in ALLOWED_PAST_OBS_SIZES:
            raise ValueError(
                f"past_obs_size must be one of {ALLOWED_PAST_OBS_SIZES}, got {past_obs_size}."
            )
        if past_obs_size % RAW_FRAMES_PER_CLEAN_LATENT:
            raise ValueError("past_obs_size must land on a complete clean latent block.")

        # These checks deliberately repeat the derivation at construction
        # time.  A future edit to any primitive constant must fail closed
        # instead of silently changing the approved longwam.benchmarks.
        if FUTURE_RAW_STEPS != 32:
            raise RuntimeError("GR1 imagined-future span must remain 32 raw steps.")
        if ACTION_HORIZON != 16 or IMAGINE_FRAMES != 2:
            raise RuntimeError("GR1 H16/k2 action contract changed.")
        if VIDEO_SAMPLE_STRIDE != 4 or VAE_TEMPORAL_FACTOR != 4:
            raise RuntimeError("GR1 stride-4/VAE-factor-4 contract changed.")
        if self.total_latent_frames != self.action_conditioning_frames:
            raise RuntimeError(
                "The observation window must contain exactly M clean plus k imagined latent frames."
            )
        if self.sampled_offsets[-1] != FUTURE_RAW_STEPS:
            raise RuntimeError("The sampled video window does not end at +32.")
        if self.history_offsets[-1] != 0:
            raise RuntimeError("The history window must end at the current frame.")

    @classmethod
    def from_past_obs_size(cls, past_obs_size: int) -> "TemporalSpec":
        return cls(past_obs_size=past_obs_size)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TemporalSpec":
        """Rebuild and authenticate an exact JSON-shaped :meth:`as_dict` value."""

        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Temporal specification payload must be a mapping, got {type(payload).__name__}."
            )
        if "past_obs_size" not in payload:
            raise ValueError("Temporal specification is missing past_obs_size.")
        spec = cls.from_past_obs_size(payload["past_obs_size"])
        observed = dict(payload)
        expected = spec.as_dict()
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            drift = sorted(
                key for key in set(expected) & set(observed) if expected[key] != observed[key]
            )
            raise ValueError(
                "Temporal specification payload drifted: "
                f"missing={missing}, extra={extra}, changed={drift}."
            )
        return spec

    @property
    def P(self) -> int:
        return self.past_obs_size

    @property
    def N(self) -> int:
        return self.num_frames

    @property
    def I(self) -> int:  # noqa: E743 - canonical temporal-contract symbol
        return self.image_steps

    @property
    def M(self) -> int:
        return self.clean_latent_frames

    @property
    def k(self) -> int:
        return IMAGINE_FRAMES

    @property
    def has_past_memory(self) -> bool:
        return self.past_obs_size > 0

    @property
    def history_seconds(self) -> float:
        return self.past_obs_size / CONTROL_HZ

    @property
    def action_horizon_seconds(self) -> float:
        return ACTION_HORIZON / CONTROL_HZ

    @property
    def future_seconds(self) -> float:
        return FUTURE_RAW_STEPS / CONTROL_HZ

    @property
    def total_window_seconds(self) -> float:
        return (self.num_frames - 1) / CONTROL_HZ

    @property
    def num_frames(self) -> int:
        # Inclusive raw observation offsets [-P, ..., +32].
        return self.past_obs_size + FUTURE_RAW_STEPS + 1

    @property
    def sampled_video_frames(self) -> int:
        return (self.num_frames - 1) // VIDEO_SAMPLE_STRIDE + 1

    @property
    def image_steps(self) -> int:
        return self.sampled_video_frames

    @property
    def clean_sampled_video_frames(self) -> int:
        return self.past_obs_size // VIDEO_SAMPLE_STRIDE + 1

    @property
    def clean_latent_frames(self) -> int:
        return self.past_obs_size // RAW_FRAMES_PER_CLEAN_LATENT + 1

    @property
    def total_latent_frames(self) -> int:
        return (self.sampled_video_frames - 1) // VAE_TEMPORAL_FACTOR + 1

    @property
    def action_conditioning_frames(self) -> int:
        return self.clean_latent_frames + IMAGINE_FRAMES

    @property
    def raw_offsets(self) -> tuple[int, ...]:
        return tuple(range(-self.past_obs_size, FUTURE_RAW_STEPS + 1))

    @property
    def sampled_offsets(self) -> tuple[int, ...]:
        return self.raw_offsets[::VIDEO_SAMPLE_STRIDE]

    @property
    def history_offsets(self) -> tuple[int, ...]:
        """Stride-4 past-through-current offsets used by evaluation history."""

        return tuple(range(-self.past_obs_size, 1, VIDEO_SAMPLE_STRIDE))

    @property
    def action_offsets(self) -> tuple[int, ...]:
        return tuple(range(ACTION_HORIZON))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-shaped value stored in signed plans."""

        return {
            "schema": TEMPORAL_SPEC_SCHEMA,
            "version": TEMPORAL_SPEC_VERSION,
            "semantics": {
                "past_obs_size": "past_raw_steps_excluding_current",
                "history_offsets": "sampled_past_through_current",
                "zero_history": "current_frame_only_no_past_memory",
                "current_frame_always_conditioned": True,
            },
            "past_obs_size": self.past_obs_size,
            "has_past_memory": self.has_past_memory,
            "history_seconds": self.history_seconds,
            "control_hz": CONTROL_HZ,
            "action_horizon": ACTION_HORIZON,
            "action_horizon_seconds": self.action_horizon_seconds,
            "imagine_frames": IMAGINE_FRAMES,
            "future_raw_steps": FUTURE_RAW_STEPS,
            "future_seconds": self.future_seconds,
            "video_sample_stride": VIDEO_SAMPLE_STRIDE,
            "vae_temporal_factor": VAE_TEMPORAL_FACTOR,
            "raw_frames_per_clean_latent": RAW_FRAMES_PER_CLEAN_LATENT,
            "num_frames": self.num_frames,
            "sampled_video_frames": self.sampled_video_frames,
            "clean_sampled_video_frames": self.clean_sampled_video_frames,
            "clean_latent_frames": self.clean_latent_frames,
            "total_latent_frames": self.total_latent_frames,
            "action_conditioning_frames": self.action_conditioning_frames,
            "total_window_seconds": self.total_window_seconds,
            "raw_offsets": list(self.raw_offsets),
            "sampled_offsets": list(self.sampled_offsets),
            "history_offsets": list(self.history_offsets),
            "action_offsets": list(self.action_offsets),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


def get_temporal_spec(past_obs_size: int) -> TemporalSpec:
    return TemporalSpec.from_past_obs_size(past_obs_size)


def all_temporal_specs() -> tuple[TemporalSpec, ...]:
    return tuple(TemporalSpec(value) for value in ALLOWED_PAST_OBS_SIZES)
