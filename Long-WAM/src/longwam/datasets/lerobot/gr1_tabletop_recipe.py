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

"""Pinned source identities for the two official GR1 Tabletop releases."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


PromptSource = Literal["coarse_annotation", "episode_remarks"]


@dataclass(frozen=True)
class GR1TabletopSourceSpec:
    """Immutable source contract used by the shared GR1 data adapter."""

    name: str
    repo_id: str
    revision: str
    dataset_names: tuple[str, ...]
    total_episodes: int
    total_frames: int
    prompt_source: PromptSource
    annotation_keys: tuple[str, ...]
    requires_known_omissions: bool


OLD_GR1_DATASET_NAMES = (
    "gr1_arms_waist.CanToDrawer",
    "gr1_arms_waist.CupToDrawer",
    "gr1_arms_waist.CuttingboardToBasket",
    "gr1_arms_waist.CuttingboardToCardboardBox",
    "gr1_arms_waist.CuttingboardToPan",
    "gr1_arms_waist.CuttingboardToPot",
    "gr1_arms_waist.CuttingboardToTieredBasket",
    "gr1_arms_waist.PlaceBottleToCabinet",
    "gr1_arms_waist.PlaceMilkToMicrowave",
    "gr1_arms_waist.PlacematToBasket",
    "gr1_arms_waist.PlacematToBowl",
    "gr1_arms_waist.PlacematToPlate",
    "gr1_arms_waist.PlacematToTieredShelf",
    "gr1_arms_waist.PlateToBowl",
    "gr1_arms_waist.PlateToCardboardBox",
    "gr1_arms_waist.PlateToPan",
    "gr1_arms_waist.PlateToPlate",
    "gr1_arms_waist.PotatoToMicrowave",
    "gr1_arms_waist.TrayToCardboardBox",
    "gr1_arms_waist.TrayToPlate",
    "gr1_arms_waist.TrayToPot",
    "gr1_arms_waist.TrayToTieredBasket",
    "gr1_arms_waist.TrayToTieredShelf",
    "gr1_arms_waist.WineToCabinet",
)

TELEOP_GR1_DATASET_NAMES = (
    "gr1_unified.PnPBottleToCabinetClose",
    "gr1_unified.PnPCanToDrawerClose",
    "gr1_unified.PnPCupToDrawerClose",
    "gr1_unified.PnPMilkToMicrowaveClose",
    "gr1_unified.PnPPotatoToMicrowaveClose",
    "gr1_unified.PnPWineToCabinetClose",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)

OLD_GR1_SOURCE = GR1TabletopSourceSpec(
    name="old",
    repo_id="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    revision="ea7ac0b68f87da62f1e726771bba0fe74300802f",
    dataset_names=OLD_GR1_DATASET_NAMES,
    total_episodes=241_450,
    total_frames=60_570_778,
    prompt_source="coarse_annotation",
    annotation_keys=("human.coarse_action",),
    requires_known_omissions=True,
)

TELEOP_GR1_SOURCE = GR1TabletopSourceSpec(
    name="teleop",
    repo_id="nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim",
    revision="09c6de8af50168090e7e9cc01e1ec3bce788de24",
    dataset_names=TELEOP_GR1_DATASET_NAMES,
    total_episodes=24_000,
    total_frames=5_820_277,
    prompt_source="episode_remarks",
    annotation_keys=("human.coarse_action", "human.fine_action"),
    requires_known_omissions=False,
)

GR1_SOURCE_RECIPES: Mapping[str, GR1TabletopSourceSpec] = MappingProxyType(
    {source.name: source for source in (OLD_GR1_SOURCE, TELEOP_GR1_SOURCE)}
)


def get_gr1_source_spec(name: str) -> GR1TabletopSourceSpec:
    try:
        return GR1_SOURCE_RECIPES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown GR1 source_recipe {name!r}; expected one of {tuple(GR1_SOURCE_RECIPES)}."
        ) from error


__all__ = [
    "GR1_SOURCE_RECIPES",
    "GR1TabletopSourceSpec",
    "OLD_GR1_DATASET_NAMES",
    "OLD_GR1_SOURCE",
    "TELEOP_GR1_DATASET_NAMES",
    "TELEOP_GR1_SOURCE",
    "get_gr1_source_spec",
]
