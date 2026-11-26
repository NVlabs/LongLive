# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot episode scripts and the generator prompt.

An episode (``stories/robot/*.json``) is one first-frame image of a robot workspace plus eight instructions of 8 s each:
  name, title, summary
  camera        "head camera" | "fixed overhead camera" | "fixed third-person camera"
  robot         the robot as seen in the first frame, e.g. "two black dexterous robotic hands and arms ..."
  setting       the workspace, e.g. "an office space with a wooden tabletop set above a white drawer unit"
  first_frame   file name of the input image, resolved against an image folder (see ``resolve_first_frame``)
  segments      [{"instruction": ..., "memory_event": ...}]; ``memory_event`` ("none", "hide: ...", "return: ...")
                annotates the script for evaluation only and is never shown to the agent or the generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .prompts import SOUND

REQUIRED = ("name", "camera", "robot", "setting", "first_frame", "segments")


def load_episode(path: Path) -> Dict[str, Any]:
    ep = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in ep]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if not ep["segments"] or any(not s.get("instruction") for s in ep["segments"]):
        raise ValueError(f"{path}: every segment needs an instruction")
    return ep


def instructions(ep: Dict[str, Any]) -> List[str]:
    return [s["instruction"] for s in ep["segments"]]


def resolve_first_frame(ep: Dict[str, Any], image: Optional[Path] = None, image_root: Optional[Path] = None) -> Path:
    """The input image: ``image`` if given, else ``image_root / ep["first_frame"]``."""
    if image is not None:
        p = Path(image)
    elif image_root is not None:
        p = Path(image_root) / ep["first_frame"]
    else:
        raise ValueError(f"no input image for {ep['name']}: pass --image or --image-root (see stories/robot/README.md)")
    if not p.is_file():
        raise FileNotFoundError(f"input image not found: {p} (see stories/robot/README.md)")
    return p.resolve()


def header(ep: Dict[str, Any]) -> str:
    """The camera / robot / setting sentence that starts every segment description."""
    cam, robot, setting = ep["camera"], ep["robot"], ep["setting"]
    if cam == "head camera":
        return f"Realistic first-person footage from the head camera of a robot, showing its {robot}, in {setting}."
    if cam == "fixed overhead camera":
        return (f"Realistic footage from a fixed overhead camera looking down at {setting}, where a robot with {robot} "
                "works; the camera does not move.")
    return (f"Realistic footage from a fixed camera watching a robot with {robot} in {setting}; "
            "the camera does not move.")


def base_prompt(ep: Dict[str, Any], instr: str) -> str:
    """Generator prompt without memory: the sink holds only the latest frame (Picture 1)."""
    return ("Picture 1 is the latest frame of this continuous robot video, shown before the target video; the target "
            "video continues from it without a cut.\n\nintegrated_multimodal_description: [Shot 1] "
            + header(ep) + " " + instr + "\n\noverall_soundscape: " + SOUND)


def ours_prompt(ep: Dict[str, Any], text: str, pinned: Sequence[Dict[str, Any]]) -> str:
    """Generator prompt with memory: Pictures 1..k-1 are the pinned memory pictures (named by their captions) and
    Picture k is the latest frame. Without pinned pictures it is ``base_prompt`` with the agent's text."""
    if not pinned:
        return base_prompt(ep, text)
    k = len(pinned) + 1
    first = "Picture 1 is a camera frame" if k == 2 else f"Pictures 1-{k - 1} are camera frames"
    caps = " ".join(f"Picture {j + 1} shows {m['caption'].rstrip('.')}." for j, m in enumerate(pinned))
    return (f"{first} recorded earlier in this same continuous robot video and Picture {k} is its latest frame; they "
            "are shown before the target video, which continues from the latest frame without a cut. " + caps
            + " When these places and objects come back into view, they look exactly as in the earlier pictures."
            "\n\nintegrated_multimodal_description: [Shot 1] " + header(ep) + " " + text
            + "\n\noverall_soundscape: " + SOUND)
