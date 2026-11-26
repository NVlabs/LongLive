# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity cards (Method, "Entity cards").

Images carry places and a look that words do not pin down, while the text encoder reads attribute words directly, so
memory also enters as text. When p_i is revealed, every entity of the committed memory whose name occurs in p_i
(whole-word match) gets a card appended to the prompt: its name, description and persistent state, e.g.

    Remembered from earlier shots: Sam: Sam, a tall broad man in his mid-30s with sandy blonde hair.
    Current state: clothing: maroon t-shirt and grey sweatpants.

The agent writes these texts in advance (the entity store saved with every update), so only string matching remains
after the prompt. Like the images, a card reflects memory up to shot i-1: a change stated in p_i is in the prompt itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

HEADER = "Remembered from earlier shots: "


def store_path(out_dir: Path, watermark: int) -> Path:
    """The entity store the agent saved after integrating prompt ``watermark`` (setup: before the first prompt)."""
    return Path(out_dir) / "memory" / "store" / ("setup.json" if watermark < 0 else f"w{watermark:03d}.json")


def cards_for(prompt: str, entities: Dict[str, Dict]) -> List[str]:
    """One card per stored entity whose name occurs in ``prompt`` as a whole word (case-insensitive)."""
    base = prompt.rstrip()
    cards = []
    for entity in entities.values():
        name = entity.get("name")
        if not name or not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", base, re.IGNORECASE):
            continue
        facts = {a: x for a, x in (entity.get("facts") or {}).items() if x.get("persistent")}
        text = f"{name}: {entity.get('description', '').strip()}"
        if facts:
            text += " Current state: " + "; ".join(f"{a}: {facts[a]['value']}" for a in sorted(facts)) + "."
        cards.append(text)
    return cards


class CardBook:
    """Loads the saved entity stores of a run (cached per watermark) and builds the card text for a prompt."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self._stores: Dict[int, Dict[str, Dict]] = {}

    def entities(self, watermark: int) -> Dict[str, Dict]:
        if watermark not in self._stores:
            path = store_path(self.memory_dir, watermark)
            self._stores[watermark] = json.loads(path.read_text())["entities"] if path.is_file() else {}
        return self._stores[watermark]

    def text(self, prompt: str, watermark: int) -> Tuple[str, int]:
        """(" Remembered from earlier shots: <cards>", number of cards); empty when no stored entity is named."""
        cards = cards_for(prompt, self.entities(int(watermark)))
        return (" " + HEADER + " ".join(cards) if cards else ""), len(cards)
