# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt templates of the memory agent.

Each system prompt starts with a ``TASK:`` line so logs (and the offline mock models) can tell
the calls apart. All text that the memory policy writes for one revealed prompt comes from a
single planner call (``memory_plan``) – this JSON is the action space trained by RL.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

PLAN_SYSTEM = """TASK: memory_plan
You maintain the visual memory of a live, open-ended video stream. Shot prompts arrive one at a
time; each new prompt is shown to you right after it has been revealed to the video generator.
Your memory will condition the NEXT shot, whose prompt you cannot see. Write memory that remains
correct for whatever comes next.

Memory has two layers.
1. Entities: every recurring character, object and scene (location) has a stable identity
   description and a set of facts. The description holds only what never changes (face, age, build,
   skin, hair colour, species, fur pattern; for objects and places their unchanging look).
   Everything that can change (clothing, accessories, facial hair, hairstyle, make-up, stains, damage,
   wet/dirty, assembly stage, contents, placement) is a fact, never part of a description. For each
   fact decide
   - persistent: true for properties that stay until something changes them (identity, clothing,
     stains, damage, wet/dirty, assembly stage, colour, what an object has become, placement of
     an object left somewhere);
     persistent: false for momentary properties (pose, action, gaze, expression, what is being
     held right now, camera framing). Only persistent facts are drawn into memory images.
   - source: "explicit" if the prompt states it, "inferred" if you deduce it.
   Reuse the attribute names already in the ENTITY TABLE. To change a fact, write it again under the
   same attribute with the new value: that replaces the old value. Never put an attribute you write
   in "revoke". Use "revoke" only for an earlier fact that turns out to be wrong or no longer applies
   and has no new value.
   The SESSION SETUP cast cards describe the cast at the start of the session; their current
   appearance is the description plus the facts in the ENTITY TABLE.
2. Memory images: G (long-term) shows the protagonists and key long-lived objects in their current
   persistent state; M (short-term) shows the current scene as an empty set plus the entities that
   are likely to stay relevant, in neutral poses. Choose members and write their composition.

If several new prompts are listed (the memory fell behind), integrate them in order into one update
whose state is the state after the last one.

Rules: reuse existing entity ids; keep descriptions about unchanging appearance only; image prompts
describe the identity plus the CURRENT persistent facts and never repeat a value that a fact has
replaced; never invent future events; be concise. Output one JSON object and nothing else."""

PLAN_SCHEMA = {
    "entities": [{
        "id": "snake_case id (reuse existing ids)",
        "kind": "character | object | scene",
        "name": "short name",
        "description": "unchanging identity/appearance only, no clothing or other changeable state (new entities only)",
        "present": "true if it appears in the new prompt",
        "facts": [{"attr": "snake_case attribute (reuse the table's names)", "value": "value", "persistent": True,
                   "source": "explicit"}],
        "revoke": ["attributes that no longer apply and get NO new value (never an attribute written in facts)"],
    }],
    "scene": "scene entity id of the new prompt",
    "active": ["entity ids likely to matter in the next shots"],
    "retire": ["entity ids that have left the story for now"],
    "image_prompts": {"<entity id>": "appearance of this entity now (persistent facts only), for entities that are new or whose persistent facts changed"},
    "memory": {
        "g_members": ["entity ids in G (protagonists first)"],
        "m_members": ["entity ids in M besides the scene"],
        "g_prompt": "one paragraph composing G",
        "m_prompt": "one paragraph composing M in the current scene",
        "g_caption": "one sentence describing G for the video model",
        "m_caption": "one sentence describing M for the video model",
    },
}


def plan_user(setup: Dict[str, Any], table: List[Dict[str, Any]], scene: Any, recent: List[str],
              new: List[Tuple[int, str]], limits: Dict[str, int], observation: str = "",
              next_prompt: str = "") -> str:
    parts = [
        "### SESSION SETUP\n" + json.dumps(setup, ensure_ascii=False),
        "### ENTITY TABLE (current memory)\n" + json.dumps(table, ensure_ascii=False),
        f"### CURRENT SCENE\n{scene}",
    ]
    if recent:
        parts.append("### EARLIER PROMPTS (most recent last)\n" + "\n---\n".join(recent))
    if len(new) == 1:
        parts.append(f"### NEW PROMPT (shot {new[0][0]})\n{new[0][1]}")
    else:
        parts.append(f"### NEW PROMPTS (shots {new[0][0]}-{new[-1][0]}, oldest first; integrate all of them in order, "
                     "the memory must describe the state after the last one)\n"
                     + "\n---\n".join(f"[shot {i}] {text}" for i, text in new))
    if observation:
        parts.append("### OBSERVATION\n" + observation)
    if next_prompt:
        parts.append("### NEXT PROMPT (oracle upper bound only)\n" + next_prompt)
    parts.append(
        f"### OUTPUT\nAt most {limits['g']} members in G and {limits['m']} in M. JSON schema:\n"
        + json.dumps(PLAN_SCHEMA, ensure_ascii=False)
    )
    return "\n\n".join(parts)


CHECK_SYSTEM = """TASK: image_check
You check a generated memory image against its specification. Reference images (if any) show the
established identity of the same entities. Judge only what the specification requires: identity
match with the references, the listed persistent state, the requested composition, and obvious
defects (extra limbs, melted objects, text overlays, collage borders). Ignore pose and background
unless the specification asks for them. The references may show an older state: where the
specification lists a change (or a state that differs from the references), the candidate must show
the specified state, not the references' state.
"state_ok" concerns only the listed persistent state ("Current state" and listed changes): clothing,
facial hair, hairstyle, accessories, stains, damage, wet/dirty, assembly stage, what an object has
become. A required persistent state that is clearly missing or different (old clothing, a beard that
should be gone, a stain that should be there or should be gone, the wrong assembly stage) is a state
error, never a minor flaw: then "state_ok" is false, "pass" is false and the score is at most 4.
Not state errors (judge them in "score" and "issues" only): pose, position, where the entity stands,
the background or setting, what is held, expression, framing, a near-synonym of the same garment in a
similar colour (jeans vs. trousers), or the exact size of a stain that is present.
Output JSON: {"pass": bool, "score": number 0-10, "state_ok": bool, "issues": ["short actionable fixes"]}."""


def check_user(spec: str, n_refs: int) -> str:
    refs = (f"After this text come the candidate and {n_refs} reference image(s), each preceded by its label. Judge "
            "only the CANDIDATE; use the references for identity only." if n_refs else "The image is the candidate.")
    return f"{refs}\n\n### SPECIFICATION\n{spec}\n\nReturn the JSON verdict."


VIEW_STYLE = {
    "character": {
        "front": "full-body front view, neutral standing pose, arms relaxed, plain light-grey studio background",
        "side": "full-body three-quarter view, neutral standing pose, plain light-grey studio background",
    },
    "object": {
        "front": "clear product view of the whole object, centred, plain light-grey background",
        "side": "three-quarter angled product view of the whole object, plain light-grey background",
    },
    "scene": {
        "plate": "wide establishing view of the empty location, no people, natural lighting",
    },
}

ENTITY_SUFFIX = ("Photorealistic, sharp, consistent with the reference images for identity. Show only the "
                 "persistent appearance; no motion, no text, no borders, no collage.")

COMPOSE_SUFFIX = {
    "G": ("Photorealistic memory frame for a video model: the listed characters and objects together in a "
          "neutral, well-lit setting, full bodies visible, neutral poses, identities exactly as in the reference "
          "images. 16:9, no text, no borders, no collage."),
    "M": ("Photorealistic memory frame for a video model: the location from the scene reference, with the listed "
          "characters and objects placed naturally in it, neutral poses, identities and states exactly as in the "
          "reference images. 16:9, no text, no borders, no collage."),
}


Change = Tuple[str, List[str], Optional[str]]  # (attribute, old values shown by the references, new value or None)


def _attr(attr: str) -> str:
    return attr.replace("_", " ")


def change_text(changes: List[Change]) -> str:
    """For specifications and the checker: "facial hair: neatly groomed short beard -> clean-shaven"."""
    parts = []
    for attr, old, new in changes:
        before = " / ".join(old) if old else "not shown"
        parts.append(f"{_attr(attr)}: {before} -> {new if new is not None else 'no longer applies'}")
    return "; ".join(parts)


def _changes_text(changes: List[Change]) -> str:
    if not changes:
        return ""
    return ("\nThe reference images show an earlier state. Change it exactly as follows and do not copy the old "
            "state from the references: " + change_text(changes) + ". Keep the identity (face, build, hair colour, "
            "species) from the references.")


def entity_spec(name: str, appearance: str, view: str, kind: str, changes: List[Change] = ()) -> str:
    style = VIEW_STYLE.get(kind, VIEW_STYLE["object"]).get(view, "")
    return f"{name}. {appearance}{_changes_text(list(changes))}\nView: {style}. {ENTITY_SUFFIX}"


def entity_edit(name: str, changes: List[Change], view: str, kind: str) -> str:
    """Generation prompt for a redraw whose state differs from the latest entity image: an edit of that image.
    FLUX.2-klein copies the references' state from a full description (a removed beard stays) but follows an edit
    instruction."""
    steps = []
    for attr, old, new in changes:
        if new is None:
            steps.append(f"remove the {' / '.join(old)}")
        elif old:
            steps.append(f"change the {_attr(attr)} from \"{' / '.join(old)}\" to \"{new}\"")
        else:
            steps.append(f"add {new}")
    style = VIEW_STYLE.get(kind, VIEW_STYLE["object"]).get(view, "")
    return (f"Edit the reference image of {name}: " + "; ".join(steps) + ". Apply every change completely and "
            f"visibly. Keep the identity (face, hair, build, species), every other part of the appearance and the "
            f"view ({style}) unchanged. Photorealistic, no text, no borders.")


# Neutral G (agent.neutral_g, default): G shows only its members on a plain light-grey studio backdrop, without the
# planner's scene description, and its caption is the member/state text plus the backdrop. A G drawn inside a scene
# carries that place into every later shot (the generator copies it as a location); the neutral backdrop leaves the
# place to M and the prompt, and on MiniMax-H3 generators the backdrop tokens are dropped entirely (foreground G).
NEUTRAL_G_LEAD = ("The listed characters and objects together, side by side against a plain seamless light-grey "
                  "studio backdrop.")
NEUTRAL_G_SUFFIX = ("Photorealistic memory frame for a video model: only the listed characters and objects on a plain, "
                    "seamless light-grey studio background, evenly lit; no room, no building, no scenery, no furniture, "
                    "no landscape, no sky and no other location cues. Full bodies visible, neutral standing poses, "
                    "identities exactly as in the reference images. 16:9, no text, no borders, no collage.")
NEUTRAL_G_CAPTION_TAIL = "Shown together against a plain light-grey studio background."


def compose_spec(role: str, text: str, members: List[str], changes: List[Change] = (), neutral_g: bool = False) -> str:
    member_text = "; ".join(members)
    if role == "G" and neutral_g:
        return f"{NEUTRAL_G_LEAD}\nInclude: {member_text}.{_changes_text(list(changes))}\n{NEUTRAL_G_SUFFIX}"
    return f"{text}\nInclude: {member_text}.{_changes_text(list(changes))}\n{COMPOSE_SUFFIX[role]}"


CAST_SPLIT_SYSTEM = """TASK: cast_split
You prepare the memory of a long video story. Each cast card describes a character at the start of the
story. During the story anything worn or groomed can change (a shave, a change of clothes, glasses
taken off). Split every card into
- "identity": what never changes, as one sentence that keeps the card's wording: the name, age, sex,
  face, skin, build, eye colour, hair colour and texture, species, fur or skin pattern. The identity
  must NOT mention facial hair (beard, moustache, goatee, stubble), glasses, clothing, footwear,
  accessories, jewellery, collars, carried items or how the hair is worn;
- "facts": exactly those changeable things, as snake_case attributes with the card's values:
  "facial_hair", "glasses", "clothing" (the whole outfit incl. footwear), "headwear", "hairstyle"
  (bun, braid, pinned up; not the colour), "jewellery", "collar", "bag", "carried_items", make-up, stains.
  Only what the card states: never a value like "none", no attribute twice.
Example card: "Sam, a man in his forties with short grey hair, a thick moustache and wire glasses,
wearing a blue suit and a red tie." ->
{"id": "sam", "identity": "Sam, a man in his forties with short grey hair.",
 "facts": {"facial_hair": "thick moustache", "glasses": "wire glasses", "clothing": "blue suit and red tie"}}
Output JSON only: {"cast": [{"id": str, "identity": str, "facts": {"attribute": "value"}}]}"""

# words that must not stay in an identity description (they name changeable appearance)
CHANGEABLE = re.compile(
    r"\b(beards?|bearded|moustaches?|mustaches?|goatee|stubble|sideburns|glasses|spectacles|sunglasses|wearing|wears|"
    r"dressed|shirt|t-shirt|tee|blouse|sweater|jumper|cardigan|hoodie|jacket|raincoat|apron|dress|skirt|jeans|"
    r"trousers|chinos|leggings|shorts|suit|uniform|boots|shoes|sneakers|loafers|hat|cap|beanie|helmet|bandana|"
    r"scarf|collar|earrings?|necklace|bracelet|backpack|bun|braid|ponytail)\b", re.I)


def cast_split_user(cast: List[Dict[str, str]]) -> str:
    return "### CAST CARDS\n" + json.dumps(cast, ensure_ascii=False)
