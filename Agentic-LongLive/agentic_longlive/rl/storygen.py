# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Story generation with exact hidden state (programmatic state tracking), for RL training data.

Stage A (one LLM call): a story bible (characters / objects with an unchanging identity and changeable initial
attributes, locations) and a beat sheet (per shot: location, present entities, explicit state changes, beat, camera,
momentary details). The hidden per-shot state is then computed by code from the initial attributes and the changes,
so the annotation is exact by construction (no LLM bookkeeping of state).
Stage B (chunked LLM calls): the full prompt (restates location look, appearance and current persistent state) and the
terse prompt (states things once; later only names / "still at" / "back at", changes narrated as actions) per shot.

Presets:

* ``train`` (default): 64 stories of 24-32 shots written by Qwen3.6-35B-A3B on an OpenAI-compatible server. Story
  ids ``tr050``-``tr057`` are the development split, the other 56 are training stories. Themes avoid the topics of the
  held-out test stories.
* ``hard``: an entity-rich held-out set (16 stories of 38-42 shots, written by Gemini, a different model family, on
  disjoint themes) with more memory pressure: 2-3 supporting characters, 9-12 recurring objects, at least 5 entities
  that leave for 5+ shots and return, 2-4 recurring entities per shot.

Every output directory gets ``full/`` and ``terse/`` story files and a sha256 ``manifest.json``.

  python -m agentic_longlive.rl.storygen --urls http://127.0.0.1:31000/v1 --out data/rl/stories
  python -m agentic_longlive.rl.storygen --preset hard --out data/rl/hard   # needs Gemini credentials
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .common import Endpoints, chat, log, parse_json

# chat_fn(system, user, temperature, max_tokens, seed) -> reply text (one JSON object)
ChatFn = Callable[[str, str, float, int, Optional[int]], str]

THEMES = [
    "a potter throws, trims, glazes and kiln-fires a set of bowls for a weekend market stall",
    "a bike mechanic rebuilds a rusty vintage road bike for a customer's charity race",
    "a beekeeper inspects hives through a summer, harvests honey and sells jars at a county fair",
    "a teenage skateboarder and two friends build a plywood ramp and enter a neighbourhood contest",
    "a wedding florist builds a bridal bouquet and a flower arch during one frantic wedding day",
    "a marine biologist rescues a tangled sea turtle, nurses it at a station and releases it",
    "a boatbuilder builds a wooden canoe from cedar strips to its first launch on a lake",
    "a street artist paints a huge mural over a week, a rainstorm damages it, she repairs it",
    "a firefighter's long shift: drills, a kitchen fire call, a smoky cleanup and a kitten rescue",
    "a luthier carves, varnishes and strings a violin for a young violinist's first concert",
    "a family restores a collapsing barn on an old farm they just moved to",
    "a tailor makes a three-piece suit from fabric selection to the final fitting",
    "a barista trains for a latte-art championship: practice, a spilled pitcher, the final round",
    "a child raises a duckling from a hatching egg to its first swim in a pond",
    "a clockmaker repairs a broken town-hall tower clock and its bell",
    "a farmer's harvest week: a combine breakdown, a hailstorm, and the grain finally in the silo",
    "an ice sculptor carves a swan for a winter festival; a warm afternoon melts the wings, he rebuilds them",
    "a stage actor prepares for opening night: costume fittings, a torn cape repaired backstage, the bow",
    "a volcanologist collects lava samples, her boots scorched and her jacket ash-covered, then analyses them",
    "a fisherman mends his nets and boat, fishes at dawn and returns through a squall",
    "a glassblower makes a blue-and-gold vase in a hot studio, one attempt cracks, the second succeeds",
    "a youth football coach trains a muddy under-12 team through a rainy season to a final",
    "a blacksmith forges, quenches and polishes a ceremonial sword for a festival",
    "a cake designer builds a three-tier cake that is bumped and damaged, then repaired before the party",
    "a girl and her grandfather restore a rusted 1960s pickup truck to drive it to a fair",
    "a rural veterinarian treats a lame horse, a goat with a cut ear, and a parrot over one day",
    "an archaeologist excavates a painted clay jar, cleans it in a field tent and installs it in a museum",
    "a hot-air balloon pilot inflates, flies over farmland, makes a rough landing and repacks",
    "a grandmother sews a patchwork quilt for a newborn over several evenings",
    "a ski instructor teaches a nervous beginner; a crash snaps a pole and tears a glove",
    "a teenager builds a racing drone, crashes it into a tree and rebuilds it for a race",
    "a lighthouse keeper repaints the tower and repairs the lamp before a winter storm",
    "a garden designer turns an overgrown backyard into a garden with a pond and a pergola",
    "an indie rock band rehearses in a garage; a guitar string snaps and the drummer's stick breaks; gig night",
    "a wildlife photographer builds a hide in a forest and waits days to photograph a fox family",
    "a fantasy squire travels from a castle to a dragon's cave to recover a stolen golden chalice",
    "a sailor repairs a ripped sail and competes in a coastal regatta",
    "a circus acrobat rehearses, sprains her wrist, has it taped and performs the finale",
    "a bookbinder restores a water-damaged antique atlas page by page",
    "a father builds a treehouse with his two kids over a summer",
    "a wizard's apprentice brews potions; one explodes, singeing her robe and turning her hair silver",
    "an ice-hockey goalie's game day: new pads, a cracked mask replaced, a broken stick, overtime",
    "a museum conservator restores a damaged oil painting of a ship",
    "a shepherd moves a flock across a valley in spring; a lamb is born on the way",
    "a jeweller designs, casts and sets a sapphire engagement ring",
    "a scuba diver recovers a brass bell from a shipwreck and cleans it at the dock",
    "a street magician prepares props and performs at a night festival; a trick goes wrong",
    "an origami artist folds a giant paper crane installation for a gallery opening",
    "a rancher repairs storm-damaged fences and rescues a calf stuck in a creek",
    "a ballet student prepares for a recital: new pointe shoes, a ripped ribbon sewn back, the performance",
    "a mountain-bike trail builder digs and shapes a new jump line through a pine forest",
    "a wildlife ranger tracks and fits a radio collar on an injured wolf, then releases it",
    "a calligrapher hand-letters wedding invitations; ink spills on her sleeve and a stack is redone",
    "a dog-sled musher prepares the team, races through snow and reaches the finish at night",
    "an astronomer builds a backyard observatory dome and photographs a comet",
    "a fashion student sews a gown for a runway show; the hem is torn on the catwalk rehearsal",
    "a costume maker builds a giant dragon puppet for a lantern parade",
    "a lifeguard's beach day: towers, a rescue, sunburn, a lost child reunited",
]

HARD_THEMES = [
    "a hair salon's busy Saturday: three clients get dramatic haircuts and colour changes, a dryer breaks, a wedding "
    "party arrives",
    "a family moves house: packing a cluttered flat, a lamp breaks in the van, unpacking and painting the kid's new "
    "room",
    "a community theatre's costume department builds, damages and repairs costumes for a pirate musical",
    "a ferry crew's day crossing a fjord: a car spills paint on deck, a storm soaks everyone, a passenger's dog goes "
    "missing",
    "a vintage camera repairer restores three broken cameras for three different customers over a week",
    "a zookeeper bathes an elephant, repaints the giraffe shelter and nurses a sick penguin back to health",
    "a mountain rescue team searches a snowy ridge for a lost hiker, treats his injuries and carries him down",
    "a toy maker builds a set of wooden trains and a station for a children's hospital",
    "four friends on a rainy camping trip: a torn tent, a lost lantern, a muddy hike and a campfire night",
    "a car-detailing shop restores a filthy vintage convertible and a dented delivery van for two owners",
    "a school science fair: two students build a baking-soda volcano and a solar car; one erupts early, the car wins",
    "a glacier field team drills ice cores, a sled tips over in a storm, they rebuild camp and fly home",
    "a small-town post office over the holidays: snowed-in routes, a torn parcel, a lost letter finally delivered",
    "a film crew shoots a period drama in a manor: costume changes, a fake-blood mishap, a broken prop sword",
    "a horse groom prepares two horses and a carriage for a parade; a wheel cracks and is replaced",
    "a puppeteer builds marionettes; one's strings snap mid-show and she repairs it for the encore",
]

BIBLE_SYSTEM = """You design long live-streamed video stories (one shot of about 10 seconds per beat). Each story is
used to test a visual memory system, so it must contain recurring characters and objects whose PERSISTENT state changes
over time and must be remembered across many shots. Output one JSON object and nothing else."""

BIBLE_USER = """Premise: {theme}

Design a story of exactly {n} shots. Requirements:
- 1 protagonist (character), 1-2 supporting characters, 6-9 recurring objects, 3-5 locations.
- Every character/object has an "identity": its UNCHANGING look (face, age, build, skin, hair colour, species; for
  objects their material, colour, shape). Changeable appearance is NOT identity.
- Every character/object has "initial": its changeable persistent attributes when it first appears, as
  {{"attr": "value"}} with snake_case attribute names. Attributes must be independent (never two attributes that
  describe the same property):
  * characters: 2-4 of outfit, footwear, headwear, accessory, hairstyle, facial_hair, stains, injury. Wearable items
    (aprons, gloves, glasses, hats, bags worn) are character attributes, never separate objects;
  * objects: exactly "state" (ONE phrase with the whole current look: stage, condition, colour, contents) and
    optionally "placement" (where it stands). A change to an object rewrites its whole "state" phrase
    ("glazed glossy blue, fired, cool"), so the phrase is always complete and current.
  Values are short concrete visual descriptions ("navy wool coat over a white shirt", "trimmed, leather-hard, grey").
- "shots": for each shot give location id, the ids of the characters/objects visible in it, explicit persistent state
  "changes" that HAPPEN IN THAT SHOT and stay afterwards ({{"entity": id, "attr": attr, "value": new value}}), a one- or
  two-sentence "beat", a "camera" phrase, and "momentary": what is only momentary in this shot (held items, pose,
  expression, action) and does not persist.
- Memory pressure (all required):
  * the protagonist has at least 4 persistent changes spread over the story (change outfit, get a stain or injury, put
    on / take off an accessory, haircut ...), each persisting for at least 3 shots;
  * objects have at least 5 persistent changes in total (assembly or work stage, damage, repair, filled / emptied,
    painted ...);
  * at least 2 locations are revisited after at least 4 shots elsewhere;
  * at least 3 characters or objects are absent for at least 5 consecutive shots and then return (in their last state);
  * at least 7 recurring non-location entities appear in 2 or more shots;
  * shots usually show only 1-3 of the recurring entities.
- Never use these topics: graphics cards / PC building, restaurant chefs, muddy dogs, detectives, robots, space
  stations, kites, chemistry crystals.
Return JSON:
{{"title": str, "style": "one sentence visual style",
  "characters": [{{"id": "snake_case", "name": str, "protagonist": bool, "identity": str, "initial": {{}}}}],
  "objects": [{{"id": "snake_case", "name": str, "identity": str, "initial": {{}}}}],
  "locations": [{{"id": "snake_case", "name": str, "look": "visual description of the place"}}],
  "shots": [{{"location": id, "present": [ids], "changes": [], "beat": str, "camera": str, "momentary": str}}]}}"""

HARD_BIBLE_USER = BIBLE_USER.replace(
    "- 1 protagonist (character), 1-2 supporting characters, 6-9 recurring objects, 3-5 locations.",
    "- 1 protagonist (character), 2-3 supporting characters (people or animals), 9-12 recurring objects, "
    "4-6 locations.",
).replace(
    "  * at least 3 characters or objects are absent for at least 5 consecutive shots and then return (in their "
    "last state);",
    "  * at least 5 characters or objects are absent for at least 5 consecutive shots and then return (in their last\n"
    "    state), and at least 2 of them after 8 or more shots away;",
).replace(
    "  * at least 7 recurring non-location entities appear in 2 or more shots;",
    "  * at least 10 recurring non-location entities appear in 2 or more shots; supporting characters also change\n"
    "    persistently (at least 3 changes in total);",
).replace(
    "  * shots usually show only 1-3 of the recurring entities.",
    "  * shots usually show 2-4 of the recurring entities.",
)
assert HARD_BIBLE_USER != BIBLE_USER

PROMPT_SYSTEM = """You write shot prompts for a text-to-video model in a live stream. You write two versions of each
shot prompt. Output one JSON object and nothing else."""

PROMPT_USER = """Story: {title}. Style: {style}
Locations: {locations}
Entities (identity = unchanging look): {entities}

Write prompts for shots {first}-{last}. For each shot you get the location, which location visit it is, the entities
present with their persistent state AT THE START of the shot, the changes that happen during the shot, the beat, the
camera and momentary details.
{shots}

FULL version (60-140 words, three short paragraphs: place, action, camera): describe the place's look; describe every
present character by name with identity AND full current persistent state (clothing, accessories, stains ...), and
every present object with its current state; narrate the beat, the changes as visible actions and the momentary
details; then the camera.
TERSE version (35-90 words): like a real user who does not repeat what was already shown.
- Place: on the FIRST visit describe the look; when continuing at the same place write only "Still at the <name>.";
  on a revisit write only "Back at the <name>."
- A character or object appearing for the FIRST time in the story is described fully (identity and state); after that
  use only its name: never restate its appearance, clothing or persistent state.
- Changes happening in this shot are narrated as actions ("she pulls on a yellow raincoat", "the glaze cracks").
- Keep the beat, momentary details and camera.
Return {{"shots": [{{"shot": index, "full": str, "terse": str}}]}} covering every listed shot in order."""


_NONE = {"none", "n/a", "na", "null", "nothing", "no", "-", ""}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "x"


def validate_bible(b: Dict[str, Any], n: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Normalise ids, check references and memory-pressure counts; return (clean bible, problems)."""
    problems: List[str] = []
    if not isinstance(b, dict):
        return None, ["not a JSON object"]
    ents: Dict[str, Dict[str, Any]] = {}
    for kind, key in (("character", "characters"), ("object", "objects")):
        for e in b.get(key) or []:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            eid = _slug(e["id"])
            init = {(_slug(k)): str(v).strip() for k, v in (e.get("initial") or {}).items()
                    if str(v).strip() and str(v).strip().lower() not in _NONE}
            ents[eid] = {"id": eid, "kind": kind, "name": str(e.get("name") or eid),
                         "identity": str(e.get("identity", "")), "initial": init,
                         "protagonist": bool(e.get("protagonist")) and kind == "character"}
    locs = {}
    for loc in b.get("locations") or []:
        if isinstance(loc, dict) and loc.get("id"):
            lid = _slug(loc["id"])
            locs[lid] = {"id": lid, "name": str(loc.get("name") or lid), "look": str(loc.get("look", ""))}
    if not [e for e in ents.values() if e["protagonist"]]:
        chars = [e for e in ents.values() if e["kind"] == "character"]
        if chars:
            chars[0]["protagonist"] = True
        else:
            problems.append("no character")
    shots = []
    for s in (b.get("shots") or [])[:n]:
        if not isinstance(s, dict):
            continue
        loc = _slug(s.get("location", ""))
        present = [p for p in dict.fromkeys(_slug(x) for x in (s.get("present") or [])) if p in ents]
        changes = []
        for c in s.get("changes") or []:
            if (isinstance(c, dict) and _slug(c.get("entity", "")) in ents and c.get("attr")
                    and str(c.get("value", "")).strip()):
                value = str(c["value"]).strip()
                if value.lower() in _NONE:
                    value = f"no {_slug(c['attr']).replace('_', ' ')} (removed)"
                changes.append({"entity": _slug(c["entity"]), "attr": _slug(c["attr"]), "value": value})
        for c in changes:  # an entity that changes is visible
            if c["entity"] not in present:
                present.append(c["entity"])
        if loc not in locs:
            problems.append(f"unknown location {loc}")
            continue
        shots.append({"location": loc, "present": present, "changes": changes, "beat": str(s.get("beat", "")),
                      "camera": str(s.get("camera", "")), "momentary": str(s.get("momentary", ""))})
    if len(shots) < n - 2:
        problems.append(f"only {len(shots)} valid shots of {n}")
    clean = {"title": str(b.get("title", "")), "style": str(b.get("style", "")), "entities": ents, "locations": locs,
             "shots": shots}
    if shots:
        stats = pressure_stats(clean)
        if stats["protagonist_changes"] < 3:
            problems.append(f"protagonist has {stats['protagonist_changes']} persistent changes (need 4)")
        if stats["object_changes"] < 4:
            problems.append(f"objects have {stats['object_changes']} persistent changes (need 5)")
        if stats["location_revisits"] < 2:
            problems.append(f"{stats['location_revisits']} location revisits after >=4 shots (need 2)")
        if stats["entity_returns"] < 2:
            problems.append(f"{stats['entity_returns']} entities return after >=5 absent shots (need 3)")
        if stats["recurring"] < 6:
            problems.append(f"{stats['recurring']} recurring entities (need 7)")
    bad = [e["id"] for e in ents.values() if e["kind"] == "object" and set(e["initial"]) - {"state", "placement"}]
    bad += [c["entity"] for s in shots for c in s["changes"]
            if ents[c["entity"]]["kind"] == "object" and c["attr"] not in ("state", "placement")]
    if bad:
        problems.append(f"objects may only use the attributes state and placement: {sorted(set(bad))[:6]}")
    return clean, problems


def validate_hard(b: Dict[str, Any], n: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    clean, problems = validate_bible(b, n)
    if clean is not None and clean["shots"]:
        st = pressure_stats(clean)
        if st["entity_returns"] < 4:
            problems.append(f"{st['entity_returns']} entities return after >=5 absent shots (need 5)")
        if st["recurring"] < 9:
            problems.append(f"{st['recurring']} recurring entities (need 10)")
    return clean, problems


def pressure_stats(b: Dict[str, Any]) -> Dict[str, int]:
    shots, ents = b["shots"], b["entities"]
    prot = {e for e, v in ents.items() if v["protagonist"]}
    pc = sum(1 for s in shots for c in s["changes"] if c["entity"] in prot)
    oc = sum(1 for s in shots for c in s["changes"] if ents[c["entity"]]["kind"] == "object")
    revisits, last_loc = 0, {}
    for j, s in enumerate(shots):
        loc = s["location"]
        if loc in last_loc and j - last_loc[loc] > 4:
            revisits += 1
        last_loc[loc] = j
    returns, last_seen = 0, {}
    for j, s in enumerate(shots):
        for e in s["present"]:
            if e in last_seen and j - last_seen[e] > 5:
                returns += 1
            last_seen[e] = j
    counts: Dict[str, int] = {}
    for s in shots:
        for e in s["present"]:
            counts[e] = counts.get(e, 0) + 1
    return {"protagonist_changes": pc, "object_changes": oc, "location_revisits": revisits, "entity_returns": returns,
            "recurring": sum(1 for c in counts.values() if c >= 2)}


def track_state(b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Exact hidden annotation per shot: state at the END of the shot for every entity seen so far."""
    ents = b["entities"]
    state: Dict[str, Dict[str, str]] = {}
    seen_loc: Dict[str, int] = {}
    out = []
    prev_loc = None
    for j, s in enumerate(b["shots"]):
        start = {e: dict(v) for e, v in state.items()}
        first = []
        for e in s["present"]:
            if e not in state:
                state[e] = dict(ents[e]["initial"])
                start[e] = dict(state[e])
                first.append(e)
        for c in s["changes"]:
            state.setdefault(c["entity"], dict(ents[c["entity"]]["initial"]))[c["attr"]] = c["value"]
        loc = s["location"]
        visit = "same" if loc == prev_loc else ("revisit" if loc in seen_loc else "new")
        seen_loc.setdefault(loc, j)
        prev_loc = loc
        prot = [e for e in ents if ents[e]["protagonist"]]
        flat = {}
        for e in prot + [x for x in s["present"] if x not in prot]:
            for a, v in (state.get(e) or {}).items():
                flat[f"{e}.{a}"] = v
        out.append({"title_en": s["beat"][:60], "scene": loc, "camera": s["camera"], "beat": s["beat"],
                    "present": list(s["present"]), "first_appearance": first, "changes": s["changes"],
                    "location_visit": visit, "momentary": s["momentary"],
                    "start_state": {e: start[e] for e in s["present"] if e in start},
                    "entity_state": {e: dict(v) for e, v in state.items()}, "state": flat})
    return out


def _entity_line(e: Dict[str, Any]) -> str:
    return f'{e["id"]} ({e["kind"]}, name "{e["name"]}"): {e["identity"]}'


def write_prompts(chat_fn: ChatFn, b: Dict[str, Any], ann: List[Dict[str, Any]], chunk: int = 6
                  ) -> Tuple[List[str], List[str]]:
    ents, locs = b["entities"], b["locations"]
    full: List[Optional[str]] = [None] * len(b["shots"])
    terse: List[Optional[str]] = [None] * len(b["shots"])
    loc_text = "; ".join(f'{loc["id"]} = "{loc["name"]}": {loc["look"]}' for loc in locs.values())
    ent_text = "; ".join(_entity_line(e) for e in ents.values())
    for first in range(0, len(b["shots"]), chunk):
        idx = list(range(first, min(len(b["shots"]), first + chunk)))
        lines = []
        for j in idx:
            s, a = b["shots"][j], ann[j]
            pres = []
            for e in s["present"]:
                tag = " [FIRST APPEARANCE]" if e in a["first_appearance"] else ""
                state = json.dumps(a["start_state"].get(e, {}), ensure_ascii=False)
                pres.append(f'{ents[e]["name"]}{tag} state {state}')
            ch = "; ".join(f'{ents[c["entity"]]["name"]}.{c["attr"]} -> {c["value"]}' for c in s["changes"]) or "none"
            lines.append(f'[shot {j}] location {locs[s["location"]]["name"]} ({a["location_visit"]} visit); '
                         f'present: {" | ".join(pres) or "nobody"}; changes: {ch}; beat: {s["beat"]}; '
                         f'camera: {s["camera"]}; momentary: {s["momentary"]}')
        user = PROMPT_USER.format(title=b["title"], style=b["style"], locations=loc_text, entities=ent_text,
                                  first=idx[0], last=idx[-1], shots="\n".join(lines))
        for _ in range(3):
            data = parse_json(chat_fn(PROMPT_SYSTEM, user, 0.6, 8192, None)) or {}
            got = {int(x.get("shot", -1)): x for x in data.get("shots") or [] if isinstance(x, dict)}
            if all(j in got and str(got[j].get("full", "")).strip() and str(got[j].get("terse", "")).strip()
                   for j in idx):
                for j in idx:
                    full[j], terse[j] = str(got[j]["full"]).strip(), str(got[j]["terse"]).strip()
                break
        else:
            raise RuntimeError(f"prompt chunk {idx[0]}-{idx[-1]} failed")
    return [x or "" for x in full], [x or "" for x in terse]


def cast_card(e: Dict[str, Any]) -> str:
    state = ", ".join(f"{v}" for v in e["initial"].values())
    return f'{e["name"]}, {e["identity"].rstrip(". ")}' + (f", {state}." if state else ".")


def collapse_objects(b: Dict[str, Any]) -> None:
    """Objects keep one holistic ``state`` phrase (+ ``placement``): merge any extra attributes into it, in place."""
    for eid, e in b["entities"].items():
        if e["kind"] != "object":
            continue
        running = {a: v for a, v in e["initial"].items() if a != "placement"}
        if set(running) - {"state"}:
            e["initial"] = {"state": "; ".join(running.values()),
                            **({"placement": e["initial"]["placement"]} if "placement" in e["initial"] else {})}
        for s in b["shots"]:
            merged = []
            for c in s["changes"]:
                if c["entity"] != eid or c["attr"] in ("placement",):
                    merged.append(c)
                    continue
                running[c["attr"]] = c["value"]
                value = c["value"] if c["attr"] == "state" and len(running) == 1 else "; ".join(running.values())
                merged = [x for x in merged if not (x["entity"] == eid and x["attr"] == "state")]
                merged.append({"entity": eid, "attr": "state", "value": value})
            s["changes"] = merged


@dataclass(frozen=True)
class Preset:
    themes: Sequence[str]
    prefix: str
    digits: int
    n_stories: int
    shots: int
    jitter: Sequence[int]
    seed: int
    model: str
    bible_user: str
    validate: Callable[[Dict[str, Any], int], Tuple[Optional[Dict[str, Any]], List[str]]]


PRESETS = {
    "train": Preset(themes=THEMES, prefix="tr", digits=3, n_stories=64, shots=28, jitter=(-4, -2, 0, 2, 4),
                    seed=20261029, model="Qwen/Qwen3.6-35B-A3B-FP8", bible_user=BIBLE_USER, validate=validate_bible),
    "hard": Preset(themes=HARD_THEMES, prefix="hd", digits=2, n_stories=len(HARD_THEMES), shots=40, jitter=(-2, 0, 2),
                   seed=20261030, model="gemini-3.8-flash", bible_user=HARD_BIBLE_USER, validate=validate_hard),
}


def make_story(chat_fn: ChatFn, theme: str, sid: str, n: int, seed: int, preset: str = "train",
               model: str = "") -> Dict[str, Any]:
    """{"full": story, "terse": story} in the unified story format, with hidden annotations and the bible."""
    p = PRESETS[preset]
    user = p.bible_user.format(theme=theme, n=n)
    problems: List[str] = []
    bible = None
    for attempt in range(4):
        prompt = user if not problems else user + "\n\nYour previous design had problems: " + "; ".join(problems) + \
            ". Fix them and return the whole JSON again."
        bible, problems = p.validate(parse_json(chat_fn(BIBLE_SYSTEM, prompt, 0.8, 12000, seed + attempt)) or {}, n)
        if bible is not None and not problems:
            break
    if bible is None or (problems and len(bible["shots"]) < n - 2):
        raise RuntimeError(f"bible failed: {problems}")
    collapse_objects(bible)
    ann = track_state(bible)
    full, terse = write_prompts(chat_fn, bible, ann)
    ents = bible["entities"]
    cast = [{"id": e["id"], "name": e["name"], "description": cast_card(e)} for e in ents.values() if e["protagonist"]]
    base = {"source": {"generator": f"agentic_longlive.rl.storygen ({preset})", "model": model or p.model,
                       "theme": theme, "seed": seed, "residual_problems": problems, "pressure": pressure_stats(bible)},
            "setup": {"title": bible["title"], "style": bible["style"], "cast": cast},
            "bible": {"entities": ents, "locations": bible["locations"]}}
    out = {}
    for variant, prompts in (("full", full), ("terse", terse)):
        out[variant] = {"id": f"{sid}_{variant}", **base,
                        "shots": [{"prompt": pr, "annotation": a} for pr, a in zip(prompts, ann)]}
    return out


def story_jobs(preset: str = "train", n_stories: Optional[int] = None, shots: Optional[int] = None,
               seed: Optional[int] = None) -> List[Tuple[str, str, int, int]]:
    """(theme, story id, number of shots, seed) per story; ids and seeds are deterministic."""
    p = PRESETS[preset]
    n_stories = p.n_stories if n_stories is None else n_stories
    shots = p.shots if shots is None else shots
    seed = p.seed if seed is None else seed
    rng = random.Random(seed)
    jobs = []
    for k in range(n_stories):
        theme = p.themes[k % len(p.themes)]
        sid = f"{p.prefix}{k:0{p.digits}d}_" + hashlib.sha1(theme.encode()).hexdigest()[:6]
        jobs.append((theme, sid, shots + rng.choice(list(p.jitter)), seed + 1000 * k))
    return jobs


def openai_chat_fn(urls: str, model: str) -> ChatFn:
    eps = Endpoints(urls)

    def call(system: str, user: str, temperature: float, max_tokens: int, seed: Optional[int]) -> str:
        return chat(eps, model, system, user, temperature=temperature, max_tokens=max_tokens, seed=seed)["texts"][0]
    return call


def gemini_chat_fn(model: str) -> ChatFn:
    """Gemini (Vertex AI) through the agent's model registry; JSON mode, at least 16k output tokens, no seed."""
    from .score import gemini_client

    clients: Dict[int, Any] = {}

    def call(system: str, user: str, temperature: float, max_tokens: int, seed: Optional[int]) -> str:
        budget = max(max_tokens, 16000)
        if budget not in clients:
            clients[budget] = gemini_client(model, max_tokens=budget, max_concurrency=32)
        r = clients[budget].chat(system, [user], json_mode=True, temperature=temperature)
        return r.text if r.text else json.dumps(r.data or {})
    return call


def write_manifest(out: Path) -> None:
    manifest = {f"{v}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                for v in ("full", "terse") for p in sorted((out / v).glob("*.json"))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="train", choices=sorted(PRESETS))
    ap.add_argument("--urls", default="", help="OpenAI-compatible base URLs of the writer LLM (train preset)")
    ap.add_argument("--model", default="", help="writer model (default: the preset's)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stories", type=int, default=None)
    ap.add_argument("--shots", type=int, default=None, help="mean shots per story")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    p = PRESETS[args.preset]
    model = args.model or p.model
    if model.startswith("gemini"):
        chat_fn = gemini_chat_fn(model)
    elif args.urls:
        chat_fn = openai_chat_fn(args.urls, model)
    else:
        raise SystemExit("--urls is required for an OpenAI-compatible writer model")
    out = Path(args.out)
    (out / "full").mkdir(parents=True, exist_ok=True)
    (out / "terse").mkdir(parents=True, exist_ok=True)
    jobs = story_jobs(args.preset, args.n_stories, args.shots, args.seed)

    def run(job):
        theme, sid, shots, seed = job
        if (out / "terse" / f"{sid}.json").is_file():
            return sid, "exists"
        story = make_story(chat_fn, theme, sid, shots, seed, args.preset, model)
        for variant in ("full", "terse"):
            (out / variant / f"{sid}.json").write_text(json.dumps(story[variant], indent=1, ensure_ascii=False))
        return sid, json.dumps(story["terse"]["source"]["pressure"])

    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                sid, info = f.result()
                log(f"story {sid}: {info}", out / "gen.log")
            except Exception as exc:  # noqa: BLE001
                log(f"story {futs[f][1]} FAILED: {exc!r}", out / "gen.log")
    write_manifest(out)


if __name__ == "__main__":
    main()
