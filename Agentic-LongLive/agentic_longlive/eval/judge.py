# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM judges of LongStory-Online (paper Sec. 4 "Metrics" and Supp. "Benchmark and Metrics").

Four judge calls, all at temperature 0 with JSON output; the system prompts below are the ones that produced the
paper's numbers and must not be edited (a changed prompt is a different benchmark version):

* shot judge: four frames of one shot vs. its hidden annotation -> location, framing, content, protagonist identity,
  persistent-state facts, severe artifacts;
* place pair judge: two frames of a revisiting shot and two of the last earlier shot at the same planned place -> same
  place (1-5);
* protagonist pair judge: the four shot-judge frames of a shot and of the method's first shot of the protagonist ->
  same individual (1-5);
* plan visibility (text only, method independent): does the hidden plan of a shot show the protagonist recognisably?
  Precomputed for the benchmark in ``stories/longstory_online/eval/expected_visible.json``.

The judge must differ from the models that produced the video (agent, RL reward models). The paper uses Gemini 3.8
Flash (:func:`judge_spec` ``"gemini"``); any OpenAI-compatible VLM endpoint works as well (``"openai"``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import ModelSpec
from ..models.base import LLMClient
from ..story import Story

SHOT_JUDGE_SYSTEM = """You evaluate one shot of an AI-generated long video against a hidden plan. You see FRAMES of the shot \
in time order (first frame, about 1/3, about 2/3, last frame). Judge only what is visible. Return JSON:
{"opening_location": str, "location": str, "shot_size": "extreme close-up|close-up|medium|wide",
 "framing_match": bool, "content": 0|1|2, "protagonist_visible": bool, "protagonist_match": bool|null,
 "facts": [{"key": str, "verdict": "confirmed|contradicted|not_visible"}], "artifacts": bool, "notes": str}
- opening_location: where the FIRST frame takes place; location: where most of the shot takes place. Each is exactly \
one key from LOCATIONS, or "other".
- framing_match: true if the camera distance/angle and the main subject agree with PLANNED CAMERA and PLANNED \
CONTENT (e.g. a planned close-up of hands on a part must not be a medium shot of the person standing).
- content: 2 if the main subject and action of PLANNED CONTENT are clearly shown, 1 if only partly (right subject or \
place but the action is missing or different), 0 if not shown.
- protagonist_match: if the PROTAGONIST is visible, whether it matches the identity description (face, species, \
body, age; ignore clothing and accessories, which may change); null if not visible.
- facts: one entry per EXPECTED STATE key: "confirmed" if clearly visible and correct, "contradicted" if visible and \
different, "not_visible" otherwise.
- artifacts: true only for severe failures: a ghosted overlay or blend of two different images, melting or broken \
anatomy, frozen or corrupted frames."""

PAIR_JUDGE_SYSTEM = """You compare two shots, A and B, of an AI-generated story video. The story says both take place in the \
same location. You see two frames of shot A, then two frames of shot B. Judge only what is visible; a different camera \
angle, distance or time of day is expected and must not lower the score. Return JSON:
{"place": 1-5, "person": 1-5 or null, "shared": str, "differs": str}
- place: how consistent the place is. 5 = unmistakably the same place: specific furniture or landmarks, layout, \
materials and colours match; 4 = the same place with minor inconsistencies; 3 = same kind of place with a few specific \
matching details but clear differences; 2 = same kind of place, no specific details match; 1 = a different kind of \
place. If a shot is a close-up, compare whatever background, surfaces and props are visible.
- person: if the PROTAGONIST is visible in both shots, how consistent their identity is (face, hair, build, species \
and markings; clothing, dirt and accessories may legitimately change). 5 = clearly the same individual, 3 = similar but \
uncertain, 1 = clearly a different individual; null if not visible in both.
- shared / differs: the main matching and differing details (short)."""

PROTAGONIST_JUDGE_SYSTEM = """You compare the protagonist of a story across two shots, A and B, of an AI-generated video. You see four frames \
of shot A, then four frames of shot B (in time order). The shots may be far apart in the story, at different places, \
times of day and camera distances. Judge only what is visible. Return JSON:
{"visible_a": bool, "visible_b": bool, "person": 1-5 or null, "differs": str}
- visible_a / visible_b: the PROTAGONIST (as described) is clearly visible in that shot's frames.
- person: if visible in both, how consistent the protagonist's identity is: face, hair colour and texture, build, age, \
species, fur or skin markings. Clothing, dirt, wetness, accessories and pose may legitimately change and must not lower \
the score. 5 = clearly the same individual, 4 = same individual with minor inconsistencies, 3 = similar but uncertain, \
2 = probably a different individual, 1 = clearly a different individual; null if not visible in both.
- differs: the main identity differences (short)."""

VISIBILITY_JUDGE_SYSTEM = """You read the hidden plan of one shot of a story video: the planned camera, the planned content and the \
entities present. You do not see the video. Decide whether the plan shows the PROTAGONIST clearly enough to recognise \
them in this shot: their face, or for a non-human protagonist their head and body, is in frame for a substantial part \
of the shot. Answer false if the protagonist is absent or off-screen, seen only as hands or another body part, only \
from behind, only as a silhouette or a tiny distant figure, or if the camera concentrates on an object, a place or \
another character. Return JSON {"expected_visible": bool}."""


# ---------------------------------------------------------------------------------------------- judge models
def judge_spec(kind: str = "gemini", model: str = "", base_url: str = "", backend: str = "vertex",
               api_key_env: str = "", concurrency: int = 8, extra_body: Optional[Dict[str, Any]] = None) -> ModelSpec:
    """Model spec of a judge. Credentials are read from the environment, never passed on the command line.

    * ``gemini``: Gemini 3.8 Flash by default (``AGL_JUDGE_MODEL`` overrides). Vertex AI with the service-account JSON
      named by ``GEMINI_SERVICE_ACCOUNT`` (``backend="vertex"``) or the Gemini API with ``GEMINI_API_KEY``
      (``backend="studio"``).
    * ``openai``: any OpenAI-compatible chat endpoint with image input (vLLM / SGLang serving e.g. a Qwen VL model,
      OpenAI, ...). ``base_url`` (or ``AGL_JUDGE_URL``) and ``model`` (or ``AGL_JUDGE_MODEL``) are required; the API
      key, if any, is read from ``api_key_env`` (default ``OPENAI_API_KEY``).
    """
    params: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 2048, "max_image_side": 768}
    if kind == "gemini":
        params.update(backend=backend, location="global")
        return ModelSpec(provider="gemini", model=model or os.environ.get("AGL_JUDGE_MODEL", "gemini-3.8-flash"),
                         base_url=base_url, api_key_env=api_key_env or "GEMINI_API_KEY",
                         credentials_env="GEMINI_SERVICE_ACCOUNT", params=params, max_concurrency=concurrency,
                         timeout_s=60.0)
    if kind == "openai":
        url = base_url or os.environ.get("AGL_JUDGE_URL", "")
        name = model or os.environ.get("AGL_JUDGE_MODEL", "")
        if not url or not name:
            raise ValueError("the openai judge needs --judge-url/AGL_JUDGE_URL and --judge-model/AGL_JUDGE_MODEL")
        if extra_body:
            params["extra_body"] = dict(extra_body)
        return ModelSpec(provider="openai_compat", model=name, base_url=url, api_key_env=api_key_env or "OPENAI_API_KEY",
                         params=params, max_concurrency=concurrency, timeout_s=120.0)
    raise ValueError(f"unknown judge kind {kind!r} (gemini | openai)")


def spec_from_yaml(path: str, concurrency: int = 8) -> ModelSpec:
    """A judge spec from YAML: either a ModelSpec mapping, or a run/agent config whose ``agent.checker`` is used."""
    from omegaconf import OmegaConf

    from ..config import load_config

    raw = OmegaConf.load(path)
    if "agent" in raw or "defaults" in raw:
        spec = load_config(path).agent.checker
    else:
        spec = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ModelSpec), raw))
    spec.max_concurrency = concurrency
    return spec


def build_judge(spec: ModelSpec) -> LLMClient:
    from ..models.registry import build_llm

    return build_llm(spec)


# ---------------------------------------------------------------------------------------------- shot judge
def location_guide(story: Story) -> Dict[str, str]:
    """scene key -> the setting sentence (first sentence, at most 40 words) of the first prompt annotated with it."""
    guide: Dict[str, str] = {}
    for prompt, ann in zip(story.prompts, story.annotations):
        scene = (ann or {}).get("scene")
        if scene and scene not in guide:
            first = re.split(r"(?<=[.!?])\s+", prompt.strip(), maxsplit=1)[0]
            words = first.split()
            guide[scene] = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
    return guide


def _annotation(story: Story, i: int) -> Dict[str, Any]:
    return (story.annotations[i] if i < len(story.annotations) else {}) or {}


def protagonist(story: Story) -> Optional[str]:
    """The protagonist card: the description of the first cast member."""
    return story.setup.cast[0].description if story.setup.cast else None


def shot_judge_text(story: Story, i: int, guide: Optional[Dict[str, str]] = None) -> str:
    ann = _annotation(story, i)
    guide = location_guide(story) if guide is None else guide
    card = protagonist(story)
    return ("LOCATIONS:\n" + "\n".join(f"- {k}: {v}" for k, v in guide.items())
            + (f"\nPROTAGONIST: {card}" if card is not None else "")
            + f"\nPLANNED CAMERA: {ann.get('camera', '')}\nPLANNED CONTENT: {ann.get('beat', '')}"
            + "\nEXPECTED STATE:\n" + "\n".join(f"- {k}: {v}" for k, v in (ann.get("state") or {}).items()))


def judge_shot(judge: LLMClient, story: Story, i: int, frames: Sequence[Path],
               guide: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One shot-judge call; returns the normalised record stored under ``shots[i]["judge"]``."""
    guide = location_guide(story) if guide is None else guide
    result = judge.chat(SHOT_JUDGE_SYSTEM, [shot_judge_text(story, i, guide), *map(Path, frames)],
                        json_mode=True, temperature=0.0)
    data = result.data or {}

    def loc(value):
        said = str(value or "other").strip().lower().replace(" ", "_")
        return said if said in guide else "other"

    facts = {str(f.get("key")): str(f.get("verdict")) for f in data.get("facts") or [] if isinstance(f, dict)}
    content = data.get("content")
    return {"location": loc(data.get("location")), "opening_location": loc(data.get("opening_location")),
            "shot_size": data.get("shot_size"), "framing_match": bool(data.get("framing_match")),
            "content": int(content) if isinstance(content, (int, float)) else None,
            "protagonist_visible": data.get("protagonist_visible"), "protagonist_match": data.get("protagonist_match"),
            "facts": facts, "artifacts": bool(data.get("artifacts")), "notes": str(data.get("notes", ""))[:400]}


# ---------------------------------------------------------------------------------------------- pair judges
def place_pairs(story: Story, continuation: bool = False) -> List[tuple]:
    """(earlier, later, kind) shot pairs at the same planned place.

    ``revisit``: a shot whose place differs from the previous shot's but was visited before, paired with the latest
    earlier shot there. ``continuation`` (optional, supplementary): consecutive shots at the same place.
    """
    scenes = [(a or {}).get("scene") for a in story.annotations]
    pairs = []
    for i in range(1, len(scenes)):
        if not scenes[i]:
            continue
        if scenes[i] == scenes[i - 1]:
            if continuation:
                pairs.append((i - 1, i, "continuation"))
        else:
            earlier = [j for j in range(i - 1) if scenes[j] == scenes[i]]
            if earlier:
                pairs.append((earlier[-1], i, "revisit"))
    return pairs


def pair_judge_text(story: Story, later: int) -> str:
    card = protagonist(story)
    scene = _annotation(story, later).get("scene", "")
    text = f"LOCATION: {scene.replace('_', ' ')}" + (f"\nPROTAGONIST: {card}" if card is not None else "")
    return text + "\nFrames 1-2: shot A. Frames 3-4: shot B."


def judge_pair(judge: LLMClient, story: Story, later: int, frames: Sequence[Path]) -> Dict[str, Any]:
    """Place pair judge on 2 frames of the earlier shot (A) followed by 2 frames of the later shot (B)."""
    result = judge.chat(PAIR_JUDGE_SYSTEM, [pair_judge_text(story, later), *map(Path, frames)],
                        json_mode=True, temperature=0.0)
    data = result.data or {}

    def score(key):
        x = data.get(key)
        return int(x) if isinstance(x, (int, float)) and 1 <= x <= 5 else None

    return {"place": score("place"), "person": score("person"),
            "notes": (str(data.get("shared", ""))[:200] + " | " + str(data.get("differs", ""))[:200])}


def protagonist_judge_text(story: Story, ref: int, shot: int) -> str:
    return (f"PROTAGONIST: {protagonist(story)}\nFrames 1-4: shot A (story shot {ref}). "
            f"Frames 5-8: shot B (story shot {shot}).")


def judge_protagonist(judge: LLMClient, story: Story, ref: int, shot: int, frames: Sequence[Path],
                      attempts: int = 3) -> Optional[Dict[str, Any]]:
    """Protagonist pair judge on the 4 shot-judge frames of the reference shot (A) and of ``shot`` (B).

    None if every attempt failed (the pair is then left unscored and retried on the next evaluation)."""
    for _ in range(attempts):
        try:
            result = judge.chat(PROTAGONIST_JUDGE_SYSTEM, [protagonist_judge_text(story, ref, shot), *map(Path, frames)],
                                json_mode=True, temperature=0.0)
        except Exception:  # noqa: BLE001 - transient provider failure, try again
            continue
        data = result.data or {}
        p = data.get("person")
        p = int(p) if isinstance(p, (int, float)) and 1 <= p <= 5 else None
        return {"person": p, "visible_a": data.get("visible_a"), "visible_b": data.get("visible_b"),
                "differs": str(data.get("differs", ""))[:200]}
    return None


# ---------------------------------------------------------------------------------------------- plan visibility
def visibility_text(story: Story, i: int) -> str:
    cast = story.setup.cast[0]
    ann = _annotation(story, i)
    return (f"PROTAGONIST: {cast.description} (id: {cast.id})\nPLANNED CAMERA: {ann.get('camera', '')}\n"
            f"PLANNED CONTENT: {ann.get('beat', '')}\nPRESENT: {', '.join(ann.get('present') or [])}")


def judge_visibility(judge: LLMClient, story: Story, i: int, attempts: int = 4) -> Optional[bool]:
    """Text-only: does the hidden plan of shot ``i`` show the protagonist recognisably? None if undecided."""
    for _ in range(attempts):
        try:
            result = judge.chat(VISIBILITY_JUDGE_SYSTEM, [visibility_text(story, i)], json_mode=True, temperature=0.0)
        except Exception:  # noqa: BLE001
            continue
        value = (result.data or {}).get("expected_visible")
        if isinstance(value, bool):
            return value
    return None


def load_visibility(path: Path) -> Dict[str, Dict[str, bool]]:
    """{story key: {shot index (str): expected visible}} from a plan-visibility JSON file (missing: empty)."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}
