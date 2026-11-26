# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State memory: a fact replaced and revoked in one update keeps its new value;
cast cards are split into identity + changeable initial facts; redraws spell out state changes; a checker-reported
state error is never committed through the score floor."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_longlive.agent import prompts
from agentic_longlive.agent.painter import Painter, Verdict
from agentic_longlive.agent.policy import LLMPolicy
from agentic_longlive.config import AgentConfig, MemoryConfig, ModelSpec
from agentic_longlive.memory.entity import Entity, ImageVersion
from agentic_longlive.memory.store import EntityStore
from agentic_longlive.memory.update import parse_plan
from agentic_longlive.models.mock import MockLLM
from agentic_longlive.runtime.trace import Tracer
from agentic_longlive.story import CastMember, StorySetup

CARD = ("The host, a man in his late twenties with short brown hair and a neatly groomed short beard, "
        "wearing a white V-neck T-shirt and dark jeans.")
SETUP = StorySetup("tour", cast=(CastMember("host", "the host", CARD),))
SPLIT = {"host": {"identity": "The host, a man in his late twenties with short brown hair.",
                  "facts": {"facial_hair": "neatly groomed short beard", "clothing": "white V-neck T-shirt, dark jeans"}}}


def _update(facts, revoke=()):
    return parse_plan({"entities": [{"id": "host", "kind": "character", "facts": facts, "revoke": list(revoke)}],
                       "scene": "bathroom"}).update


def test_a_value_written_and_revoked_in_one_update_is_kept():
    # a planner output seen in practice: the new value plus the same attribute in "revoke"
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    result = store.apply(_update([{"attr": "facial_hair", "value": "clean-shaven"}], revoke=["facial_hair"]), 33)
    host = store.get("host")
    assert host.facts["facial_hair"].value == "clean-shaven"
    assert result.superseded == [("host", "facial_hair")] and result.revoked == []
    assert any(h["event"] == "updated" and h["old"] == "neatly groomed short beard" for h in host.history)
    # a genuine withdrawal (no new value) still removes the fact
    result = store.apply(_update([], revoke=["clothing"]), 34)
    assert "clothing" not in host.facts and result.revoked == [("host", "clothing")]


def test_cast_split_moves_changeable_appearance_into_facts():
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    host = store.get("host")
    assert "beard" not in host.description and "T-shirt" not in host.description
    assert host.visual_state() == {"clothing": "white V-neck T-shirt, dark jeans",
                                   "facial_hair": "neatly groomed short beard"}
    assert host.history[0]["event"] == "cast_split" and host.history[0]["card"] == CARD
    store.apply(_update([{"attr": "facial_hair", "value": "clean-shaven"}]), 33)
    assert "beard" not in host.describe()  # nothing in the drawn text can bring the beard back
    unsplit = EntityStore()
    unsplit.seed_cast(SETUP)  # memory.cast_split: false keeps the whole card
    assert unsplit.get("host").description == CARD and not unsplit.get("host").facts


def test_cast_split_leak_is_retried_then_stripped():
    from agentic_longlive.models.base import LLMResult

    class Leaky(MockLLM):
        calls = 0

        def _chat(self, system, content, json_mode, temperature, max_tokens):
            Leaky.calls += 1
            fixed = "### FIX" in content[0]
            identity = ("The host, a man with short brown hair." if fixed and Leaky.calls > 1 and self.fixes else
                        "The host, a man with short brown hair and a neatly groomed short beard")
            data = {"cast": [{"id": "host", "identity": identity, "facts": {"clothing": "white T-shirt"}}]}
            return LLMResult(text=json.dumps(data), data=data, model="leaky")

    good = Leaky(ModelSpec(provider="mock"))
    good.fixes = True
    out = LLMPolicy(good, AgentConfig(), MemoryConfig()).split_cast(SETUP)
    assert out["host"]["identity"] == "The host, a man with short brown hair." and Leaky.calls == 2
    stubborn = Leaky(ModelSpec(provider="mock"))
    stubborn.fixes = False
    out = LLMPolicy(stubborn, AgentConfig(), MemoryConfig()).split_cast(SETUP)
    assert "beard" not in out["host"]["identity"]
    assert out["host"]["facts"]["facial_hair"] == "a neatly groomed short beard"
    # fur is not clothing: a corgi's coat stays in its identity
    corgi = "Biscuit, a Pembroke Welsh corgi with a red-and-white coat, a white blaze and short legs."
    assert LLMPolicy._strip_changeable(corgi, {}) == corgi


def test_split_cast_parses_and_falls_back(tmp_path):
    policy = LLMPolicy(MockLLM(ModelSpec(provider="mock")), AgentConfig(), MemoryConfig())
    out = policy.split_cast(SETUP)
    assert out["host"]["facts"]["clothing"].startswith("a white V-neck")
    assert "wearing" not in out["host"]["identity"]

    class Junk(MockLLM):
        def _chat(self, system, content, json_mode, temperature, max_tokens):
            from agentic_longlive.models.base import LLMResult
            data = {"cast": [{"id": "host", "identity": "", "facts": []}, "x"]}
            return LLMResult(text=json.dumps(data), data=data, model="junk")

    assert LLMPolicy(Junk(ModelSpec(provider="mock")), AgentConfig(), MemoryConfig()).split_cast(SETUP) == {}


def test_planner_sees_no_card_appearance_after_a_split():
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    from agentic_longlive.story import AgentView

    view = AgentView(setup=SETUP, revealed=("He shaves.",))
    call = LLMPolicy(MockLLM(ModelSpec(provider="mock")), AgentConfig(), MemoryConfig()).plan(view, store)
    setup_block = call.user.split("### ENTITY TABLE")[0]
    assert "beard" not in setup_block and "T-shirt" not in setup_block
    assert "neatly groomed short beard" in call.user  # still in the table, as a fact that can be changed
    assert "Never put an attribute you write" in prompts.PLAN_SYSTEM


def test_state_changes_are_written_into_entity_and_compose_specs():
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    host = store.get("host")
    host.images.append(ImageVersion(0, {"front": "a.png"}, -1, "s0", state=dict(host.visual_state())))
    store.apply(_update([{"attr": "facial_hair", "value": "clean-shaven"}], revoke=["facial_hair"]), 33)
    changes = host.state_changes(host.visual_state(), [host.latest])
    assert changes == [("facial_hair", ["neatly groomed short beard"], "clean-shaven")]
    spec = prompts.entity_spec("the host", host.describe(), "front", "character", changes)
    assert "do not copy the old state" in spec and "facial hair: neatly groomed short beard -> clean-shaven" in spec
    edit = prompts.entity_edit("the host", changes, "front", "character")
    assert edit.startswith("Edit the reference image of the host: change the facial hair from")
    # a removed and a newly added attribute are spelled out too; images without a recorded state are ignored
    store.apply(_update([{"attr": "headwear", "value": "red wool beanie"}], revoke=["clothing"]), 34)
    changes = host.state_changes(host.visual_state(), [host.latest])
    assert ("clothing", ["white V-neck T-shirt, dark jeans"], None) in changes
    assert ("headwear", [], "red wool beanie") in changes
    edit = prompts.entity_edit("the host", changes, "front", "character")
    assert "remove the white V-neck T-shirt, dark jeans" in edit and "add red wool beanie" in edit
    assert host.state_changes(host.visual_state(), [ImageVersion(1, {"f": "b.png"}, 0, "s1")]) == []
    assert "Include: x.\n" in prompts.compose_spec("G", "t", ["x"])  # unchanged without changes


def test_pose_and_location_of_a_character_are_never_drawn():
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    store.apply(_update([{"attr": "location", "value": "standing beside the induction hob", "persistent": True},
                         {"attr": "holding_item", "value": "a bowl of tomatoes", "persistent": True},
                         {"attr": "shirt_stain", "value": "grease stain on the chest", "persistent": True}]), 19)
    host = store.get("host")
    assert "location" not in host.visual_state() and "holding_item" not in host.visual_state()
    assert host.visual_state()["shirt_stain"] == "grease stain on the chest"
    plan = parse_plan({"entities": [{"id": "basil_pot", "kind": "object", "name": "basil pot",
                                     "facts": [{"attr": "location", "value": "on the counter by the till"}]}]})
    store.apply(plan.update, 32)
    assert store.get("basil_pot").visual_state() == {"location": "on the counter by the till"}  # a placement


def test_entity_check_is_against_memory_not_the_planner_text(tmp_path):
    class Image:
        spec = ModelSpec(provider="mock")

        def __init__(self):
            self.prompts = []

        def generate(self, prompt, refs, size, priority=0):
            from agentic_longlive.models.base import ImageResult
            self.prompts.append(prompt)
            return ImageResult(png=b"png", model="m", latency_s=0.0)

    class Checker(MockLLM):
        specs = []

        def _chat(self, system, content, json_mode, temperature, max_tokens):
            from agentic_longlive.models.base import LLMResult
            Checker.specs.append(content[0])
            data = {"pass": True, "score": 9, "state_ok": True, "issues": []}
            return LLMResult(text=json.dumps(data), data=data, model="c")

    image = Image()
    painter = Painter(image, Checker(ModelSpec(provider="mock")), AgentConfig(), MemoryConfig(), tmp_path, Tracer(None))
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    host = store.get("host")
    text = "The host sits at the dining table holding a plate of pasta, white T-shirt."
    painter.paint_entity(host, text, 18, "s", dict(host.visual_state()))
    assert "sits at the dining table" in image.prompts[0]  # the planner's text drives generation
    assert "sits at the dining table" not in Checker.specs[0]  # but the check is against memory
    assert "facial_hair: neatly groomed short beard" in Checker.specs[0]
    assert "where the entity stands" in prompts.CHECK_SYSTEM


def test_failed_edit_of_the_latest_image_retries_from_the_anchor(tmp_path):
    from agentic_longlive.models.base import ImageResult, LLMResult

    class Image:
        spec = ModelSpec(provider="mock")
        calls = []

        def generate(self, prompt, refs, size, priority=0):
            Image.calls.append((prompt, [str(r) for r in refs]))
            return ImageResult(png=b"png", model="m", latency_s=0.0)

    class Checker(MockLLM):
        n = 0

        def _chat(self, system, content, json_mode, temperature, max_tokens):
            Checker.n += 1  # the two attempts from the latest (soaked) image fail, the anchor edits pass
            ok = Checker.n > 2
            data = {"pass": ok, "score": 9 if ok else 3, "state_ok": ok, "issues": [] if ok else ["still soaked"]}
            return LLMResult(text=json.dumps(data), data=data, model="c")

    painter = Painter(Image(), Checker(ModelSpec(provider="mock")), AgentConfig(), MemoryConfig(), tmp_path,
                      Tracer(None))
    dog = Entity(id="biscuit", kind="character", name="Biscuit", description="a corgi")
    dog.images.append(ImageVersion(0, {"front": "clean.png", "side": "clean_s.png"}, -1, "s0", state={"collar": "red"}))
    dog.images.append(ImageVersion(1, {"front": "wet.png", "side": "wet_s.png"}, 6, "s1",
                                   state={"collar": "red", "coat": "soaked"}))
    state = {"collar": "red", "coat": "clean and fluffy"}
    version = painter.paint_entity(dog, "x", 14, "s2", state)
    assert version is not None and "_b" in version.views["front"]
    assert Image.calls[0][1] == ["wet.png"] and Image.calls[2][1] == ["clean.png"]
    assert "add clean and fluffy" in Image.calls[2][0]


def test_redraw_of_a_changed_state_is_an_edit_of_the_latest_image(tmp_path):
    class Image:
        spec = ModelSpec(provider="mock")

        def __init__(self):
            self.calls = []

        def generate(self, prompt, refs, size, priority=0):
            from agentic_longlive.models.base import ImageResult
            self.calls.append((prompt, [str(r) for r in refs]))
            return ImageResult(png=b"png", model="m", latency_s=0.0)

    class Checker(MockLLM):
        def _chat(self, system, content, json_mode, temperature, max_tokens):
            from agentic_longlive.models.base import LLMResult
            self.specs = getattr(self, "specs", []) + [content[0]]
            data = {"pass": True, "score": 9, "state_ok": True, "issues": []}
            return LLMResult(text=json.dumps(data), data=data, model="c")

    image, checker = Image(), Checker(ModelSpec(provider="mock"))
    painter = Painter(image, checker, AgentConfig(), MemoryConfig(), tmp_path, Tracer(None))
    store = EntityStore()
    store.seed_cast(SETUP, SPLIT)
    host = store.get("host")
    first = dict(host.visual_state())
    host.images.append(ImageVersion(0, {"front": "a0.png", "side": "a1.png"}, -1, "s0", state=first))
    # nothing changed: a normal description-driven redraw
    painter.paint_entity(host, host.describe(), 0, "s", dict(host.visual_state()))
    assert not image.calls[0][0].startswith("Edit the reference")
    host.images.append(ImageVersion(1, {"front": "b0.png", "side": "b1.png"}, 0, "s1", state=first))
    store.apply(_update([{"attr": "facial_hair", "value": "clean-shaven"}]), 33)
    image.calls.clear()
    version = painter.paint_entity(host, host.describe(), 33, "s2", dict(host.visual_state()))
    (front_prompt, front_refs), (side_prompt, side_refs) = image.calls
    assert front_prompt.startswith("Edit the reference image of the host: change the facial hair")
    assert front_refs == ["b0.png"] and side_refs[0] == "b1.png"  # the latest image only (no revert to the anchor)
    assert "clean-shaven" in checker.specs[-1] and "Edit the reference" not in checker.specs[-1]  # checked on the spec
    assert version.state["facial_hair"] == "clean-shaven"


def test_state_errors_are_never_committed_through_the_score_floor(tmp_path):
    painter = Painter(None, None, AgentConfig(commit_min_score=6), MemoryConfig(), tmp_path, Tracer(None))
    assert painter.acceptable(Verdict(False, 6.0))  # a minor flaw at the floor is accepted
    assert not painter.acceptable(Verdict(False, 8.0, ["has a beard, should be clean-shaven"], state_ok=False))
    assert not painter.acceptable(Verdict(True, 9.0, [], state_ok=False))


def test_checker_images_are_labelled(tmp_path):
    seen = []

    class Checker(MockLLM):
        def _chat(self, system, content, json_mode, temperature, max_tokens):
            from agentic_longlive.models.base import LLMResult
            seen.extend(content)
            data = {"pass": True, "score": 9, "state_ok": True, "issues": []}
            return LLMResult(text=json.dumps(data), data=data, model="c")

    painter = Painter(None, Checker(ModelSpec(provider="mock")), AgentConfig(), MemoryConfig(), tmp_path, Tracer(None))
    cand, ref = tmp_path / "c.png", tmp_path / "r.png"
    painter.check(cand, "spec", [ref])
    assert seen[1:] == ["CANDIDATE (judge this image):", cand, "REFERENCE 1 (identity only; it may show an older state):",
                        ref]


def test_checker_state_ok_is_parsed(tmp_path):
    class Checker(MockLLM):
        def _chat(self, system, content, json_mode, temperature, max_tokens):
            from agentic_longlive.models.base import LLMResult
            data = {"pass": True, "score": 7, "state_ok": "false", "issues": ["still bearded"]}
            return LLMResult(text=json.dumps(data), data=data, model="c")

    painter = Painter(None, Checker(ModelSpec(provider="mock")), AgentConfig(commit_min_score=6), MemoryConfig(),
                      tmp_path, Tracer(None))
    img = tmp_path / "c.png"
    img.write_bytes(b"x")
    verdict = painter.check(img, "spec", [])
    assert verdict.state_ok is False and verdict.passed is False and not painter.acceptable(verdict)
    assert "state_ok" in prompts.CHECK_SYSTEM


def test_old_store_json_without_image_state_still_loads(tmp_path):
    store = EntityStore()
    store.seed_cast(SETUP)
    store.get("host").images.append(ImageVersion(0, {"front": "a.png"}, -1, "s0"))
    path = tmp_path / "s.json"
    store.save(path)
    data = json.loads(path.read_text())
    del data["entities"]["host"]["images"][0]["state"]  # written before the field existed
    path.write_text(json.dumps(data))
    assert EntityStore.load(Path(path)).get("host").latest.state == {}
