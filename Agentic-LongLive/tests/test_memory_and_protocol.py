# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.memory.snapshot import MemoryImage, MemorySnapshot, SnapshotBoard  # noqa: E402
from agentic_longlive.memory.store import EntityStore  # noqa: E402
from agentic_longlive.memory.update import parse_plan  # noqa: E402
from agentic_longlive.story import PromptStream, Story, StorySetup  # noqa: E402

STORY = ROOT / "stories/longstory_online/chef_mei_long_day.json"


def test_story_round_trip(tmp_path):
    story = Story.load(STORY)
    assert len(story) == 16 and story.setup.cast[0].id == "mei"
    path = tmp_path / "s.json"
    path.write_text(json.dumps(story.to_unified()))
    again = Story.load(path)
    assert again.prompts == story.prompts and again.annotations == story.annotations


def test_prompt_stream_reveals_one_at_a_time():
    story = Story.load(STORY)
    stream = PromptStream(story, max_shots=4)
    assert stream.view().revealed == ()
    for i in range(4):
        index, prompt, view = stream.reveal()
        assert index == i and prompt == story.prompts[i]
        assert view.revealed == tuple(story.prompts[: i + 1]) and view.next_prompt is None
    with pytest.raises(IndexError):
        stream.reveal()


def test_oracle_view_exposes_only_the_next_prompt():
    story = Story.load(STORY)
    stream = PromptStream(story, max_shots=3, oracle_next_prompt=True)
    _, _, view = stream.reveal()
    assert view.next_prompt == story.prompts[1]
    stream.reveal()
    _, _, view = stream.reveal()
    assert view.next_prompt is None  # beyond max_shots nothing is exposed


def _plan(facts, revoke=(), scene="kitchen"):
    return parse_plan({"entities": [{"id": "mei", "kind": "character", "facts": facts, "revoke": list(revoke)}],
                       "scene": scene}).update


def _store(split=True, revision=True):
    store = EntityStore(split=split, revision=revision)
    store.seed_cast(StorySetup("t", cast=(type("C", (), {"id": "mei", "name": "Mei", "description": "chef"})(),)))
    return store


def test_transient_facts_do_not_change_the_drawn_state():
    store = _store()
    sig = store.get("mei").signature(True)
    result = store.apply(_plan([{"attr": "holding", "value": "spoon", "persistent": "transient"}]), 0)
    assert "mei" not in result.changed and store.get("mei").signature(True) == sig
    result = store.apply(_plan([{"attr": "jacket", "value": "chilli splash", "persistent": True}]), 1)
    assert "mei" in result.changed
    no_split = _store(split=False)
    result = no_split.apply(_plan([{"attr": "holding", "value": "spoon", "persistent": False}]), 0)
    assert "mei" in result.changed  # ablation: everything is drawn


def test_revision_overrides_and_revokes_inferred_facts():
    store = _store(revision=True)
    store.apply(_plan([{"attr": "jacket", "value": "stained", "source": "inferred"}]), 0)
    store.apply(_plan([{"attr": "jacket", "value": "clean", "source": "inferred"}]), 1)
    assert store.get("mei").facts["jacket"].value == "clean"
    store.apply(_plan([], revoke=["jacket"]), 2)
    assert "jacket" not in store.get("mei").facts
    frozen = _store(revision=False)
    frozen.apply(_plan([{"attr": "jacket", "value": "stained", "source": "inferred"}]), 0)
    result = frozen.apply(_plan([{"attr": "jacket", "value": "clean", "source": "inferred"}], revoke=["jacket"]), 1)
    assert frozen.get("mei").facts["jacket"].value == "stained" and result.ignored
    frozen.apply(_plan([{"attr": "jacket", "value": "clean", "source": "explicit"}]), 2)
    assert frozen.get("mei").facts["jacket"].value == "clean"  # explicit evidence still applies


def test_scene_and_lifecycle():
    store = _store()
    store.apply(parse_plan({"entities": [{"id": "Copper Pan", "kind": "object", "facts": []}],
                            "scene": "Restaurant Kitchen", "active": ["copper_pan", "mei"]}).update, 0)
    assert store.scene == "restaurant_kitchen" and store.get("restaurant_kitchen").kind == "scene"
    assert store.active == ["copper_pan", "mei"]
    store.apply(parse_plan({"retire": ["copper_pan", "mei"]}).update, 1)
    assert store.get("copper_pan").status == "dormant" and store.get("mei").status == "active"


def _snap(board, w, score):
    g = MemoryImage("G", f"g{w}.png", "g")
    return MemorySnapshot(board.next_version(), w, g, (), score=score)


def test_snapshot_board_commit_rule():
    board = SnapshotBoard()
    assert board.commit(_snap(board, 0, 6.0))
    assert not board.commit(_snap(board, 0, 5.0))  # same watermark, worse score
    assert board.commit(_snap(board, 0, 7.0))
    assert board.commit(_snap(board, 1, 1.0))  # newer watermark always wins
    assert not board.commit(_snap(board, 0, 9.0))  # older watermark never replaces newer
    assert board.latest().watermark == 1 and board.latest(max_watermark=0).score == 7.0


def test_snapshot_board_prefers_checked_versions_and_trusts_agent_replacements():
    board = SnapshotBoard()
    g = MemoryImage("G", "g.png", "g")
    assert board.commit(MemorySnapshot(board.next_version(), 0, g, (), score=8.0, passed=False))
    assert board.commit(MemorySnapshot(board.next_version(), 0, g, (), score=6.0, passed=True))  # passing beats failing
    assert not board.commit(MemorySnapshot(board.next_version(), 0, g, (), score=9.0, passed=False))
    assert board.commit(MemorySnapshot(board.next_version(), 0, g, (), score=5.0, passed=False, replaces=True))


def test_snapshot_board_wait_complete():
    board = SnapshotBoard()
    board.commit(_snap(board, 0, 5.0))
    assert board.wait_for(0, complete=True, timeout=0.05).complete is False
    board.mark_complete(0)
    assert board.wait_for(0, complete=True, timeout=0.05).complete is True
