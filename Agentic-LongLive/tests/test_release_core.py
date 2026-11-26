# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the pieces shared by the generator backends: memory chain, scene router, entity cards, H3 prompts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.agent.router import SceneRouter  # noqa: E402
from agentic_longlive.backends.h3 import memory_prompt, members_only, with_cards  # noqa: E402
from agentic_longlive.backends.longlive import MemoryChain  # noqa: E402
from agentic_longlive.memory.cards import CardBook, cards_for  # noqa: E402


def test_memory_chain_reference_clock():
    calls = []

    def write(latent, text, pos, context):
        calls.append((latent, pos, [c[0] for c in context]))
        return (latent, pos)

    chain = MemoryChain(frames_per_block=8, max_frames=1024)
    g = (("g.png", 1), "G", "g text")
    m = lambda name, w: ((name, 1), w, name, name)  # noqa: E731
    blocks, start, info = chain.build(write, g, [], 64)
    assert calls == [("G", 0, [])] and start == 8 and info["restart"]
    blocks, start, _ = chain.build(write, g, [m("M0", 0)], 64)
    assert calls[-1] == ("M0", 8, ["G"]) and start == 16 and [b[0] for b in blocks] == ["G", "M0"]
    chain.build(write, g, [m("M0", 0), m("M3", 3)], 64)
    assert calls[-1] == ("M3", 16, ["G", "M0"])
    # a refinement round of watermark 3 replaces the first round at the same clock position and context
    blocks, start, _ = chain.build(write, g, [m("M0", 0), m("M3r1", 3)], 64)
    assert calls[-1] == ("M3r1", 16, ["G", "M0"]) and start == 24
    # the next memory sees G and the two newest older M blocks; reused blocks are not rewritten
    n = len(calls)
    blocks, start, _ = chain.build(write, g, [m("M3r1", 3), m("M5", 5)], 64)
    assert calls[n:] == [("M5", 24, ["G", "M0", "M3r1"])] and [b[0] for b in blocks] == ["G", "M3r1", "M5"]
    # a new G restarts the chain: G at 0, then the snapshot's M blocks in order
    g2 = (("g2.png", 1), "G2", "g2 text")
    n = len(calls)
    blocks, start, info = chain.build(write, g2, [m("M3r1", 3), m("M5", 5)], 64)
    assert info["restart"] and calls[n:] == [("G2", 0, []), ("M3r1", 8, ["G2"]), ("M5", 16, ["G2", "M3r1"])]
    assert start == 24


def test_memory_chain_restarts_before_the_rope_table_ends():
    chain = MemoryChain(frames_per_block=8, max_frames=128)
    write = lambda latent, text, pos, context: (latent, pos)  # noqa: E731
    g = (("g.png", 1), "G", "g")
    for w in range(12):
        prev = [((f"M{w-1}", 1), w - 1, f"M{w-1}", "")] if w else []
        _, start, _ = chain.build(write, g, prev + [((f"M{w}", 1), w, f"M{w}", "")], 64)
        assert start + 64 <= 128
    assert chain.restarts >= 2


def _m(scene, w, path=None):
    return {"scene": scene, "prompt": f"In the {scene}, something happens.", "watermark": w,
            "path": path or f"/m/{scene}_{w}.png", "caption": ""}


def test_router_bank_only_holds_memory_committed_before_the_prompt():
    router = SceneRouter("http://unused", "m")
    router.register(3, [_m("kitchen", 1), _m("garden", 2)])
    router.register(5, [_m("garden", 2), _m("kitchen", 4)])
    assert set(router.bank_at(3)) == {"kitchen", "garden"} and router.bank_at(3)["kitchen"]["watermark"] == 1
    assert router.bank_at(5)["kitchen"]["watermark"] == 4
    assert router.bank_at(4)["kitchen"]["watermark"] == 1  # registered at shot 5: not usable at shot 4


def test_router_maps_llm_answers_to_keep_recall_drop(monkeypatch):
    router = SceneRouter("http://unused", "m")
    router.register(4, [_m("kitchen", 1), _m("garden", 3)])
    current = _m("garden", 3)
    for answer, decision in (("garden", "keep"), ("kitchen", "recall"), ("NEW", "drop"), ("nonsense", "keep")):
        monkeypatch.setattr(router, "_ask", lambda p, c, b, a=answer: (a, {"choice": a}))
        out = router.route(4, "Back in the kitchen.", current)
        assert out["decision"] == decision
        assert (out["entry"] or {}).get("scene") == ("kitchen" if decision == "recall" else None)
    monkeypatch.setattr(router, "_ask", lambda *a: (_ for _ in ()).throw(OSError("down")))
    assert router.route(4, "x", current)["decision"] == "keep"


def test_router_replays_recorded_decisions():
    router = SceneRouter("http://unused", "m", decisions={4: {"decision": "recall", "scene": "kitchen"},
                                                          5: {"decision": "recall", "scene": "attic"}})
    router.register(4, [_m("kitchen", 1), _m("garden", 3)])
    assert router.route(4, "x", _m("garden", 3))["entry"]["scene"] == "kitchen"
    assert router.route(5, "x", _m("garden", 3))["decision"] == "keep"  # unknown location -> keep
    assert router.route(6, "x", _m("garden", 3))["decision"] == "keep"


def test_entity_cards_match_whole_names_and_list_persistent_state(tmp_path):
    entities = {"sam": {"name": "Sam", "description": "Sam, a tall man.",
                        "facts": {"clothing": {"value": "red coat", "persistent": True},
                                  "pose": {"value": "sitting", "persistent": False}}},
                "rex": {"name": "Rex", "description": "Rex, a collie.", "facts": {}}}
    assert cards_for("Sam waves.", entities) == ["Sam: Sam, a tall man. Current state: clothing: red coat."]
    assert cards_for("Samantha waves.", entities) == []
    store = tmp_path / "memory" / "store"
    store.mkdir(parents=True)
    (store / "w002.json").write_text(json.dumps({"entities": entities}))
    text, n = CardBook(tmp_path).text("Rex runs to sam.", 2)
    assert n == 2 and text.startswith(" Remembered from earlier shots: ")
    assert CardBook(tmp_path).text("Rex runs.", 7) == ("", 0)


def test_h3_memory_prompt_puts_cards_in_the_shot_and_pictures_before_the_cut():
    h3 = ("integrated_multimodal_description: [Shot 1] Live-action, cinematic, a collie runs through mud."
          "\n\noverall_soundscape: rain.")
    shot = with_cards(h3, " Remembered from earlier shots: Rex: Rex, a collie.")
    out = memory_prompt(shot, [members_only("Rex together. Shown against a plain light-grey studio background."),
                               "A muddy yard."], [0.0, 0.25], 2.0)
    head, desc, sound = out.split("\n\n")
    assert head.startswith("How the reference pictures align") and sound == "overall_soundscape: rain."
    assert "[Shot 1] Live-action, cinematic, Rex together, as established by Picture 1." in desc
    assert "[Shot 2] At 00:00.250, the camera cuts to a muddy yard, as established by Picture 2." in desc
    assert desc.endswith("[Shot 3] At 00:02.000, the camera cuts to a collie runs through mud. "
                         "Remembered from earlier shots: Rex: Rex, a collie.")
