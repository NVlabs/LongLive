# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the RL planner code: reward math, JSON parsing, text environment, scoring, data collection and the
trainer helpers. Every network call (planner, judge, image servers) is replaced by a local fake."""
import argparse
import base64
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.agent import prompts  # noqa: E402
from agentic_longlive.rl import collect, evaluate, imagelayer, judge, storygen  # noqa: E402
from agentic_longlive.rl import targets as T  # noqa: E402
from agentic_longlive.rl.common import parse_json, read_jsonl  # noqa: E402
from agentic_longlive.rl.reward import RewardConfig, reward  # noqa: E402
from agentic_longlive.rl.score import Scorer  # noqa: E402
from agentic_longlive.rl.textenv import apply_output, memory_view, rollout, store_from_json  # noqa: E402
from agentic_longlive.story import Story  # noqa: E402

V4 = dict(version=4, w_belief=0.35, w_precision=0.15, w_image=0.3, w_retain=0.05, w_length=0.05, len_free=1200,
          len_span=1200, w_size=0.2, w_false_count=0.1)


# --------------------------------------------------------------------------------------------------- fixtures

def _shot(prompt, present, state):
    return {"prompt": prompt, "annotation": {"present": present, "entity_state": state}}


def toy_story():
    """Five shots: Ana and her mug in the kitchen, two garden shots without the mug, back to the kitchen."""
    coat = {"outfit": "red coat"}
    return {
        "id": "toy_terse",
        "setup": {"title": "Tea", "style": "plain daylight",
                  "cast": [{"id": "ana", "name": "Ana",
                            "description": "Ana, a woman in her 30s with short black hair, in a red coat."}]},
        "bible": {"entities": {"ana": {"id": "ana", "kind": "character", "name": "Ana",
                                       "identity": "woman in her 30s, short black hair"},
                               "mug": {"id": "mug", "kind": "object", "name": "Mug", "identity": "white ceramic mug"}}},
        "shots": [
            _shot("Kitchen. Ana in a red coat puts a white mug on the table.", ["ana", "mug"],
                  {"ana": coat, "mug": {"state": "empty"}}),
            _shot("Still at the kitchen. Ana fills the mug with tea.", ["ana", "mug"],
                  {"ana": coat, "mug": {"state": "full of tea"}}),
            _shot("Garden. Ana waters the roses.", ["ana"], {"ana": coat, "mug": {"state": "full of tea"}}),
            _shot("Still in the garden. Ana sits on a bench.", ["ana"], {"ana": coat, "mug": {"state": "full of tea"}}),
            _shot("Back at the kitchen. Ana drinks from the mug.", ["ana", "mug"],
                  {"ana": coat, "mug": {"state": "half full of tea"}}),
        ],
    }


def plan_json(shot: int) -> str:
    scene = "kitchen" if shot in (0, 1, 4) else "garden"
    ents = [{"id": "ana", "kind": "character", "name": "Ana", "present": True,
             "facts": [{"attr": "clothing", "value": "red coat", "persistent": True, "source": "explicit"},
                       {"attr": "action", "value": "walking", "persistent": True, "source": "explicit"}]},
            {"id": scene, "kind": "scene", "name": scene, "description": f"a sunny {scene}", "present": True}]
    if scene == "kitchen":
        ents.append({"id": "mug", "kind": "object", "name": "Mug", "description": "white ceramic mug", "present": True,
                     "facts": [{"attr": "state", "value": "full of tea" if shot else "empty", "persistent": True}]})
    plan = {"entities": ents, "scene": scene, "active": ["ana", "mug"], "retire": [],
            "image_prompts": {"ana": "Ana in a red coat"},
            "memory": {"g_members": ["ana"], "m_members": ["mug"] if scene == "kitchen" else ["ana"],
                       "g_prompt": "Ana standing.", "m_prompt": f"The {scene}.", "g_caption": "Ana.", "m_caption": ""}}
    return "```json\n" + json.dumps(plan) + "\n```"


class FakeLLM:
    """Stands in for the planner endpoint: answers the cast split and memory plans of the toy story."""

    def __init__(self, model="fake-planner"):
        self.model = model
        self.calls = []
        self.users = []

    def chat(self, system, content, json_mode=False, temperature=None, max_tokens=None):
        user = content[0]
        self.users.append((system, user))
        if system.startswith("TASK: cast_split"):
            text = json.dumps({"cast": [{"id": "ana", "identity": "Ana, a woman in her 30s with short black hair.",
                                         "facts": {"clothing": "red coat"}}]})
        else:
            shot = int(re.search(r"NEW PROMPT \(shot (\d+)\)", user).group(1))
            text = plan_json(shot)
        self.calls.append({"latency": 0.01, "completion_tokens": len(text) // 4, "prompt_tokens": len(user) // 4,
                           "parsed": True})
        return SimpleNamespace(data=parse_json(text), text=text, model=self.model, latency_s=0.01, parsed=True)


def perfect_judge(system, user):
    """A judge that maps every ground-truth entity to the memory id of the same name and supports every fact."""
    assert system.startswith("TASK: memory_audit")
    canon = re.findall(r"^- (\w+): ", user.split("### AGENT MEMORY")[0], re.M)
    facts = re.findall(r"^(F\d+): ", user, re.M)
    return {"map": {e: e for e in canon}, "facts": {f: {"belief": "S"} for f in facts}, "gm_conflict": [],
            "false": []}


def hand_example():
    """Targets / memory / judge / image audit of one decision, small enough to score by hand."""
    targets = {
        "F": [{"id": "F1", "entity": "ana", "attr": "outfit", "value": "red coat", "weight": 1.0, "visual": True},
              {"id": "F2", "entity": "mug", "attr": "state", "value": "full of tea", "weight": 0.8, "visual": False}],
        "E": [{"entity": "ana", "absent_for": 0}, {"entity": "mug", "absent_for": 2}],
        "R": [{"entity": "cat", "return_at": 5, "absent_for": 3}],
        "truth": {}, "canon": {},
    }

    def ent(eid, kind, status, persistent):
        return {"id": eid, "kind": kind, "name": eid, "description": eid, "status": status, "persistent": persistent}
    mem = {"entities": [ent("ana", "character", "active", {"outfit": "red coat"}),
                        ent("mug", "object", "active", {"state": "empty"}),
                        ent("cat", "object", "dormant", {}),
                        ent("lamp", "object", "active", {"state": "on"}),
                        ent("kitchen", "scene", "active", {})],
           "g_members": ["ana"], "m_members": ["ana", "lamp"], "g_prompt": "", "m_prompt": ""}
    j = {"map": {"ana": "ana", "mug": "mug", "cat": "cat"}, "labels": {"F1": "S", "F2": "C"}, "gm_conflict": [],
         "false": ["mug.state"]}
    img = {"entities": {"ana": {"G": True, "M": True}, "mug": {"G": False, "M": False}}, "facts": {"F1": "C"},
           "quality": {"G": 7.0, "M": 5.0}, "rendered": {"G": True, "M": True}}
    return targets, mem, j, img


# --------------------------------------------------------------------------------------------------- hygiene

def test_spdx_headers():
    files = sorted((ROOT / "agentic_longlive" / "rl").glob("*.py")) + [Path(__file__)]
    for f in files:
        head = f.read_text().splitlines()[:2]
        assert head == ["# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
                        "All rights reserved.", "# SPDX-License-Identifier: Apache-2.0"], f


def test_configs_parse():
    pytest.importorskip("torch")
    from agentic_longlive.rl.train import build_cfg

    for path in sorted((ROOT / "configs" / "rl").rglob("*.json")):
        cfg = build_cfg(json.loads(path.read_text()))
        if cfg.mode == "grpo":
            RewardConfig(**cfg.reward)
            assert (cfg.steps, cfg.batch_states, cfg.group_size, cfg.temperature) == (80, 8, 8, 0.9)
            assert (cfg.lr, cfg.grad_clip, cfg.std_floor, cfg.pipeline) == (2e-5, 0.5, 0.1, True)
        else:
            assert (cfg.lr, cfg.sft_min_reward, cfg.sft_batch) == (1e-4, 0.2, 32)
        assert (cfg.lora_r, cfg.lora_alpha) == (32, 64)


# --------------------------------------------------------------------------------------------------- parsing

def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}
    assert parse_json('Here is the plan: {"scene": "x"} -- done') == {"scene": "x"}
    assert parse_json("[1, 2]") is None
    assert parse_json("not json {oops") is None
    assert parse_json("") is None


def test_judge_normalise():
    targets, mem, _, _ = hand_example()
    targets["canon"] = {"ana": {}, "mug": {}, "ghost": {}}
    reply = {"map": {"ana": "ana", "mug": "nope", "ghost": 3}, "facts": {"F1": {"belief": "support"}, "F2": "x"},
             "false": ["a.b", 5] + ["c.d"] * 20}
    out = judge.normalise(reply, targets, mem)
    assert out["map"] == {"ana": "ana", "mug": None, "ghost": None}
    assert out["labels"] == {"F1": "S", "F2": "M"}
    assert out["false"] == ["a.b"] + ["c.d"] * 15
    assert judge.normalise("text", targets, mem) is None
    calls = []
    empty = {"F": [], "E": [], "R": [], "canon": {}, "truth": {}}
    assert judge.judge(lambda s, u: calls.append(1), empty, mem)["labels"] == {} and not calls


# --------------------------------------------------------------------------------------------------- targets

def test_targets_from_hidden_annotation():
    meta = toy_story()
    t0 = T.compute(meta, 0)
    assert [(f["entity"], f["attr"], f["weight"], f["visual"]) for f in t0["F"]] == [
        ("ana", "outfit", 1.0, True), ("mug", "state", 1.0, True)]
    assert t0["E"] == [{"entity": "ana", "absent_for": 0}, {"entity": "mug", "absent_for": 0}]
    assert t0["R"] == []
    t1 = T.compute(meta, 1)  # the mug is needed again at shot 4 only if it is unchanged until then: it is
    assert [(f["entity"], f["first_need"], f["weight"]) for f in t1["F"]] == [("ana", 2, 1.0), ("mug", 4, 0.64)]
    t2 = T.compute(meta, 2)  # the mug is absent in shots 2 and 3 and returns at 4
    assert t2["R"] == [{"entity": "mug", "return_at": 4, "absent_for": 2}]
    assert t2["truth"]["mug"] == {"state": "full of tea"} and t2["canon"]["mug"]["name"] == "Mug"


# --------------------------------------------------------------------------------------------------- reward

def test_reward_v4_by_hand():
    targets, mem, j, img = hand_example()
    r = reward(targets, mem, j, 1500, True, RewardConfig(**V4), img=img)
    belief = (1.0 * 1.0 + 0.8 * -1.5) / 1.8  # F1 supported, F2 contradicted (-1.5)
    precision = 1 - 1 / 3  # one false fact among 3 persistent facts
    # image: Ana depicted in G and M with her only needed fact drawn wrong (half credit), the mug nowhere,
    # and the lamp is an M member that is not in the next shot
    image = (0.5 * (0.508 + 0.492) + 0.0) / 2 - 0.107
    retain = 0.0  # the returning cat is dormant
    total = (0.35 * belief + 0.15 * precision + 0.3 * image + 0.05 * retain) / 0.85
    total -= 0.05 * (300 / 1200) + 0.2 * (1 / 3) + 0.1 * (1 / 8)  # length, size (lamp unmapped), false count
    assert r["belief"] == pytest.approx(belief)
    assert r["image"] == pytest.approx(image)
    assert r["total"] == pytest.approx(total)
    assert (r["n_m_off"], r["img_g"], r["img_fact_c"], r["img_qg"]) == (1, 0.5, 1.0, 7.0)


def test_reward_v4_not_rendered_and_v1_v3():
    targets, mem, j, img = hand_example()
    nr = reward(targets, mem, j, 1500, True, RewardConfig(**dict(V4, image=False)))
    assert nr["image"] == pytest.approx((0.508 + 0.492) / 2 - 0.107)  # membership only: Ana in G and M
    v1 = reward(targets, mem, j, 1500, True, RewardConfig())
    belief1 = (1.0 - 0.8) / 1.8
    total1 = (0.4 * belief1 + 0.2 * 1.0 + 0.2 * 0.5 + 0.1 * 0.0) / 0.9 - 0.15 * (1 / 4)  # visual 1, slots 1/2
    assert v1["total"] == pytest.approx(total1)
    v3 = reward(targets, mem, j, 1500, True, RewardConfig(version=3, w_belief=0.35, w_precision=0.15, w_visual=0.15,
                                                           w_slots=0.15, w_retain=0.05, len_free=1200, len_span=1200))
    total3 = (0.35 * ((1.0 - 1.2) / 1.8) + 0.15 * (2 / 3) + 0.15 * 1.0 + 0.15 * 0.5) / 0.85
    total3 -= 0.05 * 0.25 + 0.2 * (1 / 3) + 0.1 * (1 / 8)
    assert v3["total"] == pytest.approx(total3)


def test_reward_failures():
    targets, mem, j, _ = hand_example()
    cfg = RewardConfig(**V4)
    assert reward(targets, mem, j, 10, False, cfg) == {"total": -1.0, "format": 0.0}
    assert reward(targets, mem, None, 10, True, cfg)["total"] is None
    assert reward(targets, mem, j, 10, True, cfg, img=None)["image_failed"]
    with pytest.raises(ValueError):
        RewardConfig(version=2)


# --------------------------------------------------------------------------------------------------- text env

def test_apply_output_to_stored_state():
    story = Story._from_unified(toy_story())
    from agentic_longlive.memory.store import EntityStore

    store = EntityStore(split=True, revision=True)
    store.seed_cast(story.setup)
    st, plan, g, m, ok = apply_output(store.to_json(), plan_json(0), 0)
    assert ok and st.scene == "kitchen" and g == ["ana"] and m == ["mug"]
    ana = st.get("ana")
    assert ana.facts["clothing"].value == "red coat"
    assert ana.facts["action"].persistent is False  # momentary attributes of characters are never drawn
    assert all(e.images for e in st.entities.values())  # placeholders stand in for the entity images
    mem = memory_view(st, plan, g, m)
    assert {e["id"] for e in mem["entities"]} == {"ana", "kitchen", "mug"} and "action" not in mem["entities"][0][
        "persistent"]
    assert store_from_json(st.to_json()).to_json() == st.to_json()
    for bad in ("no json at all", '{"entities": {}}', '{"entities": [], "retire": []}'):
        assert apply_output(store.to_json(), bad, 0)[4] is False
    _, _, g_rule, m_rule, _ = apply_output(st.to_json(), plan_json(1), 1, rule="recent")
    assert g_rule[0] == "ana" and "mug" in m_rule


def test_rollout_and_state_rows(tmp_path):
    meta = toy_story()
    path = tmp_path / "toy_terse.json"
    path.write_text(json.dumps(meta))
    llm = FakeLLM()
    recs = rollout(Story.load(path), llm)
    assert len(recs) == 5 and llm.users[0][0].startswith("TASK: cast_split")
    assert recs[0]["system"] == prompts.PLAN_SYSTEM and "### NEW PROMPT (shot 0)" in recs[0]["user"]
    first = recs[0]["store_before"]["entities"]["ana"]
    assert first["facts"]["clothing"]["value"] == "red coat" and first["images"]  # cast split + placeholder image
    assert (recs[0]["g"], recs[0]["m"], recs[2]["memory"]["scene"]) == (["ana"], ["mug"], "garden")
    assert recs[3]["store_before"] == recs[2]["store_after"]
    rows = collect.state_rows(recs, meta, "terse", "teacher")
    assert rows[2]["key"] == "teacher/toy_terse/2" and rows[2]["targets"]["R"][0]["entity"] == "mug"
    assert rows[0]["behaviour_output"] == plan_json(0)


# --------------------------------------------------------------------------------------------------- scoring

class FakeImageLayer:
    def __init__(self):
        self.calls = 0

    def audit(self, targets, store, plan, g, m):
        self.calls += 1
        return {"entities": {x["entity"]: {"G": True, "M": False} for x in targets["E"]}, "facts": {},
                "quality": {"G": 8.0, "M": 0.0}, "rendered": {"G": True, "M": False}}


def _states(tmp_path):
    meta = toy_story()
    path = tmp_path / "toy_terse.json"
    path.write_text(json.dumps(meta))
    return collect.state_rows(rollout(Story.load(path), FakeLLM()), meta, "terse", "teacher")


def test_scorer_end_to_end(tmp_path):
    st = _states(tmp_path)[1]
    layer = FakeImageLayer()
    r = Scorer(perfect_judge, RewardConfig(**V4), workers=2, image_layer=layer).score_one(st, plan_json(1))
    assert layer.calls == 1 and r["format"] == 1.0 and r["belief"] == pytest.approx(1.0)
    assert r["img_g"] == 1.0 and r["img_m"] == 0.0 and r["total"] is not None
    nr = Scorer(perfect_judge, RewardConfig(**dict(V4, image=False)), workers=2).score_many(
        [(st, plan_json(1)), (st, "garbage")])
    assert nr[0]["total"] > nr[1]["total"] == -1.0
    with pytest.raises(RuntimeError):
        Scorer(perfect_judge, RewardConfig(**V4)).score_one(st, plan_json(1))


def test_image_layer_render_and_audit(monkeypatch):
    seen = []

    def fake_post(url, payload, timeout=0):
        seen.append(url)
        if url.endswith("/images/generations"):
            assert payload["size"] == "768x432" and payload["seed"] == 0 and payload["num_inference_steps"] == 4
            return {"data": [{"b64_json": base64.b64encode(payload["prompt"][:16].encode()).decode()}]}
        parts = payload["messages"][1]["content"]
        assert [p["type"] for p in parts].count("image_url") == 2 and "IMAGE G and IMAGE M" in parts[0]["text"]
        return {"choices": [{"message": {"content": json.dumps(
            {"entities": {"ana": {"G": True, "M": False}}, "facts": {"F1": "c"}, "quality": {"G": 8, "M": "n/a"}})}}]}

    monkeypatch.setattr(imagelayer, "http_post", fake_post)
    st, plan, g, m, _ = apply_output({"entities": {}}, plan_json(1), 1)
    targets = {"E": [{"entity": "ana", "absent_for": 0}], "canon": {"ana": {"kind": "character", "name": "Ana"}},
               "F": [{"id": "F1", "entity": "ana", "attr": "outfit", "value": "red coat", "visual": True}]}
    layer = imagelayer.ImageLayer("http://img-a/v1,http://img-b/v1", "http://vlm/v1")
    out = layer.audit(targets, st, plan, g, m)
    assert out == {"entities": {"ana": {"G": True, "M": False}}, "facts": {"F1": "C"},
                   "quality": {"G": 8.0, "M": 0.0}, "rendered": {"G": True, "M": True}}
    g_spec, m_spec = imagelayer.compose_specs(st, plan, g, m)
    assert g_spec.startswith("Ana standing.\nInclude: Ana:") and "Location: kitchen" in m_spec
    layer.audit(targets, st, plan, g, m)
    assert layer.renderer.stats["renders"] == 2 and layer.renderer.stats["hits"] == 2  # cached by prompt
    assert sum(u.endswith("/images/generations") for u in seen) == 2


# --------------------------------------------------------------------------------------------------- collection

def test_bestofk_picks_best_of_four(tmp_path, monkeypatch):
    states = _states(tmp_path)
    (tmp_path / "states.jsonl").write_text("\n".join(json.dumps(s) for s in states[:2]) + "\n")
    worse = parse_json(plan_json(1))
    worse["entities"] = worse["entities"][:2]  # forgets the mug
    samples = ["garbage", json.dumps(worse), plan_json(1)]
    monkeypatch.setattr(collect, "chat", lambda *a, **k: {"texts": list(samples)})
    monkeypatch.setattr(collect, "openai_judge", lambda urls, model: perfect_judge)
    out = tmp_path / "bestof.jsonl"
    collect.cmd_bestofk(argparse.Namespace(states=str(tmp_path / "states.jsonl"), max_states=0,
                                           planner_urls="http://x/v1", planner_model="teacher",
                                           judge_urls="http://j/v1", judge_model="judge", k=3, temperature=0.8,
                                           out=str(out), workers=2))
    rows = {r["key"]: r for r in read_jsonl(out)}
    assert len(rows) == 2 and all(len(r["rewards"]) == 4 for r in rows.values())
    r1 = rows["teacher/toy_terse/1"]
    assert r1["rewards"][1] == -1.0 and r1["rewards"][r1["best_index"]] == max(r1["rewards"])


# --------------------------------------------------------------------------------------------------- evaluation

def test_evaluate_rollout_judge_summary(tmp_path, monkeypatch):
    stories = tmp_path / "stories"
    stories.mkdir()
    (stories / "toy_terse.json").write_text(json.dumps(toy_story()))
    monkeypatch.setattr(evaluate, "LLMAdapter", lambda eps, model, **k: FakeLLM(model))
    monkeypatch.setattr(evaluate, "openai_judge", lambda urls, model: perfect_judge)
    out = tmp_path / "eval"
    evaluate.cmd_rollout(argparse.Namespace(name="toy", urls="http://x/v1", model="m", stories=str(stories),
                                            out=str(out), temperature=0.2, oracle=False, workers=0))
    for rule, name in (("llm", "toy"), ("recent", "rule_recent")):
        evaluate.cmd_judge(argparse.Namespace(name=name, src="toy", rule=rule, stories=str(stories), out=str(out),
                                              judge="openai", judge_model="", judge_urls="http://j/v1", workers=2))
    evaluate.cmd_summary(argparse.Namespace(out=str(out)))
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"toy|openai", "rule_recent|openai"}
    s = summary["toy|openai"]
    assert s["n"] == 5 and s["format_ok"] == 1.0 and s["belief"] == pytest.approx(1.0)
    assert s["retain_curve"]["2-3"]["n"] == 1


def test_from_runs(tmp_path):
    run = tmp_path / "runs" / "toy"
    (run / "memory" / "store").mkdir(parents=True)
    story_file = tmp_path / "toy.json"
    story_file.write_text(json.dumps(toy_story()))
    (run / "config.json").write_text(json.dumps({"story": str(story_file)}))
    (run / "summary.json").write_text("{}")
    (run / "memory" / "store" / "setup.json").write_text(json.dumps({"entities": {}}))
    (run / "memory" / "store" / "w000.json").write_text(json.dumps({"entities": {}, "scene": "kitchen"}))
    events = [{"kind": "plan", "shot": 0, "output": plan_json(0), "latency_s": 1.0, "model": "p", "since": -1},
              {"kind": "commit", "shot": 0}, {"kind": "plan", "shot": 1, "output": plan_json(1), "since": 0}]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    recs = evaluate.run_records(run)
    assert [(r["story"], r["shot"]) for r in recs] == [("toy_terse", 0), ("toy_terse", 1)]
    assert recs[1]["store_before"]["scene"] == "kitchen"


# --------------------------------------------------------------------------------------------------- story generation

def test_storygen_jobs_and_state_tracking():
    jobs = storygen.story_jobs("train")
    ids = [j[1] for j in jobs]
    assert len(jobs) == 64 and len(storygen.THEMES) == 58
    assert (ids[0], ids[49], ids[50], ids[57], ids[58], ids[63]) == (
        "tr000_ed53d5", "tr049_5e1219", "tr050_ca1b01", "tr057_2fc9aa", "tr058_ed53d5", "tr063_d931c6")
    assert jobs[20][1:] == ("tr020_d869d1", 24, 20281029)
    assert [j[1][:5] for j in storygen.story_jobs("hard")][:2] == ["hd00_", "hd01_"]
    for rule in ("1-2 supporting characters", "(in their last state);", "at least 7 recurring", "show only 1-3"):
        assert rule in storygen.BIBLE_USER and rule not in storygen.HARD_BIBLE_USER


def test_storygen_make_story_with_fake_writer():
    bible = {"title": "Tea", "style": "plain",
             "characters": [{"id": "Ana", "name": "Ana", "protagonist": True, "identity": "woman, black hair",
                             "initial": {"outfit": "red coat", "headwear": "none"}}],
             "objects": [{"id": "mug", "name": "Mug", "identity": "white mug", "initial": {"state": "empty",
                                                                                          "colour": "white"}}],
             "locations": [{"id": "kitchen", "name": "Kitchen", "look": "sunny"}, {"id": "garden", "name": "Garden"}],
             "shots": [{"location": "kitchen", "present": ["ana", "mug"], "changes": []},
                       {"location": "kitchen", "present": ["ana"],
                        "changes": [{"entity": "mug", "attr": "state", "value": "full of tea"}]},
                       {"location": "garden", "present": ["ana"],
                        "changes": [{"entity": "ana", "attr": "outfit", "value": "none"}]}]}

    def writer(system, user, temperature, max_tokens, seed):
        if system == storygen.BIBLE_SYSTEM:
            return json.dumps(bible)
        shots = [int(x) for x in re.findall(r"\[shot (\d+)\]", user)]
        return json.dumps({"shots": [{"shot": i, "full": f"full {i}", "terse": f"terse {i}"} for i in shots]})

    out = storygen.make_story(writer, "tea time", "tr999_abc", 3, 7)
    ann = [s["annotation"] for s in out["terse"]["shots"]]
    assert out["terse"]["id"] == "tr999_abc_terse" and out["full"]["shots"][2]["prompt"] == "full 2"
    assert ann[1]["present"] == ["ana", "mug"]  # an entity that changes is visible
    assert ann[1]["entity_state"]["mug"] == {"state": "full of tea; white"}  # extra object attributes join "state"
    assert ann[2]["entity_state"]["ana"]["outfit"] == "no outfit (removed)" and ann[2]["location_visit"] == "new"
    assert "headwear" not in ann[0]["entity_state"]["ana"]
    assert out["terse"]["setup"]["cast"][0]["description"].startswith("Ana, woman, black hair")
    assert out["terse"]["source"]["residual_problems"]  # far too little memory pressure for a real story


# --------------------------------------------------------------------------------------------------- trainer helpers

def test_train_helpers():
    pytest.importorskip("torch")
    from agentic_longlive.rl import train

    cfg = train.build_cfg(json.loads((ROOT / "configs" / "rl" / "grpo.json").read_text()),
                          ["seed=2", "reward.image=false", "refresh_stories=[]", "out=/tmp/x"])
    assert (cfg.seed, cfg.reward["image"], cfg.reward["version"], cfg.refresh_stories) == (2, False, 4, [])
    with pytest.raises(SystemExit):
        train.build_cfg({"klein_urls": "x"})
    assert train.lr_at(cfg, 1) == pytest.approx(2e-5 / 3) and train.lr_at(cfg, 10) == pytest.approx(2e-5)
    rows = [([0] * 10, [1] * 5, "a"), ([0] * 2, [1] * 2, "b"), ([0] * 8, [1] * 8, "c")]
    batches = list(train.micro_batches(rows, 32))
    assert [[r[2] for r in b] for b in batches] == [["c", "a"], ["b"]]  # padded width 16: 2 rows fit in 32
    assert train.group_advantages([0.5, 0.5, 0.51], 0.1, 0.02) is None  # no signal
    adv = train.group_advantages([0.0, 0.1], 0.1, 0.02)  # std 0.05 < floor 0.1
    assert adv == pytest.approx([-0.5, 0.5])
    assert train.group_advantages([0.0, 1.0], 0.1, 0.02) == pytest.approx([-1.0, 1.0])


def test_merge_lora(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from agentic_longlive.rl.merge import merge

    base, ad, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    base.mkdir()
    ad.mkdir()
    w = torch.randn(4, 4).to(torch.bfloat16)
    st.save_file({"model.layers.0.self_attn.q_proj.weight": w, "lm_head.weight": torch.ones(2, 4)},
                 str(base / "model-00001-of-00001.safetensors"))
    (base / "config.json").write_text("{}")
    a, b = torch.randn(2, 4), torch.randn(4, 2)
    st.save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
                  "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b},
                 str(ad / "adapter_model.safetensors"))
    (ad / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    assert merge(base, ad, out) == 1
    merged = st.load_file(str(out / "model-00001-of-00001.safetensors"))
    expect = (w.float() + 2.0 * (b @ a)).to(torch.bfloat16)
    assert torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], expect)
    assert torch.equal(merged["lm_head.weight"], torch.ones(2, 4)) and (out / "config.json").is_file()
