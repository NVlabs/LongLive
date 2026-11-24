# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.config import load_config  # noqa: E402
from agentic_longlive.runtime.session import LiveSession  # noqa: E402
from agentic_longlive.story import Story  # noqa: E402

STORY = ROOT / "stories/longstory_online/chef_mei_long_day.json"
ABLATIONS = sorted((ROOT / "configs/ablations").glob("*.yaml"))


def _digest(items):
    h = hashlib.sha256()
    for item in items:
        h.update(hashlib.sha256(item.encode()).digest())
    return h.hexdigest()[:16]


def run(tmp_path, overrides=(), overlays=(), story=None, shots=5):
    cfg = load_config(str(ROOT / "configs/mock.yaml"),
                      [f"output_dir={tmp_path}", f"max_shots={shots}", "backend.params.shot_seconds=0.05",
                       *overrides], [str(o) for o in overlays])
    story = story or Story.load(STORY)
    session = LiveSession(cfg, story)
    summary = session.run()
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    return summary, trace, story


def test_mock_session_runs_and_respects_the_online_boundary(tmp_path):
    summary, trace, story = run(tmp_path)
    assert summary["shots"] == 5 and summary["agent"]["error"] is None
    for event in trace:
        if event["kind"] == "plan":
            w = event["shot"]
            assert event["revealed"] == w + 1 and not event["oracle"]
            assert event["fingerprint"] == _digest(story.prompts[: w + 1])
            assert all(p not in event["user"] for p in story.prompts[w + 1:])
    for shot in summary["per_shot"]:
        assert shot["snapshot"] is not None and shot["snapshot"]["watermark"] <= shot["shot"] - 1
        assert shot["lag"] >= 0


def test_future_prompts_do_not_change_memory(tmp_path):
    base = Story.load(STORY)
    changed = Story.load(STORY)
    changed.prompts = base.prompts[:3] + ["A completely different future shot on a beach." for _ in base.prompts[3:]]
    _, trace_a, _ = run(tmp_path / "a", story=base, shots=6, overrides=["runtime.mode=sync"])
    _, trace_b, _ = run(tmp_path / "b", story=changed, shots=6, overrides=["runtime.mode=sync"])
    plans_a = {e["shot"]: e["output"] for e in trace_a if e["kind"] == "plan"}
    plans_b = {e["shot"]: e["output"] for e in trace_b if e["kind"] == "plan"}
    for w in range(3):  # memory built from prompts 0..2 is identical
        assert plans_a[w] == plans_b[w]
    assert _images_until(trace_a, 2) == _images_until(trace_b, 2)


def _images_until(trace, last_shot):
    """Image prompts produced while the agent processed prompts 0..last_shot (plus session setup)."""
    shot, prompts = -1, []
    for event in trace:
        if event["kind"] == "plan":
            shot = event["shot"]
        elif event["kind"] == "image" and shot <= last_shot:
            prompts.append(event["prompt"])
    return sorted(prompts)


def test_sync_mode_uses_memory_from_the_previous_prompt(tmp_path):
    summary, _, _ = run(tmp_path, overrides=["runtime.mode=sync", "backend.params.shot_seconds=0.0"])
    assert all(s["lag"] == 0 for s in summary["per_shot"])


def test_no_aux_prefill_prefills_inline(tmp_path):
    summary, trace, _ = run(tmp_path, overrides=["runtime.aux_prefill=false"])
    assert not [e for e in trace if e["kind"] == "prefill_ready"]
    assert all(s["snapshot"] is not None for s in summary["per_shot"])


@pytest.mark.parametrize("overlay", ABLATIONS, ids=[p.stem for p in ABLATIONS])
def test_every_ablation_runs(tmp_path, overlay):
    summary, trace, _ = run(tmp_path, overlays=[overlay], shots=3)
    assert summary["shots"] == 3 and summary["agent"]["error"] is None
    if overlay.stem == "no_memory":
        assert summary["injection"] == "none" and all(s["snapshot"] is None for s in summary["per_shot"])
    if overlay.stem == "oracle_next_prompt":
        assert any(e["kind"] == "plan" and e["oracle"] for e in trace)
    if overlay.stem == "anchor_only":
        assert not [e for e in trace if e["kind"] == "plan"]


def test_slow_agent_batches_catch_up_and_pipelines_entities(tmp_path):
    summary, trace, story = run(tmp_path, shots=8, overrides=["agent.image.params.latency_s=0.15",
                                                              "backend.params.shot_seconds=0.1"])
    plans = [e for e in trace if e["kind"] == "plan"]
    assert any(e["shot"] - e["since"] > 1 for e in plans), "the slow agent should have fallen behind"
    for e in plans:  # one call integrates every prompt since the last update, none from the future
        for i in range(e["since"] + 1, e["shot"] + 1):
            assert story.prompts[i][:200] in e["user"]
        assert all(p[:200] not in e["user"] for p in story.prompts[e["shot"] + 1:])
    assert [e["shot"] for e in plans] == sorted(e["shot"] for e in plans)
    commits = [e for e in trace if e["kind"] == "commit" and e["accepted"]]
    assert commits and all(c["snapshot"]["watermark"] in {e["shot"] for e in plans} for c in commits)
    paints = [e for e in trace if e["kind"] == "entity_painted" and e["shot"] >= 0]
    first_commit = {}
    for c in commits:
        first_commit.setdefault(c["shot"], c["t"])
    # pipelining: some watermark commits round 0 before its entity redraws finish
    assert any(first_commit.get(p["shot"], 1e9) < p["t"] for p in paints)
    assert summary["agent"]["error"] is None


def test_no_pipeline_waits_for_entities(tmp_path):
    _, trace, _ = run(tmp_path, shots=4, overrides=["agent.image.params.latency_s=0.05", "runtime.mode=sync"],
                      overlays=[ROOT / "configs/ablations/no_pipeline.yaml"])
    commits = {}
    for c in (e for e in trace if e["kind"] == "commit" and e["accepted"]):
        commits.setdefault(c["shot"], c["t"])
    for p in (e for e in trace if e["kind"] == "entity_painted" and e["shot"] >= 0):
        assert p["t"] <= commits.get(p["shot"], 1e9)


def test_commit_min_score_keeps_memory_fresh_under_a_strict_checker(tmp_path):
    strict = ["agent.checker.params.pass_score=10", "runtime.mode=sync", "backend.params.shot_seconds=0.0",
              "agent.commit_first_round=false"]
    _, trace, _ = run(tmp_path / "strict", overrides=strict)
    assert not [e for e in trace if e["kind"] == "commit" and e["accepted"]], "nothing passes a strict checker"
    summary, trace, _ = run(tmp_path / "floor", overrides=strict + ["agent.commit_min_score=6"])
    commits = {e["shot"] for e in trace if e["kind"] == "commit" and e["accepted"]}
    assert commits == set(range(5)) and all(s["lag"] == 0 for s in summary["per_shot"][1:])
    assert all(e["snapshot"]["score"] >= 6 for e in trace if e["kind"] == "commit")


def test_entity_redraws_queue_behind_round_zero(tmp_path):
    _, trace, _ = run(tmp_path, shots=4, overrides=["agent.image.params.latency_s=0.02"])
    for w in range(4):
        rounds = [e["t"] for e in trace if e["kind"] == "refine_round" and e["shot"] == w and e["round"] == 0]
        if not rounds:  # prompt w was integrated into a later update (catch-up)
            continue
        images = [e for e in trace if e["kind"] == "image" and f"/w{w:03d}_r0_" in e["path"]]
        redraws = [e for e in trace if e["kind"] == "image" and "/entities/" in e["path"] and e["t"] > rounds[0]]
        if images and redraws:
            assert min(e["t"] for e in images) <= min(e["t"] for e in redraws) + 0.05


def test_refinement_yields_to_a_newer_prompt_while_waiting_for_redraws(tmp_path):
    _, trace, _ = run(tmp_path, shots=6, overrides=["agent.image.params.latency_s=0.12",
                                                    "backend.params.shot_seconds=0.15"])
    stops = [e for e in trace if e["kind"] == "refine_stopped"]
    assert any("waiting for entity redraws" in e["reason"] for e in stops)
    for e in stops:  # a stopped update never delays the next plan by a whole round
        nxt = [p for p in trace if p["kind"] == "plan" and p["shot"] > e["shot"]]
        assert not nxt or nxt[0]["t"] - nxt[0]["latency_s"] >= e["t"] - 0.2


def test_first_round_is_committed_even_if_it_fails_its_check(tmp_path):
    strict = ["agent.checker.params.pass_score=10", "runtime.mode=sync", "backend.params.shot_seconds=0.0"]
    summary, trace, _ = run(tmp_path, overrides=strict)
    commits = [e for e in trace if e["kind"] == "commit" and e["accepted"]]
    assert {e["shot"] for e in commits} == set(range(5)) and all(s["lag"] == 0 for s in summary["per_shot"][1:])
    assert all(e["round"] == 0 and e["snapshot"]["passed"] is False for e in commits)  # failing rounds never replace
    assert any(e["kind"] == "refine_round" and e["round"] > 0 for e in trace), "refinement still runs"


def test_a_slow_g_does_not_hold_back_the_m_commit(tmp_path):
    _, trace, _ = run(tmp_path, shots=4, overrides=["agent.image.params.slow_if='evenly lit; no room'",
                                                    "agent.image.params.slow_s=0.6", "agent.refine_rounds=1"])
    checked = 0
    for w in range(4):
        g = [e["t"] for e in trace if e["kind"] == "image" and f"/G/w{w:03d}_r0_" in e["path"]]
        commits = [e for e in trace if e["kind"] == "commit" and e["shot"] == w and e["accepted"]]
        if g and commits:
            checked += 1
            assert commits[0]["t"] < min(g), "M of round 0 is committed before the slow G arrives"
            assert any(c["snapshot"]["g"] and f"/G/w{w:03d}_r0_" in c["snapshot"]["g"] for c in commits[1:])
    assert checked


def test_replay_injects_the_recorded_memory_without_an_agent(tmp_path):
    from agentic_longlive.memory.snapshot import MemorySnapshot

    ref, _, _ = run(tmp_path / "ref", shots=6)
    recorded = [s["memory"] for s in ref["per_shot"]]
    assert all(m is None or MemorySnapshot.from_json(m).to_json() == m for m in recorded)
    summary, trace, _ = run(tmp_path / "replay", [f"runtime.replay_memory={tmp_path / 'ref'}"], shots=6)
    assert summary["agent"]["plans"] == 0 and summary["replay_memory"]
    assert [s["memory"] for s in summary["per_shot"]] == recorded
    assert [s["lag"] for s in summary["per_shot"]] == [s["lag"] for s in ref["per_shot"]]
    only_g, _, _ = run(tmp_path / "g", [f"runtime.replay_memory={tmp_path / 'ref'}", "memory.m_slots=0"], shots=6)
    assert all(not s["memory"] or (s["memory"]["m"] == [] and s["memory"]["g"]) for s in only_g["per_shot"])
    one_m, _, _ = run(tmp_path / "m1", [f"runtime.replay_memory={tmp_path / 'ref'}", "memory.use_g=false",
                                        "memory.m_slots=1"], shots=6)
    for s, r in zip(one_m["per_shot"], recorded):
        if s["memory"]:
            assert s["memory"]["g"] is None and s["memory"]["m"] == r["m"][-1:]
