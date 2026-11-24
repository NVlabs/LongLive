# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the robot application: a fake VLM, a fake generator worker on the file queue, ffmpeg mocked."""
import json
import re
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.robotics import agent as agent_mod  # noqa: E402
from agentic_longlive.robotics import llm as llm_mod  # noqa: E402
from agentic_longlive.robotics import video  # noqa: E402
from agentic_longlive.robotics.episode import base_prompt, header, load_episode, ours_prompt  # noqa: E402
from agentic_longlive.robotics.prompts import CAMERA_RETURN_SUFFIX  # noqa: E402
from agentic_longlive.robotics.runner import Seeds, run_episode  # noqa: E402
from agentic_longlive.workers import protocol  # noqa: E402

STORIES = ROOT / "stories" / "robot"
EVENTS = ["hide: the green tube is shut inside the drawer", "hide: the desk leaves the view",
          "return: the desk comes back exactly as left"]


# ---------------------------------------------------------------------------- fakes
class FakeLLM:
    """Answers each agent prompt with a fixed, well-formed JSON object and records every call."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def ask(self, content, max_tokens=1500, max_side=1024):
        txt = [c for c in content if isinstance(c, str)][-1]
        with self.lock:
            self.calls.append({"text": txt, "n_images": len(content) - 1, "max_tokens": max_tokens,
                               "max_side": max_side})
        new = (re.search(r'New instruction(?: for the next 8-second segment)?: "(.*)"', txt) or [None, ""])[1]
        if txt.startswith("You keep the visual memory"):                         # hybrid2 hide observer
            seg = re.search(r'rendered: "(.*)"', txt).group(1)
            if "close it" in seg:
                return {"hides": [{"what": "the green tube in the open drawer", "kind": "inside", "frame": 3},
                                  {"what": "the desk with the closed drawer", "kind": "area", "frame": 5}]}, 0.01
            return {"hides": []}, 0.01
        if "Things that were hidden earlier" in txt:                             # ROUTE6
            areas = re.findall(r"^(H\d+) \[area\]", txt, re.M)
            if "back" in new and len(areas) > 1:
                return {"returns": True, "how": "camera", "id": areas[-1], "what": "the desk"}, 0.01
            return {"returns": False, "how": "", "id": "", "what": ""}, 0.01
        if txt.startswith("A robot video is generated segment by segment"):     # v4 router
            return {"returns": "back" in new, "what": "the desk" if "back" in new else ""}, 0.01
        if "In the next segment something comes back into view" in txt:          # selector
            ids = re.findall(r'"id": "([^"]+)"', txt)
            return {"shows": ids, "prompt": "Turn back to the desk with the drawer."}, 0.01
        if "Rewrite the user's next instruction" in txt:                         # text-only rewrite
            return {"prompt": f"Rewritten: {new}"}, 0.01
        if txt.startswith("You are the memory agent"):                           # observer
            return {"view": "a desk with a drawer", "keep_frame": 3,
                    "objects": [{"name": "green tube", "visible": True, "where": "on the desk",
                                 "look": "green plastic"}],
                    "containers": [{"name": "drawer", "state": "closed", "contents": ""}],
                    "inside_view": None, "crops": [{"name": "green tube", "frame": 3, "bbox": [100, 100, 400, 500]}],
                    "home_view": True}, 0.01
        raise AssertionError(f"unexpected prompt: {txt[:120]}")


class FakeGrounder:
    def boxes(self, im, names, box_thr=0.3, text_thr=0.25):
        W, H = im.size
        return {n: ([0.1 * W, 0.1 * H, 0.4 * W, 0.5 * H], 0.9) for n in names}


def _png(path, color, size=(128, 72)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """Frames, hand-off frames and joining without ffmpeg (segments are dummy files)."""
    def frames(mp4, out_dir, idx):
        out_dir = Path(out_dir)
        return [_png(out_dir / f"f{i:03d}.png", (i % 256, 40, 90)) for i in idx]

    def last_frame(mp4, out_dir):
        return _png(Path(out_dir) / "last.png", (200, 10, 10))

    def concat(segs, out_mp4, crf=20):
        Path(out_mp4).write_text(json.dumps([str(s) for s in segs]))
        return Path(out_mp4)

    monkeypatch.setattr(video, "frames", frames)
    monkeypatch.setattr(video, "last_frame", last_frame)
    monkeypatch.setattr(video, "concat", concat)


@pytest.fixture
def worker(tmp_path):
    """A generator worker thread on a file queue: writes a dummy MP4 per job and records the jobs."""
    queue = tmp_path / "queue"
    jobs = []

    def handler(job):
        jobs.append(job)
        assert Path(job["keyframe"]).is_file() and all(Path(r).is_file() for r in job["refs"])
        Path(job["out"]).write_bytes(b"fake mp4")
        return {"fake": True}

    th = threading.Thread(target=protocol.serve, args=(queue, handler), kwargs={"poll_s": 0.005}, daemon=True)
    th.start()
    yield queue, jobs
    protocol.ensure(queue)
    (queue / "jobs" / "zzz_stop.json").write_text(json.dumps({"id": "zzz_stop", "stop": True}))
    th.join(timeout=10)


@pytest.fixture
def episode(tmp_path):
    ep = {"name": "toy", "title": "toy", "summary": "toy", "camera": "head camera", "robot": "one black gripper",
          "setting": "an office desk with a drawer", "first_frame": "toy.png",
          "segments": [{"instruction": "Put the green tube into the drawer and close it.", "memory_event": EVENTS[0]},
                       {"instruction": "Turn the camera right to the window.", "memory_event": EVENTS[1]},
                       {"instruction": "Turn the camera back to the desk.", "memory_event": EVENTS[2]}]}
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(ep))
    image = _png(tmp_path / "images" / "toy.png", (30, 120, 60))
    return load_episode(path), image


def _run(tmp_path, episode, queue, arms, policy, llm, grounder=None):
    ep, image = episode
    return run_episode(ep, image, queue, tmp_path / "out", arms=arms, policy=policy, llm=llm, grounder=grounder,
                       seeds=Seeds(), job_timeout_s=30, gen_retries=1, retry_wait_s=0)


# ---------------------------------------------------------------------------- end to end
def test_both_arms_v4(tmp_path, episode, worker, no_ffmpeg):
    queue, jobs = worker
    llm = FakeLLM()
    videos = _run(tmp_path, episode, queue, ("base", "ours"), "v4", llm)
    out = tmp_path / "out"
    assert [v.name for v in videos] == ["base.mp4", "ours.mp4"] and all(v.exists() for v in videos)

    # base: 3 segments; ours: segment 0 shared, segments 1-2 rendered -> 5 jobs
    assert len(jobs) == 5
    for job in jobs:
        n = len(job["refs"])
        assert job["ref_seconds"] == [0.25 * k for k in range(n)]
        assert job["video_start_seconds"] == 2.0 and job["ref_scale"] == 0.5 and job["duration_seconds"] == 8
        assert job["refs"][-1] == job["keyframe"]              # the latest frame is the last sink picture
    base_jobs, ours_jobs = jobs[:3], jobs[3:]
    assert [j["seed"] for j in base_jobs] == [1000, 1001, 1002] and [j["seed"] for j in ours_jobs] == [1001, 1002]
    assert base_jobs[0]["keyframe"] == str(episode[1])         # segment 0 starts from the input image
    assert all(j["prompt"].startswith("Picture 1 is the latest frame") for j in base_jobs)
    assert base_jobs[1]["prompt"] == base_prompt(episode[0], "Turn the camera right to the window.")

    base = json.loads((out / "base" / "episode.json").read_text())
    ours = json.loads((out / "ours" / "episode.json").read_text())
    assert len(base["segments"]) == 3 and ours["segments"][0]["shared_with_base"]
    seg1, seg2 = ours["segments"][1], ours["segments"][2]
    # nothing comes back in segment 1: no memory picture, text-only rewrite, planned from the input image only
    assert seg1["pinned"] == [] and seg1["plan"]["observed_through"] == -1 and "Rewritten:" in seg1["prompt"]
    # the camera turns back in segment 2: newest view + newest crop pinned, planned with segment 0 observed (lag 1)
    assert seg2["plan"]["returns"] and seg2["plan"]["observed_through"] == 0
    assert [m["id"] for m in seg2["pinned"]] == ["view:seg1", "object:green tube"]
    assert ours_jobs[1]["prompt"].startswith("Pictures 1-2 are camera frames recorded earlier in this same continuous")
    assert len(ours_jobs[1]["refs"]) == 3 and all("/agent/mem/" in r for r in ours_jobs[1]["refs"][:2])

    windows = ours["agent_windows"]
    assert [(w["plans_seg"], w["observe_seg"]) for w in windows] == [(1, -1), (2, 0)]
    agent_log = json.loads((out / "ours" / "agent.json").read_text())
    assert [t["seg"] for t in agent_log["trace"] if t["role"] == "observer"] == [-1, 0, 1, 2]
    # the agent never sees the scripts' memory-event annotations
    assert not any(e.split(": ", 1)[1] in c["text"] for e in EVENTS for c in llm.calls)


def test_ours_hybrid2_alone(tmp_path, episode, worker, no_ffmpeg):
    queue, jobs = worker
    llm = FakeLLM()
    _run(tmp_path, episode, queue, ("ours",), "hybrid2", llm, grounder=FakeGrounder())
    out = tmp_path / "out"
    assert (out / "ours.mp4").exists() and not (out / "base").exists()
    assert len(jobs) == 3 and jobs[0]["prompt"].startswith("Picture 1 is the latest frame")   # ours renders segment 0

    ours = json.loads((out / "ours" / "episode.json").read_text())
    seg2 = ours["segments"][2]
    # camera return to the area hidden in segment 0: its pre-hide snapshot + the newest object crop, instruction
    # verbatim
    assert seg2["plan"]["target"] == "view:hideH2"
    assert [m["id"] for m in seg2["pinned"]] == ["view:hideH2", "object:green tube"]
    assert seg2["plan"]["prompt"] == ("Turn the camera back to the desk. The desk with the closed drawer"
                                      + CAMERA_RETURN_SUFFIX)
    state = json.loads((out / "ours" / "agent.json").read_text())["state"]
    events = {e["id"]: e for e in state["events"]}
    assert events["H1"]["kind"] == "inside" and events["H2"]["returned_at"] == 2
    zoomed = next(b for b in state["bank"] if b["id"] == "inside:hideH1")
    assert Path(zoomed["path_crop"]).is_file()
    assert any(c["text"].startswith("You keep the visual memory") and c["n_images"] == 13 for c in llm.calls)


def test_hybrid2_container_reopen_uses_zoomed_snapshot(tmp_path):
    ag = agent_mod.HideAgent({"camera": "head camera", "robot": "r", "setting": "s"}, FakeLLM(), tmp_path,
                             grounder=FakeGrounder())
    im = Image.new("RGB", (160, 90), (1, 2, 3))
    cap = "the tube in the drawer, as it looked right before it was hidden (segment 1)"
    ent, p = ag._put("inside", "hideH1", cap, 0, im)
    ent["path_crop"] = str(p.with_name("crop.jpg"))
    pics = ag.pictures({"target": "inside:hideH1", "shows": [], "returns": True})
    assert len(pics) == 1 and pics[0]["path"].endswith("crop.jpg")
    assert pics[0]["caption"].startswith("close view of the tube")
    assert ag.pictures({"target": None, "shows": [], "returns": False}) == []


# ---------------------------------------------------------------------------- units
def test_generator_prompts():
    ep = {"camera": "head camera", "robot": "two grippers", "setting": "a kitchen"}
    assert header(ep).startswith("Realistic first-person footage from the head camera of a robot, showing its two "
                                 "grippers")
    assert "fixed overhead camera looking down at a kitchen" in header(dict(ep, camera="fixed overhead camera"))
    assert "fixed camera watching a robot" in header(dict(ep, camera="fixed third-person camera"))
    assert ours_prompt(ep, "Do it.", []) == base_prompt(ep, "Do it.")
    one = ours_prompt(ep, "Do it.", [{"caption": "the desk."}])
    assert one.startswith("Picture 1 is a camera frame recorded earlier") and "Picture 2 is its latest frame" in one
    assert "Picture 1 shows the desk. When these places" in one
    assert one.endswith("overall_soundscape: Quiet room tone with faint robot motor whirs.")
    two = ours_prompt(ep, "Do it.", [{"caption": "a"}, {"caption": "b"}])
    assert two.startswith("Pictures 1-2 are camera frames") and "Picture 3 is its latest frame" in two


def test_pick_newest_and_crop():
    bank = {"view:input": {"kind": "view"}, "object:tube": {"kind": "object"}, "view:seg1": {"kind": "view"},
            "inside:seg1": {"kind": "inside"}, "view:seg2": {"kind": "view"}}
    ids = ["view:input", "object:tube", "view:seg1", "inside:seg1", "x"]
    assert agent_mod.pick_newest(bank, ids) == ["view:seg1", "inside:seg1"]
    assert agent_mod.pick_newest(bank, ["object:tube"]) == ["object:tube"]
    c = agent_mod.crop16x9(Image.new("RGB", (1344, 768)), [100, 100, 200, 200])
    assert abs(c.width / c.height - 16 / 9) < 0.01 and c.width >= 0.3 * 1344 - 1


def test_llm_retries_unparsable_reply(monkeypatch):
    bodies, replies = [], ["no json here", '```json\n{"returns": true, "what": "the desk"}\n```']

    class Resp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.text}}]}

    def post(url, data, headers, timeout):
        bodies.append(json.loads(data))
        return Resp(replies[len(bodies) - 1])

    monkeypatch.setattr(llm_mod.requests, "post", post)
    client = llm_mod.AgentLLM("http://127.0.0.1:1/v1", retry_wait_s=0)
    out, _ = client.ask([Image.new("RGB", (2048, 1024)), "question"], max_tokens=120, max_side=448)
    assert out == {"returns": True, "what": "the desk"}
    assert bodies[0]["temperature"] == 0 and bodies[0]["max_tokens"] == 120
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[1]["temperature"] == 0.3 and bodies[1]["max_tokens"] == 240
    parts = bodies[0]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url" and parts[-1] == {"type": "text", "text": "question"}


def test_episode_scripts():
    files = sorted(STORIES.glob("*.json"))
    assert len(files) == 32
    eps = [load_episode(f) for f in files]
    bench = [e for e in eps if not e["name"].startswith("demo_")]
    assert len(bench) == 30 and all(len(e["segments"]) == 8 for e in eps)
    assert sum(s["memory_event"].startswith("return") for e in bench for s in e["segments"]) == 41
    for e in eps:
        assert e["first_frame"].endswith(".png") and e["source"]["dataset"] == "DAGroup-PKU/RoVid-X"
        assert set(e["segments"][0]) == {"instruction", "memory_event"}
