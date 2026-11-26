# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the LongStory-Online evaluation: benchmark files, frame sampling, judge plumbing with a fake judge on
a tiny synthetic run, metric definitions, aggregation / bootstrap, and the evaluate.py command line."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.eval import frames as F  # noqa: E402
from agentic_longlive.eval import judge as J  # noqa: E402
from agentic_longlive.eval import metrics as M  # noqa: E402
from agentic_longlive.eval.measure import load_measurements, measure_run  # noqa: E402
from agentic_longlive.eval.mir import mir_score  # noqa: E402
from agentic_longlive.eval.runs import find_runs, load_run  # noqa: E402
from agentic_longlive.story import Story  # noqa: E402

BENCH = ROOT / "stories" / "longstory_online"
SCENES = ["kitchen", "kitchen", "garden", "kitchen", "garden", "garden"]
FRAMES_PER_SHOT = 25


# ---------------------------------------------------------------------------------------------- benchmark files
def test_benchmark_files():
    files = sorted(BENCH.glob("*.json"))
    assert len(files) == 25
    stories = [Story.load(f) for f in files]
    assert sum(len(s) for s in stories) == 857
    hard = [s for f, s in zip(files, stories) if f.stem.startswith("hd")]
    assert len(hard) == 16 and sum(len(s) for s in hard) == 646
    for f in files + sorted((BENCH / "full").glob("*.json")):
        raw = json.loads(f.read_text())
        assert "source" not in raw
        assert all("annotation" in shot and shot["annotation"].get("scene") for shot in raw["shots"])
    # the full-prompt control has the same hidden annotations
    for f in files:
        full = json.loads((BENCH / "full" / f.name).read_text())
        assert [s["annotation"] for s in full["shots"]] == [s["annotation"] for s in json.loads(f.read_text())["shots"]]
    vis = json.loads((BENCH / "eval" / "expected_visible.json").read_text())
    assert sorted(vis) == [f.stem for f in files]
    assert sum(len(v) for v in vis.values()) == 857 and sum(sum(v.values()) for v in vis.values()) == 349
    assert sum(len(J.place_pairs(s)) for s in stories) == 140  # revisits


def test_benchmark_checksums():
    lines = (BENCH / "SHA256SUMS").read_text().split("\n")
    for line in filter(None, lines):
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((BENCH / name).read_bytes()).hexdigest() == digest, name


# ---------------------------------------------------------------------------------------------- frames
def test_frame_indices():
    assert [F.fraction_index(253, f) for f in F.JUDGE_FRACTIONS] == [10, 88, 164, 244]
    assert F.uniform_indices(160, 16)[:3] == [5, 15, 25]
    assert F.evenly_spaced(10, 4) == [0, 3, 6, 9]
    assert F.evenly_spaced(1, 4) == [0]


def _frame(shot: int, t: int):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[..., 0] = 8 * t  # frame index in the red channel
    image[..., 1] = 30 * shot  # shot index in the green channel
    return image


class _FakeVideo:
    """Stand-in for frames.Video when imageio-ffmpeg is not installed (reads the synthetic pattern directly)."""

    def __init__(self, path):
        self.path = str(path)
        self.n, self.fps = FRAMES_PER_SHOT, 24.0
        self.shot = int(re.search(r"shot_(\d+)", self.path).group(1))

    duration = F.Video.duration
    at = F.Video.at

    def frames(self, indices):
        return [_frame(self.shot, int(i)) for i in indices]


@pytest.fixture
def videos(monkeypatch):
    """Writer of tiny shot videos: real mp4 files when imageio-ffmpeg is available, else a mocked decoder."""
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401

        def write(path: Path, shot: int):
            imageio.mimwrite(str(path), [_frame(shot, t) for t in range(FRAMES_PER_SHOT)], fps=24,
                             codec="libx264", quality=10, macro_block_size=16)
    except ImportError:
        monkeypatch.setattr(F, "Video", _FakeVideo)

        def write(path: Path, shot: int):
            path.write_bytes(b"fake mp4")
    return write


def test_video_sampling(tmp_path, videos):
    path = tmp_path / "shot_003.mp4"
    videos(path, 3)
    video = F.Video(path)
    assert video.n == FRAMES_PER_SHOT
    for frame, index in zip(video.at(F.JUDGE_FRACTIONS), [F.fraction_index(video.n, f) for f in F.JUDGE_FRACTIONS]):
        assert abs(float(frame[..., 0].mean()) - 8 * index) < 10 and abs(float(frame[..., 1].mean()) - 90) < 10
    paths = F.save_judge_frames(str(path), tmp_path / "eval", 3)
    assert [p.name for p in paths] == [f"shot_003_{k}.jpg" for k in range(4)] and all(p.is_file() for p in paths)


# ---------------------------------------------------------------------------------------------- tiny run
def _story(tmp: Path) -> Path:
    shots = []
    for i, scene in enumerate(SCENES):
        shots.append({"prompt": f"Shot {i}. Mia works in the {scene}.",
                      "annotation": {"scene": scene, "camera": "medium shot", "beat": f"Mia in the {scene}",
                                     "state": {"apron": "red apron", "hands": "flour on hands"}}})
    story = {"id": "tiny_terse", "setup": {"title": "Tiny", "style": "live action",
                                          "cast": [{"id": "mia", "name": "Mia", "description": "Mia, a baker."}]},
             "shots": shots}
    path = tmp / "stories" / "tiny.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(story))
    (tmp / "stories" / "eval").mkdir(exist_ok=True)
    (tmp / "stories" / "eval" / "expected_visible.json").write_text(json.dumps(
        {"tiny": {"0": True, "1": True, "2": False, "3": True, "4": True, "5": True}}))
    return path


def _run(tmp: Path, name: str, story: Path, write) -> Path:
    run = tmp / "outputs" / name / "tiny"
    (run / "shots").mkdir(parents=True)
    per_shot = []
    for i in range(len(SCENES)):
        path = run / "shots" / f"shot_{i:03d}.mp4"
        write(path, i)
        per_shot.append({"shot": i, "video_path": f"/elsewhere/{path.name}"})  # copied run: falls back to shots/
    (run / "summary.json").write_text(json.dumps({"story": "tiny_terse", "per_shot": per_shot}))
    (run / "config.json").write_text(json.dumps({"story": "stories/longstory_online/tiny.json"}))
    return run.parent


class _Result:
    def __init__(self, data):
        self.data = data


class FakeJudge:
    """Deterministic judge. ``good=False``: shot 2 is judged elsewhere, shot 3 shows another person, shot 4 hides the
    protagonist, shot 5 has artifacts; odd shots contradict one state fact."""

    name = "fake:judge"

    def __init__(self, good: bool = False):
        self.good = good
        self.calls = []

    def chat(self, system, content, json_mode=False, temperature=None):
        assert json_mode and temperature == 0.0
        text, images = content[0], [Path(c) for c in content[1:]]
        assert all(p.is_file() for p in images)
        shots = [int(re.search(r"shot_(\d+)", p.name).group(1)) for p in images]
        self.calls.append((system, text, shots))
        if system == J.SHOT_JUDGE_SYSTEM:
            i = shots[0]
            assert len(images) == 4 and "LOCATIONS:\n- kitchen: Shot 0." in text and "EXPECTED STATE:\n- apron" in text
            bad = not self.good
            return _Result({"location": "Other" if bad and i == 2 else SCENES[i].upper(),
                            "opening_location": SCENES[i], "shot_size": "medium", "framing_match": True,
                            "content": 2, "protagonist_visible": not (bad and i == 4),
                            "protagonist_match": not (bad and i == 3),
                            "facts": [{"key": "apron", "verdict": "confirmed"},
                                      {"key": "hands", "verdict": "contradicted" if i % 2 else "confirmed"}],
                            "artifacts": bad and i == 5, "notes": ""})
        if system == J.PAIR_JUDGE_SYSTEM:
            assert len(images) == 4 and text.startswith("LOCATION: ")
            return _Result({"place": 5, "person": 4, "shared": "", "differs": ""})
        if system == J.PROTAGONIST_JUDGE_SYSTEM:
            assert len(images) == 8 and text.startswith("PROTAGONIST: Mia, a baker.")
            return _Result({"visible_a": True, "visible_b": True, "person": 4, "differs": ""})
        raise AssertionError("unexpected judge call")


def test_measure_and_score(tmp_path, videos):
    story = _story(tmp_path)
    root = _run(tmp_path, "method_a", story, videos)
    run = load_run(find_runs(root)[0], tmp_path / "stories")
    assert run.key == "tiny" and sorted(run.videos) == list(range(6))
    vis = J.load_visibility(tmp_path / "stories" / "eval" / "expected_visible.json")["tiny"]
    judge = FakeJudge()
    seen = []

    def imaging(frames):
        seen.append(len(frames))
        return 0.6

    measured = measure_run(run, judge, vis, imaging=imaging, workers=2, log=lambda *_: None)
    kinds = [c[0] for c in judge.calls]
    assert kinds.count(J.SHOT_JUDGE_SYSTEM) == 6
    assert kinds.count(J.PAIR_JUDGE_SYSTEM) == 2  # revisits only: (1, 3) and (2, 4)
    # protagonist pairs vs the first appearance (shot 0), planned-visible shots where it was seen: 1, 3, 5
    assert sorted(c[2][4] for c in judge.calls if c[0] == J.PROTAGONIST_JUDGE_SYSTEM) == [1, 3, 5]
    assert seen == [16] * 6
    assert measured["suite"]["judge"] == "fake:judge" and measured["identity"]["ref"] == 0
    assert sorted(measured["suite"]["pairs"]) == ["1:3", "2:4"]

    # cached: a second pass makes no judge call
    n = len(judge.calls)
    measure_run(run, judge, vis, workers=2, log=lambda *_: None)
    assert len(judge.calls) == n

    scores = M.run_scores(run.story, load_measurements(run), vis)
    means = scores["means"]
    assert means["location"] == pytest.approx(2 / 3)  # scene changes at shots 2, 3, 4; shot 2 judged elsewhere
    assert means["revisit"] == pytest.approx(0.5)  # (1, 3) at planned places -> 1; (2, 4) gated (shot 2) -> 0
    assert means["identity"] == pytest.approx(4 / 5)  # visible on 0, 1, 2, 3, 5; shot 3 is another person
    assert means["same_protagonist"] == pytest.approx((0.75 * 3 + 0.0) / 4)  # shots 1, 3, 5 rated 4; shot 4 unseen
    assert means["state"] == pytest.approx(0.75)
    assert means["artifacts"] == pytest.approx(1 / 6)
    assert means["framing"] == 1.0 and means["imaging"] == pytest.approx(0.6)
    assert scores["story_level"]["imaging_drift"] is None  # stories shorter than two minutes have no drift


def test_aggregation_and_bootstrap():
    def scores(values):
        per = {m: {} for m in M.SHOT_METRICS}
        per["state"] = dict(enumerate(values))
        return {"per_shot": per, "story_level": {"imaging_drift": None, "mir": None}, "judged_shots": len(values)}

    a = {"s1": scores([1.0, 1.0, 0.0]), "hd01": scores([1.0])}
    b = {"s1": scores([0.0, 1.0, 0.0]), "hd01": scores([0.0])}
    summary = M.summarize([a, a], "all")
    assert summary["state"] == pytest.approx(0.75) and summary["seeds"] == 2 and summary["stories"] == 2
    assert M.summarize([a], "hard")["state"] == 1.0 and M.summarize([a], "hand_written")["state"] == pytest.approx(2 / 3)
    d = M.paired([a], [b], "state")
    assert d["diff"] == pytest.approx(0.5) and d["n"] == 4 and d["lo"] <= d["diff"] <= d["hi"] and d["significant"]
    assert M.paired([a], [a], "state")["significant"] is False
    assert M.paired([a], [b], "state") == M.paired([a], [b], "state")  # seeded
    table = M.markdown_table({"A": summary})
    assert "| A | 2 | 2 | 4 | 75.0 |" in table


def test_mir_score():
    e1, e2, e3 = np.eye(3)
    near = (e1 + 0.1 * e2) / np.linalg.norm(e1 + 0.1 * e2)
    dets = [{"shot": 0, "emb": e1.tolist()}, {"shot": 1, "emb": near.tolist()}, {"shot": 2, "emb": e3.tolist()}]
    value, clusters = mir_score(dets)
    assert clusters == 1 and value == pytest.approx(float(e1 @ near))
    assert mir_score(dets[:1]) == (0.0, 0)


def test_cli(tmp_path, videos):
    story = _story(tmp_path)
    for name, good in (("a", False), ("b", True)):
        root = _run(tmp_path, name, story, videos)
        run = load_run(find_runs(root)[0], tmp_path / "stories")
        vis = J.load_visibility(tmp_path / "stories" / "eval" / "expected_visible.json")["tiny"]
        measure_run(run, FakeJudge(good), vis, workers=2, log=lambda *_: None)
    out = tmp_path / "results"
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--runs", f"A={tmp_path / 'outputs/a'}",
           f"B={tmp_path / 'outputs/b'}", "--compare", "A", "B", "--stories", str(tmp_path / "stories"),
           "--judge", "none", "--out", str(out), "--draws", "200"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    scores = json.loads((out / "scores.json").read_text())["methods"]
    assert scores["A"]["subsets"]["all"]["artifacts"] == pytest.approx(1 / 6)
    assert scores["B"]["subsets"]["all"]["location"] == 1.0
    compare = json.loads((out / "compare.json").read_text())["subsets"]["all"]
    assert compare["artifacts"]["diff"] == pytest.approx(1 / 6) and compare["location"]["diff"] == pytest.approx(-1 / 3)
    assert "Same prot." in (out / "table.md").read_text()
    assert (tmp_path / "outputs" / "a" / "tiny" / "eval" / "scores.json").is_file()
