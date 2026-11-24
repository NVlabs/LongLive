# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the generator workers' pure parts: layouts, foreground tokens, picture sizes, prompt rewriting, the
FastH3 request with a mocked server, and the UniServe sink patch. Tests that need torch are skipped without it."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_longlive.workers import fasth3, patch_uniserve_sink  # noqa: E402
from agentic_longlive.workers.longlive_h3 import text_encoder, worker  # noqa: E402

R5_PROMPT = (
    "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
    "mark of the target video; Picture 2 (from Shot 2) aligns with the 0.25-second mark of the target video."
    "\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, Mei, a chef in a white jacket, as "
    "established by Picture 1. [Shot 2] At 00:00.250, the camera cuts to the restaurant kitchen, as established by "
    "Picture 2. [Shot 3] At 00:02.000, the camera cuts to Mei plating pasta at the pass."
    "\n\noverall_soundscape: pans sizzle.")
EARLIER_EXPECTED = (
    "Pictures 1-2 are frames from earlier shots of this story, shown before the target video; the target video starts "
    "after a cut and shows only the shot below. Keep the people, animals and objects they show looking the same when "
    "the shot includes them.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, Mei plating pasta "
    "at the pass.\n\noverall_soundscape: pans sizzle.")


def _png(path, size, color=(120, 60, 30)):
    Image.new("RGB", size, color).save(path)
    return str(path)


# --------------------------------------------------------------------------------------------- LongLive-H3 layout
def test_sink_layout_ticks():
    offsets, shift, name = worker.memory_layout(3, [0.0, 0.25, 0.5], 2.0)
    assert (offsets, shift, name) == ([0.0, 10.0, 20.0], 80.0, "sink")
    assert worker.memory_layout(1, [0.0], 2.0) == ([0.0], 80.0, "sink")


def test_other_layouts():
    assert worker.memory_layout(2, [], 0.0) == ([0.0, 1.0], 2.0, "ref2va")
    assert worker.memory_layout(2, None, None) == ([0.0, 1.0], 2.0, "ref2va")
    assert worker.memory_layout(0, [], 0.0) == ([], 0.0, "t2va")
    assert worker.memory_layout(0, [], 2.0) == ([], 80.0, "shift")


@pytest.mark.parametrize("secs,start", [([0.5, 0.25], 2.0), ([0.0, 2.5], 2.0), ([0.0], 2.0), ([0.0, 0.25], None)])
def test_bad_sink_layouts(secs, start):
    with pytest.raises(ValueError):
        worker.memory_layout(2, secs, start)


def test_shot_frames():
    assert worker.shot_frames(None, 192) == 192
    assert worker.shot_frames(8, 192) == 192
    assert worker.shot_frames(8.0, 120) == 192
    with pytest.raises(ValueError):
        worker.shot_frames(5, 192)


def test_token_keep_synthetic_mask(tmp_path):
    mask = np.zeros((768, 1344), dtype=np.uint8)
    mask[:, :672] = 255  # left half foreground
    mask[0:32, 1312:1344] = 255  # one token at the top right
    mask[736:768, 1280:1283] = 255  # 3 of 32 columns (9.4% < 10%) of a bottom token
    Image.fromarray(mask).save(tmp_path / "m.png")
    keep = worker.token_keep(tmp_path / "m.png", 1344, 768)
    assert keep.shape == (24, 42)
    assert keep[:, :21].all() and not keep[1:, 21:].any()
    assert keep[0, 41] and keep[0, 21:41].sum() == 0
    assert not keep[23, 40]


def test_token_keep_cover_resizes_mask(tmp_path):
    small = np.zeros((384, 672), dtype=np.uint8)
    small[:, :336] = 255  # same picture at half size: same tokens
    Image.fromarray(small).save(tmp_path / "half.png")
    keep = worker.token_keep(tmp_path / "half.png", 1344, 768)
    assert keep[:, :21].all() and not keep[:, 21:].any()


def test_memory_keep(tmp_path):
    pytest.importorskip("torch")
    mask = np.zeros((768, 1344), dtype=np.uint8)
    mask[:384] = 255  # top half
    Image.fromarray(mask).save(tmp_path / "top.png")
    Image.fromarray(np.zeros_like(mask)).save(tmp_path / "empty.png")
    assert worker.memory_keep(None, 2, 1344, 768) is None
    assert worker.memory_keep([None, None], 2, 1344, 768) is None
    keep = worker.memory_keep([None, str(tmp_path / "top.png"), str(tmp_path / "empty.png")], 3, 1344, 768)
    per = 24 * 42
    assert keep.shape == (3 * per,)
    assert bool(keep[:per].all())  # no mask: every token
    assert int(keep[per:2 * per].sum()) == 12 * 42  # top half
    assert bool(keep[2 * per:].all())  # a mask without foreground keeps every token


def test_longlive_h3_geometry():
    pytest.importorskip("torch")
    from agentic_longlive.workers.longlive_h3 import core

    g = core.Geometry(192)
    assert (g.k, g.n_lat, g.n_audio, g.blocks) == (11, 57, 320, 12)
    assert (g.lh, g.lw, g.spatial_rows) == (48, 84, 1008)
    assert g.video_edges[:3] == (0, 5, 10) and g.video_edges[-1] == 57
    assert g.audio_edges[:3] == (0, 29, 57) and g.audio_edges[-1] == 320
    assert sum(g.chunk_rows(b) for b in range(g.blocks)) == 57 * 1008 + 2 * 320
    with pytest.raises(ValueError):
        core.Geometry(190)


def test_sigma_grids_and_patchify():
    torch = pytest.importorskip("torch")
    from agentic_longlive.workers.longlive_h3 import core

    vg, ag = core.sigma_grids("cpu")
    assert vg[0] == 1 and vg[-1] == 0 and ag[0] == 1 and ag[-1] == 0
    assert torch.allclose(vg, torch.tensor([1.0, 24 / 25, 6 / 7, 0.0]))
    assert torch.allclose(ag, torch.tensor([1.0, 6 / 7, 3 / 5, 0.0]))
    g = core.Geometry(22, 128, 64)
    lat = torch.randn(24, g.n_lat, g.lh, g.lw)
    rows = core.patchify(lat)
    assert rows.shape == (g.n_lat * g.spatial_rows, 96)
    audio = torch.randn(2 * g.n_audio, 32)
    video, audio_latents = core.unpack(rows, audio, g)
    assert torch.equal(video[0], lat) and audio_latents.shape == (2, 32, g.n_audio)


def test_longlive_h3_rejects_unsupported_jobs():
    w = worker.LongLiveH3Worker.__new__(worker.LongLiveH3Worker)  # no GPU: only the job checks run
    w.torch, w.core = None, None
    with pytest.raises(ValueError, match="keyframe"):
        w.handle({"keyframe": "/x.png", "prompt": "p", "seed": 0, "out": "/x.mp4"})
    with pytest.raises(ValueError, match="ref_scale"):
        w.handle({"ref_scale": 0.5, "prompt": "p", "seed": 0, "out": "/x.mp4"})


# ----------------------------------------------------------------------------------- text-encoder service helpers
@pytest.mark.parametrize("size", [(1024, 576), (1536, 1024), (700, 900), (1344, 768), (100, 50)])
def test_picture_cover_resize_size(tmp_path, size):
    arr = text_encoder.cover_resize(_png(tmp_path / "p.png", size), 1344, 768)
    assert arr.shape == (768, 1344, 3) and arr.dtype == np.uint8


def test_picture_cover_resize_center_crop(tmp_path):
    im = np.zeros((1000, 1344, 3), dtype=np.uint8)
    im[116:884] = 255  # the central 768 rows survive the crop
    Image.fromarray(im).save(tmp_path / "tall.png")
    assert text_encoder.cover_resize(tmp_path / "tall.png", 1344, 768).min() == 255


def test_request_key_is_order_independent():
    a = text_encoder.request_key({"kind": "image", "path": "/a.png", "seed": 1, "width": 1344, "height": 768})
    b = text_encoder.request_key({"height": 768, "width": 1344, "seed": 1, "path": "/a.png", "kind": "image"})
    assert a == b and len(a) == 64


# ----------------------------------------------------------------------------------------------------- FastH3
def test_earlier_prompt():
    assert fasth3.earlier_prompt(R5_PROMPT, 2) == EARLIER_EXPECTED
    assert fasth3.earlier_prompt(R5_PROMPT, 1).startswith("Picture 1 is a frame from earlier shots")
    plain = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a robot arm opens the drawer."
    assert fasth3.earlier_prompt(plain, 2) == plain


def test_fasth3_picture_sizes(tmp_path):
    assert fasth3.sink_size(0.5) == (672, 384)
    assert fasth3.sink_size(1.0) == (1344, 768)
    im = Image.open(_png(tmp_path / "p.png", (1024, 576)))
    assert fasth3.cover_resize(im, *fasth3.sink_size(0.5)).size == (672, 384)
    assert fasth3.cover_resize(Image.open(_png(tmp_path / "t.png", (600, 900))), 1344, 768).size == (1344, 768)
    assert fasth3.shrink(Image.open(_png(tmp_path / "r.png", (1536, 1024))), 0.5).size == (768, 512)
    assert fasth3.shrink(Image.open(_png(tmp_path / "s.png", (1000, 700))), 0.5).size == (512, 352)
    assert fasth3.shrink(Image.open(_png(tmp_path / "u.png", (1000, 700))), 1.0).size == (1000, 700)


def test_fasth3_sink_plan_and_body():
    assert fasth3.sink_plan([0.0, 0.25, 0.5], 2.0) == {"image_times": [0.0, 10.0, 20.0], "target_start": 80.0}
    body = fasth3.request_body("FastH3-Omni", "p", ["/m/a.png", "/m/b.png"], 7, keyframe="/m/k.png")
    assert body["task"] == "ref2va" and body["seed"] == 7
    assert body["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8}
    assert [c["role"] for c in body["conditions"]] == ["keyframe", "reference", "reference"]
    assert body["conditions"][0] == {"type": "image", "uri": "file:///m/k.png", "role": "keyframe", "frame_index": 0}


class FakeServer:
    """Mocked UniServe /v1/videos API; records each request with the sink plan visible at request time."""

    def __init__(self, plan_path, status="completed"):
        self.plan_path, self.status = Path(plan_path), status
        self.posts, self.deleted = [], []

    def __call__(self, method, url, body=None, timeout=60):
        path = url.split("127.0.0.1:18083", 1)[1]
        if method == "POST" and path == "/v1/videos":
            plan = json.loads(self.plan_path.read_text()) if self.plan_path.exists() else None
            self.posts.append((json.loads(body), plan))
            return json.dumps({"id": f"v{len(self.posts)}"}).encode()
        if method == "GET" and path.endswith("/content"):
            return b"MP4"
        if method == "GET" and path.startswith("/v1/videos/"):
            return json.dumps({"status": self.status, "error": {"message": "boom"},
                               "stage_durations": {"encoding": 0.1, "preparing": 0.5, "denoising": 4.0,
                                                   "decoding": 1.0, "finalizing": 0.6}}).encode()
        if method == "DELETE":
            self.deleted.append(path.rsplit("/", 1)[1])
            return b"{}"
        raise AssertionError(f"unexpected {method} {url}")


def _worker(tmp_path, status="completed"):
    media = tmp_path / "media"
    server = FakeServer(media / "sink_plan.json", status)
    return fasth3.FastH3Worker("http://127.0.0.1:18083", str(media), http=server, poll_s=0.0), server


def test_fasth3_sink_job(tmp_path):
    w, server = _worker(tmp_path)
    refs = [_png(tmp_path / "g.png", (1024, 576)), _png(tmp_path / "m.png", (1536, 1024))]
    job = {"id": "000_a", "out": str(tmp_path / "out/shot.mp4"), "seed": 11, "prompt": R5_PROMPT, "refs": refs,
           "ref_seconds": [0.0, 0.25], "video_start_seconds": 2.0}
    timing = w.handle(job)
    (body, plan), = server.posts
    assert plan == {"image_times": [0.0, 10.0], "target_start": 80.0}  # written before the request
    assert not (tmp_path / "media/sink_plan.json").exists()  # removed after it
    assert body["prompt"] == EARLIER_EXPECTED and body["seed"] == 11
    pics = [c["uri"][len("file://"):] for c in body["conditions"]]
    assert all(Path(p).parent == (tmp_path / "media/fasth3_pictures").resolve() for p in pics)
    assert [Image.open(p).size for p in pics] == [(672, 384), (672, 384)]
    assert Path(job["out"]).read_bytes() == b"MP4" and server.deleted == ["v1"]
    assert timing["layout"] == "sink" and timing["gen_s"] == pytest.approx(5.6) and timing["n_refs"] == 2


def test_fasth3_r5_prompt_and_ref2va_job(tmp_path):
    w, server = _worker(tmp_path)
    ref = _png(tmp_path / "m.png", (1536, 1024))
    w.handle({"id": "a", "out": str(tmp_path / "a.mp4"), "seed": 1, "prompt": R5_PROMPT, "refs": [ref],
              "ref_seconds": [0.0], "video_start_seconds": 2.0, "sink_prompt": "r5", "ref_scale": 1.0})
    w.handle({"id": "b", "out": str(tmp_path / "b.mp4"), "seed": 2, "prompt": "p", "refs": [ref], "ref_seconds": []})
    (sink_body, _), (ref_body, ref_plan) = server.posts
    assert sink_body["prompt"] == R5_PROMPT
    assert Image.open(sink_body["conditions"][0]["uri"][7:]).size == (1344, 768)
    assert ref_plan is None  # release layout
    assert Image.open(ref_body["conditions"][0]["uri"][7:]).size == (768, 512)


def test_fasth3_keyframe_job(tmp_path):
    w, server = _worker(tmp_path)
    w.handle({"id": "k", "out": str(tmp_path / "k.mp4"), "seed": 3, "prompt": "a robot arm opens the drawer",
              "refs": [_png(tmp_path / "pin.png", (1024, 576))], "ref_seconds": [0.0], "video_start_seconds": 2.0,
              "keyframe": _png(tmp_path / "kf.png", (640, 480)), "duration_seconds": 8})
    (body, plan), = server.posts
    assert [c["role"] for c in body["conditions"]] == ["keyframe", "reference"]
    assert Image.open(body["conditions"][0]["uri"][7:]).size == (1344, 768)
    assert Image.open(body["conditions"][1]["uri"][7:]).size == (672, 384)
    assert plan == {"image_times": [0.0], "target_start": 80.0}
    assert body["prompt"] == "a robot arm opens the drawer"


def test_fasth3_job_errors(tmp_path):
    w, server = _worker(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        w.handle({"id": "x", "out": str(tmp_path / "x.mp4"), "seed": 0, "prompt": "p", "refs": []})
    with pytest.raises(ValueError, match="ref_seconds"):
        w.handle({"id": "y", "out": str(tmp_path / "y.mp4"), "seed": 0, "prompt": "p",
                  "refs": [_png(tmp_path / "a.png", (64, 64))], "ref_seconds": [0.0, 0.25], "video_start_seconds": 2.0})
    assert not server.posts
    w, server = _worker(tmp_path, status="failed")
    with pytest.raises(RuntimeError, match="FastH3 failed"):
        w.handle({"id": "z", "out": str(tmp_path / "z.mp4"), "seed": 0, "prompt": "p",
                  "refs": [_png(tmp_path / "b.png", (64, 64))], "ref_seconds": [0.0], "video_start_seconds": 2.0})
    assert server.deleted == ["v1"] and not (tmp_path / "media/sink_plan.json").exists()


# --------------------------------------------------------------------------------------------- UniServe patch
MINI_PACKING = '''from __future__ import annotations

from enum import Enum


class SegmentKind(Enum):
    KEYFRAME = 1
    IMAGE = 2


def condition_segments(conditions):
    return conditions


def dense_tables(segments, num_text_tokens, positions):
    clock = float(num_text_tokens)
    keyframes = []
    for visual, segment in enumerate(segments):
        if segment.kind == SegmentKind.KEYFRAME:
            keyframes.append((segment, visual))
        elif segment.kind == SegmentKind.IMAGE:
            positions[visual, 0] = clock
            clock += 1.0
    for segment, visual in keyframes:
        positions[visual, 0] = clock * 1.0
    return clock


def segment_tables(segments, num_text_tokens, positions):
    clock = float(num_text_tokens)
    keyframes = []
    for visual, segment in enumerate(segments):
        if segment.kind == SegmentKind.KEYFRAME:
            keyframes.append((segment, visual))
        elif segment.kind == SegmentKind.IMAGE:
            if True:
                positions[visual, 0] = clock
            clock += 1.0
    for segment, visual in keyframes:
        positions[visual, 0] = clock * 1.0
    return clock
'''


def _load(source):
    namespace = {}
    exec(compile(source, "packing.py", "exec"), namespace)
    return namespace


def test_uniserve_sink_patch(tmp_path, monkeypatch):
    patched = patch_uniserve_sink.patch_source(MINI_PACKING)
    assert patched.count("_sink, _image = _sink_plan(segments), 0") == 2 and patch_uniserve_sink.patch_source(patched) == patched
    ns = _load(patched)
    seg = type("Segment", (), {})
    image, key = seg(), seg()
    image.kind, key.kind = ns["SegmentKind"].IMAGE, ns["SegmentKind"].KEYFRAME
    segments = [key, image, image]
    plan = tmp_path / "plan.json"
    monkeypatch.setenv("UNISERVE_SINK_PLAN", str(plan))
    for table in ("dense_tables", "segment_tables"):
        pos = {}  # no plan file: release layout (pictures at T, T + 1; target after them)
        assert ns[table](segments, 100, pos) == 102.0 and pos == {(1, 0): 100.0, (2, 0): 101.0, (0, 0): 102.0}
        plan.write_text(json.dumps(fasth3.sink_plan([0.0, 0.25], 2.0)))
        pos = {}
        assert ns[table](segments, 100, pos) == 180.0 and pos == {(1, 0): 100.0, (2, 0): 110.0, (0, 0): 180.0}
        plan.write_text(json.dumps(fasth3.sink_plan([0.0], 2.0)))  # picture count differs: release layout
        pos = {}
        assert ns[table](segments, 100, pos) == 102.0
        plan.unlink()


def test_uniserve_sink_patch_files(tmp_path):
    target = tmp_path / patch_uniserve_sink.PACKING
    target.parent.mkdir(parents=True)
    target.write_text(MINI_PACKING)
    assert patch_uniserve_sink.patch(str(tmp_path)) is True
    assert patch_uniserve_sink.patch(str(tmp_path)) is False  # idempotent
    assert Path(str(target) + ".orig").read_text() == MINI_PACKING
    with pytest.raises(ValueError, match="unsupported UniServe"):
        patch_uniserve_sink.patch_source(MINI_PACKING.replace("    for segment, visual in keyframes:\n", "", 1))


def test_release_headers():
    files = [ROOT / "agentic_longlive/workers" / name for name in ("fasth3.py", "patch_uniserve_sink.py")]
    files += sorted((ROOT / "agentic_longlive/workers/longlive_h3").glob("*.py")) + [Path(__file__)]
    for path in files:
        head = path.read_text().splitlines()[:2]
        assert head == ["# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
                        "# SPDX-License-Identifier: Apache-2.0"], path
    assert os.access(ROOT / "scripts/workers/serve_fasth3.sh", os.X_OK)
