# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastH3-Omni worker: serves a job queue (:mod:`.protocol`) through a UniServe server (OpenAI-style ``/v1/videos``).

FastH3-Omni (FastVideo/FastVideo-FastH3-Omni-8-Step-V1) is an 8-step distillation of MiniMax-H3's reference-to-video
model; UniServe serves it as task ``ref2va`` on 8 GPUs. This worker turns each job into one request:

  sink      refs + ref_seconds. Needs the server patched by ``patch_uniserve_sink.py`` and started with
            UNISERVE_SINK_PLAN=<--sink-plan>. The worker writes the plan (picture k at 40 * ref_seconds[k] rotary ticks
            after the prompt, the generated video at 40 * video_start_seconds) before the request and removes it after.
            Pictures are cover-resized to ``ref_scale`` x the video canvas (default 0.5: 672x384).
  ref2va    refs without ref_seconds: UniServe's release layout. With ``ref_scale`` < 1 the pictures are shrunk by that
            factor (multiples of 32 px, Lanczos), otherwise sent at their own size.
  keyframe  ``keyframe`` (image path): the video starts from this frame (cover-resized to the canvas, condition role
            ``keyframe`` at frame 0), e.g. continuous robot video; combines with sink or ref2va pictures.

Sink prompt (``--sink-prompt`` or per-job ``sink_prompt``): ``earlier`` (default) rewrites the agent's sink prompt,
which describes each memory picture as a short shot before a cut, into one sentence saying that the pictures are
frames from earlier shots, followed by the shot itself (:func:`earlier_prompt`); ``r5`` sends it unchanged.
``ref_masks`` are not used (UniServe packs whole pictures). UniServe refuses a ``ref2va`` request without reference
pictures, so the worker rejects a job with neither pictures nor a keyframe.

The sink plan is one file for the whole server, so run one worker per server. Picture files are written under the
server's ``--media-directory`` (``file://`` conditions). Every finished video is deleted from the server (it keeps
videos for an hour under a size budget, and a full budget fails new requests).

    python -m agentic_longlive.workers.fasth3 --queue QUEUE --url http://127.0.0.1:18083 --media MEDIA_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from . import protocol

WIDTH, HEIGHT = 1344, 768  # FastH3-Omni 768p 16:9 canvas
TICKS_PER_SECOND = 40.0  # H3's rotary clock
SHOT_SECONDS = 8
EARLIER_LEAD = "[Shot 1] Live-action, cinematic, "

Http = Callable[..., bytes]


def http_call(method: str, url: str, body: Optional[bytes] = None, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def earlier_prompt(prompt: str, k: int) -> str:
    """Agent sink prompt (pictures as short shots before a cut at 2 s) -> 'Pictures 1-k are frames from earlier shots
    ...' + the shot itself. A prompt without the cut line is returned unchanged."""
    m = re.search(r"\[Shot (\d+)\] At 00:02\.000, the camera cuts to ", prompt)
    if not m or "\n\nintegrated_multimodal_description: " not in prompt:
        return prompt
    _, _, rest = prompt.partition("\n\nintegrated_multimodal_description: ")
    body = rest[rest.index(m.group(0)) + len(m.group(0)):]
    pics = "Picture 1" if k == 1 else f"Pictures 1-{k}"
    return (f"{pics} {'is a frame' if k == 1 else 'are frames'} from earlier shots of this story, shown before the target "
            "video; the target video starts after a cut and shows only the shot below. Keep the people, animals and "
            "objects they show looking the same when the shot includes them.\n\nintegrated_multimodal_description: "
            + EARLIER_LEAD + body)


def sink_size(scale: float, width: int = WIDTH, height: int = HEIGHT):
    """Size of a sink picture: the canvas times ``scale``, rounded to multiples of 32 px."""
    return round(width * scale / 32) * 32, round(height * scale / 32) * 32


def cover_resize(image, width: int, height: int):
    """Scale a PIL image to cover width x height (bicubic) and center-crop it."""
    from PIL import Image

    image = image.convert("RGB")
    f = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * f)), max(height, round(image.height * f)))
    image = image.resize(size, Image.BICUBIC)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def shrink(image, scale: float):
    """Scale a PIL image by ``scale`` < 1 to multiples of 32 px (Lanczos); larger scales keep it unchanged."""
    from PIL import Image

    image = image.convert("RGB")
    if scale >= 1.0:
        return image
    w, h = image.size
    return image.resize((max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)), Image.LANCZOS)


def sink_plan(ref_seconds: Sequence[float], video_start_seconds: float) -> Dict[str, Any]:
    """The plan ``patch_uniserve_sink.py`` reads: picture and target times in rotary ticks after the prompt."""
    return {"image_times": [TICKS_PER_SECOND * float(x) for x in ref_seconds],
            "target_start": TICKS_PER_SECOND * float(video_start_seconds)}


def request_body(model: str, prompt: str, pictures: Sequence[str], seed: int, keyframe: Optional[str] = None,
                 duration_seconds: Union[int, float] = SHOT_SECONDS) -> Dict[str, Any]:
    """The ``/v1/videos`` request: optional keyframe at frame 0 first, then the reference pictures in order."""
    conditions: List[Dict[str, Any]] = []
    if keyframe:
        conditions.append({"type": "image", "uri": f"file://{keyframe}", "role": "keyframe", "frame_index": 0})
    conditions += [{"type": "image", "uri": f"file://{p}", "role": "reference"} for p in pictures]
    return {"model": model, "task": "ref2va", "prompt": prompt, "conditions": conditions,
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": duration_seconds},
            "seed": int(seed)}


class FastH3Worker:
    def __init__(self, url: str, media: str, plan: Optional[str] = None, ref_scale: float = 0.5,
                 sink_prompt: str = "earlier", model: str = "FastH3-Omni", http: Http = http_call,
                 poll_s: float = 0.02):
        self.base = url.rstrip("/")
        self.media = Path(media).resolve()
        self.pictures = self.media / "fasth3_pictures"
        self.plan = Path(plan) if plan else self.media / "sink_plan.json"
        self.ref_scale, self.sink_prompt, self.model = float(ref_scale), sink_prompt, model
        self.http, self.poll_s = http, poll_s

    # ------------------------------------------------------------------ server
    def wait_server(self, poll_s: float = 3.0) -> None:
        while True:
            try:
                self.http("GET", f"{self.base}/health", timeout=5)
                return
            except Exception:  # noqa: BLE001 -- not up yet
                time.sleep(poll_s)

    def delete(self, vid: str) -> None:
        try:
            self.http("DELETE", f"{self.base}/v1/videos/{vid}", timeout=30)
        except Exception as e:  # noqa: BLE001
            print("delete failed", vid, e, flush=True)

    def purge(self) -> int:
        """Delete every finished video the server still keeps (from earlier workers)."""
        n = 0
        try:
            for _ in range(200):
                done = [v["id"] for v in json.loads(self.http("GET", f"{self.base}/v1/videos", timeout=30)).get("data", [])
                        if v.get("status") in ("completed", "failed")]
                if not done:
                    break
                for vid in done:
                    self.delete(vid)
                    n += 1
        except Exception as e:  # noqa: BLE001
            print("purge failed", e, flush=True)
        return n

    # ---------------------------------------------------------------- pictures
    def _picture(self, path: str, kind: str, scale: float) -> str:
        """Write the prepared picture under the media directory once per (file, kind, scale); return its path."""
        from PIL import Image

        st = os.stat(path)
        tag = hashlib.md5(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}|{kind}|{scale}".encode()).hexdigest()[:12]
        dst = self.pictures / f"{Path(path).stem}_{tag}.png"
        if not dst.exists():
            self.pictures.mkdir(parents=True, exist_ok=True)
            im = Image.open(path)
            if kind == "keyframe":
                im = cover_resize(im, WIDTH, HEIGHT)
            elif kind == "sink":
                im = cover_resize(im, *sink_size(scale))
            else:
                im = shrink(im, scale)
            tmp = dst.with_name(dst.name + ".tmp.png")
            im.save(tmp)
            os.replace(tmp, dst)
        return str(dst)

    # -------------------------------------------------------------------- jobs
    def handle(self, job: Dict[str, Any], save: bool = True) -> Dict[str, Any]:
        refs = [str(r) for r in job.get("refs") or []]
        keyframe = job.get("keyframe")
        if not refs and not keyframe:
            raise ValueError("FastH3-Omni (ref2va) needs at least one memory picture or a keyframe")
        sink = bool(job.get("ref_seconds"))
        if sink and len(job["ref_seconds"]) != len(refs):
            raise ValueError("sink job: one ref_seconds entry per picture")
        prompt = job["prompt"]
        if sink and job.get("sink_prompt", self.sink_prompt) == "earlier":
            prompt = earlier_prompt(prompt, len(refs))
        scale = float(job.get("ref_scale") or self.ref_scale)
        duration = float(job.get("duration_seconds") or SHOT_SECONDS)
        duration = int(duration) if duration.is_integer() else duration  # 8, not 8.0, in the request JSON
        t0 = time.perf_counter()
        pics = [self._picture(r, "sink" if sink else "reference", scale) for r in refs]
        key = self._picture(str(keyframe), "keyframe", 1.0) if keyframe else None
        self.plan.unlink(missing_ok=True)
        if sink:
            tmp = self.plan.with_suffix(".tmp")
            tmp.write_text(json.dumps(sink_plan(job["ref_seconds"], job.get("video_start_seconds", 2.0))))
            os.replace(tmp, self.plan)
        t_pics = time.perf_counter() - t0
        try:
            body = request_body(self.model, prompt, pics, int(job["seed"]), key, duration)
            timing = self._request(body, Path(job["out"]) if save else None)
        finally:
            self.plan.unlink(missing_ok=True)
        timing.update(total_s=round(timing["gen_s"] + t_pics, 3), prep_pictures_s=round(t_pics, 3), n_refs=len(pics),
                      layout="sink" if sink else "ref2va", keyframe=bool(key), ref_scale=scale,
                      duration_seconds=duration, prompt_used=prompt)
        return timing

    def _request(self, body: Dict[str, Any], out: Optional[Path]) -> Dict[str, Any]:
        t1 = time.perf_counter()
        vid = json.loads(self.http("POST", f"{self.base}/v1/videos", json.dumps(body, ensure_ascii=False).encode(),
                                   timeout=300))["id"]
        while True:
            rec = json.loads(self.http("GET", f"{self.base}/v1/videos/{vid}", timeout=30))
            if rec.get("status") in ("completed", "failed"):
                break
            time.sleep(self.poll_s)
        t_done = time.perf_counter() - t1
        try:
            if rec.get("status") != "completed":
                raise RuntimeError(f"FastH3 failed: {json.dumps(rec.get('error'))[:400]}")
            if out is not None:
                data = self.http("GET", f"{self.base}/v1/videos/{vid}/content", timeout=600)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, out)
        finally:
            self.delete(vid)
        st = rec.get("stage_durations") or {}
        gen = sum(float(st.get(k, 0.0)) for k in ("encoding", "preparing", "denoising", "decoding"))
        return {"gen_s": round(gen, 3), "client_wall_s": round(t_done, 3), "stages_s": st,
                "inference_time_s": rec.get("inference_time_s")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True, help="job queue directory")
    ap.add_argument("--url", default="http://127.0.0.1:18083", help="UniServe server")
    ap.add_argument("--media", required=True, help="the server's --media-directory (pictures are written here)")
    ap.add_argument("--sink-plan", default=None, help="the server's UNISERVE_SINK_PLAN file (default <media>/sink_plan.json)")
    ap.add_argument("--ref-scale", type=float, default=0.5, help="picture size as a fraction of the canvas (per-job ref_scale)")
    ap.add_argument("--sink-prompt", choices=["earlier", "r5"], default="earlier")
    ap.add_argument("--model", default="FastH3-Omni", help="the server's --served-model-name")
    ap.add_argument("--warmup", default="", help="job JSON rendered once (not saved) before the worker reports ready")
    args = ap.parse_args()
    worker = FastH3Worker(args.url, args.media, args.sink_plan, args.ref_scale, args.sink_prompt, args.model)
    worker.plan.unlink(missing_ok=True)
    print("waiting for", args.url, flush=True)
    worker.wait_server()
    print("purged", worker.purge(), "videos", flush=True)
    if args.warmup:
        print("warmup", json.dumps(worker.handle(json.loads(Path(args.warmup).read_text()), save=False)), flush=True)
    info = {"worker": "fasth3", "url": args.url, "ref_scale": args.ref_scale, "sink_prompt": args.sink_prompt,
            "pid": os.getpid()}
    print("ready", json.dumps(info), flush=True)
    protocol.serve(Path(args.queue), worker.handle, ready=info)


if __name__ == "__main__":
    main()
