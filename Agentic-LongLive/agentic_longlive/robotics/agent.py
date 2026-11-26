# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observation-based memory agent for long, continuous robot videos.

A robot acts in one continuous scene, so where an object ends up or whether a drawer is still open depends on the
generated video, not on the script. The agent therefore observes the video. Its memory pictures are frames (or crops of
frames) that the stream has already shown, so it needs no image model.

  observer  after each segment: three of its frames + the instruction it rendered -> the view, every tracked object
            (visible, where, look), container states and visible contents, the frame to keep, a frame that shows the
            inside of an open container, and boxes for up to two objects. Code keeps the two parts of memory:
            * the ledger: objects and containers; an object last seen inside a container that is now closed stays
              "inside <container> (hidden)";
            * the bank: one view per segment, container interiors and the newest crop of each object, each with a
              caption and the segment it comes from (ordered oldest -> newest).
            The input image is observed before the video starts and stored like any view.
  router    before each segment, one narrow question on the instruction history: does something come back into view
            in the next segment (the camera turns back, a container is reopened, a cover is lifted, an object is taken
            out)?
  planner   only when something comes back: the bank as thumbnails -> every picture that shows the returning place,
            container inside or object; code pins the newest matching view + the newest matching close view (at most
            two) and the instruction is rewritten to name what comes back and how it looked when last seen. Otherwise
            nothing is pinned and the instruction is only rewritten to name the objects with their look.

Pinning only on returns matters: pictures pinned in (almost) every segment make the generator restore old states.

Policies (``make_agent``):
  v4       the paper's agent (:class:`RobotAgent`), as described above.
  hybrid2  :class:`HideAgent`. Besides the v4 observer and planner, a hide observer stores a snapshot keyed by every
           event that hides something (a container is closed, a cover is put on, the camera turns away), and a second
           router (ROUTE6) asks which hidden thing, if any, comes back and how. Pictures:
             container / cover reopened -> a zoomed crop of the pre-hide snapshot (strict rewrite, SELECT_STRICT);
             camera return to an area   -> the area's pre-hide snapshot + the newest matching object crop, and the
                                           instruction verbatim + "<area> comes back into view exactly as it looked
                                           before the camera turned away";
             nothing matched            -> the v4 choice if the v4 router sees a return, else nothing.
           On the 30-episode benchmark hybrid2 and v4 are statistically tied; v4 is the paper's policy.

The agent never sees the scripts' ``memory_event`` annotations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from . import video
from .prompts import (CAMERA_RETURN_SUFFIX, HIDE, INPUT_IMAGE_INSTRUCTION, OBSERVE, OBSERVE_HOME, REWRITE, ROUTE,
                      ROUTE6, SELECT, SELECT_STRICT, SELECT_WHAT_DEFAULT, START_VIEW_EVENT)

POLICIES = ("v4", "hybrid2")
MAX_BANK = 12               # memory pictures shown to the planner (the newest)
MAX_CROPS = 2               # object crops kept per observation


def crop16x9(im: Image.Image, bbox: Sequence[float], min_frac: float = 0.3) -> Image.Image:
    """bbox in 0-1000 -> a 16:9 crop around it with margin, at least ``min_frac`` of the frame width."""
    W, H = im.size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0, x1 = sorted((x0 * W / 1000, x1 * W / 1000))
    y0, y1 = sorted((y0 * H / 1000, y1 * H / 1000))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * 1.6, (y1 - y0) * 1.6 * 16 / 9, W * min_frac)
    w = min(w, W)
    h = min(w * 9 / 16, H)
    w = h * 16 / 9
    left = min(max(cx - w / 2, 0), W - w)
    top = min(max(cy - h / 2, 0), H - h)
    return im.crop((round(left), round(top), round(left + w), round(top + h)))


def _px_to_1000(box: Sequence[float], size: Tuple[int, int]) -> List[float]:
    W, H = size
    return [box[0] * 1000 / W, box[1] * 1000 / H, box[2] * 1000 / W, box[3] * 1000 / H]


def newest_view_and_close(bank: Dict[str, dict], ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    """The given bank ids, ordered oldest -> newest, split into scene views and close views (container inside or
    object crop)."""
    order = {k: i for i, k in enumerate(bank)}
    ids = sorted({k for k in ids if k in bank}, key=lambda k: order[k])
    views = [k for k in ids if bank[k]["kind"] == "view"]
    close = [k for k in ids if bank[k]["kind"] in ("inside", "object")]
    return views, close


def pick_newest(bank: Dict[str, dict], ids: Sequence[str]) -> List[str]:
    """v4: among the pictures that show the returning thing, the newest scene view (end-of-segment view or the input
    image) plus the newest close view, at most 2."""
    views, close = newest_view_and_close(bank, ids)
    return (views[-1:] + close[-1:])[:2]


class RobotAgent:
    """The v4 agent (the paper's robot policy).

    llm: any object with ``ask(content, max_tokens=..., max_side=...) -> (dict, seconds)`` (see :class:`AgentLLM`).
    crop_mode: "observer" crops objects with the observer's own boxes (the paper's real-time system); "grounding"
      localises them by name with Grounding DINO, excluding robot boxes (the paper's offline system and hybrid2).
    home: the observer also sees the input image and says whether the end frame shows that starting workspace.
    """

    policy = "v4"
    select_prompt = SELECT

    def __init__(self, ep: Dict[str, Any], llm, work: Path, crop_mode: str = "observer", home: bool = False,
                 grounder=None):
        if crop_mode not in ("observer", "grounding"):
            raise ValueError(f"crop_mode must be 'observer' or 'grounding', not {crop_mode!r}")
        self.ep, self.llm, self.work = ep, llm, Path(work)
        self.crop_mode, self.home = crop_mode, home
        if crop_mode == "grounding" and grounder is None:
            from .grounding import GroundingDino
            grounder = GroundingDino()
        self.grounder = grounder
        self.first_frame: Optional[Path] = None
        self.home_view: Optional[str] = None   # bank id of the newest end view that shows the starting workspace
        (self.work / "mem").mkdir(parents=True, exist_ok=True)
        self.ledger: Dict[str, Dict[str, dict]] = {"objects": {}, "containers": {}}
        self.bank: Dict[str, dict] = {}        # id -> {"id", "kind", "key", "caption", "seg", "path"}, oldest first
        self.trace: List[dict] = []
        self.observed = -2                     # last observed segment (-1 = the input image)

    # ------------------------------------------------------------------ helpers
    def _ctx(self) -> Dict[str, str]:
        return dict(camera=self.ep["camera"], robot=self.ep["robot"], setting=self.ep["setting"])

    def _put(self, kind: str, key: str, caption: str, seg: int, im: Image.Image) -> Tuple[dict, Path]:
        mid = f"{kind}:{key}"
        self.bank.pop(mid, None)               # re-insert at the end: the bank stays ordered oldest -> newest
        p = self.work / "mem" / f"s{seg:02d}_{re.sub(r'[^a-z0-9]+', '_', mid.lower())[:60]}.jpg"
        im.convert("RGB").save(p, quality=93)
        self.bank[mid] = {"id": mid, "kind": kind, "key": key, "caption": caption, "seg": seg, "path": str(p)}
        return self.bank[mid], p

    # ------------------------------------------------------------------ observer
    def observe(self, seg: int, frames: Sequence[Image.Image], instr: str) -> Tuple[dict, float]:
        """frames: 3 PIL images (early, middle, end) of segment ``seg`` (-1 = the input image, given 3 times)."""
        names = sorted(self.ledger["objects"])
        extra_imgs, extra_q = [], ""
        if self.home and seg >= 0:
            extra_imgs = [Image.open(self.first_frame).convert("RGB")]
            extra_q = OBSERVE_HOME
        r, dt = self.llm.ask(list(frames) + extra_imgs
                             + [OBSERVE.format(instr=instr, names=json.dumps(names), **self._ctx()) + extra_q])
        for c in r.get("containers", []) or []:
            if c.get("name") and c.get("state") in ("open", "closed"):
                self.ledger["containers"][c["name"]] = {"state": c["state"], "contents": c.get("contents", ""),
                                                         "seg": seg}
        for o in r.get("objects", []) or []:
            n = o.get("name")
            if not n:
                continue
            prev = self.ledger["objects"].get(n, {})
            if o.get("visible"):
                self.ledger["objects"][n] = {"where": o.get("where", ""), "look": o.get("look") or prev.get("look", ""),
                                             "seen_seg": seg}
            elif prev:
                where = prev.get("where", "")
                for cname, cs in self.ledger["containers"].items():
                    if cname.lower() in where.lower() and cs["state"] == "closed" and "(hidden)" not in where:
                        where = where + " (hidden)"
                self.ledger["objects"][n] = dict(prev, where=where, out_of_view_since=seg)
        label = "input image" if seg < 0 else f"end of segment {seg + 1}"
        k = min(max(int(r.get("keep_frame") or 3), 1), len(frames))
        ent, _ = self._put("view", "input" if seg < 0 else f"seg{seg + 1}", f"{label}: {r.get('view', '')}", seg,
                           frames[k - 1])
        if seg < 0 or (self.home and r.get("home_view") is True):
            self.home_view = ent["id"]
        iv = r.get("inside_view")
        if seg >= 0 and isinstance(iv, dict) and iv.get("frame"):
            try:
                f = frames[min(max(int(iv["frame"]), 1), len(frames)) - 1]
                cap = f"during segment {seg + 1}, open container: {iv.get('caption', '')}"
                self._put("inside", f"seg{seg + 1}", cap, seg, f)
            except (TypeError, ValueError):
                pass
        for c in (r.get("crops") or [])[:MAX_CROPS]:
            try:
                f = frames[min(max(int(c.get("frame", 3)), 1), len(frames)) - 1]
                n = c["name"]
                look = self.ledger["objects"].get(n, {}).get("look", "")
                cap = f"close view of the {n}" + (f" ({look})" if look else "") + f", segment {max(seg + 1, 0)}"
                if self.crop_mode == "grounding":
                    box, _ = self.grounder.boxes(f, [n])[n]
                    if box is None:
                        continue
                    bb = _px_to_1000(box, f.size)
                else:
                    bb = c["bbox"]
                ent, _ = self._put("object", n, cap, seg, crop16x9(f, bb))
                ent["bbox"] = bb
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
                self.trace.append({"role": "crop_error", "seg": seg, "error": str(e)[:200]})
        self.observed = seg
        self.trace.append({"role": "observer", "seg": seg, "s": round(dt, 2), "out": r})
        return r, dt

    def observe_input(self, image: Path) -> float:
        """Observe the user's input image (before the video starts)."""
        self.first_frame = Path(image)
        im0 = Image.open(image).convert("RGB")
        _, dt = self.observe(-1, [im0, im0, im0], INPUT_IMAGE_INSTRUCTION)
        return dt

    def observe_segment(self, seg: int, mp4: Path, instr: str) -> float:
        """Observe a rendered segment (frames 40 / 110 / 191)."""
        frames = video.load_frames(mp4, self.work / "frames" / f"seg{seg}", video.OBS_IDX)
        _, dt = self.observe(seg, frames, instr)
        return dt

    # ------------------------------------------------------------------ router + planner
    def plan(self, seg: int, instr: str, hist: Sequence[str]) -> Tuple[dict, float]:
        """Plan segment ``seg`` from its instruction and the instructions before it."""
        if not self.bank:
            return {"pins": [], "prompt": instr, "returns": False, "pinned": []}, 0.0
        hist_s = json.dumps(list(hist), ensure_ascii=False)
        ctx = dict(hist=hist_s, ledger=json.dumps(self.ledger, ensure_ascii=False), instr=instr, **self._ctx())
        route, dt1 = self.llm.ask([ROUTE.format(hist=hist_s, instr=instr)], max_tokens=120)
        returns = bool(route.get("returns"))
        if returns:
            items = list(self.bank.values())[-MAX_BANK:]
            thumbs = [Image.open(m["path"]) for m in items]
            bank = json.dumps([{"id": m["id"], "shows": m["caption"]} for m in items], ensure_ascii=False)
            r, dt2 = self.llm.ask(thumbs + [self.select_prompt.format(what=route.get("what") or SELECT_WHAT_DEFAULT,
                                                                      bank=bank, **ctx)], max_tokens=600, max_side=448)
            shows = [p for p in (r.get("shows") or r.get("pins") or []) if p in self.bank]
            pins = pick_newest(self.bank, shows)
        else:
            shows = []
            r, dt2 = self.llm.ask([REWRITE.format(**ctx)], max_tokens=400)
            pins = []
        out = {"returns": returns, "what": route.get("what", ""), "shows": shows, "pins": pins,
               "prompt": str(r.get("prompt") or instr).strip(), "observed_through": self.observed,
               "pinned": [dict(self.bank[p]) for p in pins], "route_s": round(dt1, 2), "plan_s": round(dt2, 2)}
        self.trace.append({"role": "planner", "seg": seg, "s": round(dt1 + dt2, 2), "out": out})
        return out, dt1 + dt2

    def pictures(self, plan: dict) -> List[dict]:
        """The memory pictures (bank entries) that enter the generator for the planned segment."""
        return [dict(e) for e in plan.get("pinned", [])]

    def state(self) -> Dict[str, Any]:
        return {"policy": self.policy, "ledger": self.ledger, "bank": list(self.bank.values()),
                "home_view": self.home_view}


class HideAgent(RobotAgent):
    """The hybrid2 agent: v4 observer and planner (Grounding DINO crops, home-view observer, strict rewrite) plus
    snapshots keyed by hide events and the ROUTE6 router over them."""

    policy = "hybrid2"
    select_prompt = SELECT_STRICT

    def __init__(self, ep: Dict[str, Any], llm, work: Path, grounder=None):
        super().__init__(ep, llm, work, crop_mode="grounding", home=True, grounder=grounder)
        self.events: List[dict] = []           # [{"id": "H<k>", "seg", "what", "kind", "bank", "frame"}]
        self.prev: Optional[Image.Image] = None   # the frame right before the next segment to observe

    # ------------------------------------------------------------------ observer
    def observe_input(self, image: Path) -> float:
        dt = super().observe_input(image)
        self.prev = Image.open(image).convert("RGB")
        self.events.append({"id": "H0", "seg": -1, "what": START_VIEW_EVENT, "kind": "area", "bank": "view:input"})
        return dt

    def observe_segment(self, seg: int, mp4: Path, instr: str) -> float:
        """v4 observer on frames 64 / 128 / 191, then the hide observer on the frame before the segment + 12 frames.

        A hidden thing (at most 2 per segment) is stored as "<kind>:hideH<k>" with the last frame in which it is still
        fully visible; such snapshots are never overwritten. Container / cover snapshots also get a zoomed crop around
        the hidden thing (Grounding DINO on the event text, at least 45% of the frame width)."""
        paths = video.frames(mp4, self.work / "frames" / f"seg{seg}", video.DENSE12)
        dense = [Image.open(p).convert("RGB") for p in paths]
        _, dt_obs = self.observe(seg, [dense[3], dense[7], dense[-1]], instr)
        ims = [self.prev] + dense
        r, dt = self.llm.ask(ims + [HIDE.format(n=len(ims), instr=instr, **self._ctx())], max_tokens=500, max_side=640)
        for h in (r.get("hides") or [])[:2]:
            try:
                f = min(max(int(h.get("frame") or 1), 1), len(ims))
            except (TypeError, ValueError):
                f = 1
            what = str(h.get("what") or "").strip()
            if not what:
                continue
            kind = "inside" if h.get("kind") == "inside" else "view"
            eid = f"H{len(self.events)}"
            cap = f"{what}, as it looked right before it was hidden (segment {seg + 1})"
            ent, _ = self._put(kind, f"hide{eid}", cap, seg, ims[f - 1])
            if kind == "inside":
                try:
                    im = ims[f - 1]
                    box, _ = self.grounder.boxes(im, [what])[what]
                    if box is not None:
                        q = Path(ent["path"]).with_name(Path(ent["path"]).stem + "_crop.jpg")
                        crop16x9(im, _px_to_1000(box, im.size), min_frac=0.45).save(q, quality=93)
                        ent["path_crop"] = str(q)
                except (KeyError, TypeError, ValueError, RuntimeError) as e:
                    self.trace.append({"role": "zoom_error", "seg": seg, "error": str(e)[:200]})
            self.events.append({"id": eid, "seg": seg, "what": what, "kind": kind, "bank": ent["id"], "frame": f})
        self.prev = dense[-1]
        self.trace.append({"role": "hide", "seg": seg, "s": round(dt, 2), "out": r})
        return dt_obs + dt

    # ------------------------------------------------------------------ router + planner
    def plan(self, seg: int, instr: str, hist: Sequence[str]) -> Tuple[dict, float]:
        """v4 planner (router + selector or rewrite) + ROUTE6 over the live hide events -> ``out["target"]``."""
        base, dt0 = super().plan(seg, instr, hist)
        live = [e for e in self.events if not e.get("returned_at")]
        ev = "\n".join(f"{e['id']} [{'contents' if e['kind'] == 'inside' else 'area'}]: "
                       + ("start" if e["seg"] < 0 else f"segment {e['seg'] + 1}") + f" - {e['what']}" for e in live)
        q = ROUTE6.format(hist=json.dumps(list(hist), ensure_ascii=False), instr=instr, events=ev)
        route, dt1 = self.llm.ask([q], max_tokens=200)
        target, event = None, None
        if route.get("returns"):
            event = next((e for e in live if e["id"] == str(route.get("id") or "").strip()), None)
            how = str(route.get("how") or "").strip().lower()
            # A camera return can only bring back an area, an opening only a container's contents.
            if event and ((how == "camera" and event["kind"] == "inside")
                          or (how == "open" and event["kind"] != "inside")):
                self.trace.append({"role": "route6_reject", "seg": seg, "how": how, "id": event["id"],
                                   "kind": event["kind"]})
                event = None
            if event and event["bank"] in self.bank:
                target = event["bank"]
                if event["seg"] >= 0:          # back in view now: not offered again (H0 stays)
                    event["returned_at"] = seg
        out = dict(base, route6=route, target=target, returns6=bool(route.get("returns")), route6_s=round(dt1, 2))
        if target and event["kind"] != "inside" and event["seg"] >= 0:
            # Camera return: the user's instruction already names the place; no model writes anything for it.
            w = event["what"].strip().rstrip(".")
            out["prompt_select"] = out["prompt"]
            out["prompt"] = instr.strip().rstrip(".") + ". " + w[:1].upper() + w[1:] + CAMERA_RETURN_SUFFIX
        self.trace.append({"role": "planner6", "seg": seg, "out": {k: v for k, v in out.items() if k != "pinned"}})
        return out, dt0 + dt1

    def pictures(self, plan: dict) -> List[dict]:
        bank = self.bank
        views, close = newest_view_and_close(bank, plan.get("shows", []))
        target = plan.get("target")
        if target == "view:input":             # H0 alone is too eager; the v4 router decides below
            target = None
        if target and bank[target]["kind"] == "inside":    # container / cover reopened: zoomed pre-hide snapshot
            e = dict(bank[target])
            if e.get("path_crop"):
                e["path"] = e["path_crop"]
                e["caption"] = "close view of " + e["caption"]
            return [e]
        if target:                             # camera return: the area's pre-hide snapshot + newest object crop
            objs = [k for k in close if bank[k]["kind"] == "object"]
            return [dict(bank[k]) for k in [target] + objs[-1:]]
        if plan.get("returns"):                # no hide event matched: the v4 choice
            return [dict(bank[k]) for k in views[-1:] + close[-1:]]
        return []

    def state(self) -> Dict[str, Any]:
        s = super().state()
        s.update(events=self.events)
        return s


def make_agent(policy: str, ep: Dict[str, Any], llm, work: Path, grounder=None,
               crop_mode: Optional[str] = None) -> RobotAgent:
    """policy: "v4" (default crops: the observer's boxes) or "hybrid2" (always Grounding DINO crops)."""
    if policy == "v4":
        return RobotAgent(ep, llm, work, crop_mode=crop_mode or "observer", grounder=grounder)
    if policy == "hybrid2":
        if crop_mode not in (None, "grounding"):
            raise ValueError("hybrid2 always localises crops with Grounding DINO")
        return HideAgent(ep, llm, work, grounder=grounder)
    raise ValueError(f"unknown policy {policy!r} (choose from {POLICIES})")
