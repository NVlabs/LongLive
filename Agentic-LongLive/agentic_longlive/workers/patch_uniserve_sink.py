# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add an opt-in sink layout to UniServe's MiniMax-H3 packing (both ``dense_tables`` and ``segment_tables``).

Tested against UniServe commit d58bee14b46475c690ead7540f9f2bbab7790195 (``uniserve_models/minimax_h3/packing.py``).

With UNISERVE_SINK_PLAN=<file> in the server's environment and that file present when a request is packed, holding
``{"image_times": [t_1, ..., t_k], "target_start": T}`` (rotary ticks after the prompt; H3 counts 40 ticks per second),
image reference k sits at ``num_text_tokens + t_k`` instead of ``num_text_tokens + (k - 1)``, and the generated audio and
video (and any keyframe) start at ``num_text_tokens + T`` instead of ``num_text_tokens + k``. Without the file, or when
the plan's picture count differs from the request's image references, the release layout is unchanged. The FastH3
worker (:mod:`agentic_longlive.workers.fasth3`) writes the plan before each sink request and removes it afterwards.

Idempotent; the unpatched file is kept as ``packing.py.orig``.

    python -m agentic_longlive.workers.patch_uniserve_sink UNISERVE_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKING = "uniserve_models/minimax_h3/packing.py"
MARKER = "_sink_plan("

HELPER = '''

def _sink_plan(segments) -> dict | None:
    """Sink layout of Agentic LongLive (patch_uniserve_sink.py): the plan file named by UNISERVE_SINK_PLAN, if present."""
    import json as _json
    import os as _os

    path = _os.environ.get("UNISERVE_SINK_PLAN")
    if not path or not _os.path.exists(path):
        return None
    try:
        plan = _json.loads(open(path).read())
    except (OSError, ValueError):
        return None
    images = sum(1 for segment in segments if segment.kind == SegmentKind.IMAGE)
    if len(plan.get("image_times") or []) != images or images == 0:
        return None
    import sys as _sys

    print(f"[sink-plan] {images} images at {plan['image_times']} target {plan['target_start']}", file=_sys.stderr, flush=True)
    return plan

'''


def _replace(text: str, old: str, new: str, count: int) -> str:
    if text.count(old) != count:
        raise ValueError(f"expected {count} x {old.strip()!r} in packing.py (unsupported UniServe version?)")
    return text.replace(old, new)


def patch_source(text: str) -> str:
    """Return the patched packing.py source (unchanged if it is already patched)."""
    if MARKER in text:
        return text
    # 1. the plan reader, right before condition_segments()
    text = _replace(text, "\ndef condition_segments(", HELPER + "\ndef condition_segments(", 1)
    # 2. both table builders read the plan where the media clock starts
    text = _replace(text, "    clock = float(num_text_tokens)\n",
                    "    clock = float(num_text_tokens)\n    _sink, _image = _sink_plan(segments), 0\n", 2)
    # 3. image reference k at num_text_tokens + image_times[k] instead of the running clock
    anchor = "positions[visual, 0] = clock\n"
    parts = text.split(anchor)
    if len(parts) != 3:
        raise ValueError(f"expected 2 x {anchor.strip()!r} in packing.py (unsupported UniServe version?)")
    out = parts[0]
    for part in parts[1:]:
        indent = out[out.rfind("\n") + 1:]
        out += ("positions[visual, 0] = (\n"
                + indent + "    clock if _sink is None else float(num_text_tokens) + float(_sink[\"image_times\"][_image])\n"
                + indent + ")\n" + indent + "_image += 1\n") + part
    text = out
    # 4. keyframes and the generated media start at num_text_tokens + target_start
    old = "    for segment, visual in keyframes:\n"
    return _replace(text, old, "    if _sink is not None:\n        clock = float(num_text_tokens) + float(_sink[\"target_start\"])\n"
                    + old, 2)


def patch(uniserve_dir: str) -> bool:
    """Patch ``<uniserve_dir>/uniserve_models/minimax_h3/packing.py`` in place; False if it was already patched."""
    path = Path(uniserve_dir) / PACKING
    text = path.read_text()
    patched = patch_source(text)
    if patched == text:
        return False
    orig = Path(str(path) + ".orig")
    if not orig.exists():
        orig.write_text(text)
    path.write_text(patched)
    return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(("patched " if patch(sys.argv[1]) else "already patched ") + str(Path(sys.argv[1]) / PACKING))


if __name__ == "__main__":
    main()
