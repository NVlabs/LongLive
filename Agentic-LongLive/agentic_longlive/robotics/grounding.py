# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-vocabulary object boxes (Grounding DINO) for the agent's object crops.

The hybrid2 policy (and, optionally, v4) localises an object crop by the object's name instead of the observer's own
box, and excludes boxes that are mostly the robot itself. Needs ``torch`` and ``transformers``; the model is loaded on
first use.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

Box = Optional[List[float]]

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"
ROBOT_QUERY = "robot arm. robot gripper. robot hand."


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the smaller box."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


class GroundingDino:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_id, self.device = model_id, device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            dev = self.device or ("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available() else "cpu")
            proc = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(dev).eval()
            self._model = (proc, model, dev)
        return self._model

    def _detect(self, im: Image.Image, text: str, box_thr: float, text_thr: float) -> List[Tuple[List[float], float]]:
        import torch
        proc, model, dev = self._load()
        inp = proc(images=im, text=text, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp)
        r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                                                        target_sizes=[im.size[::-1]])[0]
        return [([float(v) for v in b.tolist()], float(sc)) for b, sc in zip(r["boxes"], r["scores"])]

    def boxes(self, im: Image.Image, names: Sequence[str], box_thr: float = 0.3,
              text_thr: float = 0.25) -> Dict[str, Tuple[Box, float]]:
        """Best box (pixels) per name that is not mostly the robot (overlap < 0.5 with every robot arm / gripper box)
        and covers < 40% of the image; ``(None, 0.0)`` if there is none."""
        with self._lock:
            W, H = im.size
            robot = [b for b, _ in self._detect(im, ROBOT_QUERY, 0.35, 0.3)]
            out: Dict[str, Tuple[Box, float]] = {}
            for n in names:
                query = n.lower().strip().rstrip(".") + "."
                cands = sorted(self._detect(im, query, box_thr, text_thr), key=lambda x: -x[1])
                pick = None
                for b, sc in cands:
                    area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
                    if area < 0.4 and all(_overlap(b, r) < 0.5 for r in robot) and "robot" not in n.lower():
                        pick = ([round(v, 1) for v in b], round(sc, 3))
                        break
                out[n] = pick or (None, 0.0)
            return out
