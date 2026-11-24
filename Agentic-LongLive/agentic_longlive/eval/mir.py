# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MIR face identity, StoryEngine protocol (supplementary metric; optional, needs ``insightface`` + ``onnxruntime``).

Four evenly spaced frames per shot (first and last included); InsightFace buffalo_l detection and recognition at
640x640; every detection with score >= 0.55 is kept. In shot order, each L2-normalised embedding joins the cluster
whose mean embedding is most similar if the cosine is >= 0.45, otherwise it starts a new cluster. Single-detection
clusters are dropped unless every cluster is a singleton. MIR is the minimum over clusters of the mean cosine between
detections from different shots (clusters without such pairs are ignored), and 0 if there are fewer than two faces or
no cluster spans two shots.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import frames as F
from .runs import Run

DET_SCORE, CLUSTER_COS = 0.55, 0.45


class FaceDetector:
    def __init__(self, gpu: int = 0, root: Optional[str] = None):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError("MIR needs `pip install insightface onnxruntime-gpu`") from exc
        kwargs = {"root": root} if root else {}
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"], **kwargs)
        self.app.prepare(ctx_id=gpu, det_thresh=DET_SCORE, det_size=(640, 640))

    def detect(self, frames: Sequence) -> List[Dict[str, Any]]:
        out = []
        for frame in frames:  # RGB -> BGR for insightface
            for face in self.app.get(frame[:, :, ::-1].copy()):
                if float(face.det_score) >= DET_SCORE:
                    out.append({"score": round(float(face.det_score), 3),
                                "w": round(float(face.bbox[2] - face.bbox[0]), 1),
                                "emb": [round(float(x), 5) for x in face.normed_embedding]})
        return out


def mir_score(dets: List[Dict[str, Any]], threshold: float = CLUSTER_COS) -> Tuple[float, int]:
    """(MIR, number of scored clusters) from detections [{"shot", "emb"}] in shot order."""
    import numpy as np

    if len(dets) < 2:
        return 0.0, 0
    embs = [np.asarray(d["emb"], dtype=np.float64) for d in dets]
    embs = [e / (np.linalg.norm(e) or 1.0) for e in embs]
    clusters: List[list] = []  # [members, embedding sum]
    for k, e in enumerate(embs):
        best, index = -2.0, None
        for ci, (_, total) in enumerate(clusters):
            sim = float(e @ (total / (np.linalg.norm(total) or 1.0)))
            if sim > best:
                best, index = sim, ci
        if index is not None and best >= threshold:
            clusters[index][0].append(k)
            clusters[index][1] = clusters[index][1] + e
        else:
            clusters.append([[k], e.copy()])
    keep = [c for c in clusters if len(c[0]) > 1] or clusters
    scores = []
    for members, _ in keep:
        sims = [float(embs[a] @ embs[b]) for a, b in itertools.combinations(members, 2)
                if dets[a]["shot"] != dets[b]["shot"]]
        if sims:
            scores.append(sum(sims) / len(sims))
    return (min(scores) if scores else 0.0), len(scores)


def measure_mir(run: Run, detector: FaceDetector) -> Dict[str, Any]:
    """Detect faces (cached in ``<run>/eval/faces_se.json``) and return {"mir", "faces", "clusters"}."""
    cache = run.eval_dir / "faces_se.json"
    if cache.is_file():
        dets = json.loads(cache.read_text())
    else:
        dets = []
        for shot in sorted(run.videos):
            video = F.Video(run.videos[shot])
            for det in detector.detect(video.frames(F.evenly_spaced(video.n, F.MIR_FRAMES))):
                dets.append({"shot": shot, **det})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dets))
    value, clusters = mir_score(dets)
    return {"mir": value, "faces": len(dets), "clusters": clusters}
