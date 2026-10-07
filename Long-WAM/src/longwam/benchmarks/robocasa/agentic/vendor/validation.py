# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/validation.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Copied action/gate validation from colleague baseline 5ca5512."""

import numpy as np
from scipy.spatial.transform import Rotation
from .action_edit_kinematics import edited_targets
from .gate_assessment import validate_assessment


def angles(p):
    q = np.asarray(p["quaternion_wxyz"])
    return dict(
        p, rpy_deg=Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz", degrees=True).tolist()
    )


def _contains_cjk(text):
    return any(
        "\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U000323af" for char in text
    )


def validate_public_language(response):
    """Reject CJK in model-authored public prose; protocol/task text is untouched."""
    assessment = response.get("assessment", {})
    progress = assessment.get("task_progress", {}) if isinstance(assessment, dict) else {}
    values = [
        response.get("reason"),
        *(
            assessment.get(key)
            for key in (
                "current_subgoal",
                "execution_evidence",
                "expected_next_intent",
                "predicted_next_intent",
                "intent_evidence",
            )
        ),
        progress.get("currently_attempting"),
        *progress.get("verified_completed", []),
        *progress.get("remaining", []),
    ]
    if any(isinstance(value, str) and _contains_cjk(value) for value in values):
        raise ValueError("Public decision and assessment text must be written in English")


def validate_response(response, request, student_cap=15):
    if response.get("request_id") != request["request_id"]:
        raise ValueError("Response is stale or belongs to another observation")
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        raise ValueError("A visible observation/decision explanation is required")
    mode = response.get("mode")
    if mode not in ("student", "edit", "eef", "stop"):
        raise ValueError("Expected student/edit/eef/stop")
    if request.get("gate_policy") == "failure-or-intent":
        validate_assessment(response, request)
        validate_public_language(response)
    if mode == "stop":
        return
    n = response.get("steps")
    cap = student_cap if mode == "student" else 5
    if type(n) is not int or not 1 <= n <= cap:
        raise ValueError(f"Expected 1..{cap} execution steps")
    if mode == "edit":
        edited_targets(request["student_eef_trajectory"], n, response["edit"])
    elif mode == "eef":
        target = response["target"]
        p, q = np.asarray(target["position"], float), np.asarray(target["quaternion_wxyz"], float)
        if (
            p.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
        ):
            raise ValueError("Finite EEF position/quaternion required")
        if abs(np.linalg.norm(q) - 1) > 1e-4 or type(target["gripper_closed"]) is not bool:
            raise ValueError("Unit quaternion and explicit boolean gripper required")
        if np.linalg.norm(p - np.asarray(request["current_eef"]["position"])) > 0.05 + 1e-9:
            raise ValueError("Recovery EEF target exceeds 5 cm")
        old = request["current_eef"]["quaternion_wxyz"]
        delta = (
            Rotation.from_quat(q[[1, 2, 3, 0]])
            * Rotation.from_quat(np.asarray(old)[[1, 2, 3, 0]]).inv()
        )
        if delta.magnitude() > 0.35 + 1e-9:
            raise ValueError("Recovery EEF target exceeds 0.35 rad")
