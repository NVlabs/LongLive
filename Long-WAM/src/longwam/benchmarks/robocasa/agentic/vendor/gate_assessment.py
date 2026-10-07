# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/gate_assessment.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Require outcome/intent evidence before editing a student action proposal."""

GATE_INSTRUCTION = """Use the environment's task instruction as the objective.
Derive ordered subgoals and prerequisites from that instruction; do not assume
any particular object, destination or task family. Maintain task_progress:
verified_completed (list), currently_attempting (string), remaining (list).
Update completion only from visual execution evidence; undo a completed
subgoal if later observations show it has been lost (e.g. a structure collapses).
Keep the original task instruction as the student's input.
At each chunk boundary, assess two separate questions:
1. What happened during the LAST executed chunk? Compare before/after RGB,
   measured robot state, executed gripper commands and same-episode history.
   A closed command alone proves neither grasp success nor failure. Look for
   object motion during lifting, slipping, missed placement or sustained lack
   of progress. Distinguish pending/uncertain results from an observed failure.
2. Given the current task phase, does the NEXT student chunk pursue an
   appropriate subgoal? Infer intent from its robot-only FK trajectory and
   gripper sequence; do not claim access to the student's internal intention.
   Advancing to a dependent subgoal after its prerequisite failed is wrong.
   Also check selected object, destination, required order and grasp/release phase.
   Realigning for a retry can be appropriate: allow student self-recovery.
Return a concise assessment with task_progress, current_subgoal, execution_status,
execution_evidence, expected_next_intent, predicted_next_intent, intent_status,
and intent_evidence. Statuses: execution not_started/progressing/failed/
uncertain/recovered; intent aligned/misaligned/uncertain.
An edit/eef takeover requires execution_status=failed or intent_status=misaligned.
Uncertainty alone or an aesthetically imperfect pose is not a takeover reason.
After recovery, hand back when the current state and student subgoal are suitable.
Do not rewind. Do not use object truth, reward or future simulated object states.
"""


def validate_assessment(response, request):
    assessment = response.get("assessment")
    if not isinstance(assessment, dict):
        raise ValueError("Outcome and intent assessment required")
    progress = assessment.get("task_progress")
    if not isinstance(progress, dict):
        raise ValueError("Generic task_progress required")
    for key in ("verified_completed", "remaining"):
        if not isinstance(progress.get(key), list) or any(
            not isinstance(item, str) or not item.strip() for item in progress[key]
        ):
            raise ValueError(f"task_progress.{key} must be a list of subgoal strings")
    if (
        not isinstance(progress.get("currently_attempting"), str)
        or not progress["currently_attempting"].strip()
    ):
        raise ValueError("task_progress.currently_attempting required")
    for key in (
        "current_subgoal",
        "execution_evidence",
        "expected_next_intent",
        "predicted_next_intent",
        "intent_evidence",
    ):
        if not isinstance(assessment.get(key), str) or not assessment[key].strip():
            raise ValueError(f"Assessment requires visible-evidence text: {key}")
    execution = assessment.get("execution_status")
    intent = assessment.get("intent_status")
    if execution not in ("not_started", "progressing", "failed", "uncertain", "recovered"):
        raise ValueError("Unknown execution assessment status")
    if intent not in ("aligned", "misaligned", "uncertain"):
        raise ValueError("Unknown intent assessment status")
    if (request["step_id"] == 0) != (execution == "not_started"):
        raise ValueError("not_started is required only before the first control step")
    failure, wrong_intent = execution == "failed", intent == "misaligned"
    if response["mode"] in ("edit", "eef"):
        if not (failure or wrong_intent):
            raise ValueError("Takeover requires observed failure or wrong task intent")
        return (
            "both"
            if failure and wrong_intent
            else ("execution_failure" if failure else "wrong_intent")
        )
    # A failed action does not mandate takeover if the student can self-recover.
    return "none"


def previous_observation(request, record):
    """Only prior RGB/robot measurements and executed command history."""
    if request is None:
        return None
    return dict(
        step_id=request["step_id"],
        images=request["images"],
        current_eef=request["current_eef"],
        current_state=request["current_state"],
        executed_steps=record["executed_steps"],
        executed_gripper_closed=record["executed_gripper_closed"],
        source_detail=record["source_detail"],
    )
