# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/tools.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""File-backed blocking rollout tools exposed to the configured agent (Codex dynamic tools).

Mirrors ``RoboDojoTools`` of the hybrid framework: the agent schedules every
start -> infer -> execute cycle; nothing is inferred, retried or corrected implicitly.
Single arm (PandaOmron): ``edit``/``eef`` carry one target, not left/right.
"""

from __future__ import annotations
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from longwam.benchmarks.robocasa.agentic import contract as C
from longwam.benchmarks.robocasa.agentic.session import (
    RoboCasaSession,
    pose_from_state,
    robot_only_fk,
)
from longwam.benchmarks.robocasa.agentic.vendor.action_edit_kinematics import edited_targets
from longwam.benchmarks.robocasa.agentic.vendor.gate_assessment import (
    GATE_INSTRUCTION,
    previous_observation,
    validate_assessment,
)
from longwam.agentic._vendor.io import InputError, write_json
from longwam.benchmarks.robocasa.agentic.vendor.schema import _obj
from longwam.benchmarks.robocasa.agentic.vendor.validation import (
    angles,
    validate_public_language,
    validate_response,
)

CONTEXT_VERSION = "robocasa365-v1"
TOOL_START, TOOL_INFER, TOOL_EXECUTE = ("robocasa_start", "longwam_infer", "robocasa_execute")


def response_schema(
    request_id=None,
    *,
    allow_subgoal_prompt=False,
    student_cap=C.STUDENT_MAX_STEPS,
    modes=("student", "edit", "eef", "stop"),
):
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    vector4 = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    string = {"type": "string"}
    progress = _obj(
        dict(
            verified_completed={"type": "array", "items": string},
            currently_attempting=string,
            remaining={"type": "array", "items": string},
        )
    )
    assessment = _obj(
        dict(
            task_progress=progress,
            current_subgoal=string,
            execution_status={
                "type": "string",
                "enum": ["not_started", "progressing", "failed", "uncertain", "recovered"],
            },
            execution_evidence=string,
            expected_next_intent=string,
            predicted_next_intent=string,
            intent_status={"type": "string", "enum": ["aligned", "misaligned", "uncertain"]},
            intent_evidence=string,
        )
    )
    identifier = string if request_id is None else {"type": "string", "enum": [request_id]}
    return _obj(
        dict(
            request_id=identifier,
            mode={"type": "string", "enum": list(modes)},
            steps={"type": "integer", "minimum": 1, "maximum": student_cap},
            reason=string,
            edit=_obj(
                dict(
                    delta_position=vector3,
                    delta_rotation_vector=vector3,
                    gripper={"type": "string", "enum": ["keep", "open", "closed"]},
                )
            ),
            target=_obj(
                dict(position=vector3, quaternion_wxyz=vector4, gripper_closed={"type": "boolean"})
            ),
            assessment=assessment,
        )
    )


def tool_specs(method: str):
    if method != "hybrid_decompose":
        raise ValueError("Only hybrid_decompose is a public experiment")
    string = {"type": "string"}

    def spec(name, description, **properties):
        return dict(
            type="function", name=name, description=description, inputSchema=_obj(properties)
        )

    infer_props = dict(observation_path=string, output_dir=string)
    infer_desc = "Blocking: run one Long-WAM inference from the latest observation (the policy keeps its own 2.4 s observation history); save the 32x12 proposal and return its robot-only EEF goal trajectory integrated from the measured pose. Diagnostics are not a gate verdict."
    infer_props["instruction"] = string
    infer_desc += " `instruction` is the text given to the student: pass the original task instruction, or ONE atomic sub-instruction (an imperative sentence describing only the current subgoal, e.g. 'open the cabinet door') when the composite instruction is not what the student should pursue right now."
    plan_specs = []
    return [
        spec(
            TOOL_START,
            "Blocking: start the requested RoboCasa episode; save and return the current RGB views, 16-D state and measured EEF pose.",
            task=string,
            output_dir=string,
        ),
        *plan_specs,
        spec(TOOL_INFER, infer_desc, **infer_props),
        spec(
            TOOL_EXECUTE,
            "Blocking: validate and execute a reviewed proposal prefix (student, 1-15 steps) or a short EEF correction (edit/eef, 1-5 steps); save and return the new observation and terminal result.",
            proposal_path=string,
            response=response_schema(),
            output_dir=string,
        ),
    ]


class RoboCasaTools:
    profile = "robocasa365_pandaomron"

    def __init__(
        self,
        output: Path,
        session: RoboCasaSession,
        *,
        method: str,
        episode_index: int,
        max_decisions: int = 0,
        prompt_sha256: str = "",
        student_identity: dict | None = None,
    ):
        self.output = Path(output).resolve()
        self.session, self.method = (session, method)
        self.episode_index = int(episode_index)
        self.max_decisions, self.prompt_sha256 = (int(max_decisions), prompt_sha256)
        self.student_identity = student_identity or {}
        self.phase = "start"
        self.history: list[dict[str, Any]] = []
        self.preceding = None
        self.counters = dict(
            student_steps=0, edited_steps=0, recovery_steps=0, predictions=0, subgoal_prompts=0
        )
        self.source = "student"
        self.started = time.monotonic()
        self.actions = self.fk = self.request = self.prediction = None
        self.observation_path = self.proposal_path = None
        self.plan: list[str] | None = None
        self.plan_revisions = 0

    @property
    def tick(self):
        return self.session.step_id

    @property
    def episode(self):
        return self.session.episode_id

    def next_call(self):
        index = len(self.history)
        if self.phase == "start":
            return dict(
                tool=TOOL_START,
                task=self.session.task,
                output_dir=str(self.output / "observations" / "000"),
            )
        if self.phase == "infer":
            call = dict(
                tool=TOOL_INFER,
                observation_path=str(self.observation_path),
                output_dir=str(self.output / "proposals" / f"{index:03d}"),
            )
            call["instruction"] = "<original instruction or one atomic sub-instruction>"
            return call
        if self.phase == "execute":
            return dict(
                tool=TOOL_EXECUTE,
                proposal_path=str(self.proposal_path),
                output_dir=str(self.output / "observations" / f"{index + 1:03d}"),
                response="Supply your gate assessment and student/edit/eef decision; this benchmark waits for native success or the native step limit",
            )
        return None

    def _check(self, name, arguments, *, free_keys=()):
        expected = self.next_call()
        if expected is None or name != expected["tool"]:
            raise InputError(f"Wrong order or completed rollout; next call: {expected}")
        for key, value in expected.items():
            if key in ("tool", "response", *free_keys):
                continue
            if arguments.get(key) != value:
                raise InputError(f"{key} must equal {value!r}")
        destination = Path(arguments["output_dir"])
        if destination.exists() or not destination.resolve().is_relative_to(self.output):
            raise InputError("Output must be a fresh directory under this rollout root")
        return destination

    def _observation(self, directory: Path, *, result=None):
        directory.mkdir(parents=True)
        state = self.session.state()
        cams = self.session.cameras()
        np.savez_compressed(directory / "observation.npz", state=state, **cams)
        images = []
        for camera in C.CAMERA_ORDER:
            path = directory / f"{camera}.png"
            Image.fromarray(cams[camera]).save(path)
            images.append(dict(camera=camera, path=str(path)))
        teacher_images = None
        self.observation_path = directory / "observation.json"
        packet = dict(
            episode_id=self.episode,
            step_id=self.tick,
            task=self.session.task,
            task_group=self.session.group,
            max_episode_steps=self.session.horizon,
            instruction=self.session.instruction,
            remaining_steps=self.session.remaining_steps(),
            require_native_termination=True,
            control_hz=C.CONTROL_HZ,
            current_state=state.tolist(),
            current_eef=angles(pose_from_state(state)),
            gripper_closed_estimate=bool(np.mean(state[C.S_GRIPPER]) > 0.02),
            images=teacher_images or images,
            student_images=images if teacher_images else None,
            robot_profile=self.profile,
            observation_path=str(self.observation_path),
            arrays_path=str(directory / "observation.npz"),
            history_path=str(self.output / "history.json"),
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters),
            result=result,
            rollout_finished=self.phase == "done",
            next_call=self.next_call(),
        )
        if self.tick == 0:
            packet["task_context"] = dict(
                benchmark="RoboCasa-365",
                success="native task success only (no partial score)",
                composite="composite tasks chain several atomic kitchen subtasks that must all be completed in a sensible order",
                arm_return_required=False,
            )
        self.observation_packet = packet
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check(TOOL_START, arguments)
        reset = self.session.reset(self.episode_index)
        self.run = dict(
            schema="robocasa_gpt6_rollout.run.v1",
            teacher="codex_tools",
            context_version=CONTEXT_VERSION,
            method=self.method,
            task=self.session.task,
            task_group=self.session.group,
            instruction=reset["instruction"],
            env_seed=self.session.env_seed,
            split=self.session.split,
            episode_index=self.episode_index,
            reset_seed=reset["reset_seed"],
            horizon=reset["horizon"],
            robot_profile=self.profile,
            student=self.student_identity,
            action_horizon=C.STUDENT_HORIZON,
            action_dim=C.ACTION_DIM,
            control_dt=C.CONTROL_DT,
            gate_policy="failure-or-intent",
            no_rollback=True,
            gate_instruction_sha256=hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest(),
            teacher_prompt_sha256=self.prompt_sha256,
            max_decisions=self.max_decisions,
            require_native_termination=True,
        )
        write_json(self.output / "run.json", self.run)
        write_json(self.output / "history.json", self.history)
        self.phase = "infer"
        packet = self._observation(directory)
        return packet

    def infer(self, **arguments):
        directory = self._check(TOOL_INFER, arguments, free_keys=("instruction",))
        prompt = None
        prompt = arguments.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InputError(
                "`instruction` must be the original task instruction or one atomic sub-instruction"
            )
        prompt = prompt.strip()
        if prompt != self.session.instruction:
            self.counters["subgoal_prompts"] += 1
        directory.mkdir(parents=True)
        decision = len(self.history)
        self.proposal_path = directory / "actions.npz"
        before = time.monotonic()
        self.actions = self.session.infer(prompt)
        state = self.session.state()
        np.savez_compressed(
            self.proposal_path,
            actions=self.actions,
            state=state,
            prompt=np.asarray(prompt or self.session.instruction),
        )
        self.prediction = dict(
            prediction_id=uuid.uuid4().hex,
            inference_index=self.session.inference_calls - 1,
            inference_seconds=time.monotonic() - before,
            student_prompt=prompt or self.session.instruction,
        )
        self.counters["predictions"] += 1
        self.fk = robot_only_fk(state, self.actions)
        a = self.actions
        diagnostics = dict(
            shape=list(a.shape),
            finite=bool(np.isfinite(a).all()),
            gripper_closed_steps=int(np.sum(a[:, C.A_GRIP] > 0)),
            base_mode_steps=int(np.sum(a[:, C.A_MODE] > 0)),
            max_abs_base_motion=float(np.max(np.abs(a[:, C.A_BASE]))),
            max_step_translation_m=float(
                np.max(np.linalg.norm(a[:, C.A_POS], axis=1)) * C.OSC_POS_SCALE_M
            ),
            max_step_rotation_rad=float(
                np.max(np.linalg.norm(a[:, C.A_ROT], axis=1)) * C.OSC_ROT_SCALE_RAD
            ),
        )
        self.request = dict(
            request_id=uuid.uuid4().hex,
            episode_id=self.episode,
            step_id=self.tick,
            decision=decision,
            gate_policy="failure-or-intent",
            gate_instruction=GATE_INSTRUCTION,
            task=self.session.task,
            instruction=self.session.instruction,
            sim_time_s=self.tick * C.CONTROL_DT,
            max_episode_steps=self.session.horizon,
            remaining_steps=self.session.remaining_steps(),
            require_native_termination=True,
            images=self.observation_packet["images"],
            current_state=self.observation_packet["current_state"],
            current_eef=self.observation_packet["current_eef"],
            student_eef_trajectory=[angles(p) for p in self.fk],
            robot_profile=self.profile,
            action_diagnostics=diagnostics,
            previous_observation=previous_observation(self.preceding, self.history[-1])
            if self.history
            else None,
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters),
            **self.prediction,
        )
        write_json(self.output / f"request_{decision:03d}.json", self.request)
        self.phase = "execute"
        packet = {k: v for k, v in self.request.items() if k != "gate_instruction"}
        packet.update(
            proposal_path=str(self.proposal_path),
            rollout_finished=False,
            request_path=str(self.output / f"request_{decision:03d}.json"),
            next_call=self.next_call(),
        )
        write_json(directory / "proposal.json", packet)
        return packet

    def _correction_targets(self, response):
        if response["mode"] == "eef":
            return [response["target"]] * response["steps"]
        targets = edited_targets(self.fk, response["steps"], response["edit"])
        if response["edit"]["gripper"] == "keep":
            for t, original in zip(targets, self.fk):
                t["gripper_closed"] = original["gripper_closed"]
        return targets

    def execute(self, **arguments):
        directory = self._check(TOOL_EXECUTE, arguments)
        response = arguments.get("response")
        try:
            validate_response(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        if response["mode"] == "stop":
            raise InputError(
                f"This benchmark requires native termination; model stop is not allowed. {self.session.remaining_steps()} control steps remain. Submit a student/edit/eef decision."
            )
        recent = [r["source_detail"] for r in self.history[-C.MAX_CONSECUTIVE_CORRECTIONS :]]
        if (
            response["mode"] in ("edit", "eef")
            and len(recent) == C.MAX_CONSECUTIVE_CORRECTIONS
            and all((m != "student" for m in recent))
        ):
            raise InputError(
                f"You have issued {C.MAX_CONSECUTIVE_CORRECTIONS} consecutive corrections without handing back. Hand control back now: submit mode=student (>= 5 steps of the fresh proposal) and reassess after the student acts. Fine manipulation (grasp, press, release) is the student's strength."
            )
        decision = len(self.history)
        write_json(self.output / f"response_{decision:03d}.json", response)
        trigger = validate_assessment(response, self.request)
        self.source = "student" if response["mode"] == "student" else "gpt_eef"
        start_tick = self.tick
        if response["mode"] == "student":
            rows = self.session.run_student_prefix(self.actions, response["steps"])
        else:
            rows = self.session.run_eef_targets(
                self._correction_targets(response), source="gpt_" + response["mode"]
            )
            write_json(
                self.output / f"edit_{decision:03d}.json",
                [
                    dict(
                        target=r["edited_target"],
                        diagnostics=r["eef_diagnostics"],
                        executed_action=r["executed_action"],
                    )
                    for r in rows
                ],
            )
        key = dict(student="student_steps", edit="edited_steps", eef="recovery_steps")[
            response["mode"]
        ]
        self.counters[key] += len(rows)
        executed = np.asarray([r["executed_action"] for r in rows], np.float32)
        np.savez_compressed(
            self.output / f"execution_{decision:03d}.npz",
            actions=executed,
            states=np.asarray([r["state_after"] for r in rows], np.float32),
        )
        record = dict(
            decision=decision,
            start_tick=start_tick,
            end_tick=self.tick,
            response=response,
            executed_steps=len(rows),
            source=self.source,
            source_detail=response["mode"],
            prediction_id=self.prediction["prediction_id"],
            student_prompt=self.prediction["student_prompt"],
            selected_student_steps=response["steps"] if response["mode"] == "student" else 0,
            discarded_student_steps=len(self.actions) - len(rows)
            if response["mode"] == "student"
            else len(self.actions),
            executed_gripper_closed=[bool(a[C.A_GRIP] > 0) for a in executed],
            assessment=response["assessment"],
            takeover_trigger=trigger,
            terminal=self.session.done,
            native_success=self.session.success,
        )
        self.history.append(record)
        self.preceding = self.request
        write_json(self.output / "history.json", self.history)
        write_json(
            self.output / "progress.json",
            dict(episode_id=self.episode, step_id=self.tick, **self.counters),
        )
        self.phase = "infer"
        budget = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        final = (
            self.finish("terminal" if self.session.done else "decision_budget")
            if self.session.done or budget
            else None
        )
        return self._observation(directory, result=final)

    def finish(self, reason):
        self.phase = "done"
        final = dict(
            self.session.summary(),
            reason=reason,
            decisions=len(self.history),
            **self.counters,
            context_version=CONTEXT_VERSION,
            wall_seconds=time.monotonic() - self.started,
            history_path=str(self.output / "history.json"),
        )
        final["complete"] = bool(final.get("complete")) and reason == "terminal"
        write_json(self.output / "result.json", final)
        try:
            self.session.write_video(self.output / "rollout.mp4")
            final["video_path"] = str(self.output / "rollout.mp4")
        except Exception as exc:
            final["video_error"] = str(exc)
        write_json(self.output / "result.json", final)
        return final
