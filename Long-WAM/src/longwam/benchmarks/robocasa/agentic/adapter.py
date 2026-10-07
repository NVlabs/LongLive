# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/codex_host.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""RoboCasa365 prompts and tool binding; the model runtime lives in longwam.agentic."""

import hashlib
import shutil
import sys
from pathlib import Path

from longwam.agentic import AgentSpec, InputError
from longwam.agentic._vendor.io import write_json
from .tools import CONTEXT_VERSION, TOOL_EXECUTE, TOOL_INFER, TOOL_START, tool_specs

SKILL_ROOT = Path(__file__).parent / "skill"
CONTROLLER_VERSION = "robocasa_recoverable_tool_input_v1"


def prepare_workspace(audit: Path, method: str) -> Path:
    """Agent working directory beside the host's audit files (skill + context + notes)."""
    agent = audit / "agent"
    agent.mkdir()
    (agent / "scratch").mkdir()
    shutil.copytree(SKILL_ROOT / "context", agent / "context")
    skill_name = "robocasa-hybrid-decompose-rollout"
    skill = agent / ".agents/skills" / skill_name
    skill.mkdir(parents=True)
    shutil.copy2(skill_file(method), skill / "SKILL.md")
    shutil.copy2(SKILL_ROOT / "gate_prompt.md", skill / "gate_prompt.md")
    shutil.copytree(agent / "context", skill / "context")
    controller = audit.parent
    workspace = dict(
        controller_output=str(controller),
        robot_profile="robocasa365_pandaomron",
        evaluation_method=method,
        python_executable=sys.executable,
        history_path=str(controller / "history.json"),
        observations_path=str(controller / "observations"),
        proposals_path=str(controller / "proposals"),
        execution_files=str(controller / "execution_NNN.npz"),
        eef_diagnostics_files=str(controller / "edit_NNN.json"),
        scratch=str(agent / "scratch"),
        notes_path=str(agent / "NOTES.md"),
        context_files={
            str(p.relative_to(agent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (agent / "context").rglob("*")
            if p.is_file()
        },
    )
    write_json(agent / "workspace.json", workspace)
    (agent / "NOTES.md").write_text("# Episode notes (agent-owned)\n")
    return agent


def skill_file(method: str) -> Path:
    if method != "hybrid_decompose":
        raise ValueError("Only the selected hybrid_decompose experiment is released")
    return SKILL_ROOT / "SKILL_hybrid_decompose.md"


def rejected_input(error, rollout, arguments, count):
    packet = dict(
        error=str(error),
        error_type="recoverable_tool_input",
        no_execution=True,
        retryable=True,
        controller_version=CONTROLLER_VERSION,
        rejected_calls=count,
        step_id=rollout.tick,
        next_call=rollout.next_call(),
        correction="Correct the arguments and retry next_call in this same episode. Do not reset or replay already executed actions.",
    )
    if getattr(rollout, "observation_path", None):
        packet["observation_path"] = str(rollout.observation_path)
    request = getattr(rollout, "request", None)
    if isinstance(request, dict) and rollout.phase in ("act", "execute"):
        packet["request_id"] = request.get("request_id")
        packet["current_eef"] = request.get("current_eef")
    return packet


class RoboCasaAdapter:
    def __init__(self, method="hybrid_decompose"):
        skill_file(method)  # Reject unsupported variants before any model call.
        self.method = method

    def prepare(self, audit: Path) -> AgentSpec:
        method = self.method
        agent_workspace = prepare_workspace(audit, method)
        skill = skill_file(method).read_text()
        gate = (SKILL_ROOT / "gate_prompt.md").read_text()
        context = (SKILL_ROOT / "context/teacher_context.md").read_text()
        instructions = (
            skill
            + ("\n\n# Unchanged baseline gate prompt\n\n" + gate)
            + "\n\n"
            + context
            + "\n\nAgent working directory: "
            + str(agent_workspace)
            + "\nResolve context/ and workspace.json relative to that directory.\n"
        )
        return AgentSpec(
            workspace=agent_workspace,
            instructions=instructions,
            tools=tool_specs(method),
            image_tools=frozenset({TOOL_START, TOOL_EXECUTE}),
            initial_message=(
                "Act as the autonomous policy agent for this single RoboCasa-365 simulation rollout. "
                "Use your normal file, image, shell/code and planning tools as useful, alongside the rollout "
                "service tools. Read workspace.json for paths and interpreter; keep working notes in your "
                "agent workspace. Use English for every public explanation, assessment, note and report. "
                "The user authorizes sending this episode's three RGB images, robot state, robot-only EEF "
                "trajectories, task text and same-episode history to OpenAI Codex for online decisions."
            ),
            metadata=dict(
                benchmark="robocasa365",
                evaluation_method=method,
                context_version=CONTEXT_VERSION,
                controller_version=CONTROLLER_VERSION,
            ),
        )


class RoboCasaEpisode:
    """Bridge the existing, unchanged start/infer/execute rollout to AgenticEpisode."""

    def __init__(self, rollout):
        self.rollout = rollout
        self.handlers = {
            TOOL_START: rollout.start,
            TOOL_INFER: rollout.infer,
            TOOL_EXECUTE: rollout.execute,
        }

    @property
    def finished(self):
        return self.rollout.phase == "done"

    @property
    def step_id(self):
        return self.rollout.tick

    def next_call(self):
        return self.rollout.next_call()

    def call_tool(self, name, arguments):
        if name not in self.handlers:
            raise InputError(f"Unknown RoboCasa tool: {name}")
        return self.handlers[name](**arguments)

    def rejected_input(self, error, count):
        return rejected_input(error, self.rollout, None, count)
