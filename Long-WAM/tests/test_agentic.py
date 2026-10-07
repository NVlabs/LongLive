# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""Offline Agentic API contracts: no model calls, GPUs or simulator assets."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from longwam.agentic import AgenticConfig, AgentSpec, CodexAgent, InputError
from longwam.agentic import codex as host
from longwam.agentic._vendor import transport
from longwam.evaluation import read_settings

ROOT = Path(__file__).resolve().parents[1]


class SyntheticAdapter:
    """An unrelated benchmark: no RoboCasa tools, state or action dimensions."""

    def prepare(self, audit):
        workspace = audit / "agent"
        workspace.mkdir()
        return AgentSpec(
            workspace=workspace,
            instructions="Use the synthetic_step tool.",
            tools=[
                dict(
                    name="synthetic_step",
                    description="Complete the synthetic task",
                    inputSchema=dict(type="object", properties={}),
                )
            ],
            initial_message="Run one synthetic task.",
            metadata={"benchmark": "synthetic"},
        )


class SyntheticEpisode:
    finished = False
    step_id = 0

    def next_call(self):
        return None if self.finished else {"tool": "synthetic_step"}

    def call_tool(self, name, arguments):
        assert name == "synthetic_step"
        if arguments:
            raise InputError("Expected no arguments")
        self.step_id += 1
        self.finished = True
        return {"done": True}

    def rejected_input(self, error, count):
        return {"error": str(error), "count": count, "no_execution": True}


def tool_call(arguments=None, **overrides):
    params = dict(
        threadId="thread",
        turnId="turn",
        tool="synthetic_step",
        callId="call",
        arguments={} if arguments is None else arguments,
    )
    params.update(overrides)
    return dict(id=99, method="item/tool/call", params=params)


def completed(status="completed", error=None, turn="turn"):
    return dict(
        method="turn/completed",
        params=dict(threadId="thread", turn=dict(id=turn, status=status, error=error)),
    )


@pytest.fixture
def fake_codex(monkeypatch):
    instances = []
    events = [tool_call(), completed()]
    response_override = {}

    class FakeTransport:
        def __init__(self, argv, workspace):
            self.argv = argv
            self.calls = []
            self.replies = []
            self.closed = False
            self.process = SimpleNamespace(pid=12345)
            self.events = iter(events)
            instances.append(self)

        def request(self, method, params, timeout):
            self.calls.append((method, params))
            if method == "thread/start":
                return dict(
                    dict(
                        thread={"id": "thread"},
                        model=params["model"],
                        reasoningEffort=params["config"]["model_reasoning_effort"],
                    ),
                    **response_override,
                )
            if method == "turn/start":
                return {"turn": {"id": "turn"}}
            return {}

        def notify(self, method, params):
            self.calls.append((method, params))

        def next_message(self, timeout):
            try:
                return next(self.events)
            except StopIteration as error:
                raise TimeoutError("No more fake events") from error

        def reply(self, identifier, result):
            self.replies.append((identifier, result))

        def close(self):
            self.closed = True

    monkeypatch.setattr(transport, "StdioAppServer", FakeTransport)
    monkeypatch.setattr(host.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="fake-cli"))
    monkeypatch.setattr(host.time, "sleep", lambda _: None)
    return SimpleNamespace(instances=instances, events=events, override=response_override)


def settings(**kwargs):
    return AgenticConfig(
        **dict(
            {"allow_model_upload": True, "max_total_tokens": 1000},
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "model,effort",
    [
        ("gpt-6-astra", "xhigh"),
        ("test-sol-model-id", "high"),
    ],
)
def test_other_benchmark_and_exact_model_pass_through(tmp_path, fake_codex, model, effort):
    # Ambient values cannot override an explicit instance's settings.
    episode = SyntheticEpisode()
    with CodexAgent(
        tmp_path / "audit", adapter=SyntheticAdapter(), config=settings(model=model, effort=effort)
    ) as agent:
        agent.run(episode)
    client = fake_codex.instances[0]
    assert episode.finished and episode.step_id == 1 and client.closed
    assert len(client.replies) == 1 and client.replies[0][1]["success"]
    requests = dict(client.calls)
    assert requests["thread/start"]["model"] == model
    assert requests["thread/start"]["config"]["model_reasoning_effort"] == effort
    assert requests["thread/start"]["allowProviderModelFallback"] is False
    assert requests["turn/start"]["model"] == model
    assert requests["turn/start"]["effort"] == effort
    assert f'model="{model}"' in client.argv
    record = json.loads((tmp_path / "audit/worker.json").read_text())
    assert (record["model"], record["reasoning_effort"]) == (model, effort)
    assert record["adapter_metadata"] == {"benchmark": "synthetic"}


def test_defaults_do_not_follow_ambient_environment(monkeypatch):
    monkeypatch.setenv("ROLLOUT_MODEL", "some-other-model")
    monkeypatch.setenv("ROLLOUT_EFFORT", "low")
    cfg = AgenticConfig()
    assert (cfg.model, cfg.effort) == ("gpt-6-astra", "xhigh")


def test_model_shell_does_not_inherit_host_credentials(tmp_path, fake_codex, monkeypatch):
    secret = "synthetic-private-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", secret)
    with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()):
        pass
    record = json.loads((tmp_path / "audit/launch.json").read_text())
    cfg = record["config"]
    assert cfg["shell_environment_policy.inherit"] == "none"
    assert cfg["shell_environment_policy.ignore_default_excludes"] is False
    assert cfg["shell_environment_policy.experimental_use_profile"] is False
    assert cfg["shell_environment_policy.set"] == {
        "PATH": os.defpath, "HOME": str(tmp_path / "audit/agent"), "LANG": "C.UTF-8",
    }
    assert secret not in json.dumps(record)
    assert (tmp_path / "audit").stat().st_mode & 0o777 == 0o700
    # Authentication is still available to the caller's Codex client, not its model tools.
    assert os.environ["OPENAI_API_KEY"] == secret


def test_relative_output_paths_are_resolved_before_codex_launch(tmp_path, fake_codex, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with CodexAgent(Path("outputs/audit"), adapter=SyntheticAdapter(), config=settings()) as agent:
        assert agent.workspace == tmp_path / "outputs/audit"
        assert agent.agent_workspace.is_absolute()
        agent.run(SyntheticEpisode())


def test_backend_cannot_silently_substitute_model(tmp_path, fake_codex):
    fake_codex.override["model"] = "unexpected-model"
    with pytest.raises(RuntimeError, match="model/effort differs"):
        CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings())
    assert fake_codex.instances[0].closed
    assert not any(name == "turn/start" for name, _ in fake_codex.instances[0].calls)


def test_timeout_closes_transport(tmp_path, fake_codex):
    fake_codex.events.clear()
    with pytest.raises(TimeoutError):
        with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
            agent.run(SyntheticEpisode())
    assert fake_codex.instances[0].closed


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"allow_model_upload": False}, "allow_model_upload"),
        ({"max_total_tokens": None}, "budget"),
        ({"max_total_tokens": 0}, "budget"),
        ({"max_total_tokens": True}, "budget"),
        ({"max_total_tokens": 1.5}, "budget"),
        ({"model": ""}, "model"),
        ({"model": " gpt-6-astra"}, "model"),
        ({"effort": None}, "effort"),
        ({"effort": "high effort"}, "effort"),
        ({"codex": ""}, "codex"),
        ({"timeout_seconds": float("nan")}, "timeout"),
        ({"timeout_seconds": float("inf")}, "timeout"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"backend": "unknown"}, "backend"),
    ],
)
def test_invalid_config_fails_before_process_or_files(tmp_path, fake_codex, changes, match):
    with pytest.raises(ValueError, match=match):
        CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings(**changes))
    assert not fake_codex.instances
    assert not (tmp_path / "audit").exists()


def test_input_error_is_recoverable_without_action_replay(tmp_path, fake_codex):
    fake_codex.events[:] = [tool_call('{"invalid": true}'), tool_call(), completed()]
    episode = SyntheticEpisode()
    with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
        agent.run(episode)
    assert episode.step_id == 1
    assert [r["success"] for _, r in fake_codex.instances[0].replies] == [False, True]


def test_usage_budget_stops_before_tools(tmp_path, fake_codex):
    fake_codex.events.insert(
        0,
        dict(
            method="thread/tokenUsage/updated",
            params=dict(
                threadId="thread",
                tokenUsage={"total": {"totalTokens": 1000}},
            ),
        ),
    )
    episode = SyntheticEpisode()
    with pytest.raises(RuntimeError, match="budget"):
        with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
            agent.run(episode)
    assert not episode.finished and episode.step_id == 0
    assert fake_codex.instances[0].closed
    assert (tmp_path / "token_usage.json").is_file()


def test_foreign_thread_cannot_execute(tmp_path, fake_codex):
    fake_codex.events[:] = [tool_call(threadId="other")]
    episode = SyntheticEpisode()
    with pytest.raises(RuntimeError, match="another thread"):
        with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
            agent.run(episode)
    assert episode.step_id == 0


def test_non_network_failure_is_not_retried(tmp_path, fake_codex):
    fake_codex.events[:] = [completed("failed", {"message": "unauthorized"})]
    with pytest.raises(RuntimeError, match="turn ended"):
        with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
            agent.run(SyntheticEpisode())
    calls = fake_codex.instances[0].calls
    assert sum(method == "turn/start" for method, _ in calls) == 1


def test_network_continuation_keeps_thread_and_does_not_replay(tmp_path, fake_codex):
    fake_codex.events[:] = [
        completed("failed", {"message": "stream disconnected"}),
        tool_call(),
        completed(),
    ]
    episode = SyntheticEpisode()
    with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
        agent.run(episode)
    calls = fake_codex.instances[0].calls
    assert sum(method == "thread/start" for method, _ in calls) == 1
    assert sum(method == "turn/start" for method, _ in calls) == 2
    assert episode.step_id == 1


def test_incomplete_turns_are_bounded(tmp_path, fake_codex):
    fake_codex.events[:] = [completed(), completed(), completed()]
    with pytest.raises(RuntimeError, match="repeatedly"):
        with CodexAgent(tmp_path / "audit", adapter=SyntheticAdapter(), config=settings()) as agent:
            agent.run(SyntheticEpisode())


def test_core_imports_without_any_benchmark_or_ml_runtime():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('longwam.benchmarks', 'torch', 'scipy', 'robocasa', 'robosuite')):
            raise RuntimeError('Core must not import ' + fullname)
sys.meta_path.insert(0, Guard())
from longwam.agentic import CodexAgent, AgenticConfig, AgenticAdapter
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True)


@pytest.mark.parametrize(
    "filename,namespace",
    [
        ("robocasa365_agentic.yaml", "agentic"),
        ("robocasa365_gpt6.yaml", "gpt6"),
    ],
)
def test_new_and_legacy_configs(filename, namespace):
    from longwam.benchmarks.robocasa.agentic.evaluate import validate_agentic_settings

    cfg = read_settings(
        ROOT / "configs/eval" / filename,
        [
            "agentic.model=test-sol-model-id",
            "agentic.effort=high",
            "gpt6.allow_model_upload=true",
            "gpt6.max_total_tokens=100",
        ],
    )
    assert cfg[namespace].model == "test-sol-model-id"
    assert validate_agentic_settings(cfg).effort == "high"
    with pytest.raises(ValueError, match="Duplicate"):
        read_settings(ROOT / "configs/eval" / filename, ["gpt6.model=one", "agentic.model=two"])


def test_robocasa_adapter_exposes_only_original_tools_and_bounds(tmp_path):
    from longwam.benchmarks.robocasa.agentic.adapter import RoboCasaAdapter
    from longwam.benchmarks.robocasa.agentic.contract import (
        CORRECTION_MAX_STEPS,
        MAX_CONSECUTIVE_CORRECTIONS,
        STUDENT_MAX_STEPS,
    )

    spec = RoboCasaAdapter().prepare(tmp_path)
    assert [s["name"] for s in spec.tools] == [
        "robocasa_start",
        "longwam_infer",
        "robocasa_execute",
    ]
    assert spec.image_tools == frozenset({"robocasa_start", "robocasa_execute"})
    assert (STUDENT_MAX_STEPS, CORRECTION_MAX_STEPS, MAX_CONSECUTIVE_CORRECTIONS) == (15, 5, 3)
    assert spec.workspace.is_relative_to(tmp_path)
    assert "robot_profile" in json.loads((spec.workspace / "workspace.json").read_text())


def test_incomplete_robocasa_result_is_not_complete(tmp_path):
    from longwam.benchmarks.robocasa.agentic.tools import RoboCasaTools
    import time

    session = SimpleNamespace(
        summary=lambda: {"complete": True, "success": True}, write_video=lambda _: None
    )
    rollout = RoboCasaTools(tmp_path, session, method="hybrid_decompose", episode_index=0)
    rollout.started = time.monotonic()
    assert rollout.finish("decision_budget")["complete"] is False


def test_legacy_imports_still_resolve():
    from longwam.benchmarks.robocasa_gpt6.evaluate import validate_gpt_settings
    from longwam.benchmarks.robocasa_gpt6.codex_host import skill_file
    from longwam.benchmarks.robocasa_gpt6.vendor.transport import StdioAppServer
    from longwam.benchmarks.robocasa_gpt6.vendor.io import InputError as LegacyInputError

    assert callable(validate_gpt_settings) and callable(StdioAppServer)
    assert LegacyInputError is InputError
    assert skill_file("hybrid_decompose").is_file()


def test_unsupported_benchmark_does_not_silently_use_standard_eval():
    with pytest.raises(ValueError, match="requires"):
        read_settings(ROOT / "configs/eval/robocasa365_agentic.yaml", ["benchmark=robocasa_gr1"])


def test_robocasa_binding_dispatches_full_tool_cycle(tmp_path, fake_codex):
    from longwam.benchmarks.robocasa.agentic.adapter import RoboCasaAdapter, RoboCasaEpisode

    rollout = SimpleNamespace(phase="start", tick=0)
    calls = []
    phases = {"start": "robocasa_start", "infer": "longwam_infer", "execute": "robocasa_execute"}
    rollout.next_call = lambda: {"tool": phases[rollout.phase]} if rollout.phase != "done" else None

    def advance(name, next_phase):
        def run(**arguments):
            calls.append(name)
            rollout.phase = next_phase
            rollout.tick += 1
            return {"phase": next_phase}

        return run

    rollout.start = advance("start", "infer")
    rollout.infer = advance("infer", "execute")
    rollout.execute = advance("execute", "done")
    fake_codex.events[:] = [tool_call(tool=name) for name in phases.values()] + [completed()]
    with CodexAgent(tmp_path / "audit", adapter=RoboCasaAdapter(), config=settings()) as agent:
        agent.run(RoboCasaEpisode(rollout))
    assert calls == ["start", "infer", "execute"]
    assert rollout.phase == "done"


def test_evaluation_forwards_explicit_settings_and_records_model(tmp_path, monkeypatch):
    from longwam.benchmarks.robocasa.agentic import evaluate
    from longwam.benchmarks.robocasa import rollout as runtime

    model_config, stats = tmp_path / "config.yaml", tmp_path / "stats.json"
    model_config.write_text("example: true")
    stats.write_text("{}")
    cfg = read_settings(
        ROOT / "configs/eval/robocasa365_agentic.yaml",
        [
            "checkpoint=not-loaded.pt",
            f"model_config={model_config}",
            f"stats={stats}",
            f"output_dir={tmp_path}",
            "episodes=1",
            "agentic.allow_model_upload=true",
            "agentic.max_total_tokens=1000",
            "agentic.model=test-sol-model-id",
            "agentic.effort=high",
        ],
    )
    monkeypatch.setattr(runtime, "_load_robocasa_runtime", lambda: ({}, None, None, None))
    monkeypatch.setattr(runtime, "resolve_tasks", lambda *args: ["SyntheticTask"])

    def fake_run(command, **kwargs):
        assert command[2] == "longwam.benchmarks.robocasa.agentic.run_rollout"
        assert command[command.index("--model") + 1] == "test-sol-model-id"
        assert command[command.index("--effort") + 1] == "high"
        assert command[command.index("--max-total-tokens") + 1] == "1000"
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "result.json").write_text(
            json.dumps(dict(complete=True, reason="terminal", success=True))
        )

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    evaluate.run(cfg)
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["agentic"]["model"] == "test-sol-model-id"
    assert result["complete"] and len(result["episodes"]) == 1
