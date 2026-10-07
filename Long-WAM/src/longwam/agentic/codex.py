# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/codex_host.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# Changes: Minimal model-tool environment and private episode audit directory.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Benchmark-independent Codex sessions, tool dispatch, budgets and audit logs."""

from __future__ import annotations
import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from .api import AgenticAdapter, AgenticEpisode
from .config import AgenticConfig
from longwam.agentic._vendor.image_preview import image_max_edge, prepare_image
from longwam.agentic._vendor.io import InputError, write_json
from longwam.agentic._vendor.network_recovery import (
    NETWORK_CONTINUE_DELAYS,
    closed_network_turn,
    is_network_error,
)

PROVIDER = "openai"
DEFAULT_IMAGE_MAX_EDGE = 480
CONTROLLER_VERSION = "longwam_agentic_v1"
NATIVE_WORK_ITEMS = frozenset(
    (
        "commandExecution",
        "fileChange",
        "imageView",
        "webSearch",
        "mcpToolCall",
        "collabAgentToolCall",
    )
)


def toml_value(value):
    if isinstance(value, dict):
        return (
            "{" + ", ".join((json.dumps(k) + " = " + toml_value(v) for k, v in value.items())) + "}"
        )
    return json.dumps(value)


def agent_config(audit: Path, agent: Path, settings: AgenticConfig):
    state = Path(
        os.environ.get("ROLLOUT_CODEX_STATE_DIR", os.environ.get("CODEX_HOME", str(audit)))
    )
    return dict(
        model=settings.model,
        model_provider=PROVIDER,
        model_reasoning_effort=settings.effort,
        sqlite_home=str(state / "runtime_db"),
        log_dir=str(state / "runtime_logs"),
        default_permissions="rollout_agent",
        **{
            "features.shell_tool": True,
            "features.view_image": True,
            # Model tools must not inherit the app-server's authentication environment.
            "shell_environment_policy.inherit": "none",
            "shell_environment_policy.ignore_default_excludes": False,
            "shell_environment_policy.experimental_use_profile": False,
            "shell_environment_policy.set": {
                "PATH": os.defpath, "HOME": str(agent), "LANG": "C.UTF-8",
            },
            "permissions.rollout_agent.extends": ":workspace",
            "permissions.rollout_agent.filesystem": {
                str(audit.parent): "read",
                str(agent): "write",
            },
        },
    )


def content_items(packet, *, images, max_image_edge):
    visible, attachments = (packet, [])
    if images:
        visible = dict(packet, images=[])
        for item in packet.get("images", []):
            descriptor, payload = prepare_image(item, max_image_edge)
            visible["images"].append(descriptor)
            attachments.append(
                dict(
                    type="inputImage",
                    imageUrl="data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
                )
            )
    return [dict(type="inputText", text=json.dumps(visible, separators=(",", ":"))), *attachments]


class CodexAgent:
    def __init__(self, workspace: Path, *, adapter: AgenticAdapter, config: AgenticConfig):
        self.config = config.validate()
        self.transport = None
        from longwam.agentic._vendor.transport import StdioAppServer

        self.image_max_edge = image_max_edge(
            os.environ.get("CODEX_IMAGE_MAX_EDGE", str(DEFAULT_IMAGE_MAX_EDGE))
        )
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.spec = adapter.prepare(self.workspace)
        self.agent_workspace = Path(self.spec.workspace).resolve()
        if not self.agent_workspace.is_relative_to(self.workspace.resolve()):
            raise ValueError("Adapter workspace must be below its audit root")
        self.timeout = config.timeout_seconds
        self.prompt = self.spec.instructions
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace / "PROMPT.md").write_text(self.prompt)
        specs = self.spec.tools
        self.tool_names = {s["name"] for s in specs}
        if len(self.tool_names) != len(specs):
            raise ValueError("Duplicate Agentic tool names")
        write_json(self.workspace / "tools.json", specs)
        version = subprocess.run(
            [config.codex, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        runtime_config = agent_config(self.workspace, self.agent_workspace, config)
        argv = [config.codex, "app-server", "--stdio", "--strict-config"]
        for key, value in runtime_config.items():
            argv += ["-c", key + "=" + toml_value(value)]
        write_json(self.workspace / "launch.json", dict(argv=argv, config=runtime_config))
        self.transport = StdioAppServer(argv, self.workspace)
        try:
            self.transport.request(
                "initialize",
                dict(
                    clientInfo=dict(name="longwam-agentic", version="1"),
                    capabilities=dict(experimentalApi=True),
                ),
                self.timeout,
            )
            self.transport.notify("initialized", {})
            response = self.transport.request(
                "thread/start",
                dict(
                    cwd=str(self.agent_workspace),
                    model=self.config.model,
                    modelProvider=PROVIDER,
                    config=dict(model_reasoning_effort=self.config.effort),
                    developerInstructions=self.prompt,
                    dynamicTools=specs,
                    ephemeral=False,
                    allowProviderModelFallback=False,
                    approvalPolicy="never",
                    approvalsReviewer="auto_review",
                    permissions="rollout_agent",
                    runtimeWorkspaceRoots=[str(self.agent_workspace)],
                ),
                self.timeout,
            )
            if (
                response.get("model") != self.config.model
                or response.get("reasoningEffort") != self.config.effort
            ):
                raise RuntimeError(
                    f"Codex model/effort differs from {self.config.model}/{self.config.effort}: {response.get('model')}/{response.get('reasoningEffort')}"
                )
            self.thread_id = response["thread"]["id"]
            write_json(
                self.workspace / "worker.json",
                dict(
                    model=self.config.model,
                    model_provider=PROVIDER,
                    reasoning_effort=self.config.effort,
                    controller_version=CONTROLLER_VERSION,
                    adapter_metadata=self.spec.metadata,
                    codex_version=version,
                    thread_id=self.thread_id,
                    pid=self.transport.process.pid,
                    prompt_sha256=self.prompt_sha256,
                    tools=[s["name"] for s in specs],
                    codex_image_max_edge=self.image_max_edge,
                ),
            )
        except BaseException:
            self.close()
            raise

    def _turn(self, text):
        result = self.transport.request(
            "turn/start",
            dict(
                threadId=self.thread_id,
                model=self.config.model,
                effort=self.config.effort,
                approvalPolicy="never",
                approvalsReviewer="auto_review",
                permissions="rollout_agent",
                cwd=str(self.agent_workspace),
                runtimeWorkspaceRoots=[str(self.agent_workspace)],
                input=[dict(type="text", text=text)],
            ),
            self.timeout,
        )
        return result["turn"]["id"]

    def _network_continue(self, rollout, old_turn, error, consecutive, serial):
        delay = NETWORK_CONTINUE_DELAYS[min(consecutive, len(NETWORK_CONTINUE_DELAYS) - 1)]
        write_json(
            self.workspace / f"network_continue_{serial:02d}.json",
            dict(previous_turn=old_turn, error=error, delay=delay, step_id=rollout.step_id),
        )
        time.sleep(delay)
        return self._turn(
            "The previous model response failed with a network error; the simulator did not move. Continue the same episode; next call: "
            + json.dumps(rollout.next_call())
        )

    def run(self, rollout: AgenticEpisode):
        turn_id = self._turn(
            self.spec.initial_message + " First call: " + json.dumps(rollout.next_call())
        )
        call_index = errors = continuations = completed_turns = 0
        network_error = None
        network_consecutive = network_serial = 0
        token_limit = self.config.max_total_tokens
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError(
                        "Codex made no completed service or native tool call within the configured timeout"
                    )
                event = self.transport.next_message(remaining)
            except TimeoutError:
                if not network_error:
                    raise
                response = self.transport.request(
                    "thread/read",
                    dict(threadId=self.thread_id, includeTurns=True),
                    min(30, self.timeout),
                )
                turn = closed_network_turn(response, self.thread_id, turn_id)
                if turn is None:
                    raise
                event = dict(
                    method="turn/completed", params=dict(threadId=self.thread_id, turn=turn)
                )
            method, params = (event.get("method"), event.get("params", {}))
            if method == "thread/tokenUsage/updated" and params.get("threadId") == self.thread_id:
                usage = params.get("tokenUsage", {})
                write_json(
                    self.workspace.parent / "token_usage.json",
                    dict(
                        model=self.config.model,
                        effort=self.config.effort,
                        usage=usage,
                        configured_limit=token_limit,
                    ),
                )
                print(
                    json.dumps(dict(event="token_usage", total=usage.get("total", {}))), flush=True
                )
                if token_limit and usage.get("total", {}).get("totalTokens", 0) >= token_limit:
                    raise RuntimeError(
                        "Configured total-token budget reached; stopping this episode"
                    )
            elif method == "item/tool/call":
                if params.get("threadId") != self.thread_id or params.get("turnId") != turn_id:
                    raise RuntimeError("Tool call belongs to another thread or turn")
                name, arguments = params["tool"], params["arguments"]
                print(
                    json.dumps(
                        dict(
                            event="tool_start",
                            tool=name,
                            call=call_index,
                            finished=rollout.finished,
                            step_id=rollout.step_id,
                        )
                    ),
                    flush=True,
                )
                write_json(
                    self.workspace / f"call_{call_index:04d}_request.json",
                    dict(call_id=params["callId"], tool=name, arguments=arguments),
                )
                try:
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(
                                f"Tool arguments must be a JSON object: {error}"
                            ) from error
                    if name not in self.tool_names:
                        raise InputError(
                            "Unknown host service; native Codex tools are executed by app-server"
                        )
                    if not isinstance(arguments, dict):
                        raise InputError("Tool arguments must be a JSON object")
                    packet = rollout.call_tool(name, arguments)
                except InputError as error:
                    errors += 1
                    packet, success = rollout.rejected_input(error, errors), False
                else:
                    success, network_consecutive, network_error = True, 0, None
                write_json(self.workspace / f"call_{call_index:04d}_result.json", packet)
                self.transport.reply(
                    event["id"],
                    dict(
                        success=success,
                        contentItems=content_items(
                            packet,
                            images=success and name in self.spec.image_tools,
                            max_image_edge=self.image_max_edge,
                        ),
                    ),
                )
                call_index += 1
                print(
                    json.dumps(
                        dict(
                            event="tool_done",
                            tool=name,
                            step_id=rollout.step_id,
                            finished=rollout.finished,
                        )
                    ),
                    flush=True,
                )
                deadline = time.monotonic() + self.timeout
            elif method in ("item/started", "item/completed"):
                if params.get("threadId") != self.thread_id or params.get("turnId") != turn_id:
                    continue
                item = params.get("item", {})
                kind = item.get("type")
                if kind in NATIVE_WORK_ITEMS or (
                    kind == "agentMessage" and method == "item/completed"
                ):
                    with (self.workspace / "agent_events.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(dict(method=method, **params), ensure_ascii=False) + "\n"
                        )
                    record = dict(
                        event="agent_activity",
                        phase=method.split("/")[-1],
                        item_type=kind,
                        step_id=rollout.step_id,
                        status=item.get("status"),
                    )
                    if kind == "agentMessage":
                        record["text"] = item.get("text", "")[:400]
                    elif kind == "commandExecution":
                        record["command"] = item.get("command")
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    if kind in NATIVE_WORK_ITEMS:
                        deadline = time.monotonic() + self.timeout
            elif method == "turn/completed" and params.get("threadId") == self.thread_id:
                turn = params.get("turn", {})
                if turn.get("id") != turn_id:
                    continue
                write_json(self.workspace / f"turn_{completed_turns:02d}_completed.json", turn)
                completed_turns += 1
                failure = turn.get("error") or network_error
                if turn.get("status") == "failed" and is_network_error(failure):
                    if rollout.finished:
                        return
                    turn_id = self._network_continue(
                        rollout, turn_id, failure, network_consecutive, network_serial
                    )
                    network_serial += 1
                    network_consecutive += 1
                    network_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get("status") != "completed":
                    raise RuntimeError(f"Codex turn ended: {turn.get('status')}")
                if rollout.finished:
                    return
                if continuations >= 2:
                    raise RuntimeError("Codex ended repeatedly before completing the episode")
                continuations += 1
                turn_id = self._turn(
                    "The same episode is unfinished. Continue the skill; next call: "
                    + json.dumps(rollout.next_call())
                )
                deadline = time.monotonic() + self.timeout
            elif method in ("error", "turn/failed"):
                if params.get("threadId") not in (None, self.thread_id) or params.get(
                    "turnId"
                ) not in (None, turn_id):
                    continue
                if is_network_error(params.get("error", params)):
                    network_error = params.get("error", params)
                if method == "error" and params.get("willRetry") is True:
                    print(
                        json.dumps(
                            dict(
                                event="codex_transport_retry",
                                error=params.get("error"),
                                step_id=rollout.step_id,
                            )
                        ),
                        flush=True,
                    )
                    continue
                if network_error and is_network_error(params.get("error", params)):
                    if rollout.finished:
                        return
                    deadline = min(deadline, time.monotonic() + 30)
                    continue
                raise RuntimeError(f"Codex error: {params}")
            elif "id" in event and "method" in event:
                raise RuntimeError(f"Unexpected Codex capability request: {method}")

    def close(self):
        if self.transport is not None:
            self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
