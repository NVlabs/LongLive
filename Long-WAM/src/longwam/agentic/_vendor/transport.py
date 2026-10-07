# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/transport.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# Changes: Credential-redacted RPC/stderr audit logs; wire payloads stay unchanged.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Local Codex JSON-RPC transport; no dependency on the retained pipeline."""

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
from .io import redact_credentials, redact_log_line


class StdioAppServer:
    """Small JSON-RPC transport for a single local Codex app-server process."""

    def __init__(self, argv, workspace, *, popen=subprocess.Popen):
        self.workspace = Path(workspace)
        self._incoming = queue.Queue()
        self._responses = {}
        self._response_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._closed = False
        self._stdout_closed = threading.Event()
        self._in_log = (self.workspace / "rpc_in.jsonl").open("a", buffering=1)
        self._out_log = (self.workspace / "rpc_out.jsonl").open("a", buffering=1)
        self._stderr_log = (self.workspace / "stderr.log").open("a", buffering=1)
        agent_cwd = self.workspace / "agent"
        self.process = popen(
            argv,
            cwd=agent_cwd if agent_cwd.is_dir() else self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            raise RuntimeError("Codex app-server stdio pipes are unavailable")
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self._out_log.write(redact_log_line(line))
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    self._incoming.put(RuntimeError(f"Non-JSON Codex app-server output: {error}"))
                    continue
                identifier = message.get("id")
                if identifier is not None and "method" not in message:
                    with self._response_lock:
                        destination = self._responses.get(identifier)
                    if destination is not None:
                        destination.put(message)
                        continue
                self._incoming.put(message)
        finally:
            self._stdout_closed.set()
            # Startup/config errors close stdout before replying to initialize.
            # Wake outstanding RPCs immediately instead of waiting their full timeout.
            with self._response_lock:
                for destination in self._responses.values():
                    try:
                        destination.put_nowait(
                            dict(
                                error=dict(
                                    message="Codex app-server stdout closed; inspect stderr.log"
                                )
                            )
                        )
                    except queue.Full:
                        pass
            self._incoming.put(EOFError("Codex app-server stdout closed"))

    def _read_stderr(self):
        for line in self.process.stderr:
            self._stderr_log.write(redact_log_line(line))

    def _send(self, message):
        if self._closed or self._stdout_closed.is_set() or self.process.poll() is not None:
            raise RuntimeError("Codex app-server is not running")
        line = json.dumps(message, separators=(",", ":")) + "\n"
        with self._write_lock:
            self._in_log.write(redact_log_line(line))
            self.process.stdin.write(line)
            self.process.stdin.flush()

    def request(self, method, params, timeout):
        with self._response_lock:
            identifier = self._next_id
            self._next_id += 1
            destination = queue.Queue(maxsize=1)
            self._responses[identifier] = destination
        try:
            self._send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
            try:
                response = destination.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError(f"Timed out waiting for Codex RPC {method}") from error
            if "error" in response:
                raise RuntimeError(f"Codex RPC {method} failed: {redact_credentials(response['error'])}")
            return response.get("result")
        finally:
            with self._response_lock:
                self._responses.pop(identifier, None)

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def reply(self, identifier, result):
        self._send({"jsonrpc": "2.0", "id": identifier, "result": result})

    def next_message(self, timeout):
        try:
            message = self._incoming.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for a Codex turn event") from error
        if isinstance(message, BaseException):
            raise message
        return message

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        for stream in (self._in_log, self._out_log, self._stderr_log):
            stream.close()
