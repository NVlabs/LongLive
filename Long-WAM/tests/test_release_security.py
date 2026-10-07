# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM release tooling.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""Offline release/audit guards; all credentials here are synthetic."""

import importlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest

from longwam.agentic._vendor.io import redact_credentials, redact_log_line, write_json
from longwam.agentic._vendor.transport import StdioAppServer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_guard", ROOT / "scripts/check_public_release.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_release_has_no_detected_credentials_or_private_paths():
    count, findings = guard.scan_checkout(ROOT)
    assert count > 100
    assert findings == []


@pytest.mark.parametrize("prefix", ["sk-", "hf_", "ghp_", "github_pat_", "nvapi-"])
def test_scanner_and_log_redaction_detect_synthetic_tokens(prefix):
    secret = prefix + "a" * 40
    assert guard.content_findings(secret.encode()) == [(1, "credential token")]
    assert secret not in redact_log_line("error: " + secret + "\n")


@pytest.mark.parametrize("field", ["api_key", "OPENAI_API_KEY", "access_token", "refresh_token", "authorization", "client_secret"])
def test_structured_and_plaintext_credentials_are_redacted(field, tmp_path):
    secret = "synthetic-private-value"
    original = {"nested": [{field: secret}], "usage": {"totalTokens": 42}}
    redacted = redact_credentials(original)
    assert redacted["nested"][0][field] == "[REDACTED]"
    assert original["nested"][0][field] == secret
    assert redacted["usage"] == {"totalTokens": 42}
    for line in (json.dumps(original), f"{field}={secret}\n", f"{field}: Bearer {secret}\n"):
        assert secret not in redact_log_line(line)
    path = tmp_path / "audit.json"
    write_json(path, original)
    assert json.loads(path.read_text()) == redacted
    assert list(tmp_path.iterdir()) == [path]


def test_transport_redacts_audit_but_keeps_wire_messages():
    secret = "hf_" + "b" * 40
    message = {"id": 7, "params": {"access_token": secret}}
    client = object.__new__(StdioAppServer)
    client._closed = False
    client._stdout_closed = threading.Event()
    client._write_lock = threading.Lock()
    client._response_lock = threading.Lock()
    client._responses = {}
    client._incoming = queue.Queue()
    client._in_log, client._out_log, client._stderr_log = io.StringIO(), io.StringIO(), io.StringIO()
    client.process = SimpleNamespace(
        poll=lambda: None, stdin=io.StringIO(),
        stdout=io.StringIO(json.dumps(message) + "\n"),
        stderr=io.StringIO("OPENAI_API_KEY=" + secret + "\n"),
    )
    client._send(message)
    client._read_stdout()
    client._read_stderr()
    assert json.loads(client.process.stdin.getvalue()) == message
    assert client._incoming.get_nowait() == message
    for stream in (client._in_log, client._out_log, client._stderr_log):
        assert secret not in stream.getvalue()
        assert "[REDACTED]" in stream.getvalue()


def test_history_scan_finds_deleted_key_without_printing_it(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    private = tmp_path / "example.txt"
    secret = "nvapi-" + "x" * 30
    private.write_text(secret)
    guard.git(tmp_path, "add", "example.txt")
    guard.git(tmp_path, "-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    private.unlink()
    guard.git(tmp_path, "add", "-u")
    guard.git(tmp_path, "-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "-qm", "remove fixture")
    assert guard.scan_checkout(tmp_path)[1] == []
    count, findings = guard.scan_history(tmp_path)
    assert count == 1 and findings[0][2] == "credential token"
    assert secret not in repr(findings)


def test_scanner_refuses_symlink_without_reading_target(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "link").symlink_to("/nonexistent-private-file")
    assert guard.scan_checkout(tmp_path)[1] == [("link", 0, "review symlink before release")]


def test_monorepo_scan_is_scoped_to_the_project(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    project = tmp_path / "Long-WAM"
    neighbor = tmp_path / "AnotherProject"
    project.mkdir()
    neighbor.mkdir()
    (project / "README.md").write_text("Synthetic public project")
    (neighbor / "auth.json").write_text(json.dumps({"api_key": "sk-" + "z" * 30}))
    guard.git(tmp_path, "add", ".")
    guard.git(tmp_path, "-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    assert guard.scan_checkout(project) == (1, [])
    assert guard.scan_history(project) == (1, [])
    assert guard.scan_history(tmp_path)[1]  # The synthetic neighbor is intentionally not clean.


@pytest.mark.parametrize("name", [".env", ".env.local", "auth.json", "private.pem", ".codex/config.toml"])
def test_private_filenames_are_blocked(name):
    assert guard.forbidden_name(name)
    assert not guard.forbidden_name(".env.example")


@pytest.mark.parametrize("benchmark", ["robocasa", "robocasa_gr1"])
def test_policy_socket_is_private_before_listening(benchmark, tmp_path, monkeypatch):
    policy = importlib.import_module(f"longwam.benchmarks.{benchmark}.policy")
    real_socket = socket.socket
    path = tmp_path / "private/policy.sock"
    checked = []

    class StopAfterPermissionCheck(Exception):
        pass

    class BoundSocket:
        def __init__(self, *args):
            self.socket = real_socket(*args)

        def bind(self, name):
            self.socket.bind(name)

        def listen(self, backlog):
            checked.append(path.stat().st_mode & 0o777)
            raise StopAfterPermissionCheck

        def close(self):
            self.socket.close()

    monkeypatch.setattr(policy.socket, "socket", BoundSocket)
    with pytest.raises(StopAfterPermissionCheck):
        policy.UnixPolicyServer(path, None).serve_forever()
    assert checked == [0o600]
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not path.exists()
