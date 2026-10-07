#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM release tooling.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""Offline credential/path guard. Reports locations, never matched secret values.

Checks tracked and non-ignored files, optionally every blob reachable from HEAD.
This is a high-confidence release check, not a general vulnerability scanner.
"""

import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "credential token": re.compile(
        rb"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
        rb"github_pat_[A-Za-z0-9_]{30,}|nvapi-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}\b|"
        rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
    ),
    "private key": re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    "private cluster path": re.compile(rb"/(?:lustre|gpfs)/|/(?:home|Users)/weihua/"),
}


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def forbidden_name(name):
    path = Path(name)
    return (
        path.name in {".env", "auth.json", "credentials.json", "id_rsa", "id_ed25519",
                      ".DS_Store", ".netrc", ".pypirc", ".npmrc", ".git-credentials"}
        or path.name.startswith(".env.") and path.name != ".env.example"
        or path.suffix in {".pem", ".key"}
        or any(part in {".codex", ".aws", ".ssh"} for part in path.parts)
    )


def content_findings(data):
    return sorted({(data.count(b"\n", 0, match.start()) + 1, rule)
                   for rule, pattern in RULES.items() for match in pattern.finditer(data)})


def inspect(name, data):
    findings = [(name, 0, "credential/private file")] if forbidden_name(name) else []
    findings.extend((name, line, rule) for line, rule in content_findings(data))
    return findings


def scan_checkout(root):
    paths = set(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"))
    findings, count = [], 0
    for raw in sorted(paths - {b""}):
        name = raw.decode("utf-8")
        path = root / name
        if path.is_symlink():
            # Never follow symlinks to a user's secrets outside the release.
            findings.append((name, 0, "review symlink before release"))
        elif path.is_file():
            count += 1
            findings.extend(inspect(name, path.read_bytes()))
    return count, findings


def scan_history(root):
    objects = {}
    findings = set()
    commits = git(root, "rev-list", "HEAD").decode().splitlines()
    for commit in commits:
        for entry in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not entry:
                continue
            meta, name = entry.split(b"\t", 1)
            _, kind, oid = meta.split()
            name = name.decode("utf-8")
            if forbidden_name(name):
                findings.add((f"{commit[:12]}:{name}", 0, "credential/private file"))
            if kind == b"blob":
                objects.setdefault(oid, f"{commit[:12]}:{name}")
    # Keep blob contents out of command output, even when a match is found.
    for oid, label in objects.items():
        for line, rule in content_findings(git(root, "cat-file", "blob", oid.decode())):
            findings.add((label, line, rule))
    return len(objects), sorted(findings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also scan all blobs reachable from HEAD")
    args = parser.parse_args(argv)
    count, findings = scan_checkout(ROOT)
    print(f"Checked {count} release files.")
    if args.history:
        count, history = scan_history(ROOT)
        findings.extend(history)
        print(f"Checked {count} unique history blobs reachable from HEAD.")
    for name, line, rule in findings:
        print(f"{name}:{line}: {rule}")
    print(f"Findings: {len(findings)}. Matched values are never printed.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
