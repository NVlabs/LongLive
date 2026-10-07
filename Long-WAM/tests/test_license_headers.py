# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM release tooling.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""Keep release attribution complete without changing parser-sensitive files."""

import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("license_check", ROOT / "scripts/check_license_headers.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_all_release_files_have_headers_or_sidecars():
    failures = {str(p.relative_to(ROOT)): checker.header_errors(p)
                for p in checker.release_files() if p.is_file() and checker.header_errors(p)}
    assert failures == {}


def test_no_comments_were_inserted_in_json_configs():
    for path in (ROOT / "configs").rglob("*.json"):
        json.loads(path.read_text())
        assert Path(str(path) + ".license").is_file()


def test_license_texts_keep_original_owners():
    assert "Apache License" in (ROOT / "LICENSE").read_text()
    assert "The FastWAM Authors" in (ROOT / "licenses/FastWAM-MIT.txt").read_text()
    assert "Meta Platforms" in (ROOT / "licenses/PyTorch3D-BSD-3-Clause.txt").read_text()
    assert "Jack Cook" in (ROOT / "stage1/third_party/LongLive/fouroversix/LICENSE.md").read_text()


def test_agent_prompts_keep_frontmatter_and_payload_free_of_headers():
    prompts = ROOT / "src/longwam/benchmarks/robocasa/agentic/skill"
    assert (prompts / "SKILL_hybrid_decompose.md").read_text().startswith("---\n")
    for path in prompts.rglob("*.md"):
        assert checker.needs_sidecar(path)
        assert Path(str(path) + ".license").is_file()
        assert "Long-WAM file attribution;" not in path.read_text()


def test_plain_python_without_header_is_rejected(tmp_path):
    path = tmp_path / "example.py"
    path.write_text("x = 1\n")
    assert "missing SPDX license" in checker.header_errors(path)


def test_missing_sidecar_is_rejected(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}\n")
    assert checker.header_errors(path) == ["missing .license sidecar"]


def test_binary_gif_uses_attribution_sidecar(tmp_path):
    path = tmp_path / "demo.gif"
    path.write_bytes(b"GIF89a\xff")
    assert checker.needs_sidecar(path)
    assert checker.header_errors(path) == ["missing .license sidecar"]
    Path(str(path) + ".license").write_text(
        "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.\n"
        "SPDX-License-Identifier: Apache-2.0\n"
        "Provenance: Modified Long-WAM project-page demo.\n"
        "Source: https://nvlabs.github.io/LongLive/Long-WAM/\n"
    )
    assert checker.header_errors(path) == []


def test_headers_must_not_break_shell_shebang(tmp_path):
    path = tmp_path / "run.sh"
    path.write_text("# header\n#!/bin/bash\necho ok\n")
    assert "shell shebang must remain the first line" in checker.header_errors(path)


def test_monorepo_attribution_does_not_inspect_sibling_projects(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    project = tmp_path / "Long-WAM"
    project.mkdir()
    (tmp_path / "README.md").write_text("Other projects have separate attribution rules")
    tracked = project / "tracked.py"
    untracked = project / "new.py"
    tracked.write_text("# Synthetic fixture")
    untracked.write_text("# Synthetic fixture")
    subprocess.run(["git", "add", "README.md", "Long-WAM/tracked.py"], cwd=tmp_path, check=True)
    assert checker.release_files(project) == sorted([tracked, untracked])
