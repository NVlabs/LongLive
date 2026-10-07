# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/runtime/test_native_build.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Exercise real device patches without compiling or installing dependencies."""

from pathlib import Path
import subprocess
import shutil

import pytest
import torch

from longwam.runtime.backends import build
from longwam.runtime.backends import CAPABILITIES
from longwam.paths import repository_root


@pytest.mark.parametrize("hardware", ["rtx5090", "spark", "thor"])
def test_device_build_copy_is_pristine_repeatable_and_rejects_changes(
    hardware, tmp_path, monkeypatch
):
    source = repository_root() / "third_party/fouroversix"
    names = build._files(source)
    before = build._fingerprint(source, names)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: CAPABILITIES[hardware])
    actual_run = subprocess.run
    installs = []

    def run(command, **kwargs):
        if command[:2] == ["git", "apply"]:
            return actual_run(command, **kwargs)
        assert command[1:4] == ["-m", "pip", "install"]
        installs.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build.subprocess, "run", run)
    target = tmp_path / hardware
    first = build.build_extension(hardware, build_dir=target)
    second = build.build_extension(hardware, build_dir=target)
    assert first == second
    assert len(installs) == 2
    assert build._fingerprint(source, names) == before
    assert (target / ".longwam-build.json").exists()
    with (target / "setup.py").open("a") as stream:
        stream.write("\n# local change\n")
    with pytest.raises(ValueError, match="different source, patch or local changes"):
        build.build_extension(hardware, build_dir=target)
    assert len(installs) == 2


def test_auto_cache_tracks_source_patch_environment_and_preserves_local_edits(
    tmp_path, monkeypatch
):
    import longwam.paths

    root = tmp_path / "checkout"
    source = root / "third_party/fouroversix"
    shutil.copytree(repository_root() / "third_party/fouroversix", source)
    monkeypatch.setattr(longwam.paths, "repository_root", lambda: root)
    monkeypatch.setenv("LONGWAM_BUILD_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: CAPABILITIES["rtx5090"])
    environment = {"torch": "test-abi"}
    monkeypatch.setattr(build, "_build_environment", lambda *args: dict(environment))
    actual_run = subprocess.run
    installs = []

    def run(command, **kwargs):
        if command[:2] == ["git", "apply"]:
            return actual_run(command, **kwargs)
        assert command[1:4] == ["-m", "pip", "install"]
        target = Path(command[-1])
        installs.append(target)
        # Editable builds can place generated binaries and metadata in the tree.
        (target / "src/fouroversix/_C.test.so").write_bytes(b"generated")
        (target / "fouroversix.egg-info").mkdir(exist_ok=True)
        (target / "fouroversix.egg-info/PKG-INFO").write_text("generated")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build.subprocess, "run", run)
    original = build.build_extension("rtx5090")
    build.build_extension("rtx5090")
    assert installs[0] == installs[1]

    with (source / "README.md").open("a") as stream:
        stream.write("\nLocal documentation update.\n")
    with pytest.warns(UserWarning, match="continuing with patch compatibility"):
        changed_source = build.build_extension("rtx5090")
    assert changed_source["source_sha256"] != original["source_sha256"]
    assert installs[2] != installs[1]

    environment["torch"] = "different-abi"
    with pytest.warns(UserWarning):
        build.build_extension("rtx5090")
    assert installs[3] != installs[2]
    modified = installs[3] / "setup.py"
    modified.write_text(modified.read_text() + "\n# user modification\n")
    with pytest.warns(UserWarning) as messages:
        build.build_extension("rtx5090")
    assert any("Preserving changed build copies" in str(m.message) for m in messages)
    assert installs[4] != installs[3]
    assert "# user modification" in modified.read_text()
    with pytest.warns(UserWarning):
        build.build_extension("rtx5090")
    assert installs[5] == installs[4]

    patch = Path(build.__file__).parent / "patches/rtx5090.patch"
    original_read = Path.read_bytes

    def read_bytes(path, *args, **kwargs):
        content = original_read(path, *args, **kwargs)
        return content + b"\n" if path == patch else content

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.warns(UserWarning):
        changed_patch = build.build_extension("rtx5090")
    assert changed_patch["patch_sha256"] != original["patch_sha256"]
    assert installs[6] != installs[5]
