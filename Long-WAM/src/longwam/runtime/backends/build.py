# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/backends/build.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Build a device-specific copy of bundled, unpatched FourOverSix source."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings


def _files(source):
    return sorted(
        p.relative_to(source)
        for p in source.rglob("*")
        if p.is_file()
        and p.name not in {"UPSTREAM.json", ".longwam-build.json"}
        and p.suffix not in {".pyc", ".pyo", ".so", ".pyd", ".o", ".a"}
        and not any(
            part in {".git", "__pycache__", "build", "dist"} or part.endswith(".egg-info")
            for part in p.relative_to(source).parts
        )
    )


def _fingerprint(source, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(
            name.as_posix().encode() + b"\0" + hashlib.sha256((source / name).read_bytes()).digest()
        )
    return digest.hexdigest()


def _patch(source, patch, *, check=False):
    subprocess.run(
        ["git", "apply", *(["--check"] if check else []), str(patch)], cwd=source, check=True
    )


def _build_environment(torch, hardware):
    """Identify the active ABI/toolchain without prescribing dependency versions."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    if cuda_home and (Path(cuda_home) / "bin/nvcc").is_file():
        nvcc = str(Path(cuda_home) / "bin/nvcc")
    nvcc_version = None
    if nvcc:
        nvcc_version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_home": cuda_home,
        "nvcc": nvcc,
        "nvcc_version": nvcc_version,
        "cc": os.environ.get("CC"),
        "cxx": os.environ.get("CXX"),
        "cuda_archs": "110" if hardware == "thor" else "120",
    }


def _matches(target, identity):
    try:
        previous = json.loads((target / ".longwam-build.json").read_text())
        return previous.get("identity") == identity and _fingerprint(
            target, _files(target)
        ) == previous.get("patched_sha256")
    except (OSError, ValueError):
        return False


def build_extension(hardware, *, build_dir=None, check_only=False):
    from longwam.paths import repository_root
    from . import CAPABILITIES

    if hardware not in CAPABILITIES:
        raise ValueError(f"Choose hardware from {tuple(CAPABILITIES)}")
    root = repository_root()
    source = root / "third_party/fouroversix"
    upstream = json.loads((source / "UPSTREAM.json").read_text())
    names = _files(source)
    source_sha256 = _fingerprint(source, names)
    if source_sha256 != upstream["source_sha256"]:
        warnings.warn(
            "Bundled FourOverSix source differs from the recorded upstream snapshot; "
            "continuing with patch compatibility checks and the local source.",
            stacklevel=2,
        )
    patch = Path(__file__).parent / "patches" / f"{hardware}.patch"
    identity = {
        "hardware": hardware,
        "upstream": upstream,
        "source_sha256": source_sha256,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    }
    # Check in an isolated temporary tree; the pristine source is never patched.
    if check_only:
        print(json.dumps(identity, indent=2))
        with tempfile.TemporaryDirectory(prefix="longwam-native-check-") as directory:
            target = Path(directory) / "fouroversix"
            shutil.copytree(source, target)
            _patch(target, patch, check=True)
        return identity
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != CAPABILITIES[hardware]
    ):
        raise ValueError(
            f"Building {hardware} requires a visible device with CC {CAPABILITIES[hardware]}"
        )
    identity["environment"] = _build_environment(torch, hardware)
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache_root = Path(os.environ.get("LONGWAM_BUILD_ROOT", Path.home() / ".cache/longwam"))
    base = (cache_root / hardware / f"fouroversix-{cache_key}").expanduser().resolve()
    target = Path(build_dir).expanduser().resolve() if build_dir else base
    if target.is_relative_to(root):
        raise ValueError("Choose a build directory outside the Long-WAM checkout")
    if not build_dir:
        suffix = 0
        while target.exists() and not _matches(target, identity):
            suffix += 1
            target = base.with_name(f"{base.name}-{suffix}")
        if suffix:
            warnings.warn(f"Preserving changed build copies; using {target}", stacklevel=2)
    stamp = target / ".longwam-build.json"
    if target.exists():
        if not stamp.is_file():
            raise FileExistsError(f"Refusing to replace an existing directory: {target}")
        if not _matches(target, identity):
            raise ValueError(
                "Existing build has different source, patch or local changes; choose a new --build-dir"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Do not leave a partial build directory if patching fails.
        with tempfile.TemporaryDirectory(prefix=".longwam-build-", dir=target.parent) as directory:
            staged = Path(directory) / "fouroversix"
            shutil.copytree(source, staged)
            _patch(staged, patch, check=True)
            _patch(staged, patch)
            (staged / stamp.name).write_text(
                json.dumps(
                    {"identity": identity, "patched_sha256": _fingerprint(staged, _files(staged))},
                    indent=2,
                )
                + "\n"
            )
            staged.rename(target)
    print(json.dumps({**identity, "build_dir": str(target)}, indent=2))
    env = dict(os.environ, FORCE_BUILD="1", CUDA_ARCHS="110" if hardware == "thor" else "120")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(target),
        ],
        env=env,
        check=True,
    )
    return identity
