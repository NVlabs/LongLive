#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM release tooling.
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

"""Check release-file attribution without importing ML libraries or using GPUs."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SIDECAR_SUFFIXES = {".json", ".patch", ".sha256", ".txt", ".gif", ".png"}
LICENSE_IDS = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0"}


def is_license_text(path):
    return path.name.upper().startswith(("LICENSE", "COPYING", "LICENCE")) or (
        path.parent.name == "licenses" and path.suffix == ".txt"
    )


def release_files(root=ROOT):
    """Include new, non-ignored files as well as tracked files in a checkout."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    return sorted({root / p for p in result.stdout.decode().split("\0") if p})


def needs_sidecar(path):
    # These Markdown files are literal agent inputs, including YAML frontmatter.
    prompt = (
        path.suffix == ".md" and "skill" in path.parts
        and any(name in path.parts for name in ("robocasa_gpt6", "agentic"))
    )
    return path.suffix in SIDECAR_SUFFIXES or prompt


def header_errors(path):
    if is_license_text(path) or path.suffix == ".license":
        return []
    target = Path(str(path) + ".license") if needs_sidecar(path) else path
    if not target.is_file():
        return ["missing .license sidecar"]
    text = target.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:65])
    errors = []
    if "SPDX-FileCopyrightText:" not in head:
        errors.append("missing copyright header")
    matches = re.findall(r"(?m)^(?:# |// )?SPDX-License-Identifier:[ \t]*([^\r\n]+)$", head)
    if not matches:
        errors.append("missing SPDX license")
    for expression in matches:
        ids = re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]*", expression)
        unknown = set(ids) - LICENSE_IDS - {"AND", "OR"}
        if unknown:
            errors.append(f"unsupported license expression: {expression}")
    if "Provenance:" not in head:
        errors.append("missing original/copied/modified provenance")
    if "Source:" not in head and "Original Long-WAM" not in head:
        errors.append("third-party file is missing a source reference")
    if path.suffix == ".sh" and not path.read_bytes().startswith(b"#!"):
        errors.append("shell shebang must remain the first line")
    return errors


def main():
    files = release_files()
    failures = []
    covered = 0
    for path in files:
        if not path.is_file():
            continue  # Tracked deletions are not part of the release payload.
        if path.suffix == ".license":
            if not path.with_suffix("").is_file():
                failures.append((path, ["orphan .license sidecar"]))
            continue
        if not is_license_text(path):
            covered += 1
        errors = header_errors(path)
        if errors:
            failures.append((path, errors))
    for path, errors in failures:
        print(f"{path.relative_to(ROOT)}: {'; '.join(errors)}")
    print(f"Attribution: {covered} covered files; {len(failures)} failures.")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
