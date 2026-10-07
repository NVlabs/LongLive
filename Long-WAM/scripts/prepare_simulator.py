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

"""Apply version-checked benchmark patches to a user-specified checkout.

Does not clone, delete, reset, install dependencies, download assets or reserve GPUs.
"""

import argparse
from pathlib import Path
import subprocess

from longwam.paths import repository_root

COMMITS = {
    "robotwin2": "bf44be51cf5717a5595ce59447f2cf5263d2aa95",
    "domino": "93d485a34a6e710bb4c895d83e2bb0520a9f169e",
    "robocasa365": "b4684e6ee37d377cc392e98302a6b916d588b415",
}


def prepare(benchmark, checkout, *, arm=False, check_only=False):
    checkout = Path(checkout).expanduser().resolve(strict=True)
    root = repository_root()
    observed = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != COMMITS[benchmark]:
        raise ValueError(f"{benchmark} requires {COMMITS[benchmark]}, found {observed}")
    if benchmark == "robocasa365":
        patches = [root / "patches/robocasa365.patch"]
    elif benchmark == "robotwin2":
        patches = [root / "src/longwam/benchmarks/robotwin/patches/robotwin_eval_policy.patch"]
    else:
        base = root / "src/longwam/benchmarks/domino/patches"
        patches = [base / "domino_eval_policy.patch", base / "domino_lerobot_compat.patch"]
        if arm:
            patches.append(base / "domino_arm_runtime.patch")
    # Check every patch before applying any: a failed check must not partly edit a checkout.
    pending = []
    for patch in patches:
        command = ["git", "-C", str(checkout), "apply"]
        already = (
            subprocess.run(
                command + ["--reverse", "--check", str(patch)], capture_output=True
            ).returncode
            == 0
        )
        if not already:
            subprocess.run(command + ["--check", str(patch)], check=True)
            pending.append(patch)
    for patch in pending:
        print(f"{'Would apply' if check_only else 'Applying'} {patch.name}")
        if not check_only:
            subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=COMMITS)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    prepare(args.benchmark, args.checkout, arm=args.arm, check_only=args.check_only)
