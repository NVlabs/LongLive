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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/cli.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Small command router; model/simulator imports happen only in their workers."""

import argparse
import os
import subprocess
import sys

from .paths import repository_root

BENCHMARKS = ("libero", "robotwin2", "domino", "robocasa_gr1", "robocasa365")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="longwam")
    parser.add_argument("command", choices=("train", "eval", "infer", "list", "build", "deploy"))
    parser.add_argument("target", nargs="?", help="benchmark name, or hardware for build")
    parser.add_argument("--config", help="evaluation or robot deployment YAML")
    parser.add_argument(
        "--check-only", action="store_true", help="validate native patch without building"
    )
    parser.add_argument("--build-dir", help="native build directory outside the checkout")
    args, overrides = parser.parse_known_args(argv)
    if args.command == "list":
        print("\n".join(BENCHMARKS))
        return 0
    if args.command == "build":
        if args.config or overrides:
            parser.error("build accepts a hardware target, --check-only and --build-dir")
        from .runtime.backends.build import build_extension

        build_extension(args.target, build_dir=args.build_dir, check_only=args.check_only)
        return 0
    if args.check_only or args.build_dir:
        parser.error("--check-only/--build-dir are only used by build")
    if args.command == "deploy":
        if args.target not in {"yam", "franka", "g1"}:
            parser.error("choose a robot from yam, franka or g1")
        from .runtime.deploy import main as deploy_main

        return deploy_main([
            "--robot", args.target, "--config",
            str(args.config or repository_root() / f"configs/deploy/{args.target}.yaml"),
            *overrides,
        ])
    if args.target not in BENCHMARKS:
        parser.error(f"choose a benchmark from {BENCHMARKS}")
    if args.command == "train" and args.config:
        parser.error("train uses benchmark recipes and Hydra key=value overrides")
    root = repository_root()
    if args.command == "train":
        command = ["bash", str(root / f"scripts/{args.target}/train.sh"), *overrides]
    else:
        benchmark = args.target
        command = [
            sys.executable,
            "-m",
            "longwam.evaluation",
            "--config",
            str(args.config or root / f"configs/eval/{benchmark}.yaml"),
        ]
        if args.command == "infer":
            command.append("--serve-only")
        command.extend(overrides)
    environment = os.environ.copy()
    environment.setdefault("PYTHON", sys.executable)
    return subprocess.call(command, cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
