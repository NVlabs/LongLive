# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/run_rollout.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""One hybrid_decompose v1 episode. Prefer scripts/robocasa365/eval.sh --agentic."""

import argparse
import json
import os
from pathlib import Path
import signal
import traceback

from longwam.agentic import AgenticConfig, CodexAgent
from longwam.agentic.config import DEFAULT_MODEL, DEFAULT_EFFORT
from . import contract as C
from longwam.agentic._vendor.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=C.METHODS, default="hybrid_decompose")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default=os.environ.get("ROLLOUT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--effort", default=os.environ.get("ROLLOUT_EFFORT", DEFAULT_EFFORT))
    parser.add_argument(
        "--max-total-tokens", type=int, default=os.environ.get("CODEX_MAX_TOTAL_TOKENS")
    )
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--codex-timeout", type=float, default=900)
    parser.add_argument("--student-identity-json", type=Path)
    parser.add_argument("--allow-model-upload", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_upload:
        parser.error("Explicit --allow-model-upload is required for simulator RGB/state")
    try:
        settings = AgenticConfig(
            model=args.model,
            effort=args.effort,
            codex=args.codex,
            timeout_seconds=args.codex_timeout,
            max_total_tokens=args.max_total_tokens,
            allow_model_upload=args.allow_model_upload,
        ).validate()
    except ValueError as error:
        parser.error(str(error))
    if args.episode_index < 0 or args.max_decisions < 0 or args.codex_timeout <= 0:
        parser.error("Invalid episode index, decision limit or timeout")
    from .session import RoboCasaSession
    from .adapter import RoboCasaAdapter, RoboCasaEpisode
    from .tools import RoboCasaTools

    def terminate(signum, frame):
        raise KeyboardInterrupt("Stopping this rollout")

    signal.signal(signal.SIGTERM, terminate)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = (
        json.loads(args.student_identity_json.read_text()) if args.student_identity_json else {}
    )
    session = worker = rollout = None
    try:
        session = RoboCasaSession(
            task=args.task,
            split=args.split,
            env_seed=args.env_seed,
            policy_socket=args.policy_socket,
            use_policy=True,
        )
        worker = CodexAgent(
            output / "codex_workspace", adapter=RoboCasaAdapter(args.method), config=settings
        )
        rollout = RoboCasaTools(
            output,
            session,
            method=args.method,
            episode_index=args.episode_index,
            max_decisions=args.max_decisions,
            prompt_sha256=worker.prompt_sha256,
            student_identity=identity,
        )
        worker.run(RoboCasaEpisode(rollout))
    except BaseException:
        write_json(output / "failure.json", dict(error=traceback.format_exc(), complete=False))
        if rollout is not None and rollout.phase != "done" and session.episode_id is not None:
            rollout.finish("controller_error")
        raise
    finally:
        if worker is not None:
            worker.close()
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
