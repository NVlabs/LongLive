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

"""Agentic + Long-WAM Target50 evaluation; no API calls during planning."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def agentic_settings(cfg):
    """Accept the legacy config namespace without duplicating runtime logic."""
    from longwam.agentic import AgenticConfig

    if cfg.get("agentic") is not None and cfg.get("gpt6") is not None:
        raise ValueError("Use agentic or legacy gpt6 settings, not both")
    settings = cfg.get("agentic", cfg.get("gpt6"))
    if settings is None:
        raise ValueError("Missing agentic settings")
    return AgenticConfig(**{k: v for k, v in settings.items() if k != "max_decisions"})


def validate_agentic_settings(cfg):
    settings = agentic_settings(cfg).validate()
    section = cfg.get("agentic", cfg.get("gpt6"))
    limit = section.max_decisions
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("agentic.max_decisions must be a non-negative integer")
    return settings


# Legacy Python entry point.
validate_gpt_settings = validate_agentic_settings


def run(cfg):
    from longwam.benchmarks.robocasa import rollout as r

    settings = validate_agentic_settings(cfg)
    section = cfg.get("agentic", cfg.get("gpt6"))
    registry, _, _, _ = r._load_robocasa_runtime()
    tasks = r.resolve_tasks(registry, str(cfg.task_set), list(cfg.tasks) or None)
    output = Path(cfg.output_dir)
    result_path = output / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    identity = output / "student.json"
    identity.write_text(
        json.dumps(
            {
                "checkpoint": cfg.checkpoint,
                "model_config_sha256": hashlib.sha256(
                    Path(cfg.model_config).read_bytes()
                ).hexdigest(),
                "stats_sha256": hashlib.sha256(Path(cfg.stats).read_bytes()).hexdigest(),
                "policy_seed": cfg.policy_seed,
                "action_nfe": cfg.num_inference_steps,
            },
            indent=2,
        )
    )
    env = os.environ.copy()
    result = dict(
        benchmark="robocasa365",
        experiment="hybrid_decompose",
        agentic=dict(backend=settings.backend, model=settings.model, effort=settings.effort),
        complete=False,
        expected_episodes=len(tasks) * int(cfg.episodes),
        episodes=[],
    )
    r.atomic_write_json(result_path, result)
    for task in tasks:
        for index in range(int(cfg.episodes)):
            destination = output / task / f"episode_{index:03d}"
            command = [
                sys.executable,
                "-m",
                "longwam.benchmarks.robocasa.agentic.run_rollout",
                "--method",
                "hybrid_decompose",
                "--task",
                task,
                "--episode-index",
                str(index),
                "--env-seed",
                str(cfg.env_seed),
                "--split",
                str(cfg.split),
                "--policy-socket",
                str(cfg.socket),
                "--output",
                str(destination),
                "--model",
                settings.model,
                "--effort",
                settings.effort,
                "--max-total-tokens",
                str(settings.max_total_tokens),
                "--codex",
                str(settings.codex),
                "--codex-timeout",
                str(settings.timeout_seconds),
                "--max-decisions",
                str(section.max_decisions),
                "--student-identity-json",
                str(identity),
                "--allow-model-upload",
            ]
            subprocess.run(command, env=env, check=True)
            episode = json.loads((destination / "result.json").read_text())
            if not episode.get("complete") or episode.get("reason") != "terminal":
                raise RuntimeError(f"Incomplete episode {task}/{index}; not a benchmark result")
            result["episodes"].append(episode)
            result["successes"] = sum(bool(ep["success"]) for ep in result["episodes"])
            result["success_rate"] = result["successes"] / len(result["episodes"])
            r.atomic_write_json(result_path, result)
    result["complete"] = len(result["episodes"]) == result["expected_episodes"]
    r.atomic_write_json(result_path, result)
