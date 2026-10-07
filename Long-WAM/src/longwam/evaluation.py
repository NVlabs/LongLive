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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/evaluation.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""YAML-driven local evaluation; no Slurm, personal paths or historical receipts."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .paths import repository_root, setup_model_paths

BENCHMARKS = {"libero", "robotwin2", "domino", "robocasa_gr1", "robocasa365"}


def read_settings(path, overrides=()):
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(
        {
            "execution": "sync",
            "execute_steps": None,
            "trigger_stride": None,
            "streaming_vae": False,
            "hardware": "reference",
            "optimization": {},
        },
        OmegaConf.load(path),
    )
    overrides = list(overrides)
    if cfg.get("experiment") in {"agentic", "gpt6"}:
        if cfg.benchmark != "robocasa365":
            raise ValueError("Only the RoboCasa365 Agentic benchmark adapter is currently released")
        if "agentic" in cfg and "gpt6" in cfg:
            raise ValueError("Use agentic or legacy gpt6 settings, not both")
        namespace = "agentic" if "agentic" in cfg else "gpt6"
        normalized = []
        seen = set()
        for override in overrides:
            key, separator, value = override.partition("=")
            if key.startswith(("gpt6.", "agentic.")):
                key = namespace + "." + key.split(".", 1)[1]
                if key in seen:
                    raise ValueError(f"Duplicate Agentic override: {key}")
                seen.add(key)
            normalized.append(key + separator + value)
        overrides = normalized
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.optimization, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if "agentic" in cfg or "gpt6" in cfg:
        if cfg.benchmark != "robocasa365" or cfg.get("experiment") not in {"agentic", "gpt6"}:
            raise ValueError("Agentic evaluation currently requires benchmark=robocasa365 and experiment=agentic")
    if cfg.benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {cfg.benchmark}")
    if cfg.get("inference_mode", "idm") not in {"idm", "codenoise"}:
        raise ValueError("inference_mode must be idm or codenoise")
    for name in ("episodes", "num_inference_steps", "n_envs", "max_control_steps", "replan_steps"):
        if name in cfg and (isinstance(cfg[name], bool) or int(cfg[name]) <= 0):
            raise ValueError(f"{name} must be positive")
    if cfg.benchmark in {"libero", "robotwin2", "domino"}:
        from .benchmarks.evaluation_plan import native_plan

        native_plan(cfg)
    from .runtime.settings import normalize_settings

    return normalize_settings(cfg)


def validate_inputs(cfg):
    for name in ("checkpoint", "model_config", "stats"):
        path = Path(cfg[name]).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{name} must be a file: {path}")
        cfg[name] = str(path)
    if cfg.benchmark == "robocasa_gr1":
        cfg.text_cache = str(Path(cfg.text_cache).expanduser().resolve(strict=True))
    if "simulator_root" in cfg:
        cfg.simulator_root = str(Path(cfg.simulator_root).expanduser().resolve(strict=True))
    cfg.output_dir = str(Path(cfg.output_dir).expanduser().resolve())


def serve(cfg):
    from .inference import load_benchmark_policy

    if cfg.benchmark == "robocasa_gr1":
        from .benchmarks.robocasa_gr1.policy import PolicyRequestDispatcher, UnixPolicyServer
    elif cfg.benchmark == "robocasa365":
        from .benchmarks.robocasa.policy import PolicyRequestDispatcher, UnixPolicyServer
    else:
        raise ValueError(
            "Use eval.sh for the native LIBERO/RoboTwin/Domino policy adapters; "
            "the standalone socket server supports RoboCasa GR1 and RoboCasa365."
        )
    policy = load_benchmark_policy(cfg)
    server = UnixPolicyServer(Path(cfg.socket), PolicyRequestDispatcher(policy))

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def run_gr1(cfg):
    from .benchmarks.robocasa_gr1 import rollout as r
    from .inference import load_run_config
    from .benchmarks.robocasa_gr1.temporal_contract import TemporalSpec

    output = Path(cfg.output_dir) / "results.json"
    if output.exists():
        raise FileExistsError(f"Choose a new output_dir; results already exist: {output}")
    temporal = TemporalSpec.from_past_obs_size(
        int(load_run_config(cfg.model_config).data.train.past_obs_size)
    )
    tasks = r.resolve_tasks(list(cfg.tasks))
    results = {
        "benchmark": cfg.benchmark,
        "complete": False,
        "episodes": [],
        "temporal_spec": temporal.to_dict(),
        "protocol": {
            "episodes_per_task": int(cfg.episodes),
            "n_envs": int(cfg.n_envs),
            "max_control_steps": int(cfg.max_control_steps),
            "reset_seed": None,
            "action_horizon": 16,
            "execute_actions": 16,
        },
    }
    r.atomic_write_json(output, results)
    with r.UnixPolicyClient(Path(cfg.socket)) as client:
        for task in tasks:
            r.evaluate_task(
                task=task,
                client=client,
                env_factory=r._load_env_factory(),
                episode_count=int(cfg.episodes),
                n_envs=int(cfg.n_envs),
                max_control_steps=int(cfg.max_control_steps),
                on_group_complete=lambda group: r._append_group(output, results, group),
            )
    wanted = {(task.dataset_suffix, i) for task in tasks for i in range(int(cfg.episodes))}
    actual = {(str(ep["task"]), int(ep["episode_index"])) for ep in results["episodes"]}
    if wanted != actual or len(actual) != len(results["episodes"]):
        raise RuntimeError("Incomplete or duplicate GR1 episode inventory")
    results.update(summary=r.summarize(results["episodes"]), complete=True)
    r.atomic_write_json(output, results)


def run_robocasa(cfg):
    from .benchmarks.robocasa import rollout as r
    import numpy as np

    output = Path(cfg.output_dir) / "results.json"
    if output.exists():
        raise FileExistsError(f"Choose a new output_dir; results already exist: {output}")
    registry, horizon_for, make_env, convert_action = r._load_robocasa_runtime()
    tasks = r.resolve_tasks(registry, str(cfg.task_set), list(cfg.tasks) or None)
    results = {
        "benchmark": cfg.benchmark,
        "complete": False,
        "episodes": [],
        "protocol": {
            "task_set": cfg.task_set,
            "split": cfg.split,
            "env_seed": cfg.env_seed,
            "episodes_per_task": cfg.episodes,
            "replan_steps": cfg.replan_steps,
        },
    }
    r.atomic_write_json(output, results)
    np.random.seed(int(cfg.env_seed))
    client = r.UnixPolicyClient(
        cfg.socket, response_timeout_seconds=900, startup_timeout_seconds=300
    )
    try:
        client.connect()
        client.ping()
        for task in tasks:
            env = make_env(f"robocasa/{task}", split=str(cfg.split), seed=int(cfg.env_seed))
            try:
                for index in range(int(cfg.episodes)):
                    episode = r.run_episode(
                        env=env,
                        client=client,
                        task=task,
                        episode_index=index,
                        env_seed=int(cfg.env_seed),
                        split=str(cfg.split),
                        horizon=int(horizon_for(task)),
                        replan_steps=int(cfg.replan_steps),
                        convert_action=convert_action,
                        video_recorder=None,
                        diagnostic_trace_path=None,
                    )
                    results["episodes"].append(episode)
                    results["summary"] = r.summarize_results(results["episodes"], registry)
                    r.atomic_write_json(output, results)
            finally:
                env.close()
        wanted = {(task, i) for task in tasks for i in range(int(cfg.episodes))}
        actual = {(str(ep["task"]), int(ep["episode_index"])) for ep in results["episodes"]}
        if wanted != actual or len(actual) != len(results["episodes"]):
            raise RuntimeError("Incomplete or duplicate RoboCasa episode inventory")
        results["complete"] = True
        r.atomic_write_json(output, results)
    finally:
        client.close()


def native_config(settings):
    """Join simulator defaults to the checkpoint's exact architecture/data schema."""
    from omegaconf import OmegaConf
    from .inference import load_run_config, model_config_for_inference

    domain = "domino" if settings.benchmark == "domino" else "robotwin"
    run = load_run_config(settings.model_config, domain=domain)
    evaluation = OmegaConf.create(OmegaConf.to_container(settings.policy_options, resolve=True))
    evaluation.output_dir = str(settings.output_dir)
    evaluation.num_inference_steps = int(settings.num_inference_steps)
    horizon = settings.get("action_horizon")
    evaluation.action_horizon = int(run.data.train.action_chunk if horizon is None else horizon)
    evaluation.replan_steps = int(settings.replan_steps)
    evaluation.dataset_stats_path = str(settings.stats)
    evaluation.device = str(settings.device)
    if not 1 <= evaluation.replan_steps <= evaluation.action_horizon:
        raise ValueError("replan_steps must be between 1 and the evaluation action horizon")
    if int(run.model.get("longwam_past_obs_size", 0)) != int(
        run.data.train.get("past_obs_size", 0)
    ):
        raise ValueError("Checkpoint model and data history lengths disagree")
    run.model = model_config_for_inference(run)
    if settings.benchmark in {"libero", "robotwin2"}:
        mode = "codenoise" if run.model.get("longwam_joint_denoise", False) else "idm"
        if str(settings.inference_mode) != mode:
            raise ValueError(
                f"Requested {settings.inference_mode}, but checkpoint config describes {mode}. "
                "Supply the matching checkpoint and resolved config; the attention regime "
                "must not be changed silently at evaluation."
            )
        if int(run.model.get("longwam_num_imagine_frames", 0)) <= 0:
            raise ValueError("IDM and CodeDenoise require future-video latents")
    run.ckpt = str(settings.checkpoint)
    run.seed = int(settings.seed)
    run.gpu_id = int(settings.gpu_id)
    run.checkpoint_strict = True
    run.EVALUATION = evaluation
    if settings.benchmark == "libero":
        evaluation.task_suite_name = str(settings.suite)
        evaluation.task_id = settings.task_id
        evaluation.num_trials = int(settings.episodes)
        evaluation.record_replay = False
    else:
        evaluation.robotwin_root = str(settings.simulator_root)
        evaluation.task_name = str(settings.task)
        evaluation.task_config = str(settings.setting)
        evaluation.eval_num_episodes = int(settings.episodes)
    return run


def run_native_cell(cfg):
    from omegaconf import OmegaConf

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    runtime = native_config(cfg)
    from .benchmarks.robotwin.eval_robotwin_single import (
        _ensure_policy_symlink,
        _append_override,
        _result_filename_for_task_config,
    )

    root = Path(cfg.simulator_root)
    policy = repository_root() / "src/longwam/benchmarks/robotwin/longwam_policy"
    _ensure_policy_symlink(root, policy)
    result_name = _result_filename_for_task_config(str(cfg.setting))
    if (Path(cfg.output_dir) / result_name).exists():
        raise FileExistsError("Results already exist; choose a new output_dir")
    runtime_settings_path = Path(cfg.output_dir) / "runtime_settings.yaml"
    OmegaConf.save(cfg, runtime_settings_path, resolve=True)
    overrides = []
    values = dict(
        runtime_config=str(runtime_settings_path),
        task_name=cfg.task,
        task_config=cfg.setting,
        ckpt_setting=cfg.checkpoint,
        policy_name="longwam_policy",
        instruction_type="unseen",
        seed=cfg.seed,
        eval_num_episodes=cfg.episodes,
        eval_video_log=False,
        eval_output_dir=cfg.output_dir,
        eval_result_filename=result_name,
        skip_get_obs_within_replan=False,
    )
    for key, value in values.items():
        _append_override(overrides, key, value)
    if cfg.benchmark == "domino":
        os.environ.setdefault("DOMINO_RT_DENOISER", "oidn")
    subprocess.run(
        [
            sys.executable,
            "-u",
            "script/eval_policy.py",
            "--config",
            "policy/longwam_policy/deploy_policy.yml",
            "--overrides",
            *overrides,
        ],
        cwd=root,
        check=True,
    )
    if not (Path(cfg.output_dir) / result_name).is_file():
        raise RuntimeError(
            "Simulator exited without the expected score file; check evaluator patch"
        )


def run_libero(cfg, plan, save):
    """Reuse a single model across the complete LIBERO task inventory."""
    from .benchmarks.libero import eval_libero_single as r

    runtime = native_config(cfg)
    from .runtime import create_policy
    r.set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    model = processor = None
    height, width = map(int, runtime.data.train.video_size)
    suites = r.benchmark.get_benchmark_dict()
    from .benchmarks.libero.libero_utils import load_initial_states
    inventory = []
    for cell in plan:
        task = suites[cell["suite"]]().get_task(cell["task_id"])
        states = list(load_initial_states(task))
        if not states:
            raise RuntimeError("LIBERO task has no initial states")
        inventory.append((cell, task, states))
    policy = create_policy(cfg)
    try:
        for cell, task, states in inventory:
            runtime.EVALUATION.task_suite_name = cell["suite"]
            runtime.EVALUATION.task_id = cell["task_id"]
            states = [states[index % len(states)] for index in range(int(cfg.episodes))]
            output = Path(cfg.output_dir) / cell["suite"] / str(cell["task_id"])
            output.mkdir(parents=True, exist_ok=True)
            result = r.run_single_task(
                task,
                states,
                model,
                processor,
                runtime,
                output / "videos",
                output / "predicted_videos",
                action_horizon=int(runtime.EVALUATION.action_horizon),
                input_w=width,
                input_h=height,
                model_device=str(cfg.device),
                policy=policy,
            )
            indices = result["success_episodes"] + result["failure_episodes"]
            if sorted(indices) != list(range(int(cfg.episodes))):
                raise RuntimeError("Incomplete or duplicate LIBERO episode inventory")
            result.update(cell, total_episodes=int(cfg.episodes))
            (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            save(result)
    finally:
        policy.close()


def run_native(cfg):
    from omegaconf import OmegaConf
    from .benchmarks.evaluation_plan import native_plan, native_result, summarize_cells

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    plan = native_plan(cfg)
    results = {"benchmark": cfg.benchmark, "cells": [], "plan": plan}
    output = Path(cfg.output_dir) / "results.json"
    if output.exists():
        raise FileExistsError(output)

    def save(cell=None):
        if cell is not None:
            results["cells"].append(cell)
        results.update(summarize_cells(results["cells"], len(plan)))
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(output)

    save()
    if cfg.benchmark == "libero":
        run_libero(cfg, plan, save)
        return
    for cell in plan:
        settings = OmegaConf.merge(cfg, cell)
        settings.output_dir = str(Path(cfg.output_dir) / cell["setting"] / cell["task"])
        Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
        run_native_cell(settings)
        save(native_result(settings, settings.output_dir))


def main(argv=None):
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--worker", choices=("policy", "simulator"), help=argparse.SUPPRESS)
    args, overrides = parser.parse_known_args(argv)
    cfg = read_settings(args.config, overrides)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=False))
        return 0
    if cfg.get("experiment") in {"agentic", "gpt6"}:
        from .benchmarks.robocasa.agentic.evaluate import validate_agentic_settings

        validate_agentic_settings(cfg)
    validate_inputs(cfg)
    setup_model_paths()
    if args.worker == "policy":
        serve(cfg)
        return 0
    if args.worker == "simulator":
        if cfg.benchmark == "robocasa_gr1":
            run_gr1(cfg)
        elif cfg.benchmark == "robocasa365":
            if cfg.get("experiment") in {"agentic", "gpt6"}:
                from .benchmarks.robocasa.agentic.evaluate import run

                run(cfg)
            else:
                run_robocasa(cfg)
        else:
            run_native(cfg)
        return 0
    if args.serve_only:
        if not cfg.get("socket"):
            parser.error("infer requires socket=/path/to/private.sock")
        serve(cfg)
        return 0
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "evaluation.yaml"
    if snapshot.exists():
        raise FileExistsError(f"Choose a new output_dir; refusing to overwrite {snapshot}")
    # A private socket and a process handle prevent interference with other runs.
    with tempfile.TemporaryDirectory(prefix="longwam-") as temp:
        if "socket" in cfg and cfg.socket is None:
            cfg.socket = str(Path(temp) / "policy.sock")
        OmegaConf.save(cfg, snapshot, resolve=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repository_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
        base = ["-m", "longwam.evaluation", "--config", str(snapshot)]
        server = None
        worker = None
        try:
            if cfg.benchmark in {"robocasa_gr1", "robocasa365"}:
                server = subprocess.Popen(
                    [sys.executable, *base, "--worker", "policy"], env=env, start_new_session=True
                )
            simulator_python = cfg.simulator_python or sys.executable
            worker = subprocess.Popen(
                [simulator_python, *base, "--worker", "simulator"], env=env, start_new_session=True
            )
            while True:
                try:
                    code = worker.wait(timeout=1)
                    if code:
                        raise subprocess.CalledProcessError(code, worker.args)
                    break
                except subprocess.TimeoutExpired:
                    if server is not None and server.poll() is not None:
                        raise RuntimeError(
                            f"Policy server exited unexpectedly ({server.returncode})"
                        )
        finally:
            for process in (worker, server):
                if process is not None:
                    # Include native simulator children, but never another run's processes.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
