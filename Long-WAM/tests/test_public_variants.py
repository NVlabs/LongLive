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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/test_public_variants.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Regression coverage for supported variants, not historical experiment launchers."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from test_release_contracts import ROOT, recipe
from longwam.evaluation import native_config, read_settings
from longwam.training_contract import check_training_contract
from longwam.utils.checkpoint_retention import prune_checkpoint_pairs


@pytest.mark.parametrize("history", [0, 48, 96, 192, 384, 768])
def test_gr1_scaling_is_matched_end_to_end(history):
    cfg = recipe("robocasa_gr1", [f"context=p{history}"])
    check = check_training_contract(cfg, world_size=16)
    assert check["global_batch"] == 480
    assert check["history"] == history
    assert cfg.data.train.num_frames == history + 33
    assert cfg.data.train.processor.num_obs_steps == history + 33
    assert cfg.data.train.processor.num_image_steps == (history + 32) // 4 + 1
    assert cfg.data.train.action_chunk == 16
    assert cfg.max_steps == cfg.scheduler_max_steps == 30000
    assert cfg.learning_rate == 3e-5
    assert cfg.lr_scheduler_type == "cosine_hf"
    assert cfg.adam_beta1 == 0.95 and cfg.adam_beta2 == 0.999


def test_robocasa365_is_fresh_100k_not_continuation():
    cfg = recipe("robocasa365")
    assert cfg.max_steps == cfg.scheduler_max_steps == 100000
    assert cfg.resume is None and cfg.init_checkpoint is None
    assert cfg.batch_size == 16 and cfg.gradient_accumulation_steps == 1
    assert cfg.learning_rate == 1e-4 and cfg.lr_warmup_ratio == 0.05
    assert cfg.lr_eta_min_ratio == 0.01
    assert cfg.adam_beta1 == 0.9 and cfg.adam_beta2 == 0.95
    assert cfg.weight_decay == 0.01 and cfg.seed == cfg.model_init_seed == 42
    assert cfg.sampler.samples_per_epoch == 512000
    assert dict(cfg.sampler.domain_weights) == {"atomic": 0.36, "composite": 0.64}
    assert cfg.data.train.concat_multi_camera == "robocasa_left_main"
    assert cfg.eval_every == 0
    assert cfg.checkpoint_keep_last == 3


@pytest.mark.parametrize("benchmark", ["libero", "robotwin2"])
@pytest.mark.parametrize("mode", ["idm", "codenoise"])
def test_supported_denoising_variants_and_checkpoint_pairing(benchmark, mode, tmp_path):
    cfg = recipe(benchmark, [f"denoising={mode}"])
    check_training_contract(cfg, world_size=8)
    assert cfg.model.longwam_joint_denoise == (mode == "codenoise")
    assert cfg.model.longwam_imagine_sigma == (0 if mode == "codenoise" else 0.9)
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    settings = read_settings(
        ROOT / f"configs/eval/{benchmark}.yaml",
        [f"model_config={path}", "checkpoint=/unused", "stats=/unused", f"inference_mode={mode}"],
    )
    runtime = native_config(settings)
    assert runtime.model.longwam_joint_denoise == (mode == "codenoise")
    settings.inference_mode = "idm" if mode == "codenoise" else "codenoise"
    with pytest.raises(ValueError, match="matching checkpoint"):
        native_config(settings)


@pytest.mark.parametrize("joint", [False, True])
def test_denoising_attention_keeps_history_causal(joint):
    from longwam.models.wan22.longwam_base import LongWAMBase

    class Video:
        def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
            return torch.tril(torch.ones(video_seq_len, video_seq_len, dtype=torch.bool))

    model = object.__new__(LongWAMBase)
    torch.nn.Module.__init__(model)
    model.num_clean_frames = 2
    model.joint_denoise = joint
    model.video_expert = Video()
    mask = model._build_mot_attention_mask(4, 2, 1, torch.device("cpu"), num_imagine_frames=2)
    assert not bool(mask[:2, 2:].any())
    assert bool(mask[4:, :4].all())
    assert bool(mask[2:4, 4:].all()) == joint


def test_only_public_recipe_families_remain():
    assert {p.stem for p in (ROOT / "configs/task").glob("*.yaml")} == {
        "libero",
        "robotwin2",
        "domino",
        "robocasa_gr1",
        "robocasa365",
    }
    assert not list((ROOT / "configs").rglob("*libero_plus*"))
    assert not list((ROOT / "configs").rglob("*ablation*"))
    assert not list((ROOT / "configs").glob("sim_*.yaml"))


def test_libero_adapter_import_is_package_local_and_resolvers_are_idempotent(monkeypatch):
    import importlib
    import sys
    import types
    from longwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    outer = types.ModuleType("libero")
    outer.__path__ = []
    inner = types.ModuleType("libero.libero")
    inner.__path__ = []
    inner.get_libero_path = lambda name: "/unused"
    inner.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {})
    envs = types.ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = envs.SubprocVectorEnv = object
    monkeypatch.setitem(sys.modules, "libero", outer)
    monkeypatch.setitem(sys.modules, "libero.libero", inner)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    adapter = importlib.import_module("longwam.benchmarks.libero.eval_libero_single")
    assert adapter._build_obs_window.__module__ == "longwam.runtime.policy"
    register_default_resolvers()


@pytest.mark.parametrize("joint,expected", [(False, (4, 10)), (True, (10, 10))])
def test_video_action_step_schedules(joint, expected):
    from longwam.models.wan22.longwam_base import LongWAMBase

    model = object.__new__(LongWAMBase)
    torch.nn.Module.__init__(model)
    model.joint_denoise = joint
    model.imagine_infer_steps = 4
    assert model._resolve_joint_ar_inference_steps(
        action_num_inference_steps=10, video_num_inference_steps=None) == expected
    if joint:
        with pytest.raises(ValueError, match="one shared"):
            model._resolve_joint_ar_inference_steps(
                action_num_inference_steps=10, video_num_inference_steps=4)


def test_retention_never_deletes_uncommitted_or_pinned_checkpoints(tmp_path):
    weights, states = tmp_path / "weights", tmp_path / "state"
    weights.mkdir()
    states.mkdir()
    for step in range(1, 7):
        name = f"step_{step:06d}"
        (weights / f"{name}.pt").write_bytes(b"test")
        state = states / name
        state.mkdir()
        (state / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        if step != 2:
            (state / ".longwam-complete").touch()
    (states / "step_000001/.keep").touch()
    removed = prune_checkpoint_pairs(weights, states, 2, protected_steps=[3])
    assert removed == [4]
    assert (weights / "step_000001.pt").exists()
    assert (weights / "step_000002.pt").exists()
    assert (weights / "step_000003.pt").exists()
    assert (weights / "step_000006.pt").exists()


@pytest.mark.parametrize(
    "stage,steps,sp,frames", [("S", 28602, 2, 64), ("M", 11155, 4, 128), ("L", 3972, 8, 192)]
)
def test_video_curriculum_contract_without_gpu(stage, steps, sp, frames, monkeypatch, tmp_path):
    monkeypatch.setenv("LONGWAM_VIDEO_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LONGWAM_VIDEO_INDEX_CACHE", str(tmp_path / "index"))
    spec = importlib.util.spec_from_file_location("longwam_stage1_launch", ROOT / "stage1/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        mode="train",
        stage=stage,
        resume_step=0,
        stop_step=None,
        training_step=None,
        output=tmp_path / "output",
        assets_root=tmp_path,
        generator=tmp_path / "absent.pt",
        generator_sha256="unspecified",
    )
    cfg, _ = module.prepare(args)
    assert cfg.training.max_iters == steps
    assert cfg.infra.sequence_parallel_size == sp
    assert cfg.training.batch_size == cfg.training.gradient_accumulation_steps == 1
    assert cfg.data.image_or_video_shape[1] == frames
    assert cfg.training.lr == 5e-6
    assert not args.output.exists()


def test_gpt_experiment_is_opt_in_budgeted_and_has_only_three_tools():
    from longwam.benchmarks.robocasa_gpt6.evaluate import validate_gpt_settings
    from longwam.benchmarks.robocasa_gpt6.tools import tool_specs
    from longwam.benchmarks.robocasa_gpt6.codex_host import skill_file

    cfg = read_settings(ROOT / "configs/eval/robocasa365_gpt6.yaml")
    with pytest.raises(ValueError, match="allow_model_upload"):
        validate_gpt_settings(cfg)
    cfg.gpt6.allow_model_upload = True
    with pytest.raises(ValueError, match="budget"):
        validate_gpt_settings(cfg)
    cfg.gpt6.max_total_tokens = 100
    validate_gpt_settings(cfg)
    tools = tool_specs("hybrid_decompose")
    assert [tool["name"] for tool in tools] == [
        "robocasa_start",
        "longwam_infer",
        "robocasa_execute",
    ]
    assert (
        tools[-1]["inputSchema"]["properties"]["response"]["properties"]["steps"]["maximum"] == 15
    )
    assert skill_file("hybrid_decompose").is_file()
    with pytest.raises(ValueError):
        tool_specs("decompose_v5")
