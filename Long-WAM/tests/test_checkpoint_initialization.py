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

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

import longwam.utils.checkpoint_initialization as checkpoint_init
from longwam.utils.checkpoint_initialization import (
    LIBERO_TO_ROBOCASA,
    initialize_from_checkpoint,
)


ACTION_ENCODER = "mixtures.action.action_encoder.weight"
ACTION_ENCODER_BIAS = "mixtures.action.action_encoder.bias"
ACTION_HEAD = "mixtures.action.head.weight"
ACTION_HEAD_BIAS = "mixtures.action.head.bias"


@pytest.fixture
def action_conversion():
    import runpy

    script = Path(__file__).resolve().parents[1] / "tools/weights/preprocess_action_dit_backbone.py"
    converter = runpy.run_path(str(script))
    video_cfg, action_cfg = converter["_load_model_config"](
        script.parents[2] / "configs/model/long_wam.yaml"
    )
    shared = dict(num_layers=1, num_heads=2, attn_head_dim=6, text_dim=16, freq_dim=8)
    video_cfg.update(shared, hidden_dim=12, ffn_dim=24, in_dim=4, out_dim=4)
    action_cfg.update(shared, hidden_dim=8, ffn_dim=16, action_dim=29)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        state = converter["WanVideoDiT"](**video_cfg).state_dict()
    return script, converter, video_cfg, action_cfg, state


@pytest.mark.parametrize("container", ["generator", "bare", "joint"])
def test_action_conversion_loads_backbone_and_keeps_random_interfaces(
    action_conversion, container, tmp_path
):
    _, converter, video_cfg, action_cfg, state = action_conversion
    wrapped = {
        "generator": {f"model.{k}": v for k, v in state.items()},
        "bare": state,
        "joint": {"mot": {f"mixtures.video.{k}": v for k, v in state.items()}},
    }[container]
    payload, counts = converter["convert_backbone"](
        converter["_video_state"](wrapped), video_cfg, action_cfg
    )
    assert counts["copied"] > 0 and counts["interpolated"] > 0 and counts["skipped"] == 4
    backbone = payload["backbone_state_dict"]
    assert not any(k.startswith(("action_encoder.", "head.")) for k in backbone)
    checkpoint = tmp_path / "action.pt"
    torch.save(payload, checkpoint)
    cls = converter["ActionDiT"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = cls(**action_cfg).state_dict()
        torch.manual_seed(42)
        model = cls.from_pretrained(
            action_cfg, str(checkpoint), device="cpu", torch_dtype=torch.float32
        )
    for key, value in model.state_dict().items():
        assert torch.equal(value, backbone.get(key, expected[key])), key


@pytest.mark.parametrize("problem", ["missing", "shape", "nonfinite"])
def test_action_conversion_rejects_invalid_source(action_conversion, problem):
    _, converter, video_cfg, action_cfg, state = action_conversion
    key = "text_embedding.0.weight"
    if problem == "missing":
        del state[key]
    elif problem == "shape":
        state[key] = state[key][:1]
    else:
        state[key].fill_(float("nan"))
    with pytest.raises(ValueError):
        converter["convert_backbone"](state, video_cfg, action_cfg)


def test_action_conversion_one_command_download_and_no_overwrite(
    action_conversion, tmp_path, monkeypatch
):
    import hashlib
    import runpy
    import sys

    import huggingface_hub
    from omegaconf import OmegaConf

    script, _, video_cfg, action_cfg, state = action_conversion
    root = tmp_path / "project"
    (root / "configs/model").mkdir(parents=True)
    source = tmp_path / "model.pt"
    torch.save({f"model.{k}": v for k, v in state.items()}, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    release = dict(repo_id="example/Robot-S", revision="a" * 40, sha256=digest)
    OmegaConf.save(OmegaConf.create({"stage1_robot_s": release}), root / "configs/checkpoints.yaml")
    OmegaConf.save(
        OmegaConf.create({"video_dit_config": video_cfg, "action_dit_config": action_cfg}),
        root / "configs/model/long_wam.yaml",
    )
    monkeypatch.setenv("LONGWAM_ROOT", str(root))
    monkeypatch.delenv("LONGWAM_CHECKPOINT_ROOT", raising=False)
    monkeypatch.delenv("LONGWAM_ROBOT_VIDEO_CHECKPOINT", raising=False)
    monkeypatch.chdir(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(sys, "argv", [str(script), "--threads", "1"])
    runpy.run_path(str(script), run_name="__main__")
    output = root / "checkpoints/longwam/ActionDiT_from_longlive_ar_1024hdim.pt"
    payload = torch.load(output, weights_only=True)
    assert payload["source"] == {**release, "filename": "model.pt"}
    assert calls == [
        dict(
            repo_id=release["repo_id"],
            revision=release["revision"],
            filename="model.pt",
            local_dir=root / "checkpoints/LongLive2.0-Robot-S",
        )
    ]
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    assert output.read_bytes() == original and len(calls) == 1
    assert not list(output.parent.glob(".*.tmp"))


@pytest.mark.parametrize("container", ["generator", "bare", "joint"])
def test_robot_video_initializer_loads_strictly_on_cpu(container, tmp_path):
    from longwam.runtime.factory import _load_longlive_video_expert_weights

    # Small synthetic tensors test the public generator export format without
    # downloading a full model, allocating a GPU, or creating conversion files.
    expected = {"weight": torch.full((3, 2), 7.0), "bias": torch.full((3,), 11.0)}
    if container == "generator":
        payload = {f"model.{key}": value for key, value in expected.items()}
    elif container == "joint":
        payload = {"mot": {f"mixtures.video.{key}": value for key, value in expected.items()}}
        payload["mot"]["mixtures.action.unused"] = torch.ones(1)
    else:
        payload = expected
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    target = torch.nn.Linear(2, 3)
    assert _load_longlive_video_expert_weights(target, checkpoint) == checkpoint.resolve()
    for key, value in expected.items():
        assert torch.equal(target.state_dict()[key], value)
    report = target._longwam_strict_load_report
    assert report["strict"] is True
    assert report["missing_keys"] == report["unexpected_keys"] == []


def test_robot_video_initializer_rejects_missing_weights(tmp_path):
    from longwam.runtime.factory import _load_longlive_video_expert_weights

    target = torch.nn.Linear(2, 3)
    with pytest.raises(FileNotFoundError, match="video weights do not exist"):
        _load_longlive_video_expert_weights(target, tmp_path / "missing.pt")
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"model.weight": torch.ones(3, 2)}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_longlive_video_expert_weights(target, checkpoint)


def test_robot_video_initializer_resolves_relative_paths_from_project_root(tmp_path, monkeypatch):
    from longwam.runtime.factory import _load_longlive_video_expert_weights

    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    relative = Path("checkpoints/LongLive2.0-Robot-S/model.pt")
    checkpoint = project / relative
    checkpoint.parent.mkdir(parents=True)
    expected = {"weight": torch.full((3, 2), 7.0), "bias": torch.full((3,), 11.0)}
    torch.save({f"model.{key}": value for key, value in expected.items()}, checkpoint)
    monkeypatch.setenv("LONGWAM_ROOT", str(project))
    monkeypatch.chdir(tmp_path)
    target = torch.nn.Linear(2, 3)
    assert _load_longlive_video_expert_weights(target, relative) == checkpoint.resolve()
    assert torch.equal(target.weight, expected["weight"])


class _StateHolder:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


class _TinyTarget:
    def __init__(self):
        source, target = _mot_states()
        self.source_state = source
        self.target_state = target
        self.mot = _StateHolder(self.target_state)
        self.proprio_encoder = torch.nn.Linear(16, 4)
        with torch.no_grad():
            self.proprio_encoder.weight.fill_(-31.0)
            self.proprio_encoder.bias.fill_(-32.0)


def _mot_states():
    source = OrderedDict()
    target = OrderedDict()
    for index in range(1645):
        key = f"mixtures.shared.param_{index:04d}"
        source[key] = torch.tensor([float(index + 1)])
        target[key] = torch.tensor([-float(index + 1)])

    source[ACTION_ENCODER_BIAS] = torch.tensor([7.0, 8.0, 9.0])
    target[ACTION_ENCODER_BIAS] = torch.full((3,), -20.0)
    source[ACTION_ENCODER] = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    target[ACTION_ENCODER] = torch.full((3, 12), -21.0)
    source[ACTION_HEAD] = torch.arange(21, dtype=torch.float32).reshape(7, 3) + 100.0
    target[ACTION_HEAD] = torch.full((12, 3), -22.0)
    source[ACTION_HEAD_BIAS] = torch.arange(7, dtype=torch.float32) + 200.0
    target[ACTION_HEAD_BIAS] = torch.full((12,), -23.0)
    assert len(source) == len(target) == 1649
    return source, target


def _write_checkpoint(tmp_path, source_state):
    path = tmp_path / "libero.pt"
    torch.save(
        {
            "mot": source_state,
            "proprio_encoder": {
                "weight": torch.full((4, 8), 41.0),
                "bias": torch.full((4,), 42.0),
            },
            "step": 86790,
        },
        path,
    )
    return path


def test_maps_only_eef_and_mmaps_checkpoint_on_cpu(monkeypatch, caplog, tmp_path):
    model = _TinyTarget()
    checkpoint = _write_checkpoint(tmp_path, model.source_state)
    initial_encoder = model.target_state[ACTION_ENCODER].clone()
    initial_head = model.target_state[ACTION_HEAD].clone()
    initial_bias = model.target_state[ACTION_HEAD_BIAS].clone()
    initial_proprio = model.proprio_encoder.weight.detach().clone()

    load_args = {}
    real_load = torch.load

    def _load_spy(*args, **kwargs):
        load_args.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_init.torch, "load", _load_spy)
    report = initialize_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        adapter=LIBERO_TO_ROBOCASA,
    )

    assert load_args["map_location"] == "cpu"
    assert load_args["mmap"] is True
    assert load_args["weights_only"] is True
    assert report.loaded_keys == 1646
    assert len(report.mapped_keys) == 3
    assert report.checkpoint_step == 86790
    assert report.gripper_mapped is False

    assert torch.equal(
        model.target_state["mixtures.shared.param_1000"],
        model.source_state["mixtures.shared.param_1000"],
    )
    assert torch.equal(
        model.target_state[ACTION_ENCODER_BIAS],
        model.source_state[ACTION_ENCODER_BIAS],
    )
    assert torch.equal(
        model.target_state[ACTION_ENCODER][:, :5],
        initial_encoder[:, :5],
    )
    assert torch.equal(
        model.target_state[ACTION_ENCODER][:, 5:11],
        model.source_state[ACTION_ENCODER][:, :6],
    )
    assert torch.equal(
        model.target_state[ACTION_ENCODER][:, 11],
        initial_encoder[:, 11],
    )
    assert torch.equal(model.target_state[ACTION_HEAD][:5], initial_head[:5])
    assert torch.equal(
        model.target_state[ACTION_HEAD][5:11],
        model.source_state[ACTION_HEAD][:6],
    )
    assert torch.equal(model.target_state[ACTION_HEAD][11], initial_head[11])
    assert torch.equal(model.target_state[ACTION_HEAD_BIAS][:5], initial_bias[:5])
    assert torch.equal(
        model.target_state[ACTION_HEAD_BIAS][5:11],
        model.source_state[ACTION_HEAD_BIAS][:6],
    )
    assert torch.equal(model.target_state[ACTION_HEAD_BIAS][11], initial_bias[11])
    assert torch.equal(model.proprio_encoder.weight, initial_proprio)
    assert "action[11] remains newly initialized" in caplog.text
    assert "loaded=1646 mapped=3" in caplog.text
    assert "target proprio 16D keeps its new initialization" in caplog.text


def test_explicit_inverse_gripper_ranges_enable_sign_flip(tmp_path):
    model = _TinyTarget()
    checkpoint = _write_checkpoint(tmp_path, model.source_state)
    # Raw ranges differ (LIBERO 0..1, RoboCasa -1..1), but both model inputs are
    # min/max-normalized to -1..1 and their open/close semantics are reversed.
    source_raw_range = (0.0, 1.0)
    target_raw_range = (-1.0, 1.0)
    assert source_raw_range != target_raw_range

    report = initialize_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        adapter=LIBERO_TO_ROBOCASA,
        gripper={
            "normalized_source_min": -1.0,
            "normalized_source_max": 1.0,
            "normalized_target_min": -1.0,
            "normalized_target_max": 1.0,
        },
    )

    assert report.gripper_mapped is True
    assert torch.equal(
        model.target_state[ACTION_ENCODER][:, 11],
        -model.source_state[ACTION_ENCODER][:, 6],
    )
    assert torch.equal(
        model.target_state[ACTION_HEAD][11],
        -model.source_state[ACTION_HEAD][6],
    )
    assert torch.equal(
        model.target_state[ACTION_HEAD_BIAS][11],
        -model.source_state[ACTION_HEAD_BIAS][6],
    )


def test_rejects_non_allowlisted_shape_mismatch_before_copy(tmp_path):
    model = _TinyTarget()
    checkpoint = _write_checkpoint(tmp_path, model.source_state)
    untouched = model.target_state["mixtures.shared.param_0001"].clone()
    model.target_state["mixtures.shared.param_0000"] = torch.zeros(2)

    with pytest.raises(ValueError, match="Shape mismatch allowlist violation"):
        initialize_from_checkpoint(
            model,
            checkpoint_path=checkpoint,
            adapter=LIBERO_TO_ROBOCASA,
        )

    assert torch.equal(model.target_state["mixtures.shared.param_0001"], untouched)


def test_rejects_implicit_or_non_sign_flip_gripper_mapping(tmp_path):
    model = _TinyTarget()
    checkpoint = _write_checkpoint(tmp_path, model.source_state)

    with pytest.raises(ValueError, match="requires explicit"):
        initialize_from_checkpoint(
            model,
            checkpoint_path=checkpoint,
            adapter=LIBERO_TO_ROBOCASA,
            gripper={"normalized_source_min": -1.0},
        )

    with pytest.raises(ValueError, match="exact gripper sign flip"):
        initialize_from_checkpoint(
            model,
            checkpoint_path=checkpoint,
            adapter=LIBERO_TO_ROBOCASA,
            gripper={
                "normalized_source_min": 0.0,
                "normalized_source_max": 1.0,
                "normalized_target_min": 0.0,
                "normalized_target_max": 1.0,
            },
        )
