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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/test_release_contracts.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Portable public interfaces; no GPU, simulator, data or released weights needed."""

import ast
from pathlib import Path
import re
import runpy
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
import yaml

from longwam.evaluation import native_config, read_settings
from longwam.inference import migrate_config_names, model_config_for_inference, TextCache
from longwam.paths import repository_root
from longwam.training_contract import check_training_contract
from longwam.utils.config_resolvers import register_default_resolvers

ROOT = repository_root()
RECIPES = [
    ("libero", 8, 128, 48, 32, 1e-4),
    ("robotwin2", 8, 1024, 48, 32, 1e-4),
    ("domino", 8, 1024, 48, 32, 3e-6),
    ("robocasa_gr1", 16, 480, 384, 16, 3e-5),
    ("robocasa365", 16, 256, 48, 32, 1e-4),
]


def recipe(name, overrides=()):
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=[f"task={name}", *overrides])


@pytest.mark.parametrize("name,world,batch,history,chunk,lr", RECIPES)
def test_benchmark_train_recipe_and_independent_scripts(name, world, batch, history, chunk, lr):
    cfg = recipe(name)
    result = check_training_contract(cfg, world_size=world)
    assert result["global_batch"] == batch
    assert result["history"] == history
    assert result["action_chunk"] == chunk
    assert result["learning_rate"] == lr
    for command in ("train", "eval"):
        path = ROOT / f"scripts/{name}/{command}.sh"
        assert path.is_file()
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_changed_world_size_fails_before_model_or_data_loading():
    with pytest.raises(ValueError, match="Global batch mismatch"):
        check_training_contract(recipe("libero"), world_size=4)
    cfg = recipe("libero", ["gradient_accumulation_steps=4"])
    assert check_training_contract(cfg, world_size=4)["global_batch"] == 128


def test_good_gr1_data_and_scaling_initialization():
    cfg = recipe("robocasa_gr1")
    assert cfg.data.train.source_recipe == "teleop"
    assert len(cfg.data.train.dataset_dirs) == 24
    assert cfg.data.train.expected_dataset_revision == "09c6de8af50168090e7e9cc01e1ec3bce788de24"
    assert cfg.resume is None and cfg.init_checkpoint is None
    assert cfg.model.skip_action_dit_load_from_pretrain is False
    assert cfg.max_steps == cfg.scheduler_max_steps == 30000


def test_domino_bestmix_is_the_only_public_domino_recipe():
    cfg = recipe("domino")
    assert dict(cfg.sampler.domain_weights) == {"robotwin": 0.5, "domino": 0.5}
    assert cfg.sampler.samples_per_epoch == 307200
    assert cfg.max_steps == 300
    assert cfg.model.skip_action_dit_load_from_pretrain is True
    assert len(cfg.data.train.domains.domino.dataset_dirs) == 11
    assert not (ROOT / "configs/task/domino_mix.yaml").exists()


def test_repository_config_references_do_not_depend_on_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = recipe("robocasa365")
    assert len(cfg.data.train.dataset_dirs) == 300


def test_history_override_requires_matching_data_and_model():
    cfg = recipe("libero", ["model.longwam_past_obs_size=96"])
    with pytest.raises(ValueError, match="history mismatch"):
        check_training_contract(cfg, world_size=8)


def test_old_configs_map_names_but_never_weight_keys_or_paths():
    source = {
        "model": {"_target_": "fastwam.runtime.create_arwam", "arwam_past_obs_size": 192},
        "path": "/weights/fastwam/my-model.pt",
        "tensor": "mot.mixtures.video.blocks.0.weight",
    }
    migrated = migrate_config_names(source)
    assert migrated["model"] == {
        "_target_": "longwam.runtime.create_longwam",
        "longwam_past_obs_size": 192,
    }
    assert migrated["path"] == source["path"]
    assert migrated["tensor"] == source["tensor"]
    assert "arwam_past_obs_size" in source["model"]


def test_inference_does_not_reload_old_training_initializers():
    cfg = recipe("libero")
    model_cfg = model_config_for_inference(cfg)
    assert model_cfg.longlive_video_weights is None
    assert model_cfg.action_dit_pretrained_path is None
    assert model_cfg.skip_video_dit_load_from_pretrain
    assert model_cfg.skip_action_dit_load_from_pretrain
    assert cfg.model.longlive_video_weights is not None


@pytest.mark.parametrize(
    "benchmark", ["libero", "robotwin2", "domino", "robocasa_gr1", "robocasa365"]
)
def test_eval_config_accepts_overrides_and_rejects_typos(benchmark):
    path = ROOT / f"configs/eval/{benchmark}.yaml"
    cfg = read_settings(path, ["episodes=2", "output_dir=/tmp/example-only"])
    assert cfg.episodes == 2
    with pytest.raises(Exception):
        read_settings(path, ["episoeds=2"])
    with pytest.raises(ValueError):
        read_settings(path, ["episodes=0"])


def test_model_config_for_native_eval_uses_exported_history_and_chunk(tmp_path):
    cfg = recipe("robotwin2")
    # Store a resolved model/data config like the trainer; no actual checkpoint load.
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    settings = read_settings(
        ROOT / "configs/eval/robotwin2.yaml",
        [
            f"model_config={path}",
            "checkpoint=/unused.pt",
            "stats=/unused.json",
        ],
    )
    native = native_config(settings)
    assert native.checkpoint_strict is True
    assert native.EVALUATION.action_horizon == 32
    assert native.model.longwam_past_obs_size == 48
    assert native.model.longlive_video_weights is None


def test_gr1_text_cache_preserves_training_padding_and_mask(tmp_path):
    import torch
    from longwam.datasets.lerobot.robot_video_dataset import get_text_cache_path

    prompt = "instruction"
    path = Path(get_text_cache_path(tmp_path, prompt, 128))
    context = torch.ones((128, 4096))
    mask = torch.arange(128) < 7
    torch.save({"context": context, "mask": mask}, path)
    actual, actual_mask = TextCache(tmp_path)(prompt)
    assert actual.shape == (128, 4096)
    assert bool(actual_mask.all())
    assert bool((actual[:7] == 1).all())
    assert bool((actual[7:] == 0).all())


def test_all_local_python_syntax_and_no_server_paths():
    for path in [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").rglob("*.py")]:
        source = path.read_text()
        ast.parse(source, filename=str(path))
        assert "/users/weihua/" not in source
        assert "nvr_elm_llm" not in source
    for path in ROOT.joinpath("configs").rglob("*.yaml"):
        assert "/users/weihua/" not in path.read_text()


def test_missing_checkpoint_url_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/download_checkpoint.py"),
            "domino",
            "--output",
            str(tmp_path / "download"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "intentionally TBD" in result.stderr
    assert not (tmp_path / "download").exists()


def test_public_checkpoint_registry_matches_readme():
    registry = yaml.safe_load((ROOT / "configs/checkpoints.yaml").read_text())
    readme = (ROOT / "README.md").read_text()
    releases = {entry["repo_id"]: entry["revision"] for entry in registry.values()
                if entry["repo_id"] is not None}
    assert len(releases) == 13
    for repo_id, revision in releases.items():
        assert repo_id.startswith("Efficient-Large-Model/")
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
        assert f"https://huggingface.co/{repo_id}" in readme
    for alias, variant in {
        "libero": "libero_idm", "robotwin2": "robotwin2_idm",
        "robocasa_gr1": "robocasa_gr1_p384", "stage1_robot_video": "stage1_robot_s",
    }.items():
        assert registry[alias] == registry[variant]
    for history, duration in {0: "0", 48: "2.4", 96: "4.8", 192: "9.6", 384: "19.2"}.items():
        assert registry[f"robocasa_gr1_p{history}"]["repo_id"] == (
            f"Efficient-Large-Model/Long-WAM-RoboCasa-GR1-{duration}s"
        )


def test_checkpoint_downloader_uses_pinned_release_without_network(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw))
    script = ROOT / "scripts/download_checkpoint.py"
    output = tmp_path / "gr1"
    monkeypatch.setattr(sys, "argv", [str(script), "robocasa_gr1_p192", "--output", str(output)])
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [{
        "repo_id": "Efficient-Large-Model/Long-WAM-RoboCasa-GR1-9.6s",
        "revision": "d3457a907cdefd5035bedac6474cd3179cbc391e",
        "local_dir": str(output),
    }]
    assert not output.exists()


def test_robocasa_eval_uses_matched_pretrain_split():
    cfg = read_settings(ROOT / "configs/eval/robocasa365.yaml")
    assert cfg.task_set == "target50"
    assert cfg.split == "pretrain"
    assert cfg.replan_steps == 32


def test_libero_retains_explicit_historical_evaluation_override(tmp_path):
    cfg = recipe("libero")
    assert cfg.data.train.action_chunk == 32
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    settings = read_settings(
        ROOT / "configs/eval/libero.yaml",
        [
            f"model_config={path}",
            "checkpoint=/unused.pt",
            "stats=/unused.json",
        ],
    )
    native = native_config(settings)
    assert native.EVALUATION.action_horizon == 80
    assert native.EVALUATION.replan_steps == 10
    assert native.EVALUATION.num_inference_steps == 20
    assert native.EVALUATION.num_steps_wait == 30


@pytest.mark.parametrize("benchmark,count", [("libero", 40), ("robotwin2", 100), ("domino", 35)])
def test_complete_native_evaluation_plan(benchmark, count):
    from longwam.benchmarks.evaluation_plan import native_plan

    plan = native_plan(read_settings(ROOT / f"configs/eval/{benchmark}.yaml"))
    assert len(plan) == count
    assert len({tuple(sorted(cell.items())) for cell in plan}) == count


def test_partial_summary_is_not_a_complete_benchmark_result():
    from longwam.benchmarks.evaluation_plan import summarize_cells

    result = summarize_cells([{"successes": 1, "total_episodes": 5}], expected_cells=35)
    assert result["complete"] is False
    assert result["success_rate_percent"] == 20


def test_domino_result_requires_all_episodes(tmp_path):
    import json
    from longwam.benchmarks.evaluation_plan import native_result

    cfg = read_settings(ROOT / "configs/eval/domino.yaml", ["task=click_alarmclock", "episodes=2"])
    (tmp_path / "_metrics.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "success_count": 1,
                "manipulation_score_mean": 61.5,
            }
        )
    )
    (tmp_path / "_episodes_detail.json").write_text("[]")
    with pytest.raises(RuntimeError, match="Incomplete"):
        native_result(cfg, tmp_path)
    (tmp_path / "_episodes_detail.json").write_text('[{"seed": 1}, {"seed": 2}]')
    assert native_result(cfg, tmp_path)["manipulation_score"] == 61.5


def test_model_state_dict_layout_is_unchanged_under_new_class_name():
    import torch
    from longwam.models.wan22.long_wam import LongWAM
    from longwam.models.wan22.longwam_base import LongWAMBase

    assert issubclass(LongWAM, LongWAMBase)
    # State serialization belongs to PyTorch modules, not the renamed Python package.
    model = object.__new__(LongWAM)
    torch.nn.Module.__init__(model)
    model.mot = torch.nn.Module()
    model.mot.mixtures = torch.nn.ModuleDict({"video": torch.nn.Linear(2, 2)})
    assert set(model.state_dict()) == {"mot.mixtures.video.weight", "mot.mixtures.video.bias"}


@pytest.mark.parametrize("pretrained,strict", [(True, False), (False, True)])
def test_common_loader_preserves_generator_and_release_checkpoint_modes(
    pretrained, strict, tmp_path, monkeypatch,
):
    from hydra import utils
    from longwam.inference import load_model
    calls = []
    class Model(torch.nn.Module):
        def load_checkpoint(self, path, **kwargs):
            calls.append(kwargs)
            return {"step": 12}
    configurations = []
    def instantiate(config, **kwargs):
        configurations.append(OmegaConf.to_container(config))
        return Model()
    monkeypatch.setattr(utils, "instantiate", instantiate)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    cfg = OmegaConf.create({"model": {"_target_": "longwam.runtime.create_longwam"},
                            "mixed_precision": "bf16"})
    load_model(cfg, checkpoint, device="cpu", strict=strict, load_pretrained=pretrained,
               expected_step=12)
    assert calls == [{"strict": strict}]
    assert ("skip_dit_load_from_pretrain" not in configurations[0]) == pretrained



def test_task_conditioning_encodes_once_and_respects_model_argument_contract(tmp_path):
    from longwam.inference import TextConditioning
    calls = []
    class Model:
        def encode_prompt(self, prompt):
            calls.append(prompt)
            return torch.ones(1, 2, 3), torch.ones(1, 2)
    conditioning = TextConditioning(Model())
    first = conditioning('task one')
    second = conditioning('task one')
    assert calls == ['task one']
    assert first['prompt'] is None and first['context'] is second['context']
    conditioning('task two')
    assert calls == ['task one', 'task two']


def test_offline_inference_loads_once_and_passes_action_horizon(tmp_path, monkeypatch):
    from PIL import Image
    from longwam.runtime import factory
    import longwam.inference as shared
    image = tmp_path / 'frame.png'
    Image.new('RGB', (8, 8)).save(image)
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    loads, predictions = [], []
    class Model:
        device = 'cpu'
        torch_dtype = torch.float32
        def eval(self):
            return self
        def infer(self, *, action_horizon, **kwargs):
            predictions.append(action_horizon)
            return {'video': []}
    monkeypatch.setattr(shared, 'load_model', lambda *a, **k: loads.append(k) or Model())
    monkeypatch.setattr(factory, 'save_mp4', lambda *a, **k: None)
    cfg = OmegaConf.create({'mixed_precision': 'no', 'checkpoint_strict': True,
        'data': {'train': {'action_chunk': 32}}, 'inference': {
            'checkpoint_path': str(checkpoint), 'device': 'cpu',
            'input_image_path': str(image), 'width': 8, 'height': 8,
            'output_mp4': str(tmp_path / 'out.mp4'), 'prompt': 'task',
            'negative_prompt': '', 'text_cfg_scale': 1, 'action_cfg_scale': 1,
            'num_frames': 5, 'num_inference_steps': 1, 'seed': 0,
            'rand_device': 'cpu', 'tiled': False}})
    factory.run_inference(cfg)
    assert predictions == [32] and len(loads) == 1
    assert loads[0]['strict'] and not loads[0]['load_pretrained']
    cfg.inference.checkpoint_path = str(tmp_path / 'missing.pt')
    with pytest.raises(FileNotFoundError, match='Checkpoint'):
        factory.run_inference(cfg)
    assert len(loads) == 1


@pytest.mark.parametrize('hardware', ['rtx5090', 'spark', 'thor'])
def test_device_profile_configures_real_model_hooks(hardware, monkeypatch):
    from longwam.runtime.backends import CAPABILITIES, prepare_backend
    from longwam.runtime.optim.model import InferenceModelMixin
    from longwam.runtime.optim.config import InferenceOptimizationConfig
    class Model(InferenceModelMixin):
        device = 'cuda'
        joint_denoise = False
        _inference_optimization_prepared = False
        def prepare_inference_optimizations(self):
            return self._inference_optimization_config
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: CAPABILITIES[hardware])
    resolved = prepare_backend(Model(), hardware)
    assert resolved.model_quant_targets == ('video',)
    assert resolved.torch_compile and resolved.action_segmented_attention
    assert InferenceOptimizationConfig.from_mapping(resolved.to_dict()) == resolved


def test_optimization_serialization_preserves_per_target_compile_modes():
    from longwam.runtime.optim.config import InferenceOptimizationConfig
    cfg = InferenceOptimizationConfig.from_mapping({
        'torch_compile': True, 'torch_compile_targets': ['action_denoiser'],
        'torch_compile_target_modes': {'action_denoiser': 'max-autotune'},
    })
    assert InferenceOptimizationConfig.from_mapping(cfg.to_dict()) == cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('query,key', [(392, 392), (196, 588)])
def test_compiled_cudnn_attention_matches_reference_and_declared_layout(query, key):
    from longwam.runtime.optim.projections import cudnn_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if not torch.backends.cudnn.is_available():
        pytest.skip('cuDNN required')
    tensors = [torch.randn(1, length, 24, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
               for length in (query, key, key)]
    mask = torch.ones(query, key, device='cuda', dtype=torch.bool)
    if query == key:
        mask = torch.tril(mask)
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(*tensors, attn_mask=mask)
    torch.library.opcheck(cudnn_attention, (*tensors, mask))
    compiled = torch.compile(cudnn_attention, fullgraph=True)
    result = compiled(*tensors, mask)
    torch.testing.assert_close(result, expected, rtol=0.03, atol=0.02)
    assert result.stride() == (query * 24 * 128, 128, 24 * 128, 1)
