import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from model.cfg_distillation import CFGGuidanceDistillation
from utils.config import normalize_config
from utils.dataset import (
    CFGTeacherTrajectoryDataset,
    cfg_teacher_trajectory_collate_fn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(
    tmp_path: Path,
    *,
    target_shape=(2, 1, 2, 2),
    recorded_shape=(2, 1, 2, 2),
):
    shard = tmp_path / "prompt_0000.safetensors"
    tensors = {}
    for timestep_index in range(2):
        tensors[f"latent_{timestep_index:03d}"] = torch.full(
            recorded_shape, float(timestep_index), dtype=torch.bfloat16
        )
        tensors[f"target_{timestep_index:03d}"] = torch.full(
            target_shape, float(timestep_index + 1), dtype=torch.bfloat16
        )
        tensors[f"timestep_{timestep_index:03d}"] = torch.tensor(
            [999.0 if timestep_index == 0 else 100.0],
            dtype=torch.float32,
        )
        tensors[f"guidance_energy_{timestep_index:03d}"] = torch.tensor(
            [0.0, float(timestep_index + 1)], dtype=torch.float32
        )
    save_file(tensors, shard)
    shard_sha = _sha256(shard)
    records = []
    for timestep_index, timestep in enumerate((999.0, 100.0)):
        records.append(
            {
                "prompt": "A test prompt.",
                "prompt_index": 0,
                "timestep_index": timestep_index,
                "timestep": timestep,
                "shard": shard.name,
                "shard_sha256": shard_sha,
                "latent_key": f"latent_{timestep_index:03d}",
                "target_key": f"target_{timestep_index:03d}",
                "timestep_key": f"timestep_{timestep_index:03d}",
                "guidance_energy_key": (
                    f"guidance_energy_{timestep_index:03d}"
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "format": "cfg_teacher_trajectory_safetensors_v1",
        "model_name": "Wan2.2-TI2V-5B",
        "teacher_guidance_scale": 5.0,
        "teacher_sampling_steps": 2,
        "timestep_shift": 5.0,
        "latent_shape": list(recorded_shape),
        "timestep_repeat_factors": {"0": 1, "1": 3},
        "records": records,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _open_test_dataset(manifest_path, expected_shape=(2, 1, 2, 2)):
    return CFGTeacherTrajectoryDataset(
        manifest_path,
        expected_latent_shape=expected_shape,
        expected_sampling_steps=2,
    )


def test_cache_dataset_expands_timestep_repeats_and_collates(tmp_path):
    manifest_path, _ = _write_cache(tmp_path)
    dataset = _open_test_dataset(manifest_path)

    assert len(dataset) == 4
    assert [dataset[index]["cfg_timestep_index"] for index in range(4)] == [
        0,
        1,
        1,
        1,
    ]

    batch = cfg_teacher_trajectory_collate_fn([dataset[0], dataset[1]])
    assert batch["prompts"] == [["A test prompt."], ["A test prompt."]]
    assert tuple(batch["cfg_noisy_latent"].shape) == (2, 2, 1, 2, 2)
    assert tuple(batch["cfg_guided_target"].shape) == (2, 2, 1, 2, 2)
    assert tuple(batch["cfg_timestep"].shape) == (2,)
    assert tuple(batch["cfg_guidance_energy"].shape) == (2, 2)
    assert batch["cfg_timestep_index"].tolist() == [0, 1]


def test_cache_dataset_rejects_shard_hash_mismatch(tmp_path):
    manifest_path, manifest = _write_cache(tmp_path)
    manifest["records"][0]["shard_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _open_test_dataset(manifest_path)


def test_cache_dataset_rejects_tensor_and_manifest_shape_drift(tmp_path):
    manifest_path, _ = _write_cache(
        tmp_path,
        target_shape=(2, 1, 2, 1),
    )
    with pytest.raises(ValueError, match="expected"):
        _open_test_dataset(manifest_path)

    other = tmp_path / "manifest_shape"
    other.mkdir()
    manifest_path, _ = _write_cache(other)
    with pytest.raises(ValueError, match="latent_shape"):
        _open_test_dataset(manifest_path, expected_shape=(3, 1, 2, 2))


def test_cache_dataset_rejects_missing_timestep_and_bad_repeat(tmp_path):
    manifest_path, manifest = _write_cache(tmp_path)
    manifest["records"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover all"):
        _open_test_dataset(manifest_path)

    other = tmp_path / "bad_repeat"
    other.mkdir()
    manifest_path, manifest = _write_cache(other)
    manifest["timestep_repeat_factors"]["1"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        _open_test_dataset(manifest_path)


def test_cache_dataset_requires_contiguous_prompt_indices(tmp_path):
    manifest_path, manifest = _write_cache(tmp_path)
    for record in manifest["records"]:
        record["prompt_index"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous from 0"):
        _open_test_dataset(manifest_path)


def test_cache_dataset_rejects_prompt_specific_timestep_drift(tmp_path):
    manifest_path, manifest = _write_cache(tmp_path)
    first_shard = tmp_path / "prompt_0000.safetensors"
    second_shard = tmp_path / "prompt_0001.safetensors"
    second_shard.write_bytes(first_shard.read_bytes())
    second_sha = _sha256(second_shard)
    second_records = []
    for record in manifest["records"]:
        copied = dict(record)
        copied["prompt"] = "A second prompt."
        copied["prompt_index"] = 1
        copied["shard"] = second_shard.name
        copied["shard_sha256"] = second_sha
        if copied["timestep_index"] == 1:
            copied["timestep"] = 101.0
        second_records.append(copied)
    manifest["records"].extend(second_records)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="differs between prompts"):
        _open_test_dataset(manifest_path)


def test_cache_dataset_rejects_manifest_to_shard_timestep_drift(tmp_path):
    manifest_path, manifest = _write_cache(tmp_path)
    manifest["records"][1]["timestep"] = 101.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match shard tensor"):
        _open_test_dataset(manifest_path)


def test_cache_hash_skip_requires_current_preverified_manifest_sha(tmp_path):
    manifest_path, _ = _write_cache(tmp_path)
    kwargs = {
        "expected_latent_shape": (2, 1, 2, 2),
        "expected_sampling_steps": 2,
        "verify_shard_hashes": False,
    }
    with pytest.raises(ValueError, match="preverified manifest SHA256"):
        CFGTeacherTrajectoryDataset(manifest_path, **kwargs)

    dataset = CFGTeacherTrajectoryDataset(
        manifest_path,
        verified_manifest_sha256=_sha256(manifest_path),
        **kwargs,
    )
    assert len(dataset) == 4


class _NeverTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Cached training must not call the teacher.")


class _CachedStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, noisy_image_or_video, conditional_dict, timestep):
        self.calls += 1
        flow = torch.ones_like(noisy_image_or_video) * self.lora_weight
        return flow, noisy_image_or_video - flow


def _cached_model():
    model = CFGGuidanceDistillation.__new__(CFGGuidanceDistillation)
    torch.nn.Module.__init__(model)
    model.args = SimpleNamespace(i2v=False)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.num_frame_per_block = 4
    model.generator_is_causal = False
    model.teacher_guidance_scale = 5.0
    model.cfg_state_source = "teacher_trajectory_cache"
    model.loss_weighting = "relative_guidance"
    model.relative_guidance_epsilon = 1.0e-6
    model.relative_guidance_max_weight = 16.0
    model.generator = _CachedStudent()
    model.real_score = _NeverTeacher()
    return model


def test_cached_loss_is_student_only_and_relative_floor_is_finite():
    model = _cached_model()
    shape = [1, 4, 1, 1, 1]
    noisy = torch.zeros(shape)
    target = torch.full(shape, 2.0)
    energy = torch.tensor([[0.0, 0.0, 1.0e-12, 4.0]])

    loss, logs = model.generator_loss(
        image_or_video_shape=shape,
        conditional_dict={"prompt_embeds": torch.tensor([[2.0]])},
        unconditional_dict=None,
        clean_latent=None,
        cached_noisy_latent=noisy,
        cached_guided_target=target,
        cached_timestep=torch.tensor([500.0]),
        cached_guidance_energy=energy,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.real_score.calls == 0
    assert model.generator.calls == 1
    assert model.generator.lora_weight.grad is not None
    assert torch.isfinite(model.generator.lora_weight.grad)
    assert logs["relative_guidance_median_floor"] >= 1.0e-6
    assert logs["relative_guidance_weight_max"] <= 16.0


def test_cached_loss_rejects_missing_or_nonfinite_inputs():
    model = _cached_model()
    shape = [1, 4, 1, 1, 1]
    common = {
        "image_or_video_shape": shape,
        "conditional_dict": {"prompt_embeds": torch.tensor([[2.0]])},
        "unconditional_dict": None,
        "clean_latent": None,
        "cached_noisy_latent": torch.zeros(shape),
        "cached_guided_target": torch.ones(shape),
        "cached_timestep": torch.tensor([500.0]),
        "cached_guidance_energy": torch.ones(1, 4),
    }
    missing = dict(common)
    missing["cached_guided_target"] = None
    with pytest.raises(ValueError, match="requires cached"):
        model.generator_loss(**missing)

    nonfinite = dict(common)
    nonfinite["cached_guidance_energy"] = torch.tensor(
        [[1.0, 1.0, float("nan"), 1.0]]
    )
    with pytest.raises(ValueError, match="non-finite"):
        model.generator_loss(**nonfinite)


def _load_trainer_with_existing_stubs():
    helper_path = Path(__file__).with_name("test_dmd_nonar_modes.py")
    spec = importlib.util.spec_from_file_location(
        "_dmd_test_helpers_for_cached_cfg",
        helper_path,
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper._load_trainer_with_stubs()


def test_trainer_cached_step_skips_unconditional_encoding_and_forwards_cache():
    Trainer = _load_trainer_with_existing_stubs().Trainer
    trainer = Trainer.__new__(Trainer)
    trainable = torch.nn.Parameter(torch.tensor(1.0))
    encoder_prompts = []
    captured = {}

    class _Model:
        def eval(self):
            return self

        def text_encoder(self, text_prompts):
            encoder_prompts.append(list(text_prompts))
            return {"prompt_embeds": torch.ones(len(text_prompts), 1)}

        def generator_loss(self, **kwargs):
            captured.update(kwargs)
            return trainable.square(), {"cfg_distill_loss": trainable.detach()}

    trainer.uses_critic = False
    trainer.model = _Model()
    trainer.step = 1
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.cfg_state_source = "teacher_trajectory_cache"
    trainer.use_backward_simulation = False
    trainer.gradient_accumulation_steps = 1
    trainer.config = OmegaConf.create(
        {
            "image_or_video_shape": [1, 2, 1, 2, 2],
            "uniform_prompt": False,
            "i2v": False,
            "negative_prompt": "must not be encoded",
            "inference": {
                "multi_shot_sink": False,
                "multi_shot_rope_offset": 0.0,
            },
        }
    )
    batch = {
        "prompts": [["A cached prompt."]],
        "cfg_noisy_latent": torch.zeros(1, 2, 1, 2, 2),
        "cfg_guided_target": torch.ones(1, 2, 1, 2, 2),
        "cfg_timestep": torch.tensor([500.0]),
        "cfg_guidance_energy": torch.ones(1, 2),
    }

    logs = trainer.fwdbwd_one_step(batch, True)

    assert encoder_prompts == [["A cached prompt."]]
    assert captured["unconditional_dict"] is None
    assert "clean_latent" not in captured
    assert torch.equal(
        captured["cached_noisy_latent"], batch["cfg_noisy_latent"]
    )
    assert torch.equal(
        captured["cached_guided_target"], batch["cfg_guided_target"]
    )
    assert trainable.grad is not None
    assert "generator_loss" in logs


def test_cache_config_is_explicit_and_fail_closed():
    def config():
        return OmegaConf.load(REPO_ROOT / "configs" / "wan22_cfg.yaml")
    normalized = normalize_config(config())
    assert normalized.cfg_state_source == "teacher_trajectory_cache"
    assert normalized.cfg_verify_cache_hashes is False
    for field, value, match in [
        ("cfg_state_source", "student_rollout", "cfg_state_source"),
        ("teacher_sampling_steps", 4, "50"),
    ]:
        bad = config()
        bad.algorithm[field] = value
        with pytest.raises(ValueError, match=match):
            normalize_config(bad)
    bad = config()
    bad.data.data_path = "videos"
    with pytest.raises(ValueError, match="JSON manifest"):
        normalize_config(bad)
    bad = config()
    bad.data.load_raw_video = True
    with pytest.raises(ValueError, match="load_raw_video=false"):
        normalize_config(bad)

def test_cached_model_initialization_omits_teacher_and_vae(monkeypatch):
    import model.cfg_distillation as module

    created = []

    class _Backbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(("backbone", kwargs))
            self.model = torch.nn.Linear(1, 1)

        def get_scheduler(self):
            return SimpleNamespace(
                timesteps=torch.tensor([999.0]),
                num_train_timesteps=1000,
                shift=5.0,
            )

    class _TextEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(("text", kwargs))

    class _ForbiddenVAE:
        def __init__(self):
            raise AssertionError("cache training must not instantiate a VAE")

    monkeypatch.setattr(module, "WanDiffusionWrapper", _Backbone)
    monkeypatch.setattr(module, "WanTextEncoder", _TextEncoder)
    args = SimpleNamespace(
        model_kwargs={"model_name": "Wan2.2-TI2V-5B"},
        real_model_kwargs={"model_name": "Wan2.2-TI2V-5B"},
        generator_is_causal=False,
        real_score_is_causal=False,
        cfg_state_source="teacher_trajectory_cache",
    )
    model = CFGGuidanceDistillation.__new__(CFGGuidanceDistillation)
    torch.nn.Module.__init__(model)
    model._initialize_models(args, torch.device("cpu"))

    assert model.real_score is None
    assert model.vae is None
    assert [kind for kind, _ in created] == ["backbone", "text"]
    assert created[1][1] == {
        "model_name": "Wan2.2-TI2V-5B",
        "model_dir": None,
    }
