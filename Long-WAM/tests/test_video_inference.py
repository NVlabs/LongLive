# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""Video demo contracts on CPU; no model weights, network, CUDA or generated artifacts."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("video_demo", ROOT / "stage1/infer.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def arguments(**kwargs):
    return SimpleNamespace(**dict(dict(image=None, prompt=None, example=None, latent_frames=None, seed=None), **kwargs))


def test_complete_examples_have_exact_images_seeds_and_lengths():
    jobs = demo.select_jobs(arguments(example="all"))
    assert [job["name"] for job in jobs] == ["short_00", "short_03", "short_04", "short_05", "long_00", "long_03", "long_04"]
    for job in jobs:
        assert job["noise_seed"] == 20260711 + job["original_slot"]
        assert job["raw_frames"] == 4 * job["latent_frames"] - 3
        assert job["latent_frames"] == (32 if job["name"].startswith("short") else 64)
        assert Path(job["image"] + ".license").is_file()
        assert "CC-BY-4.0" in Path(job["image"] + ".license").read_text()


def test_list_has_no_ml_imports_or_output_writes(tmp_path):
    # -S excludes all site-packages: listing needs only Python's standard library.
    result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "stage1/infer.py"), "--list"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert len(result.stdout.splitlines()) == 7
    assert list(tmp_path.iterdir()) == []


def test_dry_run_is_portable_and_preserves_sampling_recipe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    demo.main(["--example", "all", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["jobs"]) == 7
    assert plan["config"]["inference"]["sampling_steps"] == 50
    assert plan["config"]["inference"]["guidance_scale"] == 3.0
    assert plan["config"]["model_kwargs"]["timestep_shift"] == 5.0
    assert list(tmp_path.iterdir()) == []


def test_custom_image_contract_and_guards(tmp_path):
    image = tmp_path / "rgb.png"
    Image.new("RGB", (1280, 704)).save(image)
    job = demo.select_jobs(arguments(image=image, prompt="Fold the cloth", latent_frames=64, seed=7))[0]
    assert (job["name"], job["raw_frames"], job["noise_seed"]) == ("custom", 253, 7)
    for overrides in ({"prompt": ""}, {"example": "short_00"}, {"latent_frames": 3}, {"seed": -1}):
        with pytest.raises(ValueError):
            demo.select_jobs(arguments(**dict(dict(image=image, prompt="test"), **overrides)))
    with pytest.raises(ValueError, match="changed"):
        demo.check_image(image, "wrong-digest")
    Image.new("RGBA", (1280, 704)).save(image)
    with pytest.raises(ValueError, match="RGB"):
        demo.check_image(image)


@pytest.mark.parametrize("overrides", [{"prompt": "test"}, {"seed": 5}, {"example": "missing"}])
def test_invalid_example_options(overrides):
    with pytest.raises(ValueError):
        demo.select_jobs(arguments(**overrides))


def test_sampling_config_guards(tmp_path):
    cfg = demo.load_config(demo.CONFIG)
    for path, value in (("inference.sampling_steps", 0), ("inference.guidance_scale", float("nan")),
                        ("inference.streaming_vae", True), ("model_kwargs.initialize_from_config", False)):
        changed = OmegaConf.create(OmegaConf.to_container(cfg))
        OmegaConf.update(changed, path, value)
        file = tmp_path / "config.yaml"
        OmegaConf.save(changed, file)
        with pytest.raises(ValueError):
            demo.load_config(file)


def test_online_weight_loading_is_strict_and_accepts_original_layouts():
    generator = torch.nn.Module()
    generator.model = torch.nn.Linear(2, 1)
    raw = generator.model.state_dict()
    wrapped = generator.state_dict()
    for state in (raw, wrapped, {"generator": wrapped}, {"model": wrapped},
                  {"generator": {"_fsdp_wrapped_module." + key: value for key, value in wrapped.items()}}):
        demo.load_weights(generator, state)
    with pytest.raises(ValueError, match="EMA"):
        demo.generator_state({"generator_ema": wrapped})
    with pytest.raises(ValueError, match="Duplicate"):
        demo.generator_state({"model.weight": 1, "_fsdp_wrapped_module.model.weight": 2})
    with pytest.raises(RuntimeError):
        demo.load_weights(generator, {"unrelated": torch.zeros(1)})


def test_generation_matches_original_preprocessing_and_cleans_cache_on_failure(tmp_path):
    path = tmp_path / "image.png"
    pixels = np.full((704, 1280, 3), 129, dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    job = demo.select_jobs(arguments(image=path, prompt="Move the object", latent_frames=8, seed=123))[0]
    seen, clears = [], []

    def encode(image):
        expected = (torch.tensor(129).float() / 255 * 2 - 1).half().bfloat16()
        assert image.shape == (1, 3, 1, 704, 1280)
        assert image.dtype == torch.bfloat16 and torch.all(image == expected)
        return torch.zeros((1, 1, 48, 44, 80), dtype=torch.bfloat16)

    def infer(**kwargs):
        seen.append(kwargs)
        return torch.zeros(1)

    pipe = SimpleNamespace(vae=SimpleNamespace(encode_to_latent=encode),
                           clear_cache=lambda: clears.append(True), inference=infer)
    demo.generate_one(pipe, job, "cpu")
    demo.generate_one(pipe, job, "cpu")
    assert len(clears) == 4
    assert seen[0]["text_prompts"] == [["Move the object"]]
    assert seen[0]["noise"].shape == (1, 8, 48, 44, 80)
    assert torch.equal(seen[0]["noise"], seen[1]["noise"])
    assert not seen[0]["initial_latent"].requires_grad

    def fail(**kwargs):
        raise RuntimeError("synthetic failure")

    pipe.inference = fail
    with pytest.raises(RuntimeError, match="synthetic failure"):
        demo.generate_one(pipe, job, "cpu")
    assert len(clears) == 6


def test_mp4_publish_does_not_overwrite_or_leave_intermediates(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    path = tmp_path / "output/demo.mp4"
    demo.save_video(path, frames)
    assert path.stat().st_size > 0 and list(path.parent.iterdir()) == [path]
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        demo.save_video(path, frames)
    assert path.read_bytes() == saved

    def fail(*args, **kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(imageio, "mimwrite", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        demo.save_video(path.parent / "second.mp4", frames)
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("stage", ["S", "M", "L"])
@pytest.mark.parametrize("mode", ["train", "validation"])
def test_video_launches_resolve_public_environment_names(stage, mode, tmp_path, monkeypatch):
    monkeypatch.setenv("LONGWAM_VIDEO_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LONGWAM_VIDEO_INDEX_CACHE", str(tmp_path / "index"))
    spec = importlib.util.spec_from_file_location("video_launch", ROOT / "stage1/run.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    args = SimpleNamespace(mode=mode, stage=stage, output=tmp_path / "output", assets_root=tmp_path,
                           generator=tmp_path / "generator.pt", generator_sha256="unspecified",
                           resume_step=0, stop_step=None, training_step=1 if mode == "validation" else None)
    config, _ = launch.prepare(args)
    resolved = json.dumps(OmegaConf.to_container(config, resolve=True))
    assert str(tmp_path / "data") in resolved and str(args.generator) in resolved
    assert not args.output.exists()


def test_renamed_manifest_reader_builds_and_reuses_real_index(tmp_path):
    spec = importlib.util.spec_from_file_location("robot_video_data", demo.CORE / "utils/dataset.py")
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    manifest = tmp_path / "manifests/S/train.jsonl"
    manifest.parent.mkdir(parents=True)
    row = dict(
        bucket="F32", conditioning=data.ROBOT_VIDEO_CONDITIONING,
        contract_version=data.ROBOT_VIDEO_ROW_CONTRACT_VERSION,
        first_frame_source="decoded frame 0 of video", id="a" * 32,
        instruction="Synthetic instruction", source_dataset="synthetic", source_id="test",
        source_manifest_id="test", source_name="synthetic", split="train", stage="S",
        temporal_padding=False, temporal_stretching=False, tier="synthetic",
        valid_duration_seconds=125 / 24, valid_latent_frames=32, valid_raw_frames=125,
        video=str(tmp_path / "not-decoded.mp4"),
    )
    manifest.write_text(json.dumps(row) + "\n")
    body = dict(
        contract_version=data.ROBOT_VIDEO_RECEIPT_CONTRACT_VERSION,
        status=data.ROBOT_VIDEO_ACCEPTED_RECEIPT_STATUS, conditioning=data.ROBOT_VIDEO_CONDITIONING,
        media_policy=dict(target_fps=24, temporal_compression_ratio=4),
        stage_policy={key: list(value) for key, value in data.ROBOT_VIDEO_STAGE_BUCKETS.items()},
        outputs=[dict(path=str(manifest), stage="S", split="train", record_count=1,
                      size_bytes=manifest.stat().st_size, sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())],
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(body, receipt_sha256=data._canonical_json_sha256(body))))
    args = dict(manifest_path=manifest, video_size=(384, 640), total_frames=253,
                expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                index_cache_dir=tmp_path / "index")
    first = data.RobotVideoManifestDataset(**args)
    try:
        assert len(first) == 1 and first._read_row(0) == row
        assert first._index_header["index_version"] == "longlive-robot-video-jsonl-offsets-v1"
        assert not first._index_header_matches(dict(first._index_header, index_version="old-cache"))
        index = first.index_path
        stamp = index.stat().st_mtime_ns
        second = data.RobotVideoManifestDataset(**args)
        try:
            assert index.stat().st_mtime_ns == stamp
            assert second.distributed_index_contract() == first.distributed_index_contract()
        finally:
            second._close_mmaps()
    finally:
        first._close_mmaps()
    assert not list(index.parent.glob(".*"))


def test_head_sharding_transforms_resolve_renamed_helpers(monkeypatch):
    spec = importlib.util.spec_from_file_location("video_heads", ROOT / "stage1/runtime/head_sharding.py")
    heads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(heads)

    def original(x, kv_cache):
        qkv_fn = lambda value: (value, value, value)
        q, k, v = qkv_fn(x)
        x = q + k + v
        x = x.flatten(2)
        return x

    monkeypatch.setattr(heads, "partition_heads", lambda q, k, v, cache: (q, k, v))
    monkeypatch.setattr(heads, "gather_heads", lambda x: x * 2)
    transformed = heads.transformed_forward(original)
    assert torch.equal(transformed(torch.ones(1, 2, 3, 4), {}), torch.full((1, 2, 12), 6.0))
    for name, digest in heads.PINS.items():
        assert hashlib.sha256((demo.CORE / name).read_bytes()).hexdigest() == digest
