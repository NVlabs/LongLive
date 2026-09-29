import hashlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from utils.config import normalize_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('recipe,steps,rank,critic', [
    ('wan21_cfg',250,128,False), ('wan22_cfg',250,64,False),
    ('wan21_dmd',600,128,True), ('wan22_dmd',1000,128,True),
])
def test_public_recipes(recipe, steps, rank, critic):
    cfg = normalize_config(OmegaConf.load(ROOT/'configs'/f'{recipe}.yaml'))
    assert cfg.max_iters == steps and cfg.adapter.rank == rank
    assert cfg.adapter.apply_to_critic is critic
    assert cfg.batch_size == cfg.image_or_video_shape[0] == 2
    assert cfg.gradient_accumulation_steps == 1
    assert not any(cfg.get(k,False) for k in ['all_causal','generator_is_causal','fake_score_is_causal','real_score_is_causal','i2v'])


@pytest.mark.parametrize('model', ['Wan2.1-T2V-1.3B', 'Wan2.2-T2V-A14B'])
def test_removed_backbones_rejected(model):
    cfg = OmegaConf.load(ROOT/'configs/wan22_dmd.yaml')
    cfg.model_kwargs.model_name = model
    with pytest.raises(ValueError, match='supports'):
        normalize_config(cfg)


def test_corrupt_cache_is_not_marked_verified(tmp_path, monkeypatch):
    import json
    import utils.cache_validation as validation
    shard = tmp_path/'sample.safetensors'
    shard.write_bytes(b'corrupt')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'records':[{'shard':shard.name, 'shard_sha256':hashlib.sha256(b'expected').hexdigest()}]}))
    monkeypatch.setattr(validation.dist,'get_rank',lambda:0)
    monkeypatch.setattr(validation.dist,'get_world_size',lambda:1)
    monkeypatch.setattr(validation.dist,'all_gather_object',lambda result, item:result.__setitem__(0,item))
    monkeypatch.delenv('CFG_TEACHER_TRAJECTORY_VERIFIED_MANIFEST_SHA256',raising=False)
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        validation.verify_distributed_cache(manifest)


def test_valid_cache_receives_the_actual_manifest_digest(tmp_path, monkeypatch):
    import json
    import os
    import utils.cache_validation as validation
    shard = tmp_path/'sample.safetensors'
    shard.write_bytes(b'valid')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'records':[{'shard':shard.name, 'shard_sha256':hashlib.sha256(b'valid').hexdigest()}]}))
    monkeypatch.setattr(validation.dist,'get_rank',lambda:0)
    monkeypatch.setattr(validation.dist,'get_world_size',lambda:1)
    monkeypatch.setattr(validation.dist,'all_gather_object',lambda result, item:result.__setitem__(0,item))
    monkeypatch.delenv('CFG_TEACHER_TRAJECTORY_VERIFIED_MANIFEST_SHA256',raising=False)
    validation.verify_distributed_cache(manifest)
    assert os.environ['CFG_TEACHER_TRAJECTORY_VERIFIED_MANIFEST_SHA256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize('section,key,value', [
    ('algorithm', 'i2v', True),
    ('algorithm', 'generator_is_causal', True),
    ('algorithm', 'generator_quant', True),
    ('algorithm', 'backward_simulation', False),
    ('algorithm', 'denoising_loss_type', 'x0'),
    ('infra', 'sequence_parallel_size', 2),
    ('inference', 'multi_shot_rope_offset', 1),
    ('training', 'num_training_frames', 16),
])
def test_removed_training_modes_fail_before_model_loading(section, key, value):
    cfg = OmegaConf.load(ROOT / 'configs/wan22_dmd.yaml')
    cfg[section][key] = value
    with pytest.raises(ValueError):
        normalize_config(cfg)
