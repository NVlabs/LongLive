import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import cache_cfg_teacher_trajectories as cache
from scripts.download_assets import prepare_prompts
from utils.prompt_utils import normalize_prompt, sha256_file
import hashlib


def _prepare_test_prompts(output):
    raw = ("\n".join(f"Candidate prompt {i}" for i in range(100)) + "\nCandidate prompt 3\n").encode()
    provenance = {"prompt_sha256": hashlib.sha256(raw).hexdigest()}
    prepare_prompts(raw, provenance, output)
    return output / "cfg_train_64.txt", output / "cfg_train_64.provenance.json"


def test_prompt_selection_is_deterministic_unique_and_disjoint(tmp_path):
    first, _ = _prepare_test_prompts(tmp_path / "first")
    second, _ = _prepare_test_prompts(tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    prompts = first.read_text().splitlines()
    heldout = (first.parent / "heldout.txt").read_text().splitlines()
    train = (first.parent / "train.txt").read_text().splitlines()
    assert len(set(map(normalize_prompt, prompts))) == 64
    assert len(heldout) == 16 and len(train) == 84
    assert not set(heldout) & set(train)
    assert set(prompts) <= set(train)


def test_prompt_outputs_and_provenance_cross_check(tmp_path):
    prompt_path, provenance_path = _prepare_test_prompts(tmp_path)
    prompts, provenance = cache.load_prompt_contract(prompt_path, provenance_path)
    assert prompts == [record["prompt"] for record in provenance["records"]]
    assert provenance["outputs"]["prompt_file_sha256"] == sha256_file(prompt_path)
    prompt_path.write_text(prompt_path.read_text().replace(prompts[0], "Tampered prompt", 1))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cache.load_prompt_contract(prompt_path, provenance_path)


def test_prompt_source_checksum_is_required(tmp_path):
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_prompts(b"tampered", {"prompt_sha256": "incorrect"}, tmp_path)


def test_collect_teacher_trajectory_preserves_cfg_order_and_dtypes(
    monkeypatch,
):
    monkeypatch.setattr(cache, "LATENT_SHAPE", (1, 2, 1, 1, 1))
    scheduler_instances = []

    class FakeScheduler:
        def __init__(self, **kwargs):
            assert kwargs == {
                "num_train_timesteps": 1000,
                "shift": 1,
                "use_dynamic_shifting": False,
            }
            scheduler_instances.append(self)

        def set_timesteps(self, steps, device, shift):
            assert steps == 50
            assert shift == 5.0
            self.timesteps = torch.arange(
                steps,
                0,
                -1,
                device=device,
                dtype=torch.float32,
            )

        def step(self, flow, timestep, latents, return_dict):
            assert return_dict is False
            assert flow.dtype == torch.float32
            return (latents + flow * 0.01,)

    call_order = []

    class FakeTeacher:
        def __call__(
            self,
            *,
            noisy_image_or_video,
            conditional_dict,
            timestep,
        ):
            assert tuple(timestep.shape) == (1, 2)
            kind = conditional_dict["kind"]
            call_order.append(kind)
            value = 1.0 if kind == "cond" else 0.0
            return torch.full_like(noisy_image_or_video, value), None

    tensors, schedule = cache.collect_teacher_trajectory(
        teacher=FakeTeacher(),
        conditional_dict={"kind": "cond"},
        unconditional_dict={"kind": "uncond"},
        base_scheduler=SimpleNamespace(
            num_train_timesteps=1000,
            shift=5.0,
        ),
        initial_latent=torch.zeros(
            (1, 2, 1, 1, 1),
            dtype=torch.float32,
        ),
        scheduler_factory=FakeScheduler,
        model_dtype=torch.float32,
    )

    assert len(scheduler_instances) == 1
    assert schedule == [float(value) for value in range(50, 0, -1)]
    assert len(call_order) == 100
    assert all(
        call_order[index : index + 2] == ["cond", "uncond"]
        for index in range(0, len(call_order), 2)
    )
    assert len(tensors) == 200
    assert tensors["latent_000"].dtype == torch.bfloat16
    assert tensors["target_000"].dtype == torch.float32
    assert tensors["timestep_000"].dtype == torch.float32
    assert tensors["guidance_energy_000"].dtype == torch.float32
    assert tuple(tensors["latent_000"].shape) == (2, 1, 1, 1)
    assert tuple(tensors["guidance_energy_000"].shape) == (2,)
    assert torch.equal(
        tensors["target_000"].float(),
        torch.full((2, 1, 1, 1), 5.0),
    )
    assert torch.equal(
        tensors["guidance_energy_000"],
        torch.full((2,), 16.0),
    )
    assert torch.allclose(
        tensors["latent_001"].float(),
        torch.full((2, 1, 1, 1), 0.05),
        atol=2e-4,
        rtol=0,
    )


def test_atomic_shard_has_loader_compatible_header(tmp_path):
    pytest.importorskip("safetensors")
    tensors = {}
    for index in range(2):
        suffix = f"{index:03d}"
        tensors[f"latent_{suffix}"] = torch.zeros(
            2, 1, 2, 2, dtype=torch.bfloat16
        )
        tensors[f"target_{suffix}"] = torch.ones(
            2, 1, 2, 2, dtype=torch.float32
        )
        tensors[f"timestep_{suffix}"] = torch.tensor(
            [999.0 - index], dtype=torch.float32
        )
        tensors[f"guidance_energy_{suffix}"] = torch.ones(
            2, dtype=torch.float32
        )
    shard = tmp_path / "shards" / "prompt_000000.safetensors"
    cache.atomic_save_safetensors(
        tensors,
        shard,
        metadata={"format": cache.FORMAT},
    )
    header = cache.validate_prompt_shard(
        shard,
        sampling_steps=2,
        latent_shape=(2, 1, 2, 2),
    )

    assert header["__metadata__"]["format"] == cache.FORMAT
    assert len(header) == 9
    assert not list(shard.parent.glob("*.tmp"))
    assert len(sha256_file(shard)) == 64


def test_repeat_factors_and_gather_contract():
    repeats = cache.build_timestep_repeat_factors(
        sampling_steps=50,
        low_noise_start_index=35,
        low_noise_repeat_factor=2,
    )
    assert set(repeats) == {str(index) for index in range(50)}
    assert sum(repeats.values()) == 65
    assert repeats["34"] == 1
    assert repeats["35"] == 2

    rank_payloads = []
    schedule = [999.0, 92.0]
    for rank in range(8):
        shard = {
            "prompt_index": rank,
            "path": f"shards/prompt_{rank:06d}.safetensors",
        }
        records = cache._prompt_records(
            prompt=f"prompt {rank}",
            prompt_index=rank,
            rank=rank,
            noise_seed=100 + rank,
            shard_relative=shard["path"],
            shard_sha256=str(rank) * 64,
            timestep_schedule=schedule,
        )
        rank_payloads.append(
            {
                "shards": [shard],
                "records": records,
                "timestep_schedule": schedule,
            }
        )
    shards, records, read_schedule = cache._validate_complete_gather(
        rank_payloads,
        prompt_count=8,
        sampling_steps=2,
    )
    assert len(shards) == 8
    assert len(records) == 16
    assert read_schedule == schedule


def test_sha256sums_is_sorted_and_contains_only_relative_shards():
    payload = cache.build_sha256sums(
        [
            {
                "path": "shards/prompt_000001.safetensors",
                "sha256": "b" * 64,
            },
            {
                "path": "shards/prompt_000000.safetensors",
                "sha256": "a" * 64,
            },
        ]
    )
    assert payload.decode("utf-8").splitlines() == [
        f"{'a' * 64}  shards/prompt_000000.safetensors",
        f"{'b' * 64}  shards/prompt_000001.safetensors",
    ]

    with pytest.raises(ValueError, match="relative safetensors shard paths"):
        cache.build_sha256sums(
            [{"path": "manifest.json", "sha256": "a" * 64}]
        )
    with pytest.raises(ValueError, match="Invalid lowercase SHA256"):
        cache.build_sha256sums(
            [{
                "path": "shards/prompt_000000.safetensors",
                "sha256": "NOT-A-DIGEST",
            }]
        )


def test_formal_cli_defaults_pin_cache_path_and_low_noise_repetition():
    args = cache.build_parser().parse_args([])
    assert args.output_dir == Path("data/wan22_cfg_cache")
    assert args.expected_world_size == 8
    assert args.expected_prompt_count == 64
    assert args.prompts_per_rank == 8
    assert args.sampling_steps == 50
    assert args.guidance_scale == 5.0
    assert args.low_noise_start_index == 35
    assert args.low_noise_repeat_factor == 2
    assert args.skip_model_weight_hashes is False


def test_model_provenance_resolves_index_shards(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_a = model_dir / "a.safetensors"
    shard_b = model_dir / "b.safetensors"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    model_index = model_dir / "model.index.json"
    model_index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": shard_a.name,
                    "layer.1": shard_b.name,
                }
            }
        ),
        encoding="utf-8",
    )
    text_encoder = model_dir / "t5.pth"
    text_encoder.write_bytes(b"t5")

    provenance = cache.model_provenance(
        model_index,
        text_encoder,
        include_weight_sha256=True,
    )
    assert provenance["model_index"]["sha256"] == sha256_file(model_index)
    assert [Path(item["path"]).name for item in provenance["model_weight_shards"]] == [
        "a.safetensors",
        "b.safetensors",
    ]
    assert all("sha256" in item for item in provenance["model_weight_shards"])
    assert provenance["text_encoder"]["sha256"] == sha256_file(text_encoder)
