#!/usr/bin/env python3
"""Cache native 50-step Wan CFG=5 teacher trajectories.

The launcher pins a model/config/shape and assigns the same positive number of
prompts to every distributed rank. One atomic safetensors shard is written per
prompt, followed by a rank-0 manifest only after every shard is gathered and
validated.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prompt_utils import (  # noqa: E402
    FORMAT as PROMPT_PROVENANCE_FORMAT,
    atomic_write_bytes,
    normalize_prompt,
    sha256_file,
    sha256_text,
)


FORMAT = "cfg_teacher_trajectory_safetensors_v1"
MODEL_NAME = "Wan2.2-TI2V-5B"
DEFAULT_MODEL_DIR = Path("wan_models/Wan2.2-TI2V-5B")
EXPECTED_WORLD_SIZE = 8
EXPECTED_PROMPT_COUNT = 64
PROMPTS_PER_RANK = 8
SAMPLING_STEPS = 50
GUIDANCE_SCALE = 5.0
TIMESTEP_SHIFT = 5.0
LATENT_SHAPE = (1, 32, 48, 22, 40)
DEFAULT_SEED = 20260724


def _nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_prompt_contract(
    prompt_path: Path,
    provenance_path: Path,
    *,
    expected_count: int = EXPECTED_PROMPT_COUNT,
) -> tuple[list[str], dict]:
    """Read back and fully cross-check the prepared prompt artifact."""

    prompt_path = prompt_path.resolve()
    provenance_path = provenance_path.resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Prompt provenance does not exist: {provenance_path}"
        )
    prompts = _nonempty_lines(prompt_path)
    if len(prompts) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} non-empty prompts, "
            f"found {len(prompts)}."
        )
    normalized = [normalize_prompt(prompt) for prompt in prompts]
    if any(not value for value in normalized):
        raise ValueError("Prompt file contains an empty normalized prompt.")
    if len(set(normalized)) != expected_count:
        raise ValueError(
            "Prompt file contains duplicates after "
            "NFKC/whitespace/casefold normalization."
        )

    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid prompt provenance JSON: {provenance_path}"
        ) from exc
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != 1
        or provenance.get("format") != PROMPT_PROVENANCE_FORMAT
    ):
        raise ValueError("Unsupported prompt provenance schema.")
    selection = provenance.get("selection")
    records = provenance.get("records")
    outputs = provenance.get("outputs")
    if not isinstance(selection, dict) or int(
        selection.get("selected_count", -1)
    ) != expected_count:
        raise ValueError("Prompt provenance selected_count is inconsistent.")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError("Prompt provenance records are incomplete.")
    if not isinstance(outputs, dict):
        raise ValueError("Prompt provenance outputs block is missing.")
    expected_sha = outputs.get("prompt_file_sha256")
    actual_sha = sha256_file(prompt_path)
    if expected_sha != actual_sha:
        raise ValueError(
            f"Prompt SHA256 mismatch: expected {expected_sha}, got {actual_sha}."
        )

    recorded_prompts = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Prompt provenance record {index} is invalid.")
        if record.get("selected_index") != index:
            raise ValueError(
                f"Prompt provenance selected_index drift at record {index}."
            )
        prompt = record.get("prompt")
        if prompt != prompts[index]:
            raise ValueError(
                f"Prompt provenance text mismatch at record {index}."
            )
        if record.get("prompt_sha256") != sha256_text(prompt):
            raise ValueError(
                f"Prompt provenance SHA256 mismatch at record {index}."
            )
        normalized_prompt = normalize_prompt(prompt)
        if record.get("normalized_prompt") != normalized_prompt:
            raise ValueError(
                f"Prompt provenance normalization mismatch at record {index}."
            )
        recorded_prompts.append(prompt)
    if recorded_prompts != prompts:
        raise AssertionError("Prompt provenance read-back ordering changed.")
    return prompts, provenance


def build_timestep_repeat_factors(
    *,
    sampling_steps: int,
    low_noise_start_index: int,
    low_noise_repeat_factor: int,
) -> dict[str, int]:
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive.")
    if not 0 <= low_noise_start_index <= sampling_steps:
        raise ValueError(
            "low_noise_start_index must be between 0 and sampling_steps."
        )
    if (
        isinstance(low_noise_repeat_factor, bool)
        or low_noise_repeat_factor <= 0
    ):
        raise ValueError("low_noise_repeat_factor must be a positive integer.")
    return {
        str(index): (
            low_noise_repeat_factor
            if index >= low_noise_start_index
            else 1
        )
        for index in range(sampling_steps)
    }


def _autocast_context(dtype: torch.dtype):
    if torch.cuda.is_available() and dtype != torch.float32:
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def collect_teacher_trajectory(
    *,
    teacher,
    conditional_dict: dict,
    unconditional_dict: dict,
    base_scheduler,
    initial_latent: torch.Tensor,
    sampling_steps: int = SAMPLING_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    expected_latent_shape: Sequence[int] | None = None,
    expected_timestep_shift: float = TIMESTEP_SHIFT,
    model_dtype: torch.dtype = torch.bfloat16,
    scheduler_factory: Callable | None = None,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Collect one trajectory as CPU tensors matching the training loader."""

    from model.cfg_distillation import combine_cfg_predictions

    if sampling_steps != SAMPLING_STEPS:
        raise ValueError("Teacher trajectory cache is pinned to 50 steps.")
    if float(guidance_scale) != GUIDANCE_SCALE:
        raise ValueError("Teacher trajectory cache is pinned to CFG=5.")
    expected_latent_shape = tuple(expected_latent_shape or LATENT_SHAPE)
    if tuple(initial_latent.shape) != expected_latent_shape:
        raise ValueError(
            f"Initial latent must have shape {expected_latent_shape}, "
            f"got {tuple(initial_latent.shape)}."
        )
    if initial_latent.dtype != torch.float32:
        raise ValueError("Teacher trajectory integration must use FP32 latents.")
    if not torch.isfinite(initial_latent).all():
        raise ValueError("Initial latent contains non-finite values.")
    if float(base_scheduler.shift) != float(expected_timestep_shift):
        raise ValueError(
            f"Base scheduler shift must be {expected_timestep_shift}, "
            f"got {base_scheduler.shift}."
        )

    if scheduler_factory is None:
        from wan_5b.utils.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler,
        )

        scheduler_factory = FlowUniPCMultistepScheduler
    sample_scheduler = scheduler_factory(
        num_train_timesteps=base_scheduler.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    sample_scheduler.set_timesteps(
        sampling_steps,
        device=initial_latent.device,
        shift=base_scheduler.shift,
    )
    if len(sample_scheduler.timesteps) != sampling_steps:
        raise ValueError(
            f"Scheduler returned {len(sample_scheduler.timesteps)} timesteps; "
            f"expected {sampling_steps}."
        )

    latents = initial_latent
    tensors: dict[str, torch.Tensor] = {}
    timestep_schedule = []
    num_frames = initial_latent.shape[1]
    for timestep_index, timestep_value in enumerate(
        sample_scheduler.timesteps
    ):
        timestep_value = timestep_value.to(
            device=latents.device,
            dtype=torch.float32,
        )
        timestep = timestep_value * torch.ones(
            (1, num_frames),
            device=latents.device,
            dtype=torch.float32,
        )
        with _autocast_context(model_dtype):
            teacher_cond, _ = teacher(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
            )
            teacher_uncond, _ = teacher(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
        guided_target = combine_cfg_predictions(
            teacher_cond,
            teacher_uncond,
            guidance_scale,
        )
        guidance_energy = (
            guided_target.float() - teacher_cond.float()
        ).square().mean(dim=(2, 3, 4))
        values_to_check = {
            "teacher conditional": teacher_cond,
            "teacher unconditional": teacher_uncond,
            "guided target": guided_target,
            "guidance energy": guidance_energy,
        }
        for label, value in values_to_check.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"{label} became non-finite at step {timestep_index}."
                )

        suffix = f"{timestep_index:03d}"
        tensors[f"latent_{suffix}"] = (
            latents[0].to(device="cpu", dtype=torch.bfloat16).contiguous()
        )
        tensors[f"target_{suffix}"] = (
            guided_target[0]
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        tensors[f"timestep_{suffix}"] = (
            timestep_value.detach().reshape(1).to(device="cpu").contiguous()
        )
        tensors[f"guidance_energy_{suffix}"] = (
            guidance_energy[0]
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        timestep_schedule.append(float(timestep_value.item()))

        latents = sample_scheduler.step(
            guided_target.to(dtype=latents.dtype),
            timestep_value,
            latents,
            return_dict=False,
        )[0]
        if latents.dtype != torch.float32:
            raise ValueError(
                f"Scheduler changed latent dtype to {latents.dtype}; "
                "expected FP32."
            )
        if not torch.isfinite(latents).all():
            raise FloatingPointError(
                f"Teacher trajectory became non-finite after step "
                f"{timestep_index}."
            )

    expected_keys = {
        f"{kind}_{index:03d}"
        for index in range(sampling_steps)
        for kind in ("latent", "target", "timestep", "guidance_energy")
    }
    if set(tensors) != expected_keys:
        raise AssertionError("Trajectory tensor key set is incomplete.")
    return tensors, timestep_schedule


def atomic_save_safetensors(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> None:
    """Atomically write a safetensors shard without exposing partial payloads."""

    from safetensors.torch import save_file

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), os.fspath(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def safetensors_header(path: Path) -> dict:
    """Read the safetensors header without materializing the large tensors."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Safetensors file has no header size: {path}")
        header_size = int.from_bytes(raw_size, "little", signed=False)
        if header_size <= 0 or header_size > file_size - 8:
            raise ValueError(
                f"Invalid safetensors header size {header_size}: {path}"
            )
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors header JSON: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    return header


def validate_prompt_shard(
    path: Path,
    *,
    sampling_steps: int,
    latent_shape: Sequence[int],
) -> dict:
    """Validate all tensor keys, shapes, and dtypes from the shard header."""

    header = safetensors_header(path)
    tensor_header = {
        key: value for key, value in header.items() if key != "__metadata__"
    }
    expected_key_count = sampling_steps * 4
    if len(tensor_header) != expected_key_count:
        raise ValueError(
            f"Expected {expected_key_count} tensors in {path}, "
            f"found {len(tensor_header)}."
        )
    expected_shape = list(latent_shape)
    num_frames = int(latent_shape[0])
    for index in range(sampling_steps):
        suffix = f"{index:03d}"
        expected = {
            f"latent_{suffix}": ("BF16", expected_shape),
            f"target_{suffix}": ("F32", expected_shape),
            f"timestep_{suffix}": ("F32", [1]),
            f"guidance_energy_{suffix}": ("F32", [num_frames]),
        }
        for key, (dtype, shape) in expected.items():
            tensor_meta = tensor_header.get(key)
            if not isinstance(tensor_meta, dict):
                raise ValueError(f"Missing tensor {key!r} in {path}.")
            if tensor_meta.get("dtype") != dtype:
                raise ValueError(
                    f"Tensor {key!r} has dtype {tensor_meta.get('dtype')}; "
                    f"expected {dtype}."
                )
            if tensor_meta.get("shape") != shape:
                raise ValueError(
                    f"Tensor {key!r} has shape {tensor_meta.get('shape')}; "
                    f"expected {shape}."
                )
    return header


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _file_provenance(path: Path, *, include_sha256: bool = True) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Provenance input does not exist: {path}")
    payload = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    if include_sha256:
        payload["sha256"] = sha256_file(path)
    return payload


def model_provenance(
    model_index_path: Path,
    t5_path: Path,
    *,
    include_weight_sha256: bool,
) -> dict:
    """Resolve every model-index shard and the exact frozen text encoder."""

    model_index_path = model_index_path.resolve()
    try:
        model_index = json.loads(
            model_index_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model index JSON: {model_index_path}") from exc
    weight_map = (
        model_index.get("weight_map")
        if isinstance(model_index, dict)
        else None
    )
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Model index contains no non-empty weight_map.")
    shard_names = sorted(set(weight_map.values()))
    shards = []
    for shard_name in shard_names:
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Model index contains an invalid shard name.")
        relative = Path(shard_name)
        if relative.is_absolute():
            raise ValueError("Model index shard paths must be relative.")
        shard_path = (model_index_path.parent / relative).resolve()
        try:
            shard_path.relative_to(model_index_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Model shard escapes checkpoint directory: {shard_name}"
            ) from exc
        shards.append(
            _file_provenance(
                shard_path,
                include_sha256=include_weight_sha256,
            )
        )
    return {
        "model_index": _file_provenance(model_index_path),
        "model_weight_shards": shards,
        "text_encoder": _file_provenance(
            t5_path,
            include_sha256=include_weight_sha256,
        ),
        "weight_sha256_included": include_weight_sha256,
    }


def _local_assignment(
    prompts: Sequence[str],
    *,
    rank: int,
    world_size: int,
    prompts_per_rank: int,
) -> list[tuple[int, str]]:
    expected = world_size * prompts_per_rank
    if len(prompts) != expected:
        raise ValueError(
            f"Prompt count {len(prompts)} does not equal "
            f"world_size*prompts_per_rank={expected}."
        )
    start = rank * prompts_per_rank
    stop = start + prompts_per_rank
    return list(enumerate(prompts[start:stop], start=start))


def _validate_complete_gather(
    rank_payloads: Sequence[dict],
    *,
    prompt_count: int,
    sampling_steps: int,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
) -> tuple[list[dict], list[dict], list[float]]:
    if len(rank_payloads) != expected_world_size:
        raise ValueError(
            f"Expected {expected_world_size} rank payloads, "
            f"found {len(rank_payloads)}."
        )
    shards = [
        shard
        for rank_payload in rank_payloads
        for shard in rank_payload["shards"]
    ]
    records = [
        record
        for rank_payload in rank_payloads
        for record in rank_payload["records"]
    ]
    if len(shards) != prompt_count:
        raise ValueError(
            f"Expected {prompt_count} prompt shards, found {len(shards)}."
        )
    if len(records) != prompt_count * sampling_steps:
        raise ValueError(
            f"Expected {prompt_count * sampling_steps} records, "
            f"found {len(records)}."
        )
    prompt_indices = {int(shard["prompt_index"]) for shard in shards}
    if prompt_indices != set(range(prompt_count)):
        raise ValueError("Gathered shard prompt indices are incomplete.")
    cells = {
        (int(record["prompt_index"]), int(record["timestep_index"]))
        for record in records
    }
    expected_cells = {
        (prompt_index, timestep_index)
        for prompt_index in range(prompt_count)
        for timestep_index in range(sampling_steps)
    }
    if cells != expected_cells:
        raise ValueError("Gathered prompt/timestep records are incomplete.")
    schedules = [rank_payload["timestep_schedule"] for rank_payload in rank_payloads]
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("Ranks observed different scheduler timestep lists.")
    shards.sort(key=lambda item: item["prompt_index"])
    records.sort(
        key=lambda item: (item["prompt_index"], item["timestep_index"])
    )
    return shards, records, schedules[0]


def _prompt_records(
    *,
    prompt: str,
    prompt_index: int,
    rank: int,
    noise_seed: int,
    shard_relative: str,
    shard_sha256: str,
    timestep_schedule: Sequence[float],
) -> list[dict]:
    records = []
    for timestep_index, timestep in enumerate(timestep_schedule):
        suffix = f"{timestep_index:03d}"
        records.append(
            {
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "normalized_prompt_sha256": sha256_text(
                    normalize_prompt(prompt)
                ),
                "prompt_index": prompt_index,
                "rank": rank,
                "noise_seed": noise_seed,
                "timestep_index": timestep_index,
                "timestep": float(timestep),
                "shard": shard_relative,
                "shard_sha256": shard_sha256,
                "latent_key": f"latent_{suffix}",
                "target_key": f"target_{suffix}",
                "timestep_key": f"timestep_{suffix}",
                "guidance_energy_key": f"guidance_energy_{suffix}",
            }
        )
    return records


def build_sha256sums(shards: Sequence[Mapping[str, object]]) -> bytes:
    """Render a deterministic checksum file containing shard paths only."""

    checksum_lines = []
    seen_paths = set()
    for shard in sorted(shards, key=lambda item: str(item["path"])):
        relative_path = str(shard["path"])
        digest = str(shard["sha256"])
        path = Path(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not relative_path.startswith("shards/")
            or path.suffix != ".safetensors"
        ):
            raise ValueError(
                "SHA256SUMS accepts only relative safetensors shard paths: "
                f"{relative_path!r}"
            )
        if "\n" in relative_path or "\r" in relative_path:
            raise ValueError(
                "Shard path cannot be represented in SHA256SUMS: "
                f"{relative_path!r}"
            )
        if relative_path in seen_paths:
            raise ValueError(
                f"Duplicate shard path in SHA256SUMS: {relative_path}"
            )
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"Invalid lowercase SHA256 for shard {relative_path}: {digest!r}"
            )
        seen_paths.add(relative_path)
        checksum_lines.append(f"{digest}  {relative_path}")
    if not checksum_lines:
        raise ValueError("Cannot build an empty SHA256SUMS file.")
    return ("\n".join(checksum_lines) + "\n").encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/wan22_cfg.yaml"),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("data/cfg_train_64.txt"),
    )
    parser.add_argument(
        "--prompt-provenance",
        type=Path,
        default=Path(
            "data/cfg_train_64.provenance.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wan22_cfg_cache"),
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Expected normalized config model name.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Exact checkpoint directory loaded by teacher and text encoder.",
    )
    parser.add_argument(
        "--model-index",
        type=Path,
        default=DEFAULT_MODEL_DIR
        / "diffusion_pytorch_model.safetensors.index.json",
    )
    parser.add_argument(
        "--text-encoder",
        type=Path,
        default=DEFAULT_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument(
        "--expected-latent-shape",
        type=int,
        nargs=5,
        default=list(LATENT_SHAPE),
        metavar=("B", "T", "C", "H", "W"),
        help="Fail-closed pin for config.data.image_or_video_shape.",
    )
    parser.add_argument(
        "--expected-timestep-shift",
        type=float,
        default=TIMESTEP_SHIFT,
        help="Fail-closed pin for the native scheduler shift.",
    )
    parser.add_argument("--sampling-steps", type=int, default=SAMPLING_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=EXPECTED_WORLD_SIZE,
    )
    parser.add_argument(
        "--expected-prompt-count",
        type=int,
        default=EXPECTED_PROMPT_COUNT,
    )
    parser.add_argument(
        "--prompts-per-rank",
        type=int,
        default=PROMPTS_PER_RANK,
    )
    parser.add_argument(
        "--low-noise-start-index",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--low-noise-repeat-factor",
        type=int,
        default=2,
        help=(
            "Manifest-only repeat factor for timestep indices at or after "
            "--low-noise-start-index. Tensor payloads are never duplicated."
        ),
    )
    parser.add_argument(
        "--skip-model-weight-hashes",
        action="store_true",
        help=(
            "Record model weight paths and sizes without SHA256. Intended only "
            "for disposable smoke caches; formal caches should keep hashes."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for label, value in (
        ("expected_world_size", args.expected_world_size),
        ("expected_prompt_count", args.expected_prompt_count),
        ("prompts_per_rank", args.prompts_per_rank),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
    if (
        args.expected_world_size * args.prompts_per_rank
        != args.expected_prompt_count
    ):
        raise ValueError(
            "expected_prompt_count must equal "
            "expected_world_size*prompts_per_rank."
        )
    if args.sampling_steps != SAMPLING_STEPS:
        raise ValueError("Formal cache generation requires 50 steps.")
    if args.guidance_scale != GUIDANCE_SCALE:
        raise ValueError("Formal cache generation requires teacher CFG=5.")
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size != args.expected_world_size:
        raise RuntimeError(
            "Launch with the pinned torchrun world size: "
            f"WORLD_SIZE={env_world_size}, expected {args.expected_world_size}."
        )
    model_dir = args.model_dir.resolve()
    expected_model_index = (
        model_dir / "diffusion_pytorch_model.safetensors.index.json"
    ).resolve()
    expected_text_encoder = (
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    ).resolve()
    if args.model_index.resolve() != expected_model_index:
        raise ValueError(
            "--model-index must identify the exact checkpoint path loaded by "
            f"WanDiffusionWrapper: {expected_model_index}."
        )
    if args.text_encoder.resolve() != expected_text_encoder:
        raise ValueError(
            "--text-encoder must identify the exact checkpoint path loaded by "
            f"WanTextEncoder: {expected_text_encoder}."
        )

    prompts, prompt_provenance = load_prompt_contract(
        args.prompt_file,
        args.prompt_provenance,
        expected_count=args.expected_prompt_count,
    )
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}"
        )
    existing_shards = list(shard_dir.glob("*.safetensors"))
    if existing_shards:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing_shards)} existing shards "
            f"in {shard_dir}."
        )
    shard_dir.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    import torch.distributed as dist
    from utils.config import CFG_ONLY_SUPPORTED_MODELS, normalize_config
    from utils.distributed import fsdp_wrap, launch_distributed_job
    from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder

    raw_config = OmegaConf.load(args.config_path)
    raw_config.auto_resume = False
    raw_config.lora_ckpt = None
    raw_config.disable_wandb = True
    raw_config.no_save = True
    raw_config.no_visualize = True
    raw_config.generate_before_train = False
    raw_config.algorithm.cfg_state_source = "data"
    # The source training recipe disables per-rank cache rehashing because its
    # launcher verifies the completed trajectory cache once. During cache
    # construction there is no trajectory cache yet, so restore the required
    # raw-data validation default before normalization.
    raw_config.algorithm.cfg_verify_cache_hashes = True
    raw_config.model_kwargs.model_dir = str(model_dir)
    raw_config.logdir = str(
        output_dir.parent / f".{output_dir.name}_trainer_state"
    )
    config = normalize_config(raw_config)
    config.config_name = f"{args.config_path.stem}_teacher_trajectory_cache"
    model_name = str(config.model_kwargs.model_name)
    latent_shape = tuple(args.expected_latent_shape)
    timestep_shift = float(args.expected_timestep_shift)
    if args.model_name not in CFG_ONLY_SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported CFG-only cache model {args.model_name!r}; expected "
            f"one of {sorted(CFG_ONLY_SUPPORTED_MODELS)}."
        )
    if model_name != args.model_name:
        raise ValueError(
            f"Cache config model {model_name!r} does not match "
            f"--model-name {args.model_name!r}."
        )
    if Path(config.model_kwargs.model_dir).resolve() != model_dir:
        raise ValueError("Student model_dir drifted from --model-dir.")
    if Path(config.real_model_kwargs.model_dir).resolve() != model_dir:
        raise ValueError("Teacher model_dir drifted from --model-dir.")
    if float(config.model_kwargs.timestep_shift) != timestep_shift:
        raise ValueError(
            "Cache config timestep_shift does not match "
            "--expected-timestep-shift."
        )
    if int(config.teacher_sampling_steps) != SAMPLING_STEPS:
        raise ValueError("Cache config must preserve 50 teacher steps.")
    if float(config.teacher_guidance_scale) != GUIDANCE_SCALE:
        raise ValueError("Cache config must preserve teacher CFG=5.")
    if tuple(config.image_or_video_shape) != latent_shape:
        raise ValueError(
            f"Cache config latent shape must be {latent_shape}, "
            f"got {tuple(config.image_or_video_shape)}."
        )

    teacher = None
    text_encoder = None
    try:
        # Cache construction needs only the frozen teacher and text encoder.
        # Avoid allocating the LoRA student, VAE, optimizer, or a training
        # dataloader through ScoreDistillationTrainer.
        launch_distributed_job()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != args.expected_world_size:
            raise ValueError(
                f"Initialized world_size={world_size}; expected "
                f"{args.expected_world_size}."
            )
        device = torch.cuda.current_device()
        model_dtype = (
            torch.bfloat16 if config.mixed_precision else torch.float32
        )
        teacher = WanDiffusionWrapper(
            **config.real_model_kwargs,
            is_causal=bool(config.real_score_is_causal),
        )
        teacher.model.requires_grad_(False)
        if hasattr(teacher.model, "num_frame_per_block"):
            teacher.model.num_frame_per_block = int(
                config.num_frame_per_block
            )
        text_encoder = WanTextEncoder(
            model_name=model_name,
            model_dir=str(model_dir),
        )
        text_encoder.requires_grad_(False)
        teacher = fsdp_wrap(
            teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )
        text_encoder = fsdp_wrap(
            text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )
        teacher.eval()
        text_encoder.eval()
        base_scheduler = teacher.get_scheduler()
        if float(base_scheduler.shift) != timestep_shift:
            raise ValueError(
                "Runtime teacher scheduler shift drifted from the config pin."
            )

        assignment = _local_assignment(
            prompts,
            rank=rank,
            world_size=world_size,
            prompts_per_rank=args.prompts_per_rank,
        )
        with torch.no_grad():
            unconditional_dict = text_encoder(
                text_prompts=[config.negative_prompt]
            )
        local_shards = []
        local_records = []
        local_timestep_schedule = None
        for prompt_index, prompt in assignment:
            with torch.no_grad():
                conditional_dict = text_encoder(
                    text_prompts=[prompt]
                )
            noise_seed = args.seed + prompt_index
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)
            initial_latent = torch.randn(
                latent_shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device)
            tensors, timestep_schedule = collect_teacher_trajectory(
                teacher=teacher,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                base_scheduler=base_scheduler,
                initial_latent=initial_latent,
                sampling_steps=args.sampling_steps,
                guidance_scale=args.guidance_scale,
                expected_latent_shape=latent_shape,
                expected_timestep_shift=timestep_shift,
                model_dtype=model_dtype,
            )
            if local_timestep_schedule is None:
                local_timestep_schedule = timestep_schedule
            elif local_timestep_schedule != timestep_schedule:
                raise ValueError(
                    "Scheduler timestep list changed between prompts."
                )

            shard_relative = (
                Path("shards") / f"prompt_{prompt_index:06d}.safetensors"
            )
            shard_path = output_dir / shard_relative
            atomic_save_safetensors(
                tensors,
                shard_path,
                metadata={
                    "format": FORMAT,
                    "model_name": model_name,
                    "prompt_index": str(prompt_index),
                    "prompt_sha256": sha256_text(prompt),
                    "noise_seed": str(noise_seed),
                    "teacher_sampling_steps": str(args.sampling_steps),
                    "teacher_guidance_scale": str(args.guidance_scale),
                    "timestep_shift": str(timestep_shift),
                    "negative_prompt_sha256": sha256_text(
                        config.negative_prompt
                    ),
                    "latent_shape": json.dumps(list(latent_shape[1:])),
                    "trajectory_integration_dtype": "float32",
                    "latent_storage_dtype": "bfloat16",
                    "target_storage_dtype": "float32",
                    "timestep_storage_dtype": "float32",
                    "guidance_energy_storage_dtype": "float32",
                },
            )
            validate_prompt_shard(
                shard_path,
                sampling_steps=args.sampling_steps,
                latent_shape=latent_shape[1:],
            )
            shard_sha = sha256_file(shard_path)
            local_shards.append(
                {
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_index": prompt_index,
                    "rank": rank,
                    "noise_seed": noise_seed,
                    "path": shard_relative.as_posix(),
                    "sha256": shard_sha,
                    "size_bytes": shard_path.stat().st_size,
                    "tensor_count": len(tensors),
                }
            )
            local_records.extend(
                _prompt_records(
                    prompt=prompt,
                    prompt_index=prompt_index,
                    rank=rank,
                    noise_seed=noise_seed,
                    shard_relative=shard_relative.as_posix(),
                    shard_sha256=shard_sha,
                    timestep_schedule=timestep_schedule,
                )
            )
            del conditional_dict, initial_latent, tensors
            dist.barrier()

        if local_timestep_schedule is None:
            raise AssertionError("Rank received no prompts.")
        local_payload = {
            "rank": rank,
            "prompt_indices": [index for index, _ in assignment],
            "shards": local_shards,
            "records": local_records,
            "timestep_schedule": local_timestep_schedule,
        }
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)

        if rank == 0:
            shards, records, timestep_schedule = _validate_complete_gather(
                gathered,
                prompt_count=args.expected_prompt_count,
                sampling_steps=args.sampling_steps,
                expected_world_size=args.expected_world_size,
            )
            for shard in shards:
                shard_path = (output_dir / shard["path"]).resolve()
                if not shard_path.is_file():
                    raise ValueError(f"Gathered shard is missing: {shard_path}")
                if shard_path.stat().st_size != shard["size_bytes"]:
                    raise ValueError(
                        f"Gathered shard size drifted: {shard_path}"
                    )
            checksum_path = output_dir / "SHA256SUMS"
            checksum_bytes = build_sha256sums(shards)
            atomic_write_bytes(checksum_path, checksum_bytes)
            repeat_factors = build_timestep_repeat_factors(
                sampling_steps=args.sampling_steps,
                low_noise_start_index=args.low_noise_start_index,
                low_noise_repeat_factor=args.low_noise_repeat_factor,
            )
            prompt_provenance_sha256 = sha256_file(
                args.prompt_provenance.resolve()
            )
            model_index_sha256 = sha256_file(args.model_index.resolve())
            negative_prompt_sha256 = sha256_text(
                config.negative_prompt
            )
            timestep_schedule_sha256 = sha256_text(
                json.dumps(
                    timestep_schedule,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            manifest = {
                "schema_version": 1,
                "format": FORMAT,
                "status": "complete",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "git_branch": _git_value("branch", "--show-current"),
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_status_porcelain": _git_value(
                    "status", "--porcelain", "--untracked-files=all"
                ),
                "distributed": {
                    "world_size": world_size,
                },
                "model_name": model_name,
                "teacher_guidance_scale": args.guidance_scale,
                "teacher_sampling_steps": args.sampling_steps,
                "timestep_shift": timestep_shift,
                "latent_shape": list(latent_shape[1:]),
                "timestep_repeat_factors": repeat_factors,
                "inputs": {
                    "config": _file_provenance(args.config_path),
                    "prompt_file": _file_provenance(args.prompt_file),
                    "prompt_provenance": _file_provenance(
                        args.prompt_provenance
                    ),
                    "prompt_selection": prompt_provenance,
                    "model": model_provenance(
                        args.model_index,
                        args.text_encoder,
                        include_weight_sha256=not (
                            args.skip_model_weight_hashes
                        ),
                    ),
                },
                "provenance_pins": {
                    "prompt_provenance_sha256": (
                        prompt_provenance_sha256
                    ),
                    "model_index_sha256": model_index_sha256,
                    "git_commit": _git_value("rev-parse", "HEAD"),
                    "negative_prompt_sha256": negative_prompt_sha256,
                    "scheduler_timestep_schedule_sha256": (
                        timestep_schedule_sha256
                    ),
                    "sha256sums_file": "SHA256SUMS",
                    "sha256sums_sha256": sha256_file(checksum_path),
                },
                "protocol": {
                    "scheduler": "FlowUniPCMultistepScheduler",
                    "timestep_schedule": timestep_schedule,
                    "negative_prompt": config.negative_prompt,
                    "negative_prompt_sha256": negative_prompt_sha256,
                    "noise_seed_base": args.seed,
                    "prompt_count": len(prompts),
                    "prompts_per_rank": args.prompts_per_rank,
                    "rank_assignment": "contiguous blocks by global rank",
                    "trajectory_integration_dtype": "float32",
                    "saved_latent_dtype": "bfloat16",
                    "saved_target_dtype": "float32",
                    "saved_timestep_dtype": "float32",
                    "saved_guidance_energy_dtype": "float32",
                    "guidance_energy_definition": (
                        "per-frame mean((v_cfg - v_cond)^2) over C,H,W"
                    ),
                },
                "summary": {
                    "prompt_count": len(shards),
                    "native_record_count": len(records),
                    "expanded_record_count": sum(
                        repeat_factors[str(record["timestep_index"])]
                        for record in records
                    ),
                    "shard_count": len(shards),
                    "shard_bytes": sum(
                        int(shard["size_bytes"]) for shard in shards
                    ),
                },
                "shards": shards,
                "records": records,
            }
            manifest_bytes = (
                json.dumps(
                    manifest,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            atomic_write_bytes(manifest_path, manifest_bytes)
            persisted = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if persisted != manifest:
                raise RuntimeError("Manifest failed atomic read-back validation.")
            print(
                json.dumps(
                    {
                        "status": "cfg_teacher_trajectory_cache_complete",
                        "manifest": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                        "prompt_count": len(shards),
                        "record_count": len(records),
                        "shard_bytes": manifest["summary"]["shard_bytes"],
                    },
                    sort_keys=True,
                )
            )
        dist.barrier()
    finally:
        if (
            "dist" in locals()
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
