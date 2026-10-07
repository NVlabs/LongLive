# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: The Self-Forcing contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: trainer/diffusion.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/trainer/diffusion.py
# Source: https://github.com/guandeh17/Self-Forcing
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
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

# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import json
import logging
import random
import shutil
import types
from collections.abc import Mapping, Sequence as SequenceABC
from pathlib import Path

import numpy as np
from model import CausalDiffusion
from wan_5b.distributed.sp_training import SequenceParallelHelper
from utils.dataset import (
    MultiTextConcatDataset,
    MultiVideoConcatDataset,
    build_distributed_sampler,
    cycle,
    eval_collate_fn,
    multi_video_collate_fn,
    resolve_resume_data_cursor,
)
try:
    # Kept separate so lightweight checkpoint-only test harnesses that stub the
    # historical dataset API can still import this module.
    from utils.dataset import RobotVideoManifestDataset
except ImportError:  # pragma: no cover - exercised only by legacy import stubs
    RobotVideoManifestDataset = None
from utils.evaluation import (
    deterministic_evaluation_seed,
    evaluation_artifact_run_modes,
    evaluation_group_specs,
    evaluation_run_modes,
    fixed_evaluation_log_key,
    fixed_evaluation_writer,
    require_equal_evaluation_value,
    select_fixed_evaluation_subset,
    synchronize_evaluation_failure,
)
from utils.config import section_get, wan_default_config
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from torchvision.io import write_video
from utils.distributed import (
    EMA_FSDP,
    FSDP,
    TRAINING_PROCESS_GROUP_TIMEOUT,
    barrier,
    fsdp_wrap,
    launch_distributed_job,
)
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)


GENERATOR_CKPT_LOAD_MODES = frozenset({"generator_only", "resume"})
RESUME_CONTRACT_VERSION = "longlive-same-stage-resume-contract-v2"
RANK_STATE_VERSION = "longlive-same-stage-rank-state-v2"
COMPLETION_MARKER_VERSION = "longlive-checkpoint-completion-marker-v2"
RESUME_CONTRACT_FILENAME = "resume_contract.json"
COMPLETION_MARKER_FILENAME = "_SUCCESS.json"
ER_RESUME_POLICY = "sp_canonical_replication_non_bitwise"
RESUME_CRITICAL_CODE_PATHS = (
    "train.py",
    "trainer/diffusion.py",
    "trainer/sp_helper.py",
    "model/diffusion.py",
    "utils/config.py",
    "utils/dataset.py",
    "utils/error_buffer.py",
    "utils/distributed.py",
    "utils/i2v_conditioning.py",
    "utils/wan_5b_wrapper.py",
    "wan_5b/distributed/sp_training.py",
    "wan_5b/distributed/sequence_parallel.py",
)


class ResumeContractError(RuntimeError):
    """A same-stage checkpoint is incomplete or belongs to another run."""


def _canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sealed_json_document(body):
    document = dict(body)
    document["contract_sha256"] = _canonical_json_sha256(body)
    return document


def _validate_sealed_json_document(document, *, context):
    if not isinstance(document, Mapping):
        raise ResumeContractError(f"{context} must be a JSON object")
    document = dict(document)
    observed = document.pop("contract_sha256", None)
    expected = _canonical_json_sha256(document)
    if observed != expected:
        raise ResumeContractError(
            f"{context} canonical SHA-256 mismatch: {observed!r} != {expected}"
        )
    return dict(document, contract_sha256=observed)


def _regular_nonempty_file(path, *, context):
    path = Path(path)
    try:
        stat_result = path.lstat()
    except FileNotFoundError as exc:
        raise ResumeContractError(f"missing {context}: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ResumeContractError(
            f"{context} must be a regular non-symlink file: {path}"
        )
    if stat_result.st_size <= 0:
        raise ResumeContractError(f"{context} is empty: {path}")
    return stat_result


def _sha256_file(path, chunk_bytes=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_document(path, *, context):
    _regular_nonempty_file(path, context=context)
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeContractError(f"cannot read {context} {path}: {exc}") from exc
    return _validate_sealed_json_document(value, context=context)


def _write_sealed_json(path, body):
    """Create one fsynced JSON file inside an unpublished staging directory."""
    path = Path(path)
    document = _sealed_json_document(body)
    rendered = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return document


def _rank_state_filename(global_rank):
    return f"rank_state_{int(global_rank):05d}.pt"


def _canonical_buffer_filename(stem, sp_rank):
    return f"{stem}_sp{int(sp_rank)}.pt"


def _checkpoint_required_filenames(
    world_size,
    sp_size,
    *,
    error_buffer_required,
    noise_error_buffer_required,
):
    """Return bounded rank-RNG plus canonical-per-SP checkpoint payloads."""
    world_size = int(world_size)
    sp_size = int(sp_size)
    if world_size <= 0 or sp_size <= 0 or world_size % sp_size:
        raise ValueError(
            f"invalid world/SP sizes: world_size={world_size}, sp_size={sp_size}"
        )
    required = ["model.pt"]
    required.extend(
        _rank_state_filename(rank) for rank in range(world_size)
    )
    if error_buffer_required:
        required.extend(
            _canonical_buffer_filename("error_buffer", rank)
            for rank in range(sp_size)
        )
    if noise_error_buffer_required:
        required.extend(
            _canonical_buffer_filename("noise_error_buffer", rank)
            for rank in range(sp_size)
        )
    return tuple(required)


def _capture_rng_state(cuda_device=None):
    """Capture every process-local RNG stream used by LongLive training."""
    numpy_state = np.random.get_state()
    state = {
        "python_random": random.getstate(),
        "numpy_random": {
            "bit_generator": str(numpy_state[0]),
            # PyTorch cannot wrap NumPy uint32 directly. Values are losslessly
            # represented as int64 and cast back to uint32 during restore.
            "keys": torch.tensor(
                numpy_state[1].astype(np.int64), dtype=torch.int64
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
        "cuda_device_index": None,
    }
    if torch.cuda.is_available():
        device_index = (
            torch.cuda.current_device()
            if cuda_device is None
            else int(cuda_device)
        )
        state["cuda_device_index"] = int(device_index)
        state["torch_cuda"] = torch.cuda.get_rng_state(device_index).clone()
    return state


def _restore_rng_state(state, cuda_device=None):
    """Strictly restore a rank's Python, NumPy, CPU and active-CUDA RNGs."""
    if not isinstance(state, Mapping):
        raise ResumeContractError("rank RNG state must be a mapping")
    expected = {
        "python_random",
        "numpy_random",
        "torch_cpu",
        "torch_cuda",
        "cuda_device_index",
    }
    if set(state) != expected:
        raise ResumeContractError(
            "rank RNG fields changed: "
            f"missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    numpy_state = state["numpy_random"]
    if not isinstance(numpy_state, Mapping):
        raise ResumeContractError("NumPy RNG state must be a mapping")
    numpy_expected = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if set(numpy_state) != numpy_expected:
        raise ResumeContractError("NumPy RNG state fields changed")
    keys = numpy_state["keys"]
    cpu_state = state["torch_cpu"]
    if not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64:
        raise ResumeContractError("NumPy RNG keys must be a torch.int64 tensor")
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise ResumeContractError("torch CPU RNG state must be a uint8 tensor")

    random.setstate(state["python_random"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(cpu_state.cpu())

    cuda_state = state["torch_cuda"]
    saved_device = state["cuda_device_index"]
    if torch.cuda.is_available():
        if cuda_state is None or saved_device is None:
            raise ResumeContractError(
                "CUDA is active but the checkpoint has no CUDA RNG state"
            )
        current_device = (
            torch.cuda.current_device()
            if cuda_device is None
            else int(cuda_device)
        )
        if int(saved_device) != current_device:
            raise ResumeContractError(
                "CUDA device index changed across resume: "
                f"{saved_device} != {current_device}"
            )
        if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8:
            raise ResumeContractError("torch CUDA RNG state must be a uint8 tensor")
        torch.cuda.set_rng_state(cuda_state.cpu(), device=current_device)
    elif cuda_state is not None or saved_device is not None:
        raise ResumeContractError(
            "checkpoint contains CUDA RNG state but CUDA is unavailable"
        )


def _validate_distributed_dataset_index(dataset):
    """Fail if ranks disagree on the integer-index to sample-ID mapping."""
    contract_method = getattr(dataset, "distributed_index_contract", None)
    if callable(contract_method):
        # Large frozen JSONL manifests provide a sidecar-backed contract.  Do
        # not turn 1.8M IDs into a Python list independently on every rank.
        local = contract_method()
        if not isinstance(local, Mapping):
            raise RuntimeError(
                "Training dataset distributed_index_contract must return a mapping"
            )
        local = dict(local)
        dataset_count = len(dataset)
    else:
        sample_ids = [folder.name for folder in dataset.folders]
        if sample_ids != sorted(sample_ids):
            raise RuntimeError("Training dataset sample IDs are not in canonical order")
        local = {
            "count": len(sample_ids),
            "ids_sha256": hashlib.sha256(
                b"".join(
                    f"{sample_id}\n".encode("utf-8")
                    for sample_id in sample_ids
                )
            ).hexdigest(),
        }
        dataset_count = len(sample_ids)
    if local.get("count") != dataset_count:
        raise RuntimeError(
            "Training dataset index contract count disagrees with __len__: "
            f"{local.get('count')} != {dataset_count}"
        )
    gathered = [local]
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    disagreements = [
        {"rank": rank, "contract": contract}
        for rank, contract in enumerate(gathered)
        if contract != local
    ]
    if disagreements:
        raise RuntimeError(
            "Distributed training dataset index mapping differs across ranks: "
            f"local={local}, disagreements={disagreements}"
        )
    return local


def _wandb_resume_mode(run_id):
    """Append to a persisted W&B run ID across Slurm restarts."""
    return "allow" if run_id else None


def _validate_generator_ckpt_load_mode(mode):
    """Validate the explicit generator-checkpoint policy before any file I/O."""
    normalized = str(mode or "resume").strip().lower()
    if normalized not in GENERATOR_CKPT_LOAD_MODES:
        allowed = ", ".join(sorted(GENERATOR_CKPT_LOAD_MODES))
        raise ValueError(
            f"generator_ckpt_load_mode must be one of {{{allowed}}}, got "
            f"{mode!r}."
        )
    return normalized


def _resolve_checkpoint_load_plan(
    *,
    auto_resume,
    output_path,
    generator_ckpt,
    generator_ckpt_load_mode,
    find_latest_checkpoint,
):
    """Resolve checkpoint path, provenance, and effective state-load policy.

    A checkpoint discovered inside the current logdir is always a full resume.
    The explicit ``generator_ckpt_load_mode`` applies only to the external
    initialization checkpoint, so ``generator_only`` cannot accidentally
    disable recovery after an interruption in the new phase.
    """
    explicit_mode = _validate_generator_ckpt_load_mode(
        generator_ckpt_load_mode
    )
    if auto_resume and output_path:
        latest = find_latest_checkpoint(output_path)
        if latest is not None:
            return latest, "auto_resume", "resume"
    if generator_ckpt:
        return str(generator_ckpt), "generator_ckpt", explicit_mode
    return None, None, None


def _torch_load_checkpoint(checkpoint_path):
    """Load a tensor-only checkpoint without eagerly paging every storage.

    Modern zip checkpoints support mmap, which is important when an 80 GB full
    training envelope is used only for its generator entry.  Fall back only for
    the legacy pre-zip torch serialization format.
    """
    checkpoint_path = os.fspath(checkpoint_path)
    kwargs = {
        "map_location": "cpu",
        "weights_only": True,
    }
    try:
        return torch.load(checkpoint_path, mmap=True, **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "mmap can only be used" not in message:
            raise
        return torch.load(checkpoint_path, **kwargs)


def _checkpoint_directory_paths(output_path, step):
    """Return hidden staging and atomically-published checkpoint directories."""
    suffix = f"checkpoint_model_{int(step):06d}"
    final_dir = os.path.join(output_path, suffix)
    staging_dir = os.path.join(output_path, f".{suffix}.incomplete")
    return staging_dir, final_dir


def _prepare_checkpoint_staging_directory(staging_dir, final_dir):
    """Create a clean hidden staging directory without touching a final save."""
    if os.path.exists(final_dir):
        raise FileExistsError(
            f"Refusing to overwrite completed checkpoint: {final_dir}"
        )
    if os.path.lexists(staging_dir):
        if os.path.isdir(staging_dir) and not os.path.islink(staging_dir):
            shutil.rmtree(staging_dir)
        else:
            os.remove(staging_dir)
    os.makedirs(staging_dir, exist_ok=False)


def _validate_resume_code_identity(code):
    if not isinstance(code, Mapping):
        raise ResumeContractError("resume critical-code identity is missing")
    expected = set(RESUME_CRITICAL_CODE_PATHS)
    observed = set(code)
    if observed != expected:
        raise ResumeContractError(
            "resume critical-code inventory changed: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    for relative_path, digest in code.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ResumeContractError(
                f"resume critical-code SHA-256 is invalid for {relative_path}"
            )


def _validate_resume_recipe_shape(training):
    if not isinstance(training, Mapping):
        raise ResumeContractError("resume training recipe is missing")
    required_nested_fields = {
        "algorithm": {
            "causal",
            "teacher_forcing",
            "i2v",
            "independent_first_frame",
            "num_train_timestep",
            "denoising_loss_type",
            "noise_augmentation_max_timestep",
        },
        "attention_and_schedule": {
            "local_attn_size",
            "sink_size",
            "timestep_shift",
            "t_scale",
            "rope_method",
        },
        "distributed_precision": {
            "mixed_precision",
            "sharding_strategy",
            "gradient_checkpointing",
            "vae_halo_latents",
        },
        "inference_conditioning": {"negative_prompt", "guidance_scale"},
    }
    for section, required_fields in required_nested_fields.items():
        value = training.get(section)
        if not isinstance(value, Mapping):
            raise ResumeContractError(
                f"resume training recipe section {section} is missing"
            )
        missing = required_fields - set(value)
        if missing:
            raise ResumeContractError(
                f"resume training recipe section {section} is missing "
                f"{sorted(missing)}"
            )


def _validate_resume_contract_shape(contract, *, expected_step=None):
    if contract.get("contract_version") != RESUME_CONTRACT_VERSION:
        raise ResumeContractError(
            "unsupported resume contract version: "
            f"{contract.get('contract_version')!r}"
        )
    if contract.get("status") != "complete_same_stage_state":
        raise ResumeContractError("resume contract status is not complete")
    step = contract.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ResumeContractError(f"invalid resume contract step: {step!r}")
    if expected_step is not None and step != int(expected_step):
        raise ResumeContractError(
            f"resume contract step {step} != directory step {expected_step}"
        )

    topology = contract.get("topology")
    if not isinstance(topology, Mapping):
        raise ResumeContractError("resume topology is missing")
    try:
        world_size = int(topology["world_size"])
        sp_size = int(topology["sequence_parallel_size"])
        dp_size = int(topology["data_parallel_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResumeContractError("resume topology is invalid") from exc
    if min(world_size, sp_size, dp_size) <= 0 or sp_size * dp_size != world_size:
        raise ResumeContractError(
            "resume topology does not satisfy world_size = SP * DP"
        )

    state = contract.get("state")
    if not isinstance(state, Mapping):
        raise ResumeContractError("resume state requirements are missing")
    expected_rank_files = [
        _rank_state_filename(rank) for rank in range(world_size)
    ]
    if state.get("model_filename") != "model.pt":
        raise ResumeContractError("resume model filename changed")
    if state.get("rank_state_version") != RANK_STATE_VERSION:
        raise ResumeContractError("resume rank-state version changed")
    if state.get("rank_state_filenames") != expected_rank_files:
        raise ResumeContractError(
            "resume contract does not enumerate every global-rank state shard"
        )
    for field in (
        "optimizer_required",
        "ema_required",
        "error_buffer_required",
        "noise_error_buffer_required",
    ):
        if type(state.get(field)) is not bool:
            raise ResumeContractError(f"resume state flag {field} must be boolean")
    if state["optimizer_required"] is not True:
        raise ResumeContractError("same-stage resume must require optimizer state")
    if state.get("er_resume_policy") != ER_RESUME_POLICY:
        raise ResumeContractError(
            "same-stage resume must explicitly use the bounded canonical-SP "
            f"ER policy {ER_RESUME_POLICY!r}"
        )
    expected_canonical_files = []
    if state["error_buffer_required"]:
        expected_canonical_files.extend(
            _canonical_buffer_filename("error_buffer", rank)
            for rank in range(sp_size)
        )
    if state["noise_error_buffer_required"]:
        expected_canonical_files.extend(
            _canonical_buffer_filename("noise_error_buffer", rank)
            for rank in range(sp_size)
        )
    if state.get("canonical_buffer_filenames") != expected_canonical_files:
        raise ResumeContractError(
            "resume contract canonical ER/noise file inventory changed"
        )

    for section in ("training", "code", "dataset", "sampler", "data_cursor"):
        if not isinstance(contract.get(section), Mapping):
            raise ResumeContractError(f"resume contract section {section} is missing")
    _validate_resume_recipe_shape(contract["training"])
    _validate_resume_code_identity(contract["code"])
    return tuple(
        _checkpoint_required_filenames(
            world_size,
            sp_size,
            error_buffer_required=state["error_buffer_required"],
            noise_error_buffer_required=state["noise_error_buffer_required"],
        )
    )


def _validate_checkpoint_directory(
    checkpoint_dir,
    *,
    expected_step=None,
    verify_hashes=True,
):
    """Validate the small publication envelope without paging model tensors."""
    checkpoint_dir = Path(checkpoint_dir)
    try:
        directory_stat = checkpoint_dir.lstat()
    except FileNotFoundError as exc:
        raise ResumeContractError(
            f"checkpoint directory is missing: {checkpoint_dir}"
        ) from exc
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ResumeContractError(
            f"checkpoint must be a regular directory: {checkpoint_dir}"
        )
    del directory_stat

    contract = _read_json_document(
        checkpoint_dir / RESUME_CONTRACT_FILENAME,
        context="resume contract",
    )
    required_payloads = _validate_resume_contract_shape(
        contract, expected_step=expected_step
    )
    marker = _read_json_document(
        checkpoint_dir / COMPLETION_MARKER_FILENAME,
        context="checkpoint completion marker",
    )
    if marker.get("marker_version") != COMPLETION_MARKER_VERSION:
        raise ResumeContractError("unsupported checkpoint completion marker")
    if marker.get("status") != "complete":
        raise ResumeContractError("checkpoint completion marker is not complete")
    if marker.get("step") != contract["step"]:
        raise ResumeContractError("checkpoint marker step disagrees with contract")
    if marker.get("resume_contract_sha256") != contract["contract_sha256"]:
        raise ResumeContractError(
            "checkpoint marker does not bind the resume contract"
        )

    components = marker.get("components")
    if not isinstance(components, Mapping):
        raise ResumeContractError("checkpoint completion inventory is missing")
    expected_components = set(required_payloads) | {RESUME_CONTRACT_FILENAME}
    if set(components) != expected_components:
        raise ResumeContractError(
            "checkpoint completion inventory changed: "
            f"missing={sorted(expected_components - set(components))}, "
            f"unexpected={sorted(set(components) - expected_components)}"
        )
    for filename in sorted(expected_components):
        component = components[filename]
        if not isinstance(component, Mapping):
            raise ResumeContractError(
                f"invalid checkpoint component inventory for {filename}"
            )
        stat_result = _regular_nonempty_file(
            checkpoint_dir / filename,
            context=f"checkpoint component {filename}",
        )
        if component.get("size_bytes") != stat_result.st_size:
            raise ResumeContractError(
                f"checkpoint component size changed for {filename}: "
                f"{stat_result.st_size} != {component.get('size_bytes')!r}"
            )
        expected_sha256 = component.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ResumeContractError(
                f"checkpoint component SHA-256 is invalid for {filename}"
            )
        if verify_hashes:
            observed_sha256 = _sha256_file(checkpoint_dir / filename)
            if observed_sha256 != expected_sha256:
                raise ResumeContractError(
                    f"checkpoint component SHA-256 changed for {filename}: "
                    f"{observed_sha256} != {expected_sha256}"
                )
    return contract


def _distributed_validate_checkpoint_directory(checkpoint_dir, *, expected_step=None):
    """Hash all large payloads once on rank zero; peers check structure/size."""
    if not (dist.is_available() and dist.is_initialized()):
        return _validate_checkpoint_directory(
            checkpoint_dir, expected_step=expected_step, verify_hashes=True
        )
    rank = dist.get_rank()
    contract = None
    error = None
    try:
        contract = _validate_checkpoint_directory(
            checkpoint_dir,
            expected_step=expected_step,
            verify_hashes=(rank == 0),
        )
    except Exception as exc:
        error = {
            "rank": int(rank),
            "type": type(exc).__name__,
            "message": str(exc)[:4096],
        }
    local = {
        "error": error,
        "contract_sha256": (
            contract.get("contract_sha256") if contract is not None else None
        ),
        "checkpoint_dir": str(Path(checkpoint_dir)),
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [item["error"] for item in gathered if item["error"]]
    if failures:
        raise ResumeContractError(
            "distributed checkpoint validation failed: "
            + "; ".join(
                f"rank={item['rank']} {item['type']}: {item['message']}"
                for item in failures
            )
        )
    identities = {
        (item["checkpoint_dir"], item["contract_sha256"])
        for item in gathered
    }
    if len(identities) != 1:
        raise ResumeContractError(
            f"ranks selected different checkpoint contracts: {sorted(identities)}"
        )
    return contract


def _write_completion_marker(staging_dir, contract):
    staging_dir = Path(staging_dir)
    required = _validate_resume_contract_shape(
        contract, expected_step=contract["step"]
    )
    components = {}
    for filename in required + (RESUME_CONTRACT_FILENAME,):
        stat_result = _regular_nonempty_file(
            staging_dir / filename,
            context=f"checkpoint component {filename}",
        )
        components[filename] = {
            "size_bytes": int(stat_result.st_size),
            "sha256": _sha256_file(staging_dir / filename),
        }
    return _write_sealed_json(
        staging_dir / COMPLETION_MARKER_FILENAME,
        {
            "marker_version": COMPLETION_MARKER_VERSION,
            "status": "complete",
            "step": int(contract["step"]),
            "resume_contract_sha256": contract["contract_sha256"],
            "components": components,
        },
    )


def _publish_checkpoint_directory(staging_dir, final_dir, *, expected_step):
    """Atomically expose a completion-marked same-stage checkpoint."""
    # _write_completion_marker has just content-hashed every component. Recheck
    # the sealed inventory and sizes here without reading ~120 GB a second time.
    _validate_checkpoint_directory(
        staging_dir, expected_step=expected_step, verify_hashes=False
    )
    if os.path.exists(final_dir):
        raise FileExistsError(
            f"Refusing to overwrite completed checkpoint: {final_dir}"
        )
    os.rename(staging_dir, final_dir)


def _shape_tuple(value):
    """Return a stable shape tuple for state-dict compatibility checks."""
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _state_dict_match_report(module, state_dict):
    """Compare a checkpoint state dict with a module without loading it."""
    expected = module.state_dict()
    expected_keys = set(expected.keys())
    checkpoint_keys = set(state_dict.keys())
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    shape_mismatches = []
    for key in sorted(expected_keys & checkpoint_keys):
        expected_shape = _shape_tuple(expected[key])
        checkpoint_shape = _shape_tuple(state_dict[key])
        if expected_shape != checkpoint_shape:
            shape_mismatches.append((key, expected_shape, checkpoint_shape))
    del expected
    return {
        "matches": not missing and not unexpected and not shape_mismatches,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def _summarize_state_dict_mismatch(report, limit=5):
    parts = []
    if report["missing"]:
        parts.append(f"missing={len(report['missing'])} {report['missing'][:limit]}")
    if report["unexpected"]:
        parts.append(
            f"unexpected={len(report['unexpected'])} {report['unexpected'][:limit]}"
        )
    if report["shape_mismatches"]:
        parts.append(
            f"shape_mismatches={len(report['shape_mismatches'])} "
            f"{report['shape_mismatches'][:limit]}"
        )
    return "; ".join(parts) or "unknown mismatch"


def _strict_load_generator_state_dict(generator, state_dict):
    """Strictly load a raw checkpoint at its one exact generator level."""
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Generator checkpoint payload must be a state-dict mapping, got "
            f"{type(state_dict).__name__}."
        )

    candidates = [("generator", generator)]
    inner_model = getattr(generator, "model", None)
    if (
        inner_model is not None
        and inner_model is not generator
        and hasattr(inner_model, "state_dict")
        and hasattr(inner_model, "load_state_dict")
    ):
        candidates.append(("generator.model", inner_model))

    reports = {}
    for name, module in candidates:
        report = _state_dict_match_report(module, state_dict)
        reports[name] = report
        if report["matches"]:
            module.load_state_dict(state_dict, strict=True)
            return name

    details = " | ".join(
        f"{name}: {_summarize_state_dict_mismatch(report)}"
        for name, report in reports.items()
    )
    raise RuntimeError(
        "Checkpoint state dict does not exactly match generator or "
        f"generator.model. Refusing a partial load. {details}"
    )


def _load_generator_checkpoint_strict(
    checkpoint_path,
    generator,
    *,
    retain_auxiliary_state=True,
):
    """Load generator weights and optionally retain full-resume state."""
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        del checkpoint
        gc.collect()
        raise TypeError(
            "Checkpoint must be a state-dict mapping or a LongLive checkpoint mapping."
        )

    envelope_key = None
    for key in ("generator", "model"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            envelope_key = key
            break
    weight_state = checkpoint if envelope_key is None else checkpoint[envelope_key]

    try:
        target_name = _strict_load_generator_state_dict(generator, weight_state)
    except Exception:
        del weight_state
        del checkpoint
        gc.collect()
        raise

    if envelope_key is None or not retain_auxiliary_state:
        resume_state = None
        del weight_state
        del checkpoint
    else:
        del checkpoint[envelope_key]
        del weight_state
        resume_state = checkpoint
    gc.collect()
    return resume_state, target_name


def _dataset_resume_identity(dataset, config, index_contract):
    """Build a stable, JSON-safe identity for the exact sample-index mapping."""
    if not isinstance(index_contract, Mapping):
        raise ResumeContractError("dataset index contract must be a mapping")
    identity = {
        "dataset_class": type(dataset).__name__,
        "data_path": str(Path(config.data_path).expanduser().resolve()),
        "index_contract": dict(index_contract),
    }
    for field in (
        "manifest_receipt_path",
        "manifest_receipt_sha256",
        "manifest_stage",
        "manifest_split",
    ):
        value = getattr(config, field, None)
        if field.endswith("_path") and value:
            value = str(Path(value).expanduser().resolve())
        identity[field] = value

    index_path = getattr(dataset, "index_path", None)
    index_header = getattr(dataset, "_index_header", None)
    if index_path is not None:
        identity["manifest_index_path"] = str(Path(index_path).resolve())
    else:
        identity["manifest_index_path"] = None
    if isinstance(index_header, Mapping):
        stable_header = {
            key: index_header[key]
            for key in (
                "index_version",
                "manifest_identity",
                "manifest_path",
                "manifest_sha256",
                "receipt_file_sha256",
                "record_count",
                "ids_sha256",
                "stage",
                "split",
                "carrier_latent_frames",
                "temporal_compression_ratio",
            )
            if key in index_header
        }
        identity["manifest_index_header_sha256"] = _canonical_json_sha256(
            stable_header
        )
    else:
        identity["manifest_index_header_sha256"] = None
    # Exercise JSON serialization now, while an actionable dataset error can be
    # reported, instead of discovering it during a multi-rank checkpoint save.
    _canonical_json_sha256(identity)
    return identity


def _training_code_identity(longlive_root=None):
    """Hash the source files that define training/resume/data/SP semantics."""
    root = (
        Path(__file__).resolve().parents[1]
        if longlive_root is None
        else Path(longlive_root).expanduser().resolve()
    )
    identity = {}
    for relative_path in RESUME_CRITICAL_CODE_PATHS:
        path = root / relative_path
        _regular_nonempty_file(path, context=f"training source {relative_path}")
        identity[relative_path] = _sha256_file(path)
    return identity


def _resume_contract_body(
    *,
    step,
    topology,
    training,
    code,
    dataset,
    sampler,
    data_cursor,
    ema_required,
    error_buffer_required,
    noise_error_buffer_required,
):
    world_size = int(topology["world_size"])
    body = {
        "contract_version": RESUME_CONTRACT_VERSION,
        "status": "complete_same_stage_state",
        "step": int(step),
        "topology": dict(topology),
        "training": dict(training),
        "code": dict(code),
        "dataset": dict(dataset),
        "sampler": dict(sampler),
        "data_cursor": dict(data_cursor),
        "state": {
            "model_filename": "model.pt",
            "rank_state_version": RANK_STATE_VERSION,
            "rank_state_filenames": [
                _rank_state_filename(rank) for rank in range(world_size)
            ],
            "optimizer_required": True,
            "ema_required": bool(ema_required),
            "error_buffer_required": bool(error_buffer_required),
            "noise_error_buffer_required": bool(noise_error_buffer_required),
            "er_resume_policy": ER_RESUME_POLICY,
            "canonical_buffer_filenames": (
                [
                    _canonical_buffer_filename("error_buffer", rank)
                    for rank in range(
                        int(topology["sequence_parallel_size"])
                    )
                ]
                if error_buffer_required
                else []
            )
            + (
                [
                    _canonical_buffer_filename("noise_error_buffer", rank)
                    for rank in range(
                        int(topology["sequence_parallel_size"])
                    )
                ]
                if noise_error_buffer_required
                else []
            ),
        },
    }
    _validate_resume_contract_shape(
        _sealed_json_document(body), expected_step=step
    )
    return body


def _validate_resume_runtime_contract(
    contract,
    *,
    topology,
    training,
    code,
    dataset,
    sampler,
    data_cursor,
    ema_required,
    error_buffer_required,
    noise_error_buffer_required,
):
    """Fail before training if any same-stage invariant changed."""
    _validate_resume_contract_shape(contract, expected_step=data_cursor["step"])
    expected_sections = {
        "topology": dict(topology),
        "training": dict(training),
        "code": dict(code),
        "dataset": dict(dataset),
        "sampler": dict(sampler),
        "data_cursor": dict(data_cursor),
    }
    mismatches = {
        name: {"checkpoint": contract.get(name), "runtime": expected}
        for name, expected in expected_sections.items()
        if contract.get(name) != expected
    }
    expected_flags = {
        "ema_required": bool(ema_required),
        "error_buffer_required": bool(error_buffer_required),
        "noise_error_buffer_required": bool(noise_error_buffer_required),
    }
    state = contract["state"]
    for name, expected in expected_flags.items():
        if state.get(name) != expected:
            mismatches[f"state.{name}"] = {
                "checkpoint": state.get(name),
                "runtime": expected,
            }
    if mismatches:
        raise ResumeContractError(
            "same-stage resume contract does not match this runtime: "
            + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
        )


def _validate_resume_model_envelope(resume_state, contract):
    """Require optimizer/EMA/step metadata promised by the resume contract."""
    if not isinstance(resume_state, Mapping):
        raise ResumeContractError("same-stage resume model envelope is missing")
    state = contract["state"]
    required = {
        "checkpoint_contract_version",
        "resume_contract_sha256",
        "generator_optimizer",
        "step",
    }
    if state["ema_required"]:
        required.add("generator_ema")
    missing = sorted(required - set(resume_state))
    if missing:
        raise ResumeContractError(
            f"same-stage resume model envelope is missing {missing}"
        )
    if resume_state.get("checkpoint_contract_version") != RESUME_CONTRACT_VERSION:
        raise ResumeContractError("model envelope resume-contract version changed")
    if resume_state.get("resume_contract_sha256") != contract["contract_sha256"]:
        raise ResumeContractError("model envelope does not bind the resume contract")
    if resume_state.get("step") != contract["step"]:
        raise ResumeContractError("model envelope step disagrees with resume contract")
    if not isinstance(resume_state.get("generator_optimizer"), Mapping):
        raise ResumeContractError("generator optimizer state is not a mapping")
    ema_present = "generator_ema" in resume_state
    if ema_present != state["ema_required"]:
        raise ResumeContractError(
            "generator EMA presence disagrees with resume contract"
        )


def _rank_state_envelope(
    *,
    global_rank,
    step,
    contract,
    topology,
    rng_state,
    error_buffer_summary,
    noise_error_buffer_summary,
):
    return {
        "rank_state_version": RANK_STATE_VERSION,
        "resume_contract_sha256": contract["contract_sha256"],
        "step": int(step),
        "global_rank": int(global_rank),
        "world_size": int(topology["world_size"]),
        "sequence_parallel_size": int(topology["sequence_parallel_size"]),
        "data_parallel_size": int(topology["data_parallel_size"]),
        "rng": rng_state,
        "error_buffer_summary": error_buffer_summary,
        "noise_error_buffer_summary": noise_error_buffer_summary,
    }


def _torch_load_rank_state(path):
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(os.fspath(path), mmap=True, **kwargs)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc).lower():
            raise
        return torch.load(os.fspath(path), **kwargs)


def _restore_rank_state(
    rank_state,
    *,
    global_rank,
    contract,
    topology,
    cuda_device,
):
    """Restore one exact global-rank state shard; no legacy fallback exists."""
    if not isinstance(rank_state, Mapping):
        raise ResumeContractError("rank-state shard must be a mapping")
    required = {
        "rank_state_version",
        "resume_contract_sha256",
        "step",
        "global_rank",
        "world_size",
        "sequence_parallel_size",
        "data_parallel_size",
        "rng",
        "error_buffer_summary",
        "noise_error_buffer_summary",
    }
    if set(rank_state) != required:
        raise ResumeContractError(
            "rank-state fields changed: "
            f"missing={sorted(required - set(rank_state))}, "
            f"unexpected={sorted(set(rank_state) - required)}"
        )
    expected_scalars = {
        "rank_state_version": RANK_STATE_VERSION,
        "resume_contract_sha256": contract["contract_sha256"],
        "step": contract["step"],
        "global_rank": int(global_rank),
        "world_size": int(topology["world_size"]),
        "sequence_parallel_size": int(topology["sequence_parallel_size"]),
        "data_parallel_size": int(topology["data_parallel_size"]),
    }
    changed = {
        key: {"checkpoint": rank_state.get(key), "runtime": value}
        for key, value in expected_scalars.items()
        if rank_state.get(key) != value
    }
    if changed:
        raise ResumeContractError(
            "rank-state identity changed: "
            + json.dumps(changed, sort_keys=True, separators=(",", ":"))
        )

    state_requirements = contract["state"]
    for stem in ("error_buffer", "noise_error_buffer"):
        summary = rank_state[f"{stem}_summary"]
        required_state = state_requirements[f"{stem}_required"]
        if required_state and not isinstance(summary, Mapping):
            raise ResumeContractError(
                f"required {stem} summary is missing on rank {global_rank}"
            )
        if not required_state and summary is not None:
            raise ResumeContractError(
                f"unexpected {stem} summary on rank {global_rank}"
            )
    _restore_rng_state(rank_state["rng"], cuda_device=cuda_device)


def _restore_canonical_error_buffer(path, buffer, *, stem):
    if buffer is None:
        raise ResumeContractError(f"runtime {stem} is absent")
    payload = _torch_load_rank_state(path)
    try:
        if not isinstance(payload, Mapping):
            raise ResumeContractError(f"canonical {stem} state must be a mapping")
        # Strict layout verifies SP position/timestep ownership, configuration,
        # and every bucket key before mutating the live buffer.
        buffer.load_state_dict(payload, strict_layout=True)
    finally:
        del payload


def _resolve_single_video_only(config):
    """Prefer the explicit option, falling back to legacy uniform_prompt."""
    missing = object()
    explicit = getattr(config, "single_video_only", missing)
    if explicit is not missing:
        return bool(explicit)
    return bool(getattr(config, "uniform_prompt", False))


def _resolve_train_drop_last(config):
    """Resolve the data-coverage policy without accepting truthy strings."""
    value = section_get(
        config,
        "data",
        "drop_last",
        True,
        aliases=("train_drop_last",),
    )
    if not isinstance(value, bool):
        raise ValueError(
            f"data.drop_last must be a boolean, got {value!r}"
        )
    return value


def _resolve_train_shuffle(config):
    """Resolve deterministic manifest order without accepting truthy strings."""
    value = section_get(
        config,
        "data",
        "shuffle",
        True,
        aliases=("train_shuffle",),
    )
    if not isinstance(value, bool):
        raise ValueError(f"data.shuffle must be a boolean, got {value!r}")
    return value


def _is_robot_video_manifest_path(data_path):
    return Path(os.fspath(data_path)).suffix.lower() == ".jsonl"


def _build_i2v_video_dataset(
    *,
    data_path,
    video_size,
    total_frames,
    target_fps,
    num_frame_per_block,
    temporal_compression_ratio,
    deterministic,
    allow_padding,
    min_latent_frames,
    single_video_only,
    independent_first_frame,
    return_image,
    max_chunks_per_shot,
    scene_cut_prefix="The scene transitions. ",
    sample_warning_seconds=60.0,
    sample_warning_interval_seconds=60.0,
    expected_manifest_stage=None,
    expected_manifest_split=None,
    manifest_receipt_path=None,
    manifest_receipt_sha256=None,
    manifest_index_path=None,
    manifest_index_cache_dir=None,
    evaluation_mode=False,
    requested_eval_latent_frames=None,
):
    """Build either the frozen-manifest reader or the legacy folder reader."""
    if _is_robot_video_manifest_path(data_path):
        if RobotVideoManifestDataset is None:
            raise RuntimeError("RobotVideoManifestDataset is unavailable")
        if not allow_padding:
            raise ValueError(
                "robot_video stage manifests require allow_padding=true so shorter "
                "buckets use the fixed carrier with an exact loss mask"
            )
        if not single_video_only:
            raise ValueError(
                "robot_video stage manifests require single_video_only=true"
            )
        if manifest_receipt_sha256 is None:
            raise ValueError(
                "robot_video stage manifests require manifest_receipt_sha256"
            )
        if not isinstance(evaluation_mode, bool):
            raise ValueError(
                f"evaluation_mode must be a boolean, got {evaluation_mode!r}"
            )
        if evaluation_mode and not deterministic:
            raise ValueError(
                "robot_video frame0 evaluation mode requires deterministic=true"
            )
        if not evaluation_mode and requested_eval_latent_frames is not None:
            raise ValueError(
                "requested_eval_latent_frames cannot be set for training"
            )
        return RobotVideoManifestDataset(
            manifest_path=data_path,
            video_size=video_size,
            total_frames=total_frames,
            target_fps=target_fps,
            num_frame_per_block=num_frame_per_block,
            temporal_compression_ratio=temporal_compression_ratio,
            expected_stage=expected_manifest_stage,
            expected_split=expected_manifest_split,
            receipt_path=manifest_receipt_path,
            expected_receipt_sha256=manifest_receipt_sha256,
            index_path=manifest_index_path,
            index_cache_dir=manifest_index_cache_dir,
            return_image=return_image,
            evaluation_mode=evaluation_mode,
            requested_eval_latent_frames=requested_eval_latent_frames,
        )
    return MultiVideoConcatDataset(
        data_dir=data_path,
        video_size=video_size,
        total_frames=total_frames,
        deterministic=deterministic,
        num_frame_per_block=num_frame_per_block,
        temporal_compression_ratio=temporal_compression_ratio,
        target_fps=target_fps,
        allow_padding=allow_padding,
        min_latent_frames=min_latent_frames,
        single_video_only=single_video_only,
        independent_first_frame=independent_first_frame,
        return_image=return_image,
        max_chunks_per_shot=max_chunks_per_shot,
        scene_cut_prefix=scene_cut_prefix,
        sample_warning_seconds=sample_warning_seconds,
        sample_warning_interval_seconds=sample_warning_interval_seconds,
    )


def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    """
    Save prompts for one generated video to a txt file.
    Consecutive identical prompts are merged, e.g.:
        [0] a, [1] a, [2] b  =>  [0,1] a\n[2] b\n
    """
    prompt_path = Path(prompt_txt_path)
    temporary_path = prompt_path.with_name(
        f".{prompt_path.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            if len(prompts_for_sample) > 0:
                current_prompt = prompts_for_sample[0]
                current_indices = [0]
                for seg_idx in range(1, len(prompts_for_sample)):
                    p = prompts_for_sample[seg_idx]
                    if p == current_prompt:
                        current_indices.append(seg_idx)
                    else:
                        indices_str = ",".join(str(i) for i in current_indices)
                        f.write(f"[{indices_str}] {current_prompt}\n")
                        current_prompt = p
                        current_indices = [seg_idx]
                # flush the last run
                indices_str = ",".join(str(i) for i in current_indices)
                f.write(f"[{indices_str}] {current_prompt}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, prompt_path)
    except Exception as e:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {e}")
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(value, output_path):
    """Write a tensor artifact without exposing a partial final file."""
    output_path = Path(output_path)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}"
    )
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_mp4_artifact(
    path,
    *,
    expected_frames,
    expected_fps,
    expected_width,
    expected_height,
):
    """Decode a just-written validation MP4 and enforce its exact media contract."""
    import av

    path = Path(path)
    with av.open(str(path), mode="r") as container:
        video_streams = list(container.streams.video)
        if len(video_streams) != 1:
            raise RuntimeError(
                f"Expected one video stream in {path}, found {len(video_streams)}"
            )
        stream = video_streams[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        rate = stream.average_rate
        fps = float(rate) if rate is not None else 0.0
        decoded_frames = sum(1 for _ in container.decode(stream))
    if (width, height) != (int(expected_width), int(expected_height)):
        raise RuntimeError(
            f"Validation MP4 geometry mismatch for {path}: "
            f"{width}x{height} != {expected_width}x{expected_height}"
        )
    if abs(fps - float(expected_fps)) > 1e-3:
        raise RuntimeError(
            f"Validation MP4 FPS mismatch for {path}: {fps} != {expected_fps}"
        )
    if decoded_frames != int(expected_frames):
        raise RuntimeError(
            f"Validation MP4 frame-count mismatch for {path}: "
            f"{decoded_frames} != {expected_frames}"
        )
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frames": decoded_frames,
    }


def _atomic_write_video_artifact(
    output_path,
    frames,
    *,
    fps,
    expected_frames,
    expected_width,
    expected_height,
):
    """Encode, decode-validate, and atomically publish one validation MP4."""
    output_path = Path(output_path)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}"
    )
    try:
        write_video(str(temporary_path), frames, fps=fps)
        metadata = _validate_mp4_artifact(
            temporary_path,
            expected_frames=expected_frames,
            expected_fps=fps,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        os.replace(temporary_path, output_path)
        return metadata
    finally:
        temporary_path.unlink(missing_ok=True)


def _allocation_stop_reached(step, stop_after_step):
    """Return whether this allocation reached its optional resume boundary."""

    if stop_after_step is None:
        return False
    stop_after_step = int(stop_after_step)
    if stop_after_step < 1:
        raise ValueError("stop_after_step must be at least 1")
    return int(step) >= stop_after_step


def _evaluation_due(step, interval, max_iters, evaluate_at_end=False):
    """Return whether the completed optimizer step needs validation.

    ``interval`` remains an absolute global-step cadence so validation stays
    stable across Slurm resumes.  ``evaluate_at_end`` adds one final panel when
    an exact epoch boundary is not divisible by that cadence.
    """

    step = int(step)
    interval = int(interval)
    max_iters = int(max_iters)
    if step < 1:
        raise ValueError("evaluation step must be at least 1")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    periodic = interval > 0 and step % interval == 0
    terminal = bool(evaluate_at_end) and step >= max_iters
    return periodic or terminal


def _checkpoint_due(step, interval, max_iters, stop_after_step=None):
    """Return whether an atomic recovery checkpoint must be written."""

    step = int(step)
    interval = int(interval)
    max_iters = int(max_iters)
    if step < 1:
        raise ValueError("checkpoint step must be at least 1")
    if interval < 1:
        raise ValueError("checkpoint interval must be at least 1")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    return (
        step % interval == 0
        or _allocation_stop_reached(step, stop_after_step)
        or step >= max_iters
    )


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        wandb_init_error = None
        if self.is_main_process and not self.disable_wandb:
            try:
                if getattr(config, "wandb_key", None):
                    wandb.login(host=config.wandb_host, key=config.wandb_key)
                wandb_config = OmegaConf.to_container(config, resolve=True)
                wandb_config.update(
                    {
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "world_size": dist.get_world_size(),
                    }
                )
                wandb_mode = os.environ.get(
                    "WANDB_MODE", getattr(config, "wandb_mode", "online")
                )
                wandb_run_id = os.environ.get(
                    "WANDB_RUN_ID", getattr(config, "wandb_run_id", None)
                )
                wandb_run_name = os.environ.get(
                    "WANDB_NAME", getattr(config, "wandb_name", config.config_name)
                )
                wandb_tags = getattr(config, "wandb_tags", None)
                if wandb_tags is not None:
                    wandb_tags = list(wandb_tags)
                wandb.init(
                    config=wandb_config,
                    name=wandb_run_name,
                    id=wandb_run_id,
                    resume=_wandb_resume_mode(wandb_run_id),
                    mode=wandb_mode,
                    entity=config.wandb_entity,
                    project=config.wandb_project,
                    group=getattr(config, "wandb_group", None),
                    tags=wandb_tags,
                    job_type=getattr(config, "wandb_job_type", "train"),
                    dir=config.wandb_save_dir,
                )
            except Exception as exc:
                wandb_init_error = exc
        # The process group already exists.  Propagate a rank-zero W&B failure
        # before any peer can enter FSDP construction or another collective.
        synchronize_evaluation_failure(
            wandb_init_error,
            phase="initialize W&B",
        )

        self.output_path = config.logdir
        auto_resume = getattr(config, "auto_resume", True)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

        # Sequence Parallel is supported only for the 5B model; world_size must
        # equal sp_size * dp_size.
        self.sequence_parallel_size = getattr(config, "sequence_parallel_size", 1)
        world_size = dist.get_world_size()
        if self.sequence_parallel_size <= 0 or world_size % self.sequence_parallel_size:
            raise ResumeContractError(
                f"world_size={world_size} is not divisible by "
                f"sequence_parallel_size={self.sequence_parallel_size}"
            )
        self.data_parallel_size = world_size // self.sequence_parallel_size if self.sequence_parallel_size > 1 else world_size
        self._resume_topology = {
            "world_size": int(world_size),
            "sequence_parallel_size": int(self.sequence_parallel_size),
            "data_parallel_size": int(self.data_parallel_size),
        }
        self._resume_training_identity = {
            "batch_size_per_data_parallel_rank": int(config.batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "global_batch_size": int(
                config.batch_size
                * self.gradient_accumulation_steps
                * self.data_parallel_size
            ),
            "max_iters": int(config.max_iters),
            "model_name": str(config.model_kwargs.model_name),
            "num_frame_per_block": int(config.num_frame_per_block),
            "image_or_video_shape": [
                int(value) for value in config.image_or_video_shape
            ],
            "optimizer": {
                "name": "AdamW",
                "lr": float(config.lr),
                "betas": [float(config.beta1), float(config.beta2)],
                "weight_decay": float(config.weight_decay),
                "max_grad_norm": float(config.max_grad_norm),
            },
            "ema": {
                "weight": (
                    None
                    if getattr(config, "ema_weight", None) is None
                    else float(config.ema_weight)
                ),
                "start_step": int(config.ema_start_step),
            },
            "algorithm": {
                "causal": bool(config.causal),
                "teacher_forcing": bool(config.teacher_forcing),
                "i2v": bool(config.i2v),
                "independent_first_frame": bool(
                    config.independent_first_frame
                ),
                "num_train_timestep": int(config.num_train_timestep),
                "denoising_loss_type": str(config.denoising_loss_type),
                "noise_augmentation_max_timestep": int(
                    config.noise_augmentation_max_timestep
                ),
            },
            "attention_and_schedule": {
                "local_attn_size": int(
                    config.model_kwargs.get("local_attn_size", -1)
                ),
                "sink_size": int(config.model_kwargs.get("sink_size", 0)),
                "timestep_shift": float(config.timestep_shift),
                "t_scale": float(config.model_kwargs.get("t_scale", 1.0)),
                "rope_method": str(
                    config.model_kwargs.get("rope_method", "linear")
                ),
            },
            "distributed_precision": {
                "mixed_precision": bool(config.mixed_precision),
                "sharding_strategy": str(config.sharding_strategy),
                "gradient_checkpointing": bool(
                    config.gradient_checkpointing
                ),
                "vae_halo_latents": int(config.vae_halo_latents),
            },
            "inference_conditioning": {
                # This unconditional text is consumed by every training step,
                # not only by standalone visualization.
                "negative_prompt": str(config.negative_prompt),
                "guidance_scale": float(config.guidance_scale),
            },
            "data_semantics": {
                "load_raw_video": bool(config.load_raw_video),
                "allow_padding": bool(getattr(config, "allow_padding", False)),
                "min_latent_frames": int(
                    getattr(config, "min_latent_frames", 0)
                ),
                "single_video_only": bool(
                    _resolve_single_video_only(config)
                ),
            },
            "error_recycling": OmegaConf.to_container(
                config.error_recycling, resolve=True
            ),
        }
        _canonical_json_sha256(self._resume_training_identity)
        self._resume_code_identity = _training_code_identity()
        self.sp_group = None
        self.dp_group = None

        if self.is_main_process and self.gradient_accumulation_steps > 1:
            eff_batch = config.batch_size * self.gradient_accumulation_steps * self.data_parallel_size
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}, effective batch size: {eff_batch}")

        if self.sequence_parallel_size > 1:
            assert config.model_kwargs.model_name == "Wan2.2-TI2V-5B", (
                f"sequence_parallel_size is only supported for Wan2.2-TI2V-5B model, but got {config.model_kwargs.model_name}"
            )
            assert world_size % self.sequence_parallel_size == 0, (
                f"world_size ({world_size}) must be divisible by sequence_parallel_size ({self.sequence_parallel_size})"
            )
            from wan_5b.distributed.sp_training import (
                validate_sequence_parallel_training_config,
            )
            validate_sequence_parallel_training_config(
                config,
                self.sequence_parallel_size,
                config.num_frame_per_block,
            )
            # Create SP process groups: each DP group contains sp_size ranks,
            # and all_to_all runs only within that group.
            from wan_5b.distributed.sp_training import (
                set_data_parallel_group,
                set_sequence_parallel_group,
            )
            sp_size = self.sequence_parallel_size
            dp_size = self.data_parallel_size
            sp_groups = []
            for g in range(dp_size):
                ranks_g = list(range(g * sp_size, (g + 1) * sp_size))
                sp_groups.append(
                    dist.new_group(
                        ranks=ranks_g,
                        timeout=TRAINING_PROCESS_GROUP_TIMEOUT,
                    )
                )
            self.sp_group = sp_groups[global_rank // sp_size]
            set_sequence_parallel_group(self.sp_group)

            # Also create DP groups: ranks with the same SP rank across DP
            # replicas own the same sequence chunk. For sp_rank=k, the DP group
            # is [k, sp+k, 2*sp+k, ..., (dp-1)*sp+k]. This lets warmup gather
            # different batches of errors for the same block efficiently.
            dp_groups = []
            for k in range(sp_size):
                ranks_k = [g * sp_size + k for g in range(dp_size)]
                dp_groups.append(
                    dist.new_group(
                        ranks=ranks_k,
                        timeout=TRAINING_PROCESS_GROUP_TIMEOUT,
                    )
                )
            self.dp_group = dp_groups[global_rank % sp_size]
            set_data_parallel_group(self.dp_group)
            if self.is_main_process:
                print(f"[SP] Sequence Parallel enabled, sp_size={sp_size}, dp_size={dp_size}, world_size={world_size}")

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.sp_helper = SequenceParallelHelper(self)

        # 2D mode only: print which GLOBAL block-position slice this rank is
        # responsible for. The LAST SP rank carries the most error-accumulated
        # tail blocks, useful when debugging position-bucketed error recycling.
        if self.model.error_buffer is not None and self.model.er_num_blocks > 0:
            lo = self.model.er_block_offset
            hi = lo + self.model.er_num_blocks
            global_rank_id = dist.get_rank()
            sp_rk = global_rank_id % max(self.sequence_parallel_size, 1)
            print(
                f"[ErrorBuffer] rank={global_rank_id} sp_rank={sp_rk} "
                f"covers GLOBAL blocks [{lo},{hi}) ({self.model.er_num_blocks} local blocks)"
            )

        # Bind the SP forward path before FSDP wrapping.
        model_name = getattr(getattr(config, "model_kwargs", None), "model_name", "") or ""
        if self.sequence_parallel_size > 1 and "Wan2.2-TI2V-5B" in model_name:
            from wan_5b.distributed.sequence_parallel import (
                sp_dit_causal_forward_train,
                sp_causal_attn_forward,
            )
            model = self.model.generator.model
            # Use the SP forward implementation in the training path.
            model._forward_train = types.MethodType(sp_dit_causal_forward_train, model)

            # Keep the original self_attn.forward so inference can temporarily
            # disable SP.
            self._sp_attn_blocks = []
            for block in model.blocks:
                sa = block.self_attn
                if not hasattr(sa, "_orig_forward"):
                    sa._orig_forward = sa.forward
                sa.forward = types.MethodType(sp_causal_attn_forward, sa)
                self._sp_attn_blocks.append(sa)

            if self.is_main_process:
                print("[SP] sp_dit_causal_forward_train and sp_causal_attn_forward are enabled")
                print("[SP] natural TF layout is the default training layout")
                if getattr(config, "load_raw_video", False):
                    print(f"[SP-VAE] chunk-halo VAE enabled, halo_latents={self.sp_helper.vae_halo_latents}")

        # ================================= NVFP4 Quantized Training =================================
        self.model_quant = getattr(config, "model_quant", False)
        if self.model_quant:
            from utils.quant import ModelQuantizationConfig, quantize_model_with_filter

            quant_cfg = ModelQuantizationConfig(
                scale_rule=getattr(config, "model_quant_scale_rule", "static_6"),
                activation_scale_rule=getattr(config, "model_quant_activation_scale_rule", "static_6"),
                weight_scale_rule=getattr(config, "model_quant_weight_scale_rule", None),
                gradient_scale_rule=getattr(config, "model_quant_gradient_scale_rule", None),
                keep_master_weights=True,
                weight_scale_2d=True,
            )
            self.model.generator.model, matched_modules = quantize_model_with_filter(
                self.model.generator.model,
                quant_config=quant_cfg,
                filtered_modules=getattr(config, "model_quant_filtered_modules", None),
                use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
                cast_model_to_bf16=False,
                materialize_for_inference=False,
                verbose=self.is_main_process,
            )
            if self.is_main_process:
                from fouroversix.matmul.cutlass.backend import CUTLASSMatmulBackend

                print(f"[NVFP4] CUTLASS available: {CUTLASSMatmulBackend.is_available()}")
                print(
                    "[NVFP4] Quantized AR training enabled "
                    "(keep_master_weights=True, weight_scale_2d=True)"
                )
                print(f"[NVFP4] {len(matched_modules)} modules excluded from quantization")

        # ================================= Load model weights (before FSDP) =================================
        # Load model weights before FSDP wrapping, while keys still match the
        # raw nn.Module. Optimizer, EMA, and step state are restored after FSDP
        # and the related objects are created, so keep raw_state.
        #
        # Priority: auto_resume from logdir > generator_ckpt for a cold start
        # > random initialization.  A logdir checkpoint always restores full
        # training state.  ``generator_ckpt_load_mode`` applies only to the
        # explicit initialization checkpoint so a generator-only phase
        # transition cannot disable later interruption recovery.
        raw_state = None
        resume_contract = None
        explicit_load_mode = getattr(
            config, "generator_ckpt_load_mode", "resume"
        )
        checkpoint_path, checkpoint_source, checkpoint_load_mode = (
            _resolve_checkpoint_load_plan(
                auto_resume=auto_resume,
                output_path=self.output_path,
                generator_ckpt=getattr(config, "generator_ckpt", None),
                generator_ckpt_load_mode=explicit_load_mode,
                find_latest_checkpoint=self.find_latest_checkpoint,
            )
        )

        if self.is_main_process:
            if checkpoint_source == "auto_resume":
                print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")
            elif auto_resume and self.output_path:
                print("Auto resume: No complete checkpoint found in logdir")
            elif auto_resume:
                print("Auto resume enabled but no logdir was specified")
            else:
                print("Auto resume disabled for this cold start")
            if checkpoint_source == "generator_ckpt":
                print(
                    f"Using explicit checkpoint: {checkpoint_path} "
                    f"(load_mode={checkpoint_load_mode})"
                )

        if checkpoint_path:
            if self.is_main_process:
                print(f"Loading checkpoint from {checkpoint_path}")
            if checkpoint_load_mode == "resume":
                checkpoint_file = Path(checkpoint_path).expanduser().resolve()
                if checkpoint_file.name != "model.pt":
                    raise ResumeContractError(
                        "same-stage resume requires the completion-marked "
                        f"checkpoint model.pt path, got {checkpoint_file}"
                    )
                cached_path = getattr(
                    self, "_validated_resume_checkpoint_path", None
                )
                cached_contract = getattr(
                    self, "_validated_resume_contract", None
                )
                if cached_path == str(checkpoint_file) and cached_contract is not None:
                    resume_contract = cached_contract
                else:
                    resume_contract = _distributed_validate_checkpoint_directory(
                        checkpoint_file.parent
                    )
            raw_state, loaded_target = _load_generator_checkpoint_strict(
                checkpoint_path,
                self.model.generator,
                retain_auxiliary_state=(checkpoint_load_mode == "resume"),
            )
            if checkpoint_load_mode == "resume":
                _validate_resume_model_envelope(raw_state, resume_contract)
            if self.is_main_process:
                print(
                    f"Strictly loaded pretrained weights into {loaded_target} "
                    f"from {checkpoint_path}"
                )

            if raw_state is not None and "step" in raw_state:
                self.step = raw_state["step"]
                if self.is_main_process:
                    print(f"Resuming from step {self.step}")
            elif raw_state is not None:
                raise ResumeContractError(
                    "same-stage resume checkpoint has no global step"
                )
            elif checkpoint_source == "generator_ckpt" and self.is_main_process:
                print(
                    "Starting a new phase from generator weights only: "
                    "step=0, optimizer/EMA/error-recycling state reset."
                )

        # ================================= FSDP Wrap =================================
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        frame_raw_height = list(config.image_or_video_shape)[3] * wan_default_config[config.model_kwargs.model_name]["spatial_compression_ratio"]
        frame_raw_width = list(config.image_or_video_shape)[4] * wan_default_config[config.model_kwargs.model_name]["spatial_compression_ratio"]
        total_frames = (list(config.image_or_video_shape)[1] - 1) * wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"] + 1
        num_frame_per_block = config.num_frame_per_block
        self.fps = wan_default_config[config.model_kwargs.model_name].get("fps", 16)

        allow_padding = getattr(config, "allow_padding", False)
        min_latent_frames = getattr(config, "min_latent_frames", 0)
        single_video_only = _resolve_single_video_only(config)
        max_chunks_per_shot = getattr(config, "max_chunks_per_shot", 0)
        dataset_sample_warning_seconds = getattr(config, "dataset_sample_warning_seconds", 60.0)
        dataset_sample_warning_interval_seconds = getattr(
            config, "dataset_sample_warning_interval_seconds", 60.0
        )
        temporal_compression_ratio = wan_default_config[
            config.model_kwargs.model_name
        ]["temporal_compression_ratio"]
        dataset = _build_i2v_video_dataset(
            data_path=config.data_path,
            video_size=(frame_raw_height, frame_raw_width),
            total_frames=total_frames,
            deterministic=False,
            num_frame_per_block=num_frame_per_block,
            temporal_compression_ratio=temporal_compression_ratio,
            target_fps=self.fps,
            allow_padding=allow_padding,
            min_latent_frames=min_latent_frames,
            single_video_only=single_video_only,
            independent_first_frame=getattr(config, "independent_first_frame", False),
            return_image=getattr(config, "i2v", False),
            max_chunks_per_shot=max_chunks_per_shot,
            sample_warning_seconds=dataset_sample_warning_seconds,
            sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
            expected_manifest_stage=getattr(config, "manifest_stage", None),
            expected_manifest_split=getattr(
                config, "manifest_split", "train"
            ),
            manifest_receipt_path=getattr(
                config, "manifest_receipt_path", None
            ),
            manifest_receipt_sha256=getattr(
                config, "manifest_receipt_sha256", None
            ),
            manifest_index_path=getattr(config, "manifest_index_path", None),
            manifest_index_cache_dir=getattr(
                config, "manifest_index_cache_dir", None
            ),
            evaluation_mode=False,
        )
        dataset_index_contract = _validate_distributed_dataset_index(dataset)
        if self.is_main_process:
            print(
                "[data] canonical distributed sample index: "
                f"count={dataset_index_contract['count']} "
                f"sha256={dataset_index_contract['ids_sha256']}"
            )
        if allow_padding and self.is_main_process:
            print(f"[Padding] Variable-length training enabled: short videos will be padded with loss masking"
                  f" (min_latent_frames={min_latent_frames})")
        if single_video_only and self.is_main_process:
            print("[single_video_only] each sample uses one video only")
        # DP replicas shuffle one shared permutation and select disjoint
        # slices. SP ranks in the same group jointly own one sample, so they
        # use the same DP rank and receive the same train/eval indices.
        sampler_seed = int(config.seed)
        if self.sequence_parallel_size > 1:
            data_rank = global_rank // self.sequence_parallel_size
            data_replicas = self.data_parallel_size
        else:
            data_rank = global_rank
            data_replicas = world_size
        train_drop_last = _resolve_train_drop_last(config)
        train_shuffle = _resolve_train_shuffle(config)
        sampler = build_distributed_sampler(
            dataset,
            rank=data_rank,
            num_replicas=data_replicas,
            seed=sampler_seed,
            shuffle=train_shuffle,
            drop_last=train_drop_last,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=2,
            prefetch_factor=1,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=multi_video_collate_fn,
        )
        resume_data_epoch, resume_data_batch = resolve_resume_data_cursor(
            optimizer_step=self.step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            batches_per_epoch=len(dataloader),
        )
        sampler.set_start_index(resume_data_batch * int(config.batch_size))
        self._resume_dataset_identity = _dataset_resume_identity(
            dataset, config, dataset_index_contract
        )
        self._resume_sampler_identity = {
            "sampler_class": type(sampler).__name__,
            "seed": int(sampler_seed),
            "shuffle": bool(train_shuffle),
            "drop_last": bool(train_drop_last),
            "num_replicas": int(data_replicas),
            "num_samples_per_replica": int(sampler.num_samples),
            "total_size": int(sampler.total_size),
            "dataset_size": int(len(dataset)),
            "batches_per_epoch": int(len(dataloader)),
        }
        self._resume_batches_per_epoch = int(len(dataloader))
        resume_data_cursor = {
            "step": int(self.step),
            "epoch": int(resume_data_epoch),
            "batch_in_epoch": int(resume_data_batch),
            "consumed_microbatches": int(
                self.step * self.gradient_accumulation_steps
            ),
        }
        if self.is_main_process:
            print(
                "Data cursor: "
                f"epoch={resume_data_epoch}, batch={resume_data_batch}/"
                f"{len(dataloader)}, optimizer_step={self.step}, "
                f"shuffle={train_shuffle}, drop_last={train_drop_last}, "
                f"distributed_padding={sampler.total_size - len(dataset)}"
            )

        # Evaluation groups run sequentially in configuration order.  Every
        # rank builds the same ordered list, so changing the temporal shape
        # between groups cannot reorder FSDP/SP collectives.
        eval_data_path = getattr(config, "eval_data_path", config.data_path)
        inference_num_frames = section_get(
            config,
            "evaluation",
            "num_frames",
            getattr(config, "inference_num_frames", 0),
        )
        fixed_eval_ids = section_get(config, "evaluation", "sample_ids", ())
        evaluation_seed = int(
            section_get(config, "evaluation", "seed", config.seed)
        )
        evaluation_config = config.get("evaluation", {})
        eval_group_specs = evaluation_group_specs(
            evaluation_config,
            default_num_frames=inference_num_frames,
            default_sample_ids=fixed_eval_ids or (),
            default_seed=evaluation_seed,
            default_data_path=eval_data_path,
            num_frame_per_block=num_frame_per_block,
        )
        if len(eval_group_specs) > 1 and not getattr(config, "i2v", False):
            raise ValueError(
                "evaluation.groups is currently supported only for I2V datasets"
            )

        chunks_per_shot = getattr(config, "chunks_per_shot", 0)
        scene_cut_prefix = getattr(config, "scene_cut_prefix", "The scene transitions. ")
        self.eval_data_rank = data_rank
        self.eval_data_replicas = data_replicas
        self.eval_sequence_parallel_rank = (
            global_rank % self.sequence_parallel_size
            if self.sequence_parallel_size > 1
            else 0
        )

        self.evaluation_groups = []
        for group_spec in eval_group_specs:
            group_num_frames = int(group_spec.num_frames)
            eval_total_frames = (
                (group_num_frames - 1) * temporal_compression_ratio + 1
                if group_num_frames > 0
                else total_frames
            )
            first_chunk_frames = 1 + (
                num_frame_per_block - 1
            ) * temporal_compression_ratio
            subsequent_chunk_frames = (
                num_frame_per_block * temporal_compression_ratio
            )
            num_blocks = 1 + (
                eval_total_frames - first_chunk_frames
            ) // subsequent_chunk_frames

            if getattr(config, "i2v", False):
                eval_dataset = _build_i2v_video_dataset(
                    data_path=group_spec.data_path,
                    video_size=(frame_raw_height, frame_raw_width),
                    total_frames=eval_total_frames,
                    deterministic=True,
                    num_frame_per_block=num_frame_per_block,
                    temporal_compression_ratio=temporal_compression_ratio,
                    target_fps=self.fps,
                    allow_padding=allow_padding,
                    min_latent_frames=min_latent_frames,
                    single_video_only=single_video_only,
                    independent_first_frame=getattr(
                        config, "independent_first_frame", False
                    ),
                    return_image=True,
                    max_chunks_per_shot=max_chunks_per_shot,
                    scene_cut_prefix=scene_cut_prefix,
                    sample_warning_seconds=dataset_sample_warning_seconds,
                    sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
                    manifest_index_cache_dir=getattr(
                        config, "manifest_index_cache_dir", None
                    ),
                    manifest_receipt_path=getattr(
                        config, "manifest_receipt_path", None
                    ),
                    manifest_receipt_sha256=getattr(
                        config, "manifest_receipt_sha256", None
                    ),
                    evaluation_mode=True,
                    requested_eval_latent_frames=group_num_frames,
                )
                eval_collate = multi_video_collate_fn
            else:
                eval_dataset = MultiTextConcatDataset(
                    data_path=group_spec.data_path,
                    num_blocks=num_blocks,
                    chunks_per_shot=chunks_per_shot,
                    scene_cut_prefix=scene_cut_prefix,
                    deterministic=True,
                )
                eval_collate = eval_collate_fn

            fixed_samples = ()
            if group_spec.sample_ids:
                if not getattr(config, "i2v", False):
                    raise ValueError(
                        "Fixed evaluation sample_ids are supported only for I2V datasets"
                    )
                eval_dataset, fixed_samples = select_fixed_evaluation_subset(
                    eval_dataset, group_spec.sample_ids
                )
                if self.is_main_process:
                    group_label = group_spec.name or "validation"
                    selected = ", ".join(
                        f"{sample.slot}:{sample.sample_id}"
                        for sample in fixed_samples
                    )
                    print(f"[{group_label}] fixed samples: {selected}")

            if dist.get_rank() == 0:
                group_label = group_spec.name or "legacy"
                print(
                    f"Using {eval_dataset.__class__.__name__} for eval "
                    f"group={group_label}: {group_spec.data_path}, "
                    f"latent_frames={group_num_frames}, "
                    f"raw_frames={eval_total_frames}, num_blocks={num_blocks}"
                )
            eval_sampler = build_distributed_sampler(
                eval_dataset,
                rank=data_rank,
                num_replicas=data_replicas,
                seed=sampler_seed,
                shuffle=False,
                drop_last=False,
            )
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=section_get(config, "evaluation", "val_batch_size", 1),
                sampler=eval_sampler,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=eval_collate,
            )
            self.evaluation_groups.append(
                {
                    "spec": group_spec,
                    "dataloader": eval_dataloader,
                    "fixed_samples": fixed_samples,
                    "raw_frames": int(eval_total_frames),
                    "dataset_size": len(eval_dataset),
                }
            )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            for group in self.evaluation_groups:
                group_label = group["spec"].name or "legacy"
                print(
                    f"EVAL DATASET SIZE [{group_label}] "
                    f"{group['dataset_size']}"
                )

        self.dataloader = cycle(dataloader, start_epoch=resume_data_epoch)
        # Keep these aliases for callers/configs that still expect one group.
        self.eval_dataloader = self.evaluation_groups[0]["dataloader"]
        self.fixed_eval_samples = self.evaluation_groups[0]["fixed_samples"]

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0) and (self.step >= config.ema_start_step):
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) enforce the same-stage contract, then restore every
        #    training-critical state. Model weights were loaded before FSDP;
        #    optimizer/EMA and the rank-local RNG/ER state are restored here.

        if raw_state is not None:
            if resume_contract is None:
                raise ResumeContractError(
                    "auxiliary checkpoint state cannot be loaded without a "
                    "completion-marked same-stage resume contract"
                )
            _validate_resume_runtime_contract(
                resume_contract,
                topology=self._resume_topology,
                training=self._resume_training_identity,
                code=self._resume_code_identity,
                dataset=self._resume_dataset_identity,
                sampler=self._resume_sampler_identity,
                data_cursor=resume_data_cursor,
                ema_required=self.generator_ema is not None,
                error_buffer_required=self.model.error_buffer is not None,
                noise_error_buffer_required=(
                    self.model.noise_error_buffer is not None
                ),
            )

            if self.generator_ema is not None:
                self.generator_ema.load_state_dict(raw_state["generator_ema"])
                if self.is_main_process:
                    print("Resuming generator EMA...")

            gen_osd = FSDP.optim_state_dict_to_load(
                self.model.generator,
                self.generator_optimizer,
                raw_state["generator_optimizer"],
            )
            del raw_state["generator_optimizer"]
            self.generator_optimizer.load_state_dict(gen_osd)
            del gen_osd
            if self.is_main_process:
                print("Resuming generator optimizer...")

            del raw_state
            gc.collect()

        ##############################################################################################################

        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

        # Same-stage ER/noise recovery is deliberately bounded: the first DP
        # replica for each SP position saves one canonical populated buffer and
        # every DP peer at that SP position loads it. Buffers diverge between
        # allocations, so this is operational/non-bitwise canonicalization, not
        # an exact per-DP continuation. Per-rank RNG remains exact and small.
        if resume_contract is not None:
            sp_rank = global_rank % self.sequence_parallel_size
            rank_state_path = (
                Path(checkpoint_path).parent
                / _rank_state_filename(global_rank)
            )

            def restore_local_rank_state():
                if self.model.error_buffer is not None:
                    _restore_canonical_error_buffer(
                        Path(checkpoint_path).parent
                        / _canonical_buffer_filename("error_buffer", sp_rank),
                        self.model.error_buffer,
                        stem="error_buffer",
                    )
                if self.model.noise_error_buffer is not None:
                    _restore_canonical_error_buffer(
                        Path(checkpoint_path).parent
                        / _canonical_buffer_filename(
                            "noise_error_buffer", sp_rank
                        ),
                        self.model.noise_error_buffer,
                        stem="noise_error_buffer",
                    )
                rank_state = _torch_load_rank_state(rank_state_path)
                try:
                    _restore_rank_state(
                        rank_state,
                        global_rank=global_rank,
                        contract=resume_contract,
                        topology=self._resume_topology,
                        cuda_device=self.device,
                    )
                finally:
                    del rank_state

            self._run_evaluation_stage(
                phase=f"restore checkpoint rank state rank={global_rank}",
                operation=restore_local_rank_state,
            )
            error_stats = (
                self.model.error_buffer.stats()
                if self.model.error_buffer is not None
                else None
            )
            noise_stats = (
                self.model.noise_error_buffer.stats()
                if self.model.noise_error_buffer is not None
                else None
            )
            print(
                f"[resume] rank={global_rank} restored exact rank RNG from "
                f"{rank_state_path.name} and replicated canonical SP{sp_rank} "
                f"ER/noise ({ER_RESUME_POLICY}); "
                f"error={error_stats} noise={noise_stats}"
            )

    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def find_latest_checkpoint(self, logdir):
        """Return only the latest completion-marked, structurally complete save."""
        if not os.path.exists(logdir):
            return None

        checkpoint_dirs = []
        for item in os.listdir(logdir):
            prefix = "checkpoint_model_"
            step_str = item[len(prefix):] if item.startswith(prefix) else ""
            if not step_str.isdigit():
                continue
            checkpoint_dir = Path(logdir) / item
            if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
                checkpoint_dirs.append((int(step_str), checkpoint_dir))
        
        if not checkpoint_dirs:
            return None

        # A visible highest-step directory is supposed to have appeared via one
        # atomic rename. If it is incomplete/tampered, fail instead of silently
        # rolling back to an older checkpoint.
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_step, latest_dir = checkpoint_dirs[-1]
        contract = _distributed_validate_checkpoint_directory(
            latest_dir, expected_step=latest_step
        )
        latest_path = str((latest_dir / "model.pt").resolve())
        self._validated_resume_checkpoint_path = latest_path
        self._validated_resume_contract = contract
        return latest_path

    def get_all_checkpoints(self, logdir):
        """Get all complete checkpoints, failing on any visible bad directory."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            prefix = "checkpoint_model_"
            step_str = item[len(prefix):] if item.startswith(prefix) else ""
            if not step_str.isdigit():
                continue
            checkpoint_dir_path = Path(logdir) / item
            if not checkpoint_dir_path.exists() and not checkpoint_dir_path.is_symlink():
                continue
            step = int(step_str)
            _validate_checkpoint_directory(
                checkpoint_dir_path, expected_step=step, verify_hashes=False
            )
            checkpoint_dirs.append((step, str(checkpoint_dir_path), item))
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")

    def save(self):
        print("Start gathering distributed model states...")

        # Snapshot the process-local streams at the optimizer-step boundary.
        # Checkpoint serialization itself must not advance the saved streams.
        rank_rng_state = _capture_rng_state(cuda_device=self.device)

        data_epoch, data_batch = resolve_resume_data_cursor(
            optimizer_step=self.step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            batches_per_epoch=self._resume_batches_per_epoch,
        )
        data_cursor = {
            "step": int(self.step),
            "epoch": int(data_epoch),
            "batch_in_epoch": int(data_batch),
            "consumed_microbatches": int(
                self.step * self.gradient_accumulation_steps
            ),
        }
        contract_body = _resume_contract_body(
            step=self.step,
            topology=self._resume_topology,
            training=self._resume_training_identity,
            code=self._resume_code_identity,
            dataset=self._resume_dataset_identity,
            sampler=self._resume_sampler_identity,
            data_cursor=data_cursor,
            ema_required=self.generator_ema is not None,
            error_buffer_required=self.model.error_buffer is not None,
            noise_error_buffer_required=(
                self.model.noise_error_buffer is not None
            ),
        )
        resume_contract = _sealed_json_document(contract_body)
        require_equal_evaluation_value(
            resume_contract["contract_sha256"],
            phase=f"checkpoint resume contract step={self.step}",
        )

        # Release large inference caches before saving when possible.
        if hasattr(self.model, "inference_pipeline") and self.model.inference_pipeline is not None:
            clear_fn = getattr(self.model.inference_pipeline, "clear_cache", None)
            if clear_fn is not None:
                try:
                    clear_fn()
                except Exception as e:
                    print(f"Warning: failed to clear inference cache before save: {e}")
            # Drop the inference pipeline reference so GC / empty_cache can
            # reclaim memory.
            self.model.inference_pipeline = None
            torch.cuda.empty_cache()
        
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            generator_state_dict  = self.model.generator.state_dict()
            generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                            self.generator_optimizer)

        state_dict = {
            "checkpoint_contract_version": RESUME_CONTRACT_VERSION,
            "resume_contract_sha256": resume_contract["contract_sha256"],
            "generator": generator_state_dict,
            "generator_optimizer": generator_opim_state_dict,
            "step": self.step,
        }
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        staging_dir, checkpoint_dir = _checkpoint_directory_paths(
            self.output_path, self.step
        )
        # Build into a hidden sibling directory.  ``find_latest_checkpoint``
        # ignores it, so a killed writer can never make a half checkpoint look
        # resumable. Publication is one same-filesystem directory rename after
        # every global-rank shard, contract, and completion marker is closed.
        self._run_evaluation_stage(
            phase=f"prepare checkpoint staging step={self.step}",
            operation=lambda: _prepare_checkpoint_staging_directory(
                staging_dir, checkpoint_dir
            ),
            main_process_only=True,
        )

        self._run_evaluation_stage(
            phase=f"write checkpoint model step={self.step}",
            operation=lambda: torch.save(
                state_dict, os.path.join(staging_dir, "model.pt")
            ),
            main_process_only=True,
        )

        del state_dict
        del generator_state_dict
        del generator_opim_state_dict
        gc.collect()

        _global_rank = dist.get_rank() if dist.is_initialized() else 0
        rank_state_filename = _rank_state_filename(_global_rank)

        def write_rank_state():
            rank_state = _rank_state_envelope(
                global_rank=_global_rank,
                step=self.step,
                contract=resume_contract,
                topology=self._resume_topology,
                rng_state=rank_rng_state,
                error_buffer_summary=(
                    self.model.error_buffer.stats()
                    if self.model.error_buffer is not None
                    else None
                ),
                noise_error_buffer_summary=(
                    self.model.noise_error_buffer.stats()
                    if self.model.noise_error_buffer is not None
                    else None
                ),
            )
            torch.save(
                rank_state,
                os.path.join(staging_dir, rank_state_filename),
            )
            error_stats = (
                self.model.error_buffer.stats()
                if self.model.error_buffer is not None
                else None
            )
            noise_stats = (
                self.model.noise_error_buffer.stats()
                if self.model.noise_error_buffer is not None
                else None
            )
            print(
                f"[rank={_global_rank}] saved exact RNG + pre-canonicalization "
                f"ER/noise summaries to {rank_state_filename}; "
                f"error={error_stats} noise={noise_stats}"
            )
            del rank_state

        self._run_evaluation_stage(
            phase=f"write checkpoint rank states step={self.step}",
            operation=write_rank_state,
        )
        del rank_rng_state

        sp_size = int(self.sequence_parallel_size)
        sp_rank = _global_rank % sp_size
        is_canonical_writer = (_global_rank // sp_size) == 0

        def write_canonical_sp_buffers():
            if not is_canonical_writer:
                return
            for stem, buffer in (
                ("error_buffer", self.model.error_buffer),
                ("noise_error_buffer", self.model.noise_error_buffer),
            ):
                if buffer is None:
                    continue
                filename = _canonical_buffer_filename(stem, sp_rank)
                torch.save(
                    buffer.state_dict(),
                    os.path.join(staging_dir, filename),
                )
                print(
                    f"[rank={_global_rank}] saved canonical SP{sp_rank} "
                    f"{stem} to {filename}: {buffer.stats()}"
                )

        self._run_evaluation_stage(
            phase=f"write canonical SP ER/noise states step={self.step}",
            operation=write_canonical_sp_buffers,
        )

        self._run_evaluation_stage(
            phase=f"write checkpoint resume contract step={self.step}",
            operation=lambda: _write_sealed_json(
                Path(staging_dir) / RESUME_CONTRACT_FILENAME,
                contract_body,
            ),
            main_process_only=True,
        )

        # The marker is written last and inventories model.pt, contract, and all
        # world-size RNG shards and all canonical SP ER/noise states. A final
        # visible directory cannot be mistaken for a resumable partial writer.
        def publish_checkpoint():
            _write_completion_marker(staging_dir, resume_contract)
            _publish_checkpoint_directory(
                staging_dir,
                checkpoint_dir,
                expected_step=self.step,
            )
            print("Checkpoint atomically published to", checkpoint_dir)

            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        self._run_evaluation_stage(
            phase=f"publish checkpoint step={self.step}",
            operation=publish_checkpoint,
            main_process_only=True,
        )

        torch.cuda.empty_cache()
        gc.collect()

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        accumulation_steps = accumulation_steps or getattr(self, "gradient_accumulation_steps", 1)
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        clean_latent_is_sp_sharded = False
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1]
        else:  # encode raw video to latent
            (
                clean_latent,
                image_latent,
                clean_latent_is_sp_sharded,
            ) = self.sp_helper.encode_raw_video_latents(
                batch,
                batch_size=batch_size,
            )

        loss_mask = self.sp_helper.build_loss_mask(
            batch, clean_latent, clean_latent_is_sp_sharded
        )
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # turn text prompts: List[List[str]] into List[str]
            text_prompts_flat = [prompt for sublist in text_prompts for prompt in sublist]

            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts_flat)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 2.5: Sequence Parallel partitions sequence-owned tensors.
        if self.sequence_parallel_size > 1:
            clean_latent, conditional_dict, image_or_video_shape = (
                self.sp_helper.partition_training_inputs(
                    image_or_video_shape=image_or_video_shape,
                    clean_latent=clean_latent,
                    conditional_dict=conditional_dict,
                    clean_latent_is_sharded=clean_latent_is_sp_sharded,
                )
            )
            image_latent = self.sp_helper.local_i2v_initial_latent(image_latent)
        loss_mask, loss_mask_global_valid_count = self.sp_helper.partition_loss_mask(
            loss_mask,
            already_sharded=clean_latent_is_sp_sharded,
        )

        # Step 3: Train the generator
        gen_kwargs = dict(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            loss_mask=loss_mask,
            loss_mask_global_valid_count=loss_mask_global_valid_count,
            global_step=self.step,
        )
        generator_loss, log_dict = self.model.generator_loss(**gen_kwargs)
        if accumulation_step == 0:
            self.generator_optimizer.zero_grad(set_to_none=True)
        scaled_loss = generator_loss / accumulation_steps
        scaled_loss.backward()
        if accumulation_step == accumulation_steps - 1:
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm)

            self.generator_optimizer.step()
            self.step += 1
        else:
            generator_grad_norm = torch.tensor(0.0, device=self.device)

        # Run the remaining logic only after a full gradient-accumulation cycle.
        if accumulation_step != accumulation_steps - 1:
            return

        # Step 4: Update EMA (if enabled and after start step)
        if (self.step >= self.config.ema_start_step) and \
                (self.generator_ema is None) and \
                (getattr(self.config, "ema_weight", None) is not None) and \
                (self.config.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        # Update EMA after optimizer step
        if self.generator_ema is not None and self.step >= self.config.ema_start_step:
            self.generator_ema.update(self.model.generator)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
            "learning_rate": self.generator_optimizer.param_groups[0]["lr"],
            "optimizer_step": self.step,
            "training_progress": self.step / float(self.config.max_iters),
            "samples_seen": (
                self.step
                * int(self.config.batch_size)
                * int(self.gradient_accumulation_steps)
                * int(self.data_parallel_size)
            ),
        }

        # Error buffer stats
        er_log_str = ""
        if "er_total_added" in log_dict:
            wandb_loss_dict["er_total_entries"] = log_dict["er_total_entries"]
            wandb_loss_dict["er_total_added"] = log_dict["er_total_added"]
            wandb_loss_dict["er_injected"] = int(log_dict["er_injected"])
            wandb_loss_dict["er_latent_injected"] = int(log_dict["er_latent_injected"])
            wandb_loss_dict["er_noise_injected"] = int(log_dict.get("er_noise_injected", False))
            wandb_loss_dict["er_noise_total_entries"] = log_dict.get("er_noise_total_entries", 0)
            ctx_flag = 'Y' if log_dict['er_injected'] else 'N'
            lat_flag = 'Y' if log_dict['er_latent_injected'] else 'N'
            noise_flag = 'Y' if log_dict.get('er_noise_injected', False) else 'N'
            er_log_str = (
                f", er_buf={log_dict['er_total_entries']}|"
                f"{log_dict.get('er_noise_total_entries', 0)} "
                f"({log_dict['er_filled_buckets']} buckets), "
                f"ctx={ctx_flag} lat={lat_flag} noise={noise_flag}"
            )

        # Step 5: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            print(
                f"[step {self.step:07d}] "
                f"generator_loss={wandb_loss_dict['generator_loss']:.6f}, "
                f"generator_grad_norm={wandb_loss_dict['generator_grad_norm']:.6f}"
                f"{er_log_str}"
            )

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def _set_sp_attn(self, enabled: bool):
        """
        Toggle SP self-attention between training and inference.
        This only applies to 5B runs with SP enabled.
        """
        if not hasattr(self, "_sp_attn_blocks"):
            return
        if self.sequence_parallel_size <= 1:
            return

        # Lazy import to avoid failures under non-5B configurations.
        try:
            from wan_5b.distributed.sequence_parallel import sp_causal_attn_forward
        except Exception:
            return

        for sa in self._sp_attn_blocks:
            if not hasattr(sa, "_orig_forward"):
                continue
            if enabled:
                sa.forward = types.MethodType(sp_causal_attn_forward, sa)
            else:
                sa.forward = sa._orig_forward

    @torch.no_grad()
    def _swap_ema_weights(self):
        """
        Bidirectionally swap model weights with EMA shadow weights.
        Calling this twice restores both the model and EMA to their original state.
        """
        with FSDP.summon_full_params(self.model.generator, writeback=True):
            for n, p in self.model.generator.module.named_parameters():
                cleaned_name = EMA_FSDP._clean_param_name(n)
                if cleaned_name in self.generator_ema.shadow:
                    ema_val = self.generator_ema.shadow[cleaned_name]
                    tmp = p.data.clone().float().cpu()
                    p.data.copy_(ema_val.to(dtype=p.dtype, device=p.device))
                    self.generator_ema.shadow[cleaned_name] = tmp

    def _run_evaluation_stage(
        self,
        *,
        phase,
        operation,
        main_process_only=False,
    ):
        """Run a validation operation and propagate its error to every rank."""

        result = None
        local_error = None
        if not main_process_only or self.is_main_process:
            try:
                result = operation()
            except Exception as exc:
                local_error = exc
        synchronize_evaluation_failure(local_error, phase=phase)
        return result

    def _run_evaluation_inference(self):
        self._run_evaluation_stage(
            phase="evaluation startup cleanup",
            operation=lambda: (
                gc.collect(),
                torch.cuda.empty_cache(),
                torch.cuda.ipc_collect(),
            ),
        )

        initialize_pipeline = require_equal_evaluation_value(
            self.model.inference_pipeline is None,
            phase="inference pipeline initialization state",
        )
        if initialize_pipeline:
            self._run_evaluation_stage(
                phase="initialize inference pipeline",
                operation=self.model._initialize_inference_pipeline,
            )

        out_dir = os.path.join(self.output_path, f"generated_video_{self.step:06d}")
        self._run_evaluation_stage(
            phase="create evaluation output directory",
            operation=lambda: os.makedirs(out_dir, exist_ok=True),
            main_process_only=True,
        )

        vis_ema = section_get(self.config, "evaluation", "use_ema", getattr(self.config, "vis_ema", False))
        vis_ema = vis_ema and self.generator_ema is not None
        weight_mode = section_get(self.config, "evaluation", "weights", None)
        if weight_mode is None:
            # Preserve the historical behavior for old configs.  Formal fixed
            # validation configs should state their weight mode explicitly.
            weight_mode = "both" if vis_ema else "model"
        run_modes = evaluation_run_modes(
            weight_mode, ema_available=self.generator_ema is not None
        )
        save_latents_only = section_get(
            self.config,
            "evaluation",
            "save_latents_only",
            self.config.get("return_latents", False),
            aliases=("return_latents", "save_latent_only"),
        )
        evaluation_groups = self.evaluation_groups
        named_groups = any(
            group["spec"].name is not None for group in evaluation_groups
        )
        require_equal_evaluation_value(
            (
                tuple(run_modes),
                bool(save_latents_only),
                tuple(
                    (
                        group["spec"].name,
                        int(group["spec"].num_frames),
                        len(group["fixed_samples"]),
                    )
                    for group in evaluation_groups
                ),
            ),
            phase="evaluation execution plan",
        )
        wandb_payload = {}
        total_fixed_videos = 0

        for group in evaluation_groups:
            group_spec = group["spec"]
            fixed_evaluation = bool(group["fixed_samples"])
            group_run_modes = evaluation_artifact_run_modes(
                run_modes,
                group_name=group_spec.name,
                fixed_evaluation=fixed_evaluation,
            )
            group_out_dir = (
                os.path.join(out_dir, group_spec.name)
                if group_spec.name is not None
                else out_dir
            )
            group_label = group_spec.name or "legacy"
            self._run_evaluation_stage(
                phase=f"create {group_label} output directory",
                operation=lambda path=group_out_dir: os.makedirs(
                    path, exist_ok=True
                ),
                main_process_only=True,
            )

            if fixed_evaluation and save_latents_only:
                raise ValueError(
                    "Fixed W&B validation requires evaluation.save_latents_only=false"
                )
            self._run_evaluation_group(
                group=group,
                out_dir=group_out_dir,
                run_modes=group_run_modes,
                save_latents_only=save_latents_only,
            )

            if fixed_evaluation:
                def record_group():
                    group_payload = self._record_fixed_evaluation(
                        out_dir=group_out_dir,
                        run_modes=group_run_modes,
                        evaluation_seed=group_spec.seed,
                        fixed_samples=group["fixed_samples"],
                        group_name=group_spec.name,
                        num_frames=group_spec.num_frames,
                        raw_frames=group["raw_frames"],
                    )
                    duplicate_keys = set(wandb_payload).intersection(group_payload)
                    if duplicate_keys:
                        raise RuntimeError(
                            "Duplicate validation W&B keys: "
                            f"{sorted(duplicate_keys)}"
                        )
                    return group_payload

                group_payload = self._run_evaluation_stage(
                    phase=f"record {group_label} fixed evaluation",
                    operation=record_group,
                    main_process_only=True,
                )
                if self.is_main_process:
                    wandb_payload.update(group_payload)
                total_fixed_videos += len(group["fixed_samples"]) * len(run_modes)

            # Clear between horizons.  With full-context local_attn_size=64,
            # F32 and F64 may reserve the same maximum cache capacity, but no
            # runtime cache/state from one rollout may leak into the next.
            def clear_group_cache():
                if (
                    hasattr(self.model, "inference_pipeline")
                    and self.model.inference_pipeline is not None
                ):
                    clear_fn = getattr(
                        self.model.inference_pipeline, "clear_cache", None
                    )
                    if clear_fn is not None:
                        clear_fn()
                torch.cuda.empty_cache()

            self._run_evaluation_stage(
                phase=f"clear {group_label} inference cache",
                operation=clear_group_cache,
            )

        def finalize_evaluation():
            if named_groups:
                self._write_evaluation_group_manifest(
                    out_dir=out_dir,
                    groups=evaluation_groups,
                    run_modes=run_modes,
                )
                if not self.disable_wandb:
                    wandb_payload["validation/num_videos"] = total_fixed_videos
                    wandb_payload["validation/group_count"] = len(evaluation_groups)

            if not self.disable_wandb and wandb_payload:
                # One history commit per training step keeps all short/long
                # panels on the same W&B row.
                wandb.log(wandb_payload, step=self.step)

        self._run_evaluation_stage(
            phase="write root manifest and log validation to W&B",
            operation=finalize_evaluation,
            main_process_only=True,
        )

    def _run_evaluation_group(
        self,
        *,
        group,
        out_dir,
        run_modes,
        save_latents_only,
    ):
        """Run one fixed-shape pass on every rank in the same call order."""

        rank = dist.get_rank()
        group_spec = group["spec"]
        group_label = group_spec.name or "legacy"
        fixed_samples = group["fixed_samples"]
        fixed_evaluation = bool(fixed_samples)
        expected_raw_frames = int(group["raw_frames"])
        spatial_ratio = int(
            wan_default_config[self.config.model_kwargs.model_name][
                "spatial_compression_ratio"
            ]
        )
        expected_height = int(self.config.image_or_video_shape[3]) * spatial_ratio
        expected_width = int(self.config.image_or_video_shape[4]) * spatial_ratio
        dataloader = group["dataloader"]
        num_batches = self._run_evaluation_stage(
            phase=f"inspect {group_label} dataloader",
            operation=lambda: len(dataloader),
        )
        require_equal_evaluation_value(
            num_batches, phase=f"{group_label} dataloader batch count"
        )
        eval_iterator = self._run_evaluation_stage(
            phase=f"create {group_label} dataloader iterator",
            operation=lambda: iter(dataloader),
        )

        for batch_index in range(num_batches):
            eval_batch = self._run_evaluation_stage(
                phase=f"load {group_label} batch {batch_index}",
                operation=lambda: next(eval_iterator),
            )

            def prepare_batch():
                prompts = eval_batch["prompts"]
                indices = eval_batch["idx"]
                images = eval_batch.get("image", None)
                return prompts, indices, images, len(prompts)

            eval_prompts, eval_idx, eval_images, batch_size_eval = (
                self._run_evaluation_stage(
                    phase=f"prepare {group_label} batch {batch_index}",
                    operation=prepare_batch,
                )
            )
            require_equal_evaluation_value(
                batch_size_eval,
                phase=f"{group_label} batch {batch_index} sample count",
            )

            for b in range(batch_size_eval):
                def prepare_sample():
                    prompts_for_sample = eval_prompts[b]
                    if self.is_main_process:
                        print(f"prompts_for_sample: {prompts_for_sample}")
                        print(len(prompts_for_sample))
                        print(prompts_for_sample[0][:60])

                    sample_idx = (
                        eval_idx[b].item()
                        if hasattr(eval_idx, "shape")
                        else int(eval_idx[b])
                    )
                    writer_sample = None
                    noise_seed = None
                    if fixed_evaluation:
                        fixed_sample = next(
                            (
                                sample
                                for sample in fixed_samples
                                if sample.dataset_index == sample_idx
                            ),
                            None,
                        )
                        if fixed_sample is None:
                            raise RuntimeError(
                                "A fixed validation folder fell back to an "
                                "unselected sample index "
                                f"({sample_idx}); refusing to compare the wrong clip"
                            )
                        writer_sample = fixed_evaluation_writer(
                            sample_idx,
                            fixed_samples,
                            data_rank=self.eval_data_rank,
                            data_replicas=self.eval_data_replicas,
                            sequence_parallel_rank=(
                                self.eval_sequence_parallel_rank
                            ),
                        )
                        noise_seed = deterministic_evaluation_seed(
                            group_spec.seed, fixed_sample.slot
                        )
                    return (
                        prompts_for_sample,
                        sample_idx,
                        writer_sample,
                        noise_seed,
                    )

                (
                    prompts_for_sample,
                    sample_idx,
                    writer_sample,
                    noise_seed,
                ) = self._run_evaluation_stage(
                    phase=(
                        f"prepare {group_label} batch {batch_index} sample {b}"
                    ),
                    operation=prepare_sample,
                )

                for suffix, use_ema, weight_label in run_modes:
                    generated_video = self._run_evaluation_stage(
                        phase=(
                            f"generate {group_label} batch {batch_index} "
                            f"sample {b} weights {weight_label}"
                        ),
                        operation=lambda use_ema=use_ema: self.generate_video(
                            self.model.inference_pipeline,
                            [prompts_for_sample],
                            (
                                eval_images[b:b + 1]
                                if eval_images is not None
                                else None
                            ),
                            use_ema=use_ema,
                            noise_seed=noise_seed,
                            inference_num_frames=group_spec.num_frames,
                        ),
                    )

                    def write_artifact():
                        should_write = (
                            not fixed_evaluation or writer_sample is not None
                        )
                        if should_write and fixed_evaluation:
                            file_stem = (
                                f"sample_{writer_sample.slot:02d}_"
                                f"{writer_sample.sample_id}{suffix}"
                            )
                            video_path = os.path.join(
                                out_dir, f"{file_stem}.mp4"
                            )
                            _atomic_write_video_artifact(
                                video_path,
                                generated_video[0],
                                fps=self.fps,
                                expected_frames=expected_raw_frames,
                                expected_width=expected_width,
                                expected_height=expected_height,
                            )
                        elif should_write and not save_latents_only:
                            video_path = os.path.join(
                                out_dir,
                                f"video{suffix}_rank{rank:02d}_"
                                f"idx{sample_idx:06d}.mp4",
                            )
                            _atomic_write_video_artifact(
                                video_path,
                                generated_video[0],
                                fps=self.fps,
                                expected_frames=expected_raw_frames,
                                expected_width=expected_width,
                                expected_height=expected_height,
                            )
                        elif should_write:
                            video_path = os.path.join(
                                out_dir,
                                f"latents{suffix}_rank{rank:02d}_"
                                f"idx{sample_idx:06d}.pt",
                            )
                            _atomic_torch_save(generated_video[0], video_path)

                        if (
                            not fixed_evaluation
                            and not self.disable_wandb
                            and self.is_main_process
                            and not save_latents_only
                        ):
                            caption = (
                                prompts_for_sample[0]
                                if len(prompts_for_sample) > 0
                                else ""
                            )
                            log_key = f"generated_video{suffix}"
                            wandb.log(
                                {
                                    log_key: wandb.Video(
                                        generated_video[0].transpose(
                                            0, 3, 1, 2
                                        ),
                                        caption=f"{caption}",
                                        fps=self.fps,
                                        format="mp4",
                                    ),
                                },
                                step=self.step,
                            )

                    self._run_evaluation_stage(
                        phase=(
                            f"write {group_label} batch {batch_index} sample "
                            f"{b} weights {weight_label} artifact"
                        ),
                        operation=write_artifact,
                    )

                    del generated_video

                def write_prompt():
                    if fixed_evaluation and writer_sample is not None:
                        prompt_txt_path = os.path.join(
                            out_dir,
                            f"sample_{writer_sample.slot:02d}_"
                            f"{writer_sample.sample_id}.txt",
                        )
                        save_prompts_to_txt(
                            prompts_for_sample, prompt_txt_path, True
                        )
                    elif not fixed_evaluation:
                        prompt_txt_path = os.path.join(
                            out_dir,
                            f"prompt_rank{rank:02d}_idx{sample_idx:06d}.txt",
                        )
                        save_prompts_to_txt(
                            prompts_for_sample,
                            prompt_txt_path,
                            self.is_main_process,
                        )

                self._run_evaluation_stage(
                    phase=(
                        f"write {group_label} batch {batch_index} sample {b} prompt"
                    ),
                    operation=write_prompt,
                )

    def _record_fixed_evaluation(
        self,
        *,
        out_dir,
        run_modes,
        evaluation_seed,
        fixed_samples,
        group_name=None,
        num_frames=None,
        raw_frames=None,
    ):
        """Validate local media, write a manifest, and log all videos together."""

        out_path = Path(out_dir)
        if num_frames is None:
            num_frames = section_get(
                self.config,
                "evaluation",
                "num_frames",
                getattr(self.config, "inference_num_frames", 0),
            )
        manifest = {
            "step": int(self.step),
            "fps": int(self.fps),
            "latent_frames": int(num_frames),
            # Historical field retained for manifest readers written before
            # multi-horizon validation was introduced.
            "num_frames": int(num_frames),
            "sampling_steps": int(getattr(self.config, "sampling_steps", 0)),
            "guidance_scale": float(getattr(self.config, "guidance_scale", 0.0)),
            "weight_modes": [weight_label for _, _, weight_label in run_modes],
            "samples": [],
        }
        if group_name is not None:
            manifest["group"] = group_name
        if raw_frames is not None:
            manifest["raw_frames"] = int(raw_frames)
            manifest["duration_seconds"] = float(raw_frames) / float(self.fps)
        wandb_payload = {}
        multiple_weight_modes = len(run_modes) > 1

        for sample in fixed_samples:
            prompt_path = out_path / (
                f"sample_{sample.slot:02d}_{sample.sample_id}.txt"
            )
            if not prompt_path.is_file():
                raise RuntimeError(f"Missing fixed validation prompt file: {prompt_path}")
            instruction = prompt_path.read_text(encoding="utf-8").strip()
            noise_seed = deterministic_evaluation_seed(evaluation_seed, sample.slot)
            sample_record = {
                "slot": sample.slot,
                "sample_id": sample.sample_id,
                "dataset_index": sample.dataset_index,
                "instruction": instruction,
                "noise_seed": noise_seed,
                "videos": {},
            }

            for suffix, _, weight_label in run_modes:
                video_path = out_path / (
                    f"sample_{sample.slot:02d}_{sample.sample_id}{suffix}.mp4"
                )
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise RuntimeError(f"Missing fixed validation video: {video_path}")
                sample_record["videos"][weight_label] = str(video_path)

                if not self.disable_wandb:
                    log_key = fixed_evaluation_log_key(
                        sample.slot,
                        weight_label=weight_label,
                        multiple_weight_modes=multiple_weight_modes,
                        group_name=group_name,
                    )
                    group_caption = (
                        f"group={group_name} | " if group_name is not None else ""
                    )
                    wandb_payload[log_key] = wandb.Video(
                        str(video_path),
                        caption=(
                            f"{group_caption}sample={sample.sample_id} | "
                            f"weights={weight_label} | seed={noise_seed} | "
                            f"{instruction}"
                        ),
                        fps=self.fps,
                        format="mp4",
                    )
            manifest["samples"].append(sample_record)

        manifest_path = out_path / "manifest.json"
        temporary_path = out_path / f".manifest.json.tmp.{os.getpid()}"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, manifest_path)

        if not self.disable_wandb:
            metric_prefix = (
                f"validation/{group_name}"
                if group_name is not None
                else "validation"
            )
            wandb_payload[f"{metric_prefix}/num_videos"] = (
                len(fixed_samples) * len(run_modes)
            )
            wandb_payload[f"{metric_prefix}/noise_seed_base"] = int(
                evaluation_seed
            )
        return wandb_payload

    def _write_evaluation_group_manifest(self, *, out_dir, groups, run_modes):
        """Write one index that makes the short/long validation pair explicit."""

        out_path = Path(out_dir)
        manifest = {
            "step": int(self.step),
            "fps": int(self.fps),
            "group_order": [group["spec"].name for group in groups],
            "weight_modes": [weight_label for _, _, weight_label in run_modes],
            "groups": [],
        }
        for group in groups:
            spec = group["spec"]
            group_manifest = out_path / spec.name / "manifest.json"
            if not group_manifest.is_file():
                raise RuntimeError(
                    f"Missing validation group manifest: {group_manifest}"
                )
            manifest["groups"].append(
                {
                    "name": spec.name,
                    "latent_frames": int(spec.num_frames),
                    "num_frames": int(spec.num_frames),
                    "raw_frames": int(group["raw_frames"]),
                    "duration_seconds": (
                        float(group["raw_frames"]) / float(self.fps)
                    ),
                    "sample_count": len(group["fixed_samples"]),
                    "sample_ids": [
                        sample.sample_id for sample in group["fixed_samples"]
                    ],
                    "manifest": f"{spec.name}/manifest.json",
                    "wandb_namespace": f"validation/{spec.name}",
                }
            )

        manifest_path = out_path / "manifest.json"
        temporary_path = out_path / f".manifest.json.tmp.{os.getpid()}"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, manifest_path)

    @torch.no_grad()
    def generate_video(
        self,
        pipeline,
        prompts,
        image=None,
        use_ema=False,
        noise_seed=None,
        inference_num_frames=None,
    ):
        # Temporarily disable SP self-attention during inference to avoid
        # interfering with KV-cache logic.
        self._set_sp_attn(False)
        ema_applied = use_ema and self.generator_ema is not None
        if ema_applied:
            self._swap_ema_weights()
        try:
            batch_size = len(prompts)
            noise_shape = list(self.config.image_or_video_shape[1:])
            if inference_num_frames is None:
                inference_num_frames = section_get(
                    self.config,
                    "evaluation",
                    "num_frames",
                    getattr(self.config, "inference_num_frames", 0),
                )
            if isinstance(inference_num_frames, SequenceABC) and not isinstance(
                inference_num_frames, (str, bytes)
            ):
                inference_num_frames = (
                    inference_num_frames[0] if len(inference_num_frames) > 0 else 0
                )
            inference_num_frames = int(inference_num_frames)
            if inference_num_frames > 0:
                noise_shape[0] = inference_num_frames
            initial_latent = None
            if image is not None:
                image = image.to(device="cuda", dtype=self.dtype)
                if image.ndim == 4:
                    image = image.unsqueeze(2)
                elif image.ndim != 5:
                    raise ValueError(f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}")
                initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=self.dtype)
                if initial_latent.shape[0] != batch_size:
                    initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
                if noise_shape[0] <= initial_latent.shape[1]:
                    raise ValueError(
                        f"evaluation.num_frames must exceed the i2v conditioning frames; "
                        f"got {inference_num_frames} and {initial_latent.shape[1]}"
                    )
            noise_generator = None
            if noise_seed is not None:
                noise_generator = torch.Generator(
                    device=torch.device("cuda", self.device)
                )
                noise_generator.manual_seed(int(noise_seed))
            sampled_noise = torch.randn(
                [batch_size] + noise_shape,
                device="cuda",
                dtype=self.dtype,
                generator=noise_generator,
            )

            save_latents_only = section_get(
                self.config,
                "evaluation",
                "save_latents_only",
                self.config.get("return_latents", False),
                aliases=("return_latents", "save_latent_only"),
            )
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                initial_latent=initial_latent,
                return_latents=save_latents_only
            )
            if not save_latents_only:
                current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
            else:
                current_video = video
        finally:
            if ema_applied:
                self._swap_ema_weights()
            # Restore SP self-attention for training.
            self._set_sp_attn(True)

        return current_video

    def _sync_batch_for_sequence_parallel(self, batch, accumulation_step: int = 0):
        return self.sp_helper.sync_batch(batch, step=self.step)

    def train(self):
        if getattr(self.config, "generate_before_train", False):
            if self.is_main_process:
                print("[generate_before_train] Running evaluation inference before training starts...")
            self._run_evaluation_inference()
            if self.is_main_process:
                print("[generate_before_train] Inference done. Exiting.")
            barrier()
            return

        acc_steps = getattr(self, "gradient_accumulation_steps", 1)
        stop_after_step = getattr(self.config, "stop_after_step", None)
        if stop_after_step is not None:
            stop_after_step = int(stop_after_step)
            if _allocation_stop_reached(self.step, stop_after_step):
                if self.is_main_process:
                    print(
                        "[allocation-stop] checkpoint step is already at or beyond "
                        f"stop_after_step={stop_after_step}; exiting without another batch"
                    )
                barrier()
                return
        while True:
            for acc in range(acc_steps):
                batch = next(self.dataloader)

                # Synchronize batch contents across ranks under Sequence Parallel.
                if self.sequence_parallel_size > 1:
                    batch = self._sync_batch_for_sequence_parallel(batch, accumulation_step=acc)

                self.train_one_step(batch, accumulation_step=acc, accumulation_steps=acc_steps)
            reached_allocation_stop = _allocation_stop_reached(
                self.step, stop_after_step
            )
            checkpoint_due = _checkpoint_due(
                self.step,
                self.config.log_iters,
                self.config.max_iters,
                stop_after_step=stop_after_step,
            )
            if (not self.config.no_save) and checkpoint_due:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            evaluation_interval = section_get(self.config, "evaluation", "interval", getattr(self.config, "generate_interval", 0))
            evaluate_at_end = section_get(
                self.config, "evaluation", "evaluate_at_end", False
            )
            if _evaluation_due(
                self.step,
                evaluation_interval,
                self.config.max_iters,
                evaluate_at_end=evaluate_at_end,
            ):
                self._run_evaluation_inference()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if reached_allocation_stop:
                if self.is_main_process:
                    print(
                        "[allocation-stop] reached "
                        f"stop_after_step={stop_after_step}; "
                        f"experiment max_iters remains {int(self.config.max_iters)}"
                    )
                break
            if self.step >= self.config.max_iters:
                break
