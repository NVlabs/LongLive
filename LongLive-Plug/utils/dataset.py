# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import hashlib
import numpy as np
import torch
import json
from pathlib import Path
import os


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> dict:
    """Read only a safetensors header, without materializing tensor payloads."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header_size = handle.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"Invalid safetensors file (missing header size): {path}")
        header_size = int.from_bytes(raw_header_size, "little", signed=False)
        if header_size <= 0 or header_size > min(file_size - 8, 100 * 1024 * 1024):
            raise ValueError(
                f"Invalid safetensors header size {header_size} for {path}"
            )
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header must be an object: {path}")
    return header


class CFGTeacherTrajectoryDataset(Dataset):
    """Strict random-access dataset for cached Wan CFG-teacher trajectory states.

    Manifest schema ``cfg_teacher_trajectory_safetensors_v1`` stores one record
    per prompt and native teacher timestep. Records may share a shard. The
    top-level ``timestep_repeat_factors`` map expands low-noise or otherwise
    underrepresented timesteps without duplicating tensor data.
    """

    FORMAT = "cfg_teacher_trajectory_safetensors_v1"
    _ALLOWED_DTYPES = {"BF16", "F16", "F32"}

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        *,
        expected_latent_shape,
        expected_model_name: str = "Wan2.2-TI2V-5B",
        expected_guidance_scale: float = 5.0,
        expected_sampling_steps: int = 50,
        expected_timestep_shift: float = 5.0,
        verify_shard_hashes: bool = True,
        verified_manifest_sha256: str | None = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise ValueError(
                f"CFG teacher trajectory manifest does not exist: "
                f"{self.manifest_path}"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid CFG teacher trajectory manifest JSON: "
                f"{self.manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError("CFG teacher trajectory manifest must be a JSON object.")
        if not isinstance(verify_shard_hashes, bool):
            raise ValueError("verify_shard_hashes must be a boolean.")
        if not verify_shard_hashes:
            actual_manifest_sha256 = _sha256_file(self.manifest_path)
            if verified_manifest_sha256 != actual_manifest_sha256:
                raise ValueError(
                    "Skipping trajectory shard hashes requires a preverified "
                    "manifest SHA256 matching the current manifest."
                )

        required_top_level = {
            "schema_version",
            "format",
            "model_name",
            "teacher_guidance_scale",
            "teacher_sampling_steps",
            "timestep_shift",
            "latent_shape",
            "timestep_repeat_factors",
            "records",
        }
        missing = sorted(required_top_level - set(manifest))
        if missing:
            raise ValueError(
                f"CFG teacher trajectory manifest is missing fields: {missing}"
            )
        if manifest["schema_version"] != 1 or manifest["format"] != self.FORMAT:
            raise ValueError(
                "Unsupported CFG teacher trajectory manifest schema: "
                f"schema_version={manifest['schema_version']!r}, "
                f"format={manifest['format']!r}."
            )
        if manifest["model_name"] != expected_model_name:
            raise ValueError(
                f"Trajectory model_name={manifest['model_name']!r} does not "
                f"match {expected_model_name!r}."
            )
        if float(manifest["teacher_guidance_scale"]) != float(
            expected_guidance_scale
        ):
            raise ValueError(
                "Trajectory teacher_guidance_scale does not match training config."
            )
        if int(manifest["teacher_sampling_steps"]) != int(
            expected_sampling_steps
        ):
            raise ValueError(
                "Trajectory teacher_sampling_steps does not match training config."
            )
        if float(manifest["timestep_shift"]) != float(expected_timestep_shift):
            raise ValueError("Trajectory timestep_shift does not match training config.")

        self.latent_shape = tuple(int(value) for value in expected_latent_shape)
        if tuple(manifest["latent_shape"]) != self.latent_shape:
            raise ValueError(
                f"Trajectory latent_shape={manifest['latent_shape']} does not "
                f"match expected {list(self.latent_shape)}."
            )
        if len(self.latent_shape) != 4 or any(value <= 0 for value in self.latent_shape):
            raise ValueError(
                f"Expected latent shape must be [F,C,H,W], got {self.latent_shape}."
            )

        repeat_payload = manifest["timestep_repeat_factors"]
        if not isinstance(repeat_payload, dict):
            raise ValueError("timestep_repeat_factors must be a JSON object.")
        expected_indices = set(range(int(expected_sampling_steps)))
        try:
            repeats = {int(key): value for key, value in repeat_payload.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timestep_repeat_factors keys must be integer timestep indices."
            ) from exc
        if set(repeats) != expected_indices:
            raise ValueError(
                "timestep_repeat_factors must cover every native teacher "
                f"index 0..{int(expected_sampling_steps) - 1} exactly."
            )
        for index, repeat in repeats.items():
            if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
                raise ValueError(
                    f"Repeat factor for timestep {index} must be a positive integer."
                )

        records = manifest["records"]
        if not isinstance(records, list) or not records:
            raise ValueError("Trajectory manifest records must be a non-empty list.")
        required_record_fields = {
            "prompt",
            "prompt_index",
            "timestep_index",
            "timestep",
            "shard",
            "shard_sha256",
            "latent_key",
            "target_key",
            "timestep_key",
            "guidance_energy_key",
        }
        cache_root = self.manifest_path.parent
        verified_shards: dict[Path, tuple[str, dict]] = {}
        normalized_records = []
        prompt_by_index = {}
        seen_pairs = set()
        timestep_indices_by_prompt: dict[int, set[int]] = {}
        timestep_value_by_index: dict[int, float] = {}

        for record_number, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Trajectory record {record_number} must be an object.")
            missing_record = sorted(required_record_fields - set(record))
            if missing_record:
                raise ValueError(
                    f"Trajectory record {record_number} is missing fields: "
                    f"{missing_record}"
                )
            prompt = record["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Trajectory record {record_number} has an empty prompt."
                )
            prompt_index = record["prompt_index"]
            timestep_index = record["timestep_index"]
            if (
                isinstance(prompt_index, bool)
                or not isinstance(prompt_index, int)
                or prompt_index < 0
            ):
                raise ValueError(
                    f"Trajectory record {record_number} has invalid prompt_index."
                )
            if (
                isinstance(timestep_index, bool)
                or not isinstance(timestep_index, int)
                or timestep_index not in expected_indices
            ):
                raise ValueError(
                    f"Trajectory record {record_number} has invalid timestep_index."
                )
            timestep = float(record["timestep"])
            if not np.isfinite(timestep) or not 0.0 <= timestep <= 1000.0:
                raise ValueError(
                    f"Trajectory record {record_number} has invalid timestep."
                )
            previous_timestep = timestep_value_by_index.setdefault(
                timestep_index,
                timestep,
            )
            if previous_timestep != timestep:
                raise ValueError(
                    "Trajectory timestep schedule differs between prompts at "
                    f"index {timestep_index}: {previous_timestep} vs {timestep}."
                )
            previous_prompt = prompt_by_index.setdefault(prompt_index, prompt)
            if previous_prompt != prompt:
                raise ValueError(
                    f"prompt_index {prompt_index} maps to multiple prompt strings."
                )
            pair = (prompt_index, timestep_index)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate trajectory prompt/timestep record: {pair}."
                )
            seen_pairs.add(pair)
            timestep_indices_by_prompt.setdefault(prompt_index, set()).add(
                timestep_index
            )

            shard_token = record["shard"]
            if not isinstance(shard_token, str) or not shard_token:
                raise ValueError(
                    f"Trajectory record {record_number} has invalid shard path."
                )
            shard_relative = Path(shard_token)
            if shard_relative.is_absolute():
                raise ValueError("Trajectory shard paths must be relative.")
            shard_path = (cache_root / shard_relative).resolve()
            try:
                shard_path.relative_to(cache_root)
            except ValueError as exc:
                raise ValueError(
                    f"Trajectory shard escapes manifest directory: {shard_token}"
                ) from exc
            if not shard_path.is_file():
                raise ValueError(f"Trajectory shard does not exist: {shard_path}")
            expected_sha = record["shard_sha256"]
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha)
            ):
                raise ValueError(
                    f"Trajectory record {record_number} has invalid shard_sha256."
                )
            if shard_path not in verified_shards:
                if verify_shard_hashes:
                    actual_sha = _sha256_file(shard_path)
                    if actual_sha != expected_sha:
                        raise ValueError(
                            f"Trajectory shard SHA256 mismatch for {shard_path}: "
                            f"expected {expected_sha}, got {actual_sha}."
                        )
                verified_shards[shard_path] = (
                    expected_sha,
                    _safetensors_header(shard_path),
                )
            elif verified_shards[shard_path][0] != expected_sha:
                raise ValueError(
                    f"Trajectory shard {shard_path} has conflicting SHA256 values."
                )

            header = verified_shards[shard_path][1]
            keys_and_shapes = (
                ("latent_key", self.latent_shape, self._ALLOWED_DTYPES),
                ("target_key", self.latent_shape, self._ALLOWED_DTYPES),
                ("timestep_key", (1,), {"F32"}),
                (
                    "guidance_energy_key",
                    (self.latent_shape[0],),
                    {"F32"},
                ),
            )
            for field, expected_shape, allowed_dtypes in keys_and_shapes:
                key = record[field]
                if not isinstance(key, str) or key not in header:
                    raise ValueError(
                        f"Trajectory shard {shard_path} is missing tensor "
                        f"{key!r} from {field}."
                    )
                tensor_meta = header[key]
                if (
                    not isinstance(tensor_meta, dict)
                    or tuple(tensor_meta.get("shape", ())) != expected_shape
                ):
                    raise ValueError(
                        f"Trajectory tensor {key!r} in {shard_path} has shape "
                        f"{tensor_meta.get('shape') if isinstance(tensor_meta, dict) else None}; "
                        f"expected {list(expected_shape)}."
                    )
                if tensor_meta.get("dtype") not in allowed_dtypes:
                    raise ValueError(
                        f"Trajectory tensor {key!r} has unsupported dtype "
                        f"{tensor_meta.get('dtype')!r}."
                    )

            normalized_records.append(
                {
                    **record,
                    "prompt": prompt.strip(),
                    "timestep": timestep,
                    "_shard_path": shard_path,
                }
            )

        for prompt_index, timestep_indices in timestep_indices_by_prompt.items():
            if timestep_indices != expected_indices:
                missing_indices = sorted(expected_indices - timestep_indices)
                raise ValueError(
                    f"Trajectory prompt_index {prompt_index} does not cover all "
                    f"teacher timesteps; missing {missing_indices}."
                )
        prompt_indices = set(prompt_by_index)
        expected_prompt_indices = set(range(len(prompt_indices)))
        if prompt_indices != expected_prompt_indices:
            raise ValueError(
                "Trajectory prompt_index values must be contiguous from 0; "
                f"got {sorted(prompt_indices)}."
            )
        ordered_timesteps = [
            timestep_value_by_index[index]
            for index in range(int(expected_sampling_steps))
        ]
        if any(
            earlier <= later
            for earlier, later in zip(
                ordered_timesteps,
                ordered_timesteps[1:],
            )
        ):
            raise ValueError(
                "Trajectory native timestep schedule must be strictly "
                "decreasing."
            )

        # Close the manifest-to-payload contract: every scalar timestep in the
        # JSON must equal its FP32 tensor in the hashed safetensors shard.
        from safetensors import safe_open

        records_by_shard: dict[Path, list[dict]] = {}
        for record in normalized_records:
            records_by_shard.setdefault(
                record["_shard_path"],
                [],
            ).append(record)
        for shard_path, shard_records in records_by_shard.items():
            with safe_open(
                os.fspath(shard_path),
                framework="pt",
                device="cpu",
            ) as handle:
                for record in shard_records:
                    stored_timestep = handle.get_tensor(
                        record["timestep_key"]
                    )
                    if (
                        stored_timestep.numel() != 1
                        or float(stored_timestep.item())
                        != record["timestep"]
                    ):
                        raise ValueError(
                            "Trajectory manifest timestep does not match "
                            f"shard tensor for prompt_index="
                            f"{record['prompt_index']}, timestep_index="
                            f"{record['timestep_index']}."
                        )

        self.records = normalized_records
        self._expanded_indices = [
            record_index
            for record_index, record in enumerate(self.records)
            for _ in range(repeats[record["timestep_index"]])
        ]
        if not self._expanded_indices:
            raise ValueError("Trajectory repeat expansion produced an empty dataset.")

    def __len__(self):
        return len(self._expanded_indices)

    def __getitem__(self, idx):
        from safetensors import safe_open

        record_index = self._expanded_indices[idx]
        record = self.records[record_index]
        with safe_open(
            os.fspath(record["_shard_path"]),
            framework="pt",
            device="cpu",
        ) as handle:
            noisy_latent = handle.get_tensor(record["latent_key"])
            guided_target = handle.get_tensor(record["target_key"])
            guidance_energy = handle.get_tensor(record["guidance_energy_key"])
        return {
            "idx": idx,
            "prompts": [record["prompt"]],
            "cfg_noisy_latent": noisy_latent,
            "cfg_guided_target": guided_target,
            "cfg_timestep": torch.tensor(record["timestep"], dtype=torch.float32),
            "cfg_guidance_energy": guidance_energy.float(),
            "cfg_prompt_index": record["prompt_index"],
            "cfg_timestep_index": record["timestep_index"],
        }


def cfg_teacher_trajectory_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty CFG teacher trajectory batch.")
    return {
        "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        "prompts": [item["prompts"] for item in batch],
        "cfg_noisy_latent": torch.stack(
            [item["cfg_noisy_latent"] for item in batch]
        ),
        "cfg_guided_target": torch.stack(
            [item["cfg_guided_target"] for item in batch]
        ),
        "cfg_timestep": torch.stack([item["cfg_timestep"] for item in batch]),
        "cfg_guidance_energy": torch.stack(
            [item["cfg_guidance_energy"] for item in batch]
        ),
        "cfg_prompt_index": torch.tensor(
            [item["cfg_prompt_index"] for item in batch], dtype=torch.long
        ),
        "cfg_timestep_index": torch.tensor(
            [item["cfg_timestep_index"] for item in batch], dtype=torch.long
        ),
    }


class TextPromptDataset(Dataset):
    """Read one full-sequence video prompt per nonempty text line."""

    def __init__(self, data_path):
        with open(data_path, encoding="utf-8") as handle:
            self._prompts = [line.rstrip() for line in handle if line.strip()]
        if not self._prompts:
            raise ValueError(f"No prompts found in {data_path}")

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, idx):
        return {"prompts": [self._prompts[idx % len(self._prompts)]], "idx": idx}


def eval_collate_fn(batch):
    """Collate for text-only datasets (no frames)."""
    prompts_list = [b["prompts"] for b in batch]
    idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    result = {
        "prompts": prompts_list,
        "idx": idx,
    }
    return result
