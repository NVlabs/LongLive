# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import hashlib

import torch
import torch.distributed as dist
import peft


def disable_incompatible_peft_torchao_dispatch(is_main_process=True):
    """Skip PEFT's optional torchao dispatcher when local torchao is too old.

    PEFT probes every optional LoRA backend before falling back to the normal
    torch.nn.Linear implementation.  Some environments expose torchao 0.11
    while PEFT 0.19 requires >=0.16; without this narrow guard, even a plain
    BF16 LoRA raises before reaching PEFT's default dispatcher.
    """

    try:
        import peft.tuners.lora.model as peft_lora_model
        import peft.tuners.lora.torchao as peft_lora_torchao
    except ImportError:
        return False
    try:
        peft_lora_torchao.is_torchao_available()
    except ImportError as exc:
        if "incompatible version of torchao" not in str(exc):
            raise

        def _skip_torchao_dispatch(*_args, **_kwargs):
            return None

        peft_lora_torchao.dispatch_torchao = _skip_torchao_dispatch
        peft_lora_model.dispatch_torchao = _skip_torchao_dispatch
        if is_main_process:
            print(
                "Disabled PEFT torchao LoRA dispatcher because the installed "
                "torchao version is incompatible with this PEFT release."
            )
        return True
    return False


def configure_lora_for_model(
    transformer,
    model_name,
    lora_config,
    is_main_process=True,
):
    """Configure LoRA for a WanDiffusionWrapper model

    Args:
        transformer: The transformer model to apply LoRA to
        model_name: 'generator' or 'fake_score'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)
    Returns:
        lora_model: The LoRA-wrapped model
    """
    disable_incompatible_peft_torchao_dispatch(is_main_process)
    target_linear_modules = set()

    if model_name not in ('generator', 'fake_score', 'critic'):
        raise ValueError(f"Invalid model name: {model_name}")
    adapter_target_modules = ['WanAttentionBlock']

    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)

    target_linear_modules = sorted(target_linear_modules)
    expected_targets = lora_config.get("expected_target_modules")
    if expected_targets is not None and len(target_linear_modules) != int(expected_targets):
        raise RuntimeError(
            f"Found {len(target_linear_modules)} LoRA target modules for "
            f"{model_name}; expected {int(expected_targets)}"
        )

    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")

    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()

    return lora_model


def trainable_parameter_digest(model):
    """Return a stable SHA-256 digest of all trainable LoRA parameters."""
    digest = hashlib.sha256()
    parameter_count = 0
    element_count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            raise RuntimeError(
                f"Unexpected non-LoRA trainable parameter before FSDP: {name}"
            )
        if parameter.is_meta:
            raise RuntimeError(f"Cannot verify meta LoRA parameter: {name}")
        if not torch.isfinite(parameter.detach()).all().item():
            raise RuntimeError(f"Non-finite LoRA parameter before FSDP: {name}")

        value = parameter.detach().contiguous().to(device="cpu")
        metadata = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(value.view(torch.uint8).numpy().tobytes())
        parameter_count += 1
        element_count += value.numel()

    if parameter_count == 0:
        raise RuntimeError("No trainable LoRA parameters found before FSDP")
    return digest.digest(), parameter_count, element_count


def assert_trainable_lora_parameters_synced(models, device):
    """Fail before FSDP if any rank constructed different LoRA parameters."""
    summaries = {}
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if isinstance(device, int):
        device = torch.device("cuda", device)

    for role, model in models.items():
        local_error = None
        try:
            local_digest, parameter_count, element_count = trainable_parameter_digest(
                model
            )
        except Exception as exc:
            # Every rank must still enter the collective so a local validation
            # error cannot strand otherwise healthy ranks in NCCL.
            local_error = str(exc)
            local_digest = bytes(32)
            parameter_count = 0
            element_count = 0
        summaries[role] = {
            "sha256": local_digest.hex(),
            "parameter_count": parameter_count,
            "element_count": element_count,
        }
        if world_size == 1:
            if local_error is not None:
                raise RuntimeError(local_error)
            continue

        local = torch.tensor(
            [*local_digest, int(local_error is not None)],
            dtype=torch.uint8,
            device=device,
        )
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        gathered_cpu = [item.cpu().tolist() for item in gathered]
        invalid_ranks = [
            rank for rank, value in enumerate(gathered_cpu) if value[-1]
        ]
        if invalid_ranks:
            local_detail = f" Local error: {local_error}" if local_error else ""
            raise RuntimeError(
                f"Pre-FSDP {role} LoRA validation failed on ranks "
                f"{invalid_ranks}.{local_detail}"
            )
        gathered_hex = [bytes(item[:-1]).hex() for item in gathered_cpu]
        if len(set(gathered_hex)) != 1:
            differing_ranks = [
                rank
                for rank, value in enumerate(gathered_hex)
                if value != gathered_hex[0]
            ]
            raise RuntimeError(
                f"Pre-FSDP {role} LoRA parameters differ across ranks; "
                f"rank0={gathered_hex[0]}, differing_ranks={differing_ranks}"
            )
    return summaries


def insert_peft_adapter_name(state_dict, adapter_name="default"):
    """Map PEFT-saved LoRA keys to in-model adapter keys.

    `get_peft_model_state_dict` saves keys such as `lora_A.weight`, while
    the wrapped model state_dict expects `lora_A.default.weight`. This mirrors
    the key remapping done by `peft.set_peft_model_state_dict` without calling
    that helper, which can import unavailable Transformers tensor-parallel
    symbols in this environment.
    """
    remapped = {}
    lora_tokens = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    )
    for key, value in state_dict.items():
        new_key = key
        for token in lora_tokens:
            marker = f".{token}."
            if marker not in key:
                continue
            prefix, suffix = key.rsplit(marker, 1)
            if not suffix.startswith(f"{adapter_name}."):
                new_key = f"{prefix}.{token}.{adapter_name}.{suffix}"
            break
        remapped[new_key] = value
    return remapped


def load_lora_checkpoint(lora_model, lora_state_dict, model_name, is_main_process=True):
    """Load LoRA weights from state dict

    Args:
        lora_model: The LoRA-wrapped model
        lora_state_dict: LoRA state dict to load
        model_name: 'generator' or 'critic'
        is_main_process: Whether this is the main process (for logging)
    """
    if is_main_process:
        print(f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint")

    adapter_name = getattr(lora_model, "active_adapter", "default")
    if callable(adapter_name):
        adapter_name = adapter_name()
    if not isinstance(adapter_name, str):
        adapter_name = "default"

    remapped = insert_peft_adapter_name(lora_state_dict, adapter_name)
    incompatible = lora_model.load_state_dict(remapped, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_lora = [
        key for key in incompatible.missing_keys
        if ".lora_" in key or key.rsplit(".", 1)[-1].startswith("lora_")
    ]
    if unexpected or missing_lora:
        raise RuntimeError(
            f"Failed to load {model_name} LoRA checkpoint with adapter '{adapter_name}'. "
            f"Unexpected keys: {unexpected[:20]}; missing LoRA keys: {missing_lora[:20]}"
        )

    if is_main_process:
        print(
            f"LoRA {model_name} weights loaded successfully with adapter "
            f"'{adapter_name}'"
        )
