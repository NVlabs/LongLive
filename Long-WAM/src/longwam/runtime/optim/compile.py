# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/compile.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable

import torch

from longwam.utils.logging_config import get_logger

from .config import InferenceOptimizationConfig

logger = get_logger(__name__)


def _compile_observation_vae(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Give clean-observation VAE encode its own Dynamo code object."""
    return fn(*args, **kwargs)


def _compile_video_denoiser(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Give the video target its own Dynamo code object and guard cache."""
    return fn(*args, **kwargs)


def _compile_video_cross_kv_prefill(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Give invariant video cross-attention K/V its own graph."""
    return fn(*args, **kwargs)


def _compile_video_clean_prefix_prefill(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Give the fixed clean-prefix prefill its own compiled graph."""
    return fn(*args, **kwargs)


def _compile_action_denoiser(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Give the action target its own Dynamo code object and guard cache."""
    return fn(*args, **kwargs)


def _compile_action_cross_kv_prefill(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Give invariant action cross-attention K/V its own graph."""
    return fn(*args, **kwargs)


def _compile_video_kv_prefill(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Give final-video KV prefill its own Dynamo code object and guard cache."""
    return fn(*args, **kwargs)


@dataclass
class CompileTargetState:
    name: str
    compile_mode: str | None = None
    prepared: bool = False
    enabled: bool = False
    failed: bool = False
    calls: int = 0
    failure_reason: str | None = None
    static_inputs: bool = False
    static_bank_prepared: bool = False
    static_stage_calls: int = 0
    static_copy_calls: int = 0
    static_copy_bytes: int = 0
    static_address_tensors: int = 0
    static_address_bytes: int = 0
    persistent_outputs: bool = False
    output_clone_calls: int = 0
    output_clone_tensors: int = 0
    output_clone_bytes: int = 0
    persistent_output_sink: bool = False
    output_sink_calls: int = 0
    output_sink_tensors: int = 0
    output_sink_bytes: int = 0
    resident_constant_tensors: int = 0
    resident_constant_bytes: int = 0
    channels_last_conv2d: int = 0
    channels_last_conv3d: int = 0
    internal_spatial_padding_conv3d: int = 0
    first_frame_conv2d: int = 0
    first_frame_conv2d_bytes: int = 0
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    cudnn_benchmark_limit: int = 0
    rope_compute_dtype: str | None = None
    rope_output_layout: str | None = None
    sdpa_backend: str | None = None
    segmented_attention_backend: str | None = None
    action_gemm_autotune: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _StaticTensorSlot:
    tensor: torch.Tensor
    source_abi: tuple[Any, ...]
    source: torch.Tensor | None = None
    source_identity: tuple[int, int] | None = None


class _StaticInputBank:
    """Fixed-address tensor tree adapted from the O003/O007 static banks.

    The original explicit-graph backends use domain-specific banks around the
    complete Action4/Video4 loops.  The quantization branch compiles the two
    denoisers and final-video KV prefill independently, so this bank preserves
    their ABIs while providing stable tensor addresses to Inductor.
    """

    def __init__(self, *, mark_static_addresses: bool = False) -> None:
        self._template: Any = None
        self.mark_static_addresses = bool(mark_static_addresses)
        self.prepared = False
        self.stage_calls = 0
        self.copy_calls = 0
        self.copy_bytes = 0
        self.static_address_tensors = 0
        self.static_address_bytes = 0

    @staticmethod
    def _source_identity(tensor: torch.Tensor) -> tuple[int, int]:
        return (int(tensor.data_ptr()), int(tensor._version))

    @staticmethod
    def _tensor_abi(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            bool(tensor.requires_grad),
            tensor.layout,
        )

    def _allocate(self, value: Any, *, path: str) -> Any:
        if torch.is_tensor(value):
            if value.layout != torch.strided:
                raise TypeError(
                    f"static input bank supports only strided tensors; "
                    f"{path} has layout {value.layout}"
                )
            destination = torch.empty_like(
                value,
                memory_format=torch.preserve_format,
            )
            if int(destination.data_ptr()) == int(value.data_ptr()):
                raise RuntimeError(f"static input destination aliases source at {path}")
            if self.mark_static_addresses:
                # The bank owns and strongly references this allocation for the
                # compiled callable's lifetime.  Let CUDA Graphs consume it
                # directly; Dynamo will re-record if the address ever changes.
                torch._dynamo.mark_static_address(destination, guard=False)
                self.static_address_tensors += 1
                self.static_address_bytes += int(destination.numel() * destination.element_size())
            # Expanded masks can have zero strides and cannot safely serve as
            # writable destinations.  Record the source ABI independently and
            # let empty_like materialize a writable fixed-address buffer.
            return _StaticTensorSlot(
                destination,
                source_abi=self._tensor_abi(value),
            )
        if isinstance(value, tuple):
            items = [
                self._allocate(item, path=f"{path}[{index}]") for index, item in enumerate(value)
            ]
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return tuple(items)
        if isinstance(value, list):
            return [
                self._allocate(item, path=f"{path}[{index}]") for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            return {
                key: self._allocate(item, path=f"{path}[{key!r}]") for key, item in value.items()
            }
        return value

    def _stage(self, template: Any, value: Any, *, path: str) -> Any:
        if isinstance(template, _StaticTensorSlot):
            if not torch.is_tensor(value):
                raise TypeError(f"static input type changed at {path}")
            destination = template.tensor
            if self._tensor_abi(value) != template.source_abi:
                raise RuntimeError(
                    f"static input ABI changed at {path}: "
                    f"shape={tuple(value.shape)} stride={tuple(value.stride())} "
                    f"dtype={value.dtype} device={value.device}"
                )
            identity = self._source_identity(value)
            # Keep a strong reference to the prior source.  Without it, both
            # CPython and CUDA's allocator may recycle an object's id/address
            # while the replacement tensor also starts at version zero.  A
            # tuple-only cache can then mistake new contents for an unchanged
            # source and replay a graph with stale staged inputs.
            if template.source is not value or template.source_identity != identity:
                destination.copy_(value)
                template.source = value
                template.source_identity = identity
                self.copy_calls += 1
                self.copy_bytes += int(value.numel() * value.element_size())
            return destination
        if isinstance(template, tuple):
            if not isinstance(value, tuple) or len(value) != len(template):
                raise RuntimeError(f"static tuple ABI changed at {path}")
            items = [
                self._stage(slot, item, path=f"{path}[{index}]")
                for index, (slot, item) in enumerate(zip(template, value))
            ]
            if hasattr(template, "_fields"):
                if type(value) is not type(template):
                    raise RuntimeError(f"static named-tuple ABI changed at {path}")
                return type(template)(*items)
            return tuple(items)
        if isinstance(template, list):
            if not isinstance(value, list) or len(value) != len(template):
                raise RuntimeError(f"static list ABI changed at {path}")
            return [
                self._stage(slot, item, path=f"{path}[{index}]")
                for index, (slot, item) in enumerate(zip(template, value))
            ]
        if isinstance(template, dict):
            if not isinstance(value, dict) or tuple(value) != tuple(template):
                raise RuntimeError(f"static mapping ABI changed at {path}")
            return {
                key: self._stage(template[key], value[key], path=f"{path}[{key!r}]")
                for key in template
            }
        if type(value) is not type(template) or value != template:
            raise RuntimeError(
                f"non-tensor static input changed at {path}: {template!r} -> {value!r}"
            )
        return template

    def stage(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        source = (args, kwargs)
        if not self.prepared:
            self._template = self._allocate(source, path="inputs")
            self.prepared = True
        staged_args, staged_kwargs = self._stage(
            self._template,
            source,
            path="inputs",
        )
        self.stage_calls += 1
        return staged_args, staged_kwargs


class SafeCompiledCallable:
    """Lazy torch.compile wrapper with observable fallback state."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        backend: str,
        mode: str | None,
        fullgraph: bool,
        dynamic: bool,
        suppress_errors: bool,
        compile_entrypoint: Callable[..., Any],
        static_inputs: bool,
        mark_static_addresses: bool,
        persistent_outputs: bool = False,
        persistent_output_sink: Callable[[Any], Any] | None = None,
    ) -> None:
        self.fn = fn
        self.name = name
        self.suppress_errors = bool(suppress_errors)
        self.state = CompileTargetState(
            name=name,
            compile_mode=mode,
            prepared=True,
            enabled=True,
            static_inputs=bool(static_inputs),
            persistent_outputs=bool(persistent_outputs),
        )
        self.static_bank = (
            _StaticInputBank(mark_static_addresses=mark_static_addresses) if static_inputs else None
        )
        self.persistent_outputs = bool(persistent_outputs)
        self.persistent_output_sink = persistent_output_sink
        self.state.persistent_output_sink = persistent_output_sink is not None

        if self.suppress_errors:
            import torch._dynamo as torch_dynamo

            torch_dynamo.config.suppress_errors = True

        kwargs: dict[str, Any] = {
            "backend": backend,
            "fullgraph": bool(fullgraph),
            "dynamic": bool(dynamic),
        }
        if mode:
            kwargs["mode"] = mode
        self.compiled_fn = torch.compile(compile_entrypoint, **kwargs)
        logger.info(
            "[torch.compile] prepared %s backend=%s mode=%s fullgraph=%s dynamic=%s",
            name,
            backend,
            mode,
            fullgraph,
            dynamic,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.state.calls += 1
        if not self.state.enabled:
            return self.fn(*args, **kwargs)
        try:
            if self.static_bank is not None:
                args, kwargs = self.static_bank.stage(args, kwargs)
                self.state.static_bank_prepared = self.static_bank.prepared
                self.state.static_stage_calls = self.static_bank.stage_calls
                self.state.static_copy_calls = self.static_bank.copy_calls
                self.state.static_copy_bytes = self.static_bank.copy_bytes
                self.state.static_address_tensors = self.static_bank.static_address_tensors
                self.state.static_address_bytes = self.static_bank.static_address_bytes
            result = self.compiled_fn(self.fn, *args, **kwargs)
            if self.persistent_outputs:
                if self.persistent_output_sink is None:
                    result, tensor_count, payload_bytes = _clone_tensor_tree(result)
                    self.state.output_clone_calls += 1
                    self.state.output_clone_tensors += tensor_count
                    self.state.output_clone_bytes += payload_bytes
                else:
                    tensor_count, payload_bytes = _tensor_tree_stats(result)
                    result = self.persistent_output_sink(result)
                    self.state.output_sink_calls += 1
                    self.state.output_sink_tensors += tensor_count
                    self.state.output_sink_bytes += payload_bytes
            return result
        except Exception as exc:
            self.state.enabled = False
            self.state.failed = True
            self.state.failure_reason = repr(exc)
            if not self.suppress_errors:
                raise
            logger.warning(
                "[torch.compile] %s failed; falling back to eager: %s",
                self.name,
                exc,
            )
            return self.fn(*args, **kwargs)


def enable_edge_inductor_gemm_templates(device: torch.device | str) -> bool:
    """Allow explicit action GEMM autotuning on validated low-SM edge GPUs."""
    resolved = torch.device(device)
    if (
        resolved.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability(resolved) not in {(11, 0), (12, 0), (12, 1)}
    ):
        return False

    from torch._inductor import utils as inductor_utils

    original = inductor_utils.is_big_gpu
    if getattr(original, "_longwam_edge_override", False):
        return True

    def edge_or_original(index_or_device: int | torch.device = 0) -> bool:
        candidate = (
            torch.device("cuda", index_or_device)
            if isinstance(index_or_device, int)
            else torch.device(index_or_device)
        )
        if (
            candidate.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(candidate) in {(11, 0), (12, 0), (12, 1)}
        ):
            return True
        return bool(original(index_or_device))

    setattr(edge_or_original, "_longwam_edge_override", True)
    inductor_utils.is_big_gpu = edge_or_original
    return True


def _clone_tensor_tree(value: Any) -> tuple[Any, int, int]:
    """Clone tensor leaves while preserving cache-entry container types.

    Inductor CUDA Graph replays may reuse graph-owned output addresses.  A
    PreparedWorldContext can outlive one replay, so final-video K/V must not
    alias buffers that a later prefill replay can overwrite.
    """
    if torch.is_tensor(value):
        clone = value.clone(memory_format=torch.preserve_format)
        return clone, 1, int(clone.numel() * clone.element_size())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        cloned_items = [_clone_tensor_tree(item) for item in value]
        return (
            type(value)(*(item[0] for item in cloned_items)),
            sum(item[1] for item in cloned_items),
            sum(item[2] for item in cloned_items),
        )
    if isinstance(value, tuple):
        cloned_items = [_clone_tensor_tree(item) for item in value]
        return (
            tuple(item[0] for item in cloned_items),
            sum(item[1] for item in cloned_items),
            sum(item[2] for item in cloned_items),
        )
    if isinstance(value, list):
        cloned_items = [_clone_tensor_tree(item) for item in value]
        return (
            [item[0] for item in cloned_items],
            sum(item[1] for item in cloned_items),
            sum(item[2] for item in cloned_items),
        )
    if isinstance(value, dict):
        cloned_items = {key: _clone_tensor_tree(item) for key, item in value.items()}
        return (
            {key: item[0] for key, item in cloned_items.items()},
            sum(item[1] for item in cloned_items.values()),
            sum(item[2] for item in cloned_items.values()),
        )
    return value, 0, 0


def _tensor_tree_stats(value: Any) -> tuple[int, int]:
    """Count tensor leaves and payload bytes without changing ownership."""
    if torch.is_tensor(value):
        return 1, int(value.numel() * value.element_size())
    if isinstance(value, (tuple, list)):
        stats = [_tensor_tree_stats(item) for item in value]
    elif isinstance(value, dict):
        stats = [_tensor_tree_stats(item) for item in value.values()]
    else:
        return 0, 0
    return sum(item[0] for item in stats), sum(item[1] for item in stats)


def _sink_action_video_kv_cache(mot: torch.nn.Module, value: Any) -> Any:
    """Copy graph-owned K/V leaves directly into the fixed action bank."""
    mot.install_action_video_kv_cache(value)
    return mot.action_video_kv_cache()


def _sink_video_cross_kv_cache(mot: torch.nn.Module, value: Any) -> Any:
    """Copy graph-owned cross K/V directly into the fixed video bank."""
    mot.install_video_cross_kv_cache(value)
    return mot.video_cross_kv_cache()


def _sink_action_cross_kv_cache(mot: torch.nn.Module, value: Any) -> Any:
    """Copy graph-owned cross K/V directly into the fixed action bank."""
    mot.install_action_cross_kv_cache(value)
    return mot.action_cross_kv_cache()


def _materialize_vae_scale_constants(model: torch.nn.Module) -> tuple[int, int]:
    """Move Wan VAE normalization constants off CPU before graph capture.

    WanVideoVAE stores ``scale`` as an ordinary list rather than registered
    buffers.  Dynamo therefore exposes the CPU tensors as graph inputs and
    Inductor refuses CUDA Graph capture.  Materializing the same FP32 values on
    the model device once keeps the formula unchanged and gives capture stable
    CUDA addresses.
    """
    vae = getattr(model, "vae", None)
    scale = getattr(vae, "scale", None)
    if not isinstance(scale, (list, tuple)) or not scale:
        return 0, 0
    if not all(torch.is_tensor(item) for item in scale):
        return 0, 0
    device = torch.device(getattr(model, "device"))
    resident = [item.detach().to(device=device) for item in scale]
    vae.scale = type(scale)(resident) if isinstance(scale, tuple) else resident
    return (
        len(resident),
        sum(int(item.numel() * item.element_size()) for item in resident),
    )


def _configure_observation_vae_channels_last(
    model: torch.nn.Module,
    device: torch.device | str,
) -> tuple[int, int, bool]:
    """Keep observation-VAE convolution weights in a cuDNN-native layout.

    The optimization is fail-closed to the two UMA Blackwell devices on which
    this path has a campaign contract.  Plan choice and performance remain
    device-specific; supporting a capability here is not performance promotion.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("observation_vae_channels_last requires CUDA")
    capability = torch.cuda.get_device_capability(resolved)
    if capability not in {(11, 0), (12, 0), (12, 1)}:
        raise RuntimeError(
            "observation_vae_channels_last is unvalidated on compute capability "
            f"{capability[0]}.{capability[1]}"
        )

    vae = getattr(model, "vae", None)
    if not isinstance(vae, torch.nn.Module):
        raise ValueError("observation_vae_channels_last requires model.vae")

    torch.backends.cudnn.deterministic = True
    conv2d = 0
    conv3d = 0
    with torch.no_grad():
        for module in vae.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.weight.data = module.weight.data.contiguous(
                    memory_format=torch.channels_last_3d
                )
                conv3d += 1
            elif isinstance(module, torch.nn.Conv2d):
                module.weight.data = module.weight.data.contiguous(
                    memory_format=torch.channels_last
                )
                conv2d += 1

    return conv2d, conv3d, bool(torch.backends.cudnn.deterministic)


def _configure_observation_vae_cudnn_benchmark(
    device: torch.device | str,
    limit: int,
) -> bool:
    """Search a bounded deterministic cuDNN plan set for fixed VAE shapes."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"cuDNN benchmark limit must be a positive integer, got {limit!r}")
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("observation VAE cuDNN benchmark requires CUDA")
    capability = torch.cuda.get_device_capability(resolved)
    if capability not in {(11, 0), (12, 0), (12, 1)}:
        raise RuntimeError(
            "observation VAE cuDNN benchmark is unvalidated on compute capability "
            f"{capability[0]}.{capability[1]}"
        )
    if not torch.backends.cudnn.deterministic:
        raise RuntimeError("observation VAE cuDNN benchmark requires deterministic cuDNN")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = limit
    return True


def _configure_observation_vae_internal_spatial_padding(
    model: torch.nn.Module,
    device: torch.device | str,
) -> int:
    """Move only symmetric H/W zero padding into validated VAE Conv3d calls."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("observation VAE internal spatial padding requires CUDA")
    capability = torch.cuda.get_device_capability(resolved)
    if capability not in {(11, 0), (12, 0), (12, 1)}:
        raise RuntimeError(
            "observation VAE internal spatial padding is unvalidated on compute "
            f"capability {capability[0]}.{capability[1]}"
        )
    vae = getattr(model, "vae", None)
    if not isinstance(vae, torch.nn.Module):
        raise ValueError("observation_vae_internal_spatial_padding requires model.vae")

    count = 0
    for module in vae.modules():
        causal_padding = getattr(module, "_padding", None)
        if (
            isinstance(module, torch.nn.Conv3d)
            and causal_padding is not None
            and len(causal_padding) == 6
            and (causal_padding[0] or causal_padding[2])
        ):
            module._internal_spatial_padding = True
            module.padding = (
                0,
                int(causal_padding[2]),
                int(causal_padding[0]),
            )
            count += 1
    return count


def _configure_observation_vae_first_frame_conv2d(
    model: torch.nn.Module,
    device: torch.device | str,
) -> tuple[int, int]:
    """Prepack only exact encoder T=1/no-cache temporal slices."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("observation VAE first-frame Conv2d requires CUDA")
    capability = torch.cuda.get_device_capability(resolved)
    if capability not in {(11, 0), (12, 0), (12, 1)}:
        raise RuntimeError(
            "observation VAE first-frame Conv2d is unvalidated on compute capability "
            f"{capability[0]}.{capability[1]}"
        )
    vae = getattr(model, "vae", None)
    if not isinstance(vae, torch.nn.Module):
        raise ValueError("observation VAE first-frame Conv2d requires model.vae")

    count = 0
    payload_bytes = 0
    with torch.no_grad():
        for name, module in vae.named_modules():
            if not (
                name.startswith("model.encoder.")
                and hasattr(module, "_first_frame_conv2d_weight")
                and tuple(module.kernel_size) == (3, 3, 3)
                and tuple(module.stride) == (1, 1, 1)
                and tuple(module.dilation) == (1, 1, 1)
                and module.groups == 1
                and getattr(module, "_internal_spatial_padding", False)
                and not (module.in_channels == 640 and module.out_channels == 640)
            ):
                continue
            module._first_frame_conv2d_weight = (
                module.weight[:, :, 2].detach().contiguous(memory_format=torch.channels_last)
            )
            count += 1
            payload_bytes += int(
                module._first_frame_conv2d_weight.numel()
                * module._first_frame_conv2d_weight.element_size()
            )
    return count, payload_bytes


def configure_torch_compile(
    model: torch.nn.Module,
    config: InferenceOptimizationConfig,
) -> dict[str, CompileTargetState]:
    enabled = config.torch_compile
    if enabled == "auto":
        enabled = torch.cuda.is_available()
    if not enabled:
        return {}

    wavefront = bool(config.video_clean_prefix_wavefront)
    target_specs = {
        "observation_vae": (
            model,
            "_encode_video_latents",
            _compile_observation_vae,
            False,
        ),
        "video_cross_kv_prefill": (
            getattr(model, "mot", None),
            "prefill_video_cross_kv_cache",
            _compile_video_cross_kv_prefill,
            False,
        ),
        "video_clean_prefix_prefill": (
            model,
            "_prefill_video_clean_prefix",
            _compile_video_clean_prefix_prefill,
            False,
        ),
        "video_denoiser": (
            model,
            ("_predict_video_noise_with_clean_prefix" if wavefront else "_predict_video_noise"),
            _compile_video_denoiser,
            False,
        ),
        "video_kv_prefill": (
            getattr(model, "mot", None),
            ("prefill_video_cache_with_clean_prefix" if wavefront else "prefill_video_cache"),
            _compile_video_kv_prefill,
            False,
        ),
        "action_cross_kv_prefill": (
            getattr(model, "mot", None),
            "prefill_action_cross_kv_cache",
            _compile_action_cross_kv_prefill,
            False,
        ),
        "action_denoiser": (
            model,
            "_predict_action_noise_with_cache",
            _compile_action_denoiser,
            False,
        ),
    }
    target_modes = dict(config.torch_compile_target_modes)
    states: dict[str, CompileTargetState] = {}
    for target in config.torch_compile_targets:
        owner, method_name, entrypoint, persistent_outputs = target_specs[target]
        target_mode = target_modes.get(target, config.torch_compile_mode)
        action_gemm_autotune = bool(
            target in {"action_cross_kv_prefill", "action_denoiser"}
            and config.action_gemm_autotune
            and getattr(model, "_edge_action_gemm_autotune_enabled", False)
        )
        if action_gemm_autotune:
            target_mode = "max-autotune"
        if target in {
            "observation_vae",
            "video_cross_kv_prefill",
            "video_clean_prefix_prefill",
            "video_kv_prefill",
            "action_cross_kv_prefill",
        }:
            persistent_outputs = target_mode in {"reduce-overhead", "max-autotune"}
        if owner is None or not hasattr(owner, method_name):
            raise ValueError(f"torch.compile target {target!r} requires {method_name!r}")
        original = getattr(owner, method_name)
        if isinstance(original, SafeCompiledCallable):
            states[target] = original.state
            continue
        persistent_output_sink: Callable[[Any], Any] | None = None
        if (
            target == "video_cross_kv_prefill"
            and persistent_outputs
            and config.video_cross_kv_cache
        ):
            persistent_output_sink = partial(
                _sink_video_cross_kv_cache,
                model.mot,
            )
        if (
            target == "video_kv_prefill"
            and persistent_outputs
            and config.action_video_kv_direct_sink
        ):
            persistent_output_sink = partial(
                _sink_action_video_kv_cache,
                model.mot,
            )
        if (
            target == "action_cross_kv_prefill"
            and persistent_outputs
            and config.action_cross_kv_cache
        ):
            persistent_output_sink = partial(
                _sink_action_cross_kv_cache,
                model.mot,
            )
        wrapper = SafeCompiledCallable(
            original,
            name=target,
            backend=config.torch_compile_backend,
            mode=target_mode,
            fullgraph=config.torch_compile_fullgraph,
            dynamic=config.torch_compile_dynamic,
            suppress_errors=config.torch_compile_suppress_errors,
            compile_entrypoint=entrypoint,
            static_inputs=config.torch_compile_static_inputs,
            mark_static_addresses=config.torch_compile_mark_static_addresses,
            persistent_outputs=persistent_outputs,
            persistent_output_sink=persistent_output_sink,
        )
        if target == "observation_vae":
            count, payload_bytes = _materialize_vae_scale_constants(model)
            wrapper.state.resident_constant_tensors = count
            wrapper.state.resident_constant_bytes = payload_bytes
        if (
            target
            in {
                "video_clean_prefix_prefill",
                "video_denoiser",
                "video_kv_prefill",
            }
            and config.video_fp32_rope
        ):
            wrapper.state.rope_compute_dtype = "float32"
            wrapper.state.rope_output_layout = "direct_bf16"
        if target == "action_denoiser" and config.action_fp32_rope:
            wrapper.state.rope_compute_dtype = "float32"
            wrapper.state.rope_output_layout = "direct_bf16"
        if target in {"action_cross_kv_prefill", "action_denoiser"}:
            wrapper.state.action_gemm_autotune = action_gemm_autotune
        if (
            target
            in {
                "video_clean_prefix_prefill",
                "video_denoiser",
                "video_kv_prefill",
            }
            and config.video_cudnn_self_sdpa
        ):
            wrapper.state.sdpa_backend = "cudnn-self-only"
        if (
            target
            in {
                "video_clean_prefix_prefill",
                "video_denoiser",
                "video_kv_prefill",
            }
            and config.video_flex_group_causal_sdpa
        ):
            wrapper.state.sdpa_backend = "flex-group-causal-self-only"
        if target == "action_denoiser" and config.action_segmented_attention:
            wrapper.state.segmented_attention_backend = "triton-split-kv"
        setattr(owner, method_name, wrapper)
        states[target] = wrapper.state
        if target == "observation_vae":
            vae = getattr(model, "vae", None)
    return states
