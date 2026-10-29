# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: scripts/preprocess_action_dit_backbone.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import argparse
import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from longwam.models.wan22.action_dit import ActionDiT
from longwam.models.wan22.wan_video_dit import WanVideoDiT
from longwam.paths import repository_root


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        # Construct inverse permutation to restore original order
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = OmegaConf.load(str(path))
    if "video_dit_config" not in cfg or "action_dit_config" not in cfg:
        raise ValueError(
            f"`{path}` must contain both `video_dit_config` and `action_dit_config` at top level."
        )

    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("`video_dit_config` and `action_dit_config` must resolve to dicts.")

    # Robot-specific I/O tensors are excluded from the generated backbone.
    for config in (video_cfg, action_cfg):
        if _is_unresolved_interpolation(config.get("action_dim")):
            config["action_dim"] = 7

    for key in ["num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim"]:
        action_cfg[key] = _resolve_from_video_cfg(action_cfg.get(key), video_cfg)

    return video_cfg, action_cfg


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def _video_state(payload: Mapping) -> dict[str, torch.Tensor]:
    """Accept the public Robot-S generator, bare video, or full-policy export."""
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Video checkpoint must contain a non-empty state dict.")
    if isinstance(payload.get("mot"), Mapping):
        prefix = "mixtures.video."
        state = {k[len(prefix) :]: v for k, v in payload["mot"].items() if k.startswith(prefix)}
    elif all(isinstance(k, str) and k.startswith("model.") for k in payload):
        state = {k[len("model.") :]: v for k, v in payload.items()}
    else:
        state = dict(payload)
    if not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Video checkpoint must contain only video tensors.")
    return state


def convert_backbone(
    video_state, video_cfg, action_cfg, *, dtype=torch.float32, apply_alpha_scaling=True
):
    """Map tensors on CPU; meta models supply shapes without allocating model weights."""
    action_cfg = dict(action_cfg)
    int_fields = [
        "hidden_dim",
        "action_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
    ]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")
    for key in ("num_heads", "attn_head_dim", "num_layers"):
        if action_cfg[key] != int(video_cfg[key]):
            raise ValueError(f"ActionDiT `{key}` must match the video expert.")
    with torch.device("meta"):
        expected_video = WanVideoDiT(**video_cfg).state_dict()
        action_expert = ActionDiT(**action_cfg)
    missing = sorted(expected_video.keys() - video_state.keys())
    unexpected = sorted(video_state.keys() - expected_video.keys())
    if missing or unexpected:
        raise ValueError(
            f"Video state dict mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for key, expected in expected_video.items():
        source = video_state[key]
        if (
            source.shape != expected.shape
            or not source.is_floating_point()
            or source.device.type != "cpu"
        ):
            raise ValueError(
                f"Invalid video tensor {key}: expected CPU float {tuple(expected.shape)}, got {source.shape}"
            )
    action_state = action_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key].to(dtype=dtype)
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        value = value.detach().to(dtype=dtype).contiguous().clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite converted tensor: {key}")
        backbone_state_dict[key] = value

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    return payload, {
        "copied": copied,
        "interpolated": interpolated,
        "skipped": len(action_state) - len(backbone_keys),
    }


def main() -> None:
    root = repository_root()
    checkpoint_root = Path(
        os.environ.get("LONGWAM_CHECKPOINT_ROOT", root / "checkpoints")
    ).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = root / checkpoint_root
    parser = argparse.ArgumentParser(
        description="Convert public LongLive2.0-Robot-S to an ActionDiT initializer on CPU."
    )
    parser.add_argument("--model-config", type=Path, default=root / "configs/model/long_wam.yaml")
    parser.add_argument(
        "--video-weights",
        type=Path,
        default=os.environ.get("LONGWAM_ROBOT_VIDEO_CHECKPOINT"),
        help="Local Robot-S model.pt; otherwise reuse/download the pinned public release.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=checkpoint_root / "longwam/ActionDiT_from_longlive_ar_1024hdim.pt",
    )
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Apply sqrt(d_video/d_action) scaling (true/false).",
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace an existing initializer."
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output}; choose --output or explicitly use --overwrite."
        )
    registry = OmegaConf.load(root / "configs/checkpoints.yaml")
    release = registry.stage1_robot_s
    source = args.video_weights
    if source is None:
        source = checkpoint_root / "LongLive2.0-Robot-S/model.pt"
        if not source.is_file():
            from huggingface_hub import hf_hub_download

            source = Path(
                hf_hub_download(
                    repo_id=release.repo_id,
                    revision=release.revision,
                    filename="model.pt",
                    local_dir=source.parent,
                )
            )
    source = source.expanduser().resolve(strict=True)
    if source == output:
        parser.error("Output must not overwrite the source video checkpoint.")
    print(f"[INFO] Reading {source} on CPU", flush=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    source_sha256 = digest.hexdigest()
    is_public_robot_s = source_sha256 == release.sha256
    if args.video_weights is None and not is_public_robot_s:
        raise ValueError(
            "Default Robot-S model.pt does not match the pinned release SHA-256; use an explicit --video-weights for custom weights."
        )
    video_cfg, action_cfg = _load_model_config(args.model_config.expanduser())
    state = _video_state(torch.load(source, map_location="cpu", mmap=True, weights_only=True))
    payload, counts = convert_backbone(
        state,
        video_cfg,
        action_cfg,
        dtype=_parse_dtype(args.dtype),
        apply_alpha_scaling=_parse_bool(args.apply_alpha_scaling),
    )
    payload["source"] = {"filename": source.name, "sha256": source_sha256}
    if is_public_robot_s:
        payload["source"].update(repo_id=release.repo_id, revision=release.revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if args.overwrite:
            os.replace(temporary, output)
        else:
            os.link(
                temporary, output
            )  # Atomic publication without overwriting another process's output.
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"[INFO] Saved {output} ({counts}); source_sha256={source_sha256}", flush=True)


if __name__ == "__main__":
    main()
