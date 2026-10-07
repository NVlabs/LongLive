# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/inference.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

"""Portable checkpoint loading and observation-policy construction.

Weight tensor names are unchanged. Only legacy Python/config names are mapped;
the legacy framework is never imported or required.
"""

from collections.abc import Mapping
from pathlib import Path


def migrate_config_names(value):
    """Read old resolved YAML without rewriting tensor keys or filesystem paths."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            name = "longwam_" + key[6:] if key.startswith("arwam_") else key
            out[name] = migrate_config_names(item)
        return out
    if isinstance(value, list):
        return [migrate_config_names(item) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("fastwam."):
        value = "longwam." + value[len("fastwam.") :]
        replacements = {
            "create_arwam": "create_longwam",
            "create_fastwam_joint": "create_longwam_joint",
            "create_fastwam_idm": "create_longwam_idm",
            "create_fastwam": "create_longwam_base",
            "processors.fastwam_processor.FastWAMProcessor": "processors.longwam_processor.LongWAMProcessor",
            "wan22.ar_wam.ARWAM": "wan22.long_wam.LongWAM",
            "wan22.fastwam.FastWAM": "wan22.longwam_base.LongWAMBase",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value.replace("${model.arwam_", "${model.longwam_")


def load_run_config(path, *, domain=None):
    from omegaconf import OmegaConf
    from .utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    raw = OmegaConf.load(Path(path).expanduser().resolve(strict=True))
    cfg = OmegaConf.create(migrate_config_names(OmegaConf.to_container(raw, resolve=False)))
    # Resolve architecture before selecting one domain of a replay mixture.
    cfg.model = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if "domains" in cfg.data.train:
        if domain not in cfg.data.train.domains:
            raise ValueError(f"Choose an inference domain from {list(cfg.data.train.domains)}")
        node = OmegaConf.to_container(cfg.data.train.domains[domain], resolve=True)
        cfg.data.train = OmegaConf.create(node)
    return cfg


def model_config_for_inference(cfg, *, load_text_encoder=True):
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = bool(load_text_encoder)
    model_cfg.skip_dit_load_from_pretrain = True
    if model_cfg.get("_target_") == "longwam.runtime.create_longwam":
        model_cfg.skip_video_dit_load_from_pretrain = True
        model_cfg.skip_action_dit_load_from_pretrain = True
        model_cfg.longlive_video_weights = None
    model_cfg.action_dit_pretrained_path = None
    return model_cfg


def load_model(cfg, checkpoint, *, device="cuda", load_text_encoder=True, expected_step=None,
               strict=True, load_pretrained=False):
    import torch
    from hydra.utils import instantiate
    from .paths import setup_model_paths
    from .runtime import _mixed_precision_to_model_dtype

    path = Path(checkpoint).expanduser().resolve(strict=True)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no implicit CPU fallback for benchmark inference.")
    setup_model_paths()
    model = instantiate(
        cfg.model if load_pretrained else model_config_for_inference(
            cfg, load_text_encoder=load_text_encoder
        ),
        model_dtype=_mixed_precision_to_model_dtype(str(cfg.get("mixed_precision", "bf16"))),
        device=device,
    )
    payload = model.load_checkpoint(str(path), strict=bool(strict))
    if expected_step is not None and int(payload.get("step", -1)) != int(expected_step):
        raise ValueError(f"Stored checkpoint step {payload.get('step')} != {expected_step}")
    return model.to(device).eval()


class TextCache:
    """Same prompt hashes and context format as training, without inode/path seals."""

    def __init__(self, directory, context_len=128):
        self.directory = Path(directory).expanduser().resolve(strict=True)
        self.context_len = int(context_len)
        self._cache = {}

    def __call__(self, prompt):
        import torch
        from .datasets.lerobot.robot_video_dataset import get_text_cache_path

        if prompt not in self._cache:
            path = Path(get_text_cache_path(self.directory, str(prompt), self.context_len))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or set(payload) != {"context", "mask"}:
                raise ValueError(f"Invalid text-cache payload: {path}")
            context = torch.as_tensor(payload["context"])
            mask = torch.as_tensor(payload["mask"], dtype=torch.bool)
            if tuple(context.shape) != (self.context_len, 4096) or tuple(mask.shape) != (
                self.context_len,
            ):
                raise ValueError(f"Incorrect text-context shape: {path}")
            if not bool(torch.isfinite(context).all()) or not bool(mask.any()):
                raise ValueError(f"Non-finite or empty text context: {path}")
            # Match training exactly: zero padded vectors and then expose the
            # entire fixed-length sequence. Do not change the attention mask.
            context = context.clone()
            context[~mask] = 0
            self._cache[prompt] = (context, torch.ones_like(mask))
        return self._cache[prompt]


class TextConditioning:
    """Encode each task once, or read its training-compatible cached embedding."""

    def __init__(self, model, directory=None, context_len=128):
        self.model = model
        self.disk = TextCache(directory, context_len) if directory else None
        self._contexts = {}

    def __call__(self, prompt):
        if prompt not in self._contexts:
            import torch
            with torch.no_grad():
                self._contexts[prompt] = (
                    self.disk(prompt) if self.disk else self.model.encode_prompt(prompt)
                )
        context, mask = self._contexts[prompt]
        return {"prompt": None, "context": context, "context_mask": mask}


def load_benchmark_policy(settings):
    if settings.benchmark in {"libero", "robotwin2"}:
        from .runtime import create_policy
        return create_policy(settings)
    from .datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    cfg = load_run_config(settings.model_config)
    if settings.benchmark == "robocasa_gr1":
        from .benchmarks.robocasa_gr1.policy import (
            RoboCasaGR1LongWAMPolicy,
            _validate_loaded_model,
            _validate_run_config,
        )
        from .datasets.lerobot.transforms.gr1_tabletop import (
            GR1ActionNormalizer,
            GR1OfficialVideoTransform,
            fieldwise_sincos_state,
        )

        temporal = _validate_run_config(cfg)
        if settings.get("expected_history") is not None:
            if int(cfg.data.train.past_obs_size) != int(settings.expected_history):
                raise ValueError("Requested GR1 history differs from checkpoint model/data config")
        text = TextCache(settings.text_cache, int(cfg.data.train.context_len))
        normalizer = GR1ActionNormalizer.from_json(
            Path(settings.stats),
            expected_dataset_revision=str(cfg.data.train.expected_dataset_revision),
        )
        model = load_model(
            cfg,
            settings.checkpoint,
            device=settings.device,
            load_text_encoder=False,
            expected_step=settings.expected_step,
        )
        _validate_loaded_model(model, temporal)
        return RoboCasaGR1LongWAMPolicy(
            model=model,
            video_transform=GR1OfficialVideoTransform(training=False),
            state_transform=fieldwise_sincos_state,
            action_normalizer=normalizer,
            prompt_template=DEFAULT_PROMPT,
            text_context=text,
            policy_seed=settings.policy_seed,
            num_inference_steps=int(settings.num_inference_steps),
            temporal_spec=temporal,
        )
    if settings.benchmark == "robocasa365":
        from hydra.utils import instantiate
        from .benchmarks.robocasa.policy import (
            RoboCasaLongWAMPolicy,
            _validate_loaded_model,
            _validate_run_config,
        )
        from .datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        layout = _validate_run_config(cfg)
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(settings.stats)))
        model = load_model(
            cfg, settings.checkpoint, device=settings.device, expected_step=settings.expected_step
        )
        _validate_loaded_model(model)
        return RoboCasaLongWAMPolicy(
            model=model,
            processor=processor,
            prompt_template=DEFAULT_PROMPT,
            policy_seed=settings.policy_seed,
            concat_multi_camera=layout,
            num_inference_steps=int(settings.num_inference_steps),
        )
    raise ValueError(f"Standalone policy server is not defined for {settings.benchmark}")
