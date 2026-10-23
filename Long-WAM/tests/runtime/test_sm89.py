# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""CPU tests of profile admission and API contracts, with no model/GPU claim."""

import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from longwam.runtime.backends import CAPABILITIES, prepare_backend
from longwam.runtime.optim import sm89
from longwam.runtime.optim.model import InferenceModelMixin
from longwam.runtime.settings import normalize_settings


class MetadataOnlyCudaTensor(torch.Tensor):
    """CPU-backed metadata fixture; never used to execute a CUDA kernel."""

    @property
    def device(self):
        return torch.device("cuda")


class FakeModel(InferenceModelMixin):
    device = "cuda"
    training = False
    joint_denoise = False
    num_clean_frames = 4
    num_imagine_frames = 2
    imagine_infer_steps = 4

    def __init__(self):
        self._inference_optimization_prepared = False
        self._inference_optimization_config = None
        self._inference_optimization_state = None
        weight = torch.ones(1, dtype=torch.bfloat16, device="cpu")
        self.video_expert = SimpleNamespace(patch_embedding=SimpleNamespace(weight=weight))
        self.action_expert = SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=SimpleNamespace(q=SimpleNamespace(weight=weight)))]
        )
        self.mot = SimpleNamespace(kv_cache_quantizer=None)
        self.adapter = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float32, device="cpu"))
        self.calls = []

    def infer_joint_ar(self, *, action_horizon, num_inference_steps, obs_window, **kwargs):
        self.calls.append(
            dict(
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                obs_window=obs_window,
                **kwargs,
            )
        )
        return "original-inference-result"

    def named_parameters(self):
        yield "adapter", self.adapter


class FakeGraph:
    def telemetry(self):
        return {"captures": 0, "replays": 0}


class SM89BackendTests(unittest.TestCase):
    def fake_apply(self, model, profile):
        self.applied.append((model, profile))
        return {"name": profile, "batch_rgb_upload": True, "wrappers": {"example": FakeGraph()}}

    def setUp(self):
        self.applied = []

    def test_admission_does_not_quantize_or_change_adapter_or_signature(self):
        model = FakeModel()
        signature = inspect.signature(model.infer_joint_ar)
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(sm89, "apply_profile", self.fake_apply),
        ):
            state = prepare_backend(model, "rtx4090", {"video_rope_grid_size": (6, 7, 14)})
            self.assertIs(prepare_backend(model, "rtx4090"), state)
        self.assertEqual(len(self.applied), 1)
        self.assertEqual(model.adapter.dtype, torch.float32)
        self.assertTrue(model.adapter.requires_grad)
        self.assertEqual(float(model.adapter.detach()), float(torch.tensor(0.4)))
        self.assertEqual(inspect.signature(model.infer_joint_ar), signature)
        telemetry = state.to_dict()
        json.dumps(telemetry)
        self.assertFalse(telemetry["model_quant"])
        self.assertFalse(telemetry["kv_quant_enabled"])
        self.assertFalse(telemetry["batch_rgb_upload"])
        self.assertEqual(CAPABILITIES["rtx4090"], (8, 9))
        self.assertEqual(CAPABILITIES["rtx5090"], (12, 0))

    def test_unsupported_hardware_training_and_overrides_fail_before_transform(self):
        for condition in ("cpu", "wrong_gpu", "training", "quant_override"):
            with self.subTest(condition=condition):
                model = FakeModel()
                if condition == "cpu":
                    model.device = "cpu"
                if condition == "training":
                    model.training = True
                capability = (8, 6) if condition == "wrong_gpu" else (8, 9)
                overrides = {"model_quant": True} if condition == "quant_override" else {}
                with (
                    patch.object(torch.cuda, "get_device_capability", return_value=capability),
                    patch.object(sm89, "apply_profile", self.fake_apply),
                ):
                    with self.assertRaises(ValueError):
                        prepare_backend(model, "rtx4090", overrides)
        self.assertEqual(self.applied, [])

    def test_protocol_guards_preserve_valid_kwargs_and_never_rewrite_steps(self):
        model = FakeModel()
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(sm89, "apply_profile", self.fake_apply),
        ):
            prepare_backend(model, "rtx4090")
        video = torch.zeros(1, 3, 13, 2, 2, dtype=torch.bfloat16, device="cpu").as_subclass(
            MetadataOnlyCudaTensor
        )
        arguments = dict(
            action_horizon=32,
            num_inference_steps=10,
            video_num_inference_steps=4,
            obs_window=video,
            seed=17,
            rand_device="cpu",
        )
        self.assertEqual(model.infer_joint_ar(**arguments), "original-inference-result")
        self.assertEqual(model.calls[0]["seed"], 17)
        self.assertIs(model.calls[0]["obs_window"], video)
        with torch.no_grad():
            model.adapter.add_(0.1)
        self.assertEqual(model.infer_joint_ar(**arguments), "original-inference-result")
        changes = (
            {"num_inference_steps": 8},
            {"video_num_inference_steps": 2},
            {"action_horizon": 16},
            {"text_cfg_scale": 2.0},
            {"clean_latents": torch.zeros(1)},
            {"tiled": True},
            {"rand_device": "cuda"},
            {"obs_window": torch.zeros(1, 3, 13, 2, 2, dtype=torch.bfloat16)},
        )
        for change in changes:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    model.infer_joint_ar(**dict(arguments, **change))
        self.assertEqual(len(model.calls), 2)
        model.adapter = torch.nn.Parameter(model.adapter.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "bindings changed"):
            model.infer_joint_ar(**arguments)

    def test_settings_keep_reference_and_other_backends_separate(self):
        cfg = normalize_settings(
            {"benchmark": "libero", "hardware": "rtx4090", "action_horizon": 32, "replan_steps": 10}
        )
        self.assertEqual(cfg.execution, "sync")
        self.assertEqual(normalize_settings({"benchmark": "libero"}).hardware, "reference")
        for change in ({"execution": "async"}, {"streaming_vae": True}, {"action_horizon": 80}):
            with self.assertRaises(ValueError):
                normalize_settings({"benchmark": "libero", "hardware": "rtx4090", **change})


if __name__ == "__main__":
    unittest.main()
