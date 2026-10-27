# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""CPU admission and installation regressions; no CUDA computation is claimed."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from longwam.runtime.backends import prepare_backend
from longwam.runtime.optim import sm89
from longwam.runtime.optim.sm89 import profiles
from test_sm89 import FakeModel, FakeGraph


def fast_model():
    model = FakeModel()
    model.video_expert.patch_size = (1, 2, 2)
    model.video_expert.video_attention_mask_mode = "per_frame_causal"
    model.mot.mixtures = {"video": model.video_expert}
    return model


def fake_apply(_model, name):
    return {"name": name, "wrappers": {}}


class FastModeTests(unittest.TestCase):
    def test_explicit_fast_identity_idempotence_and_cross_mode_rejection(self):
        model = fast_model()
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(sm89, "apply_profile", side_effect=fake_apply) as install,
        ):
            state = prepare_backend(model, "rtx4090", {"rtx4090_mode": "fast"})
            self.assertEqual(state.profile, "graph_clean_prefix")
            self.assertEqual(state.to_dict()["mode"], "fast")
            self.assertIs(prepare_backend(model, "rtx4090", {"rtx4090_mode": "fast"}), state)
            with self.assertRaisesRegex(RuntimeError, "mix"):
                prepare_backend(model, "rtx4090")
            install.assert_called_once_with(model, "graph_clean_prefix")

    def test_default_stays_strict_and_rejects_reconfiguration(self):
        model = FakeModel()
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(sm89, "apply_profile", side_effect=fake_apply),
        ):
            state = prepare_backend(model, "rtx4090")
            self.assertEqual(state.profile, "graph_first_prefix_direct")
            self.assertEqual(state.to_dict()["mode"], "strict")
            with self.assertRaisesRegex(RuntimeError, "mix"):
                prepare_backend(model, "rtx4090", {"rtx4090_mode": "fast"})

    def test_invalid_fast_conditions_fail_before_configuration_or_install(self):
        for condition in ["patch", "causality", "expert_identity", "existing_prefix", "bad_mode"]:
            with self.subTest(condition=condition):
                model = fast_model()
                mode = "fast"
                if condition == "patch":
                    model.video_expert.patch_size = (2, 2, 2)
                elif condition == "causality":
                    model.video_expert.video_attention_mask_mode = "bidirectional"
                elif condition == "expert_identity":
                    model.mot.mixtures["video"] = object()
                elif condition == "existing_prefix":
                    model._video_clean_prefix_wavefront_enabled = True
                else:
                    mode = "auto"
                original = model.infer_joint_ar
                with (
                    patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
                    patch.object(sm89, "apply_profile") as install,
                ):
                    with self.assertRaises(ValueError):
                        prepare_backend(model, "rtx4090", {"rtx4090_mode": mode})
                    install.assert_not_called()
                self.assertIsNone(model._inference_optimization_config)
                self.assertFalse(model._inference_optimization_prepared)
                self.assertEqual(model.infer_joint_ar, original)

    def test_fast_uses_existing_shared_graph_helpers_without_strict_kernels(self):
        model = fast_model()
        model.device = "cpu"
        model.action_expert.freqs = torch.ones(2, dtype=torch.complex128)
        model.video_expert.freqs = (torch.ones(2), torch.ones(2))
        model.vae = SimpleNamespace(scale=[torch.ones(2), torch.ones(2)])
        model.mot.prefill_video_cache_with_clean_prefix = lambda *a, **k: None
        model._encode_video_latents = lambda *a, **k: None
        model._prefill_video_clean_prefix = lambda *a, **k: None
        model._predict_video_noise_with_clean_prefix = lambda *a, **k: None
        model._predict_action_noise_with_cache = lambda *a, **k: None
        model.get_submodule = lambda name: model.mot if name == "mot" else None
        with (
            patch.object(profiles, "VideoGemmShapes") as gemm,
            patch.object(profiles, "RoundedPointwise") as rounding,
            patch.object(profiles, "FirstPassPrefix") as first,
            patch.object(profiles, "RequestContextCache", return_value=FakeGraph()) as context,
            patch.object(
                profiles, "StaticGraphCallable", side_effect=lambda fn, **kw: FakeGraph()
            ) as graph,
        ):
            telemetry = profiles.apply_profile(model, "graph_clean_prefix")
        gemm.assert_not_called()
        rounding.assert_not_called()
        first.assert_not_called()
        context.assert_called_once_with(model, capture_projections=True)
        self.assertEqual(graph.call_count, 5)
        self.assertTrue(model._video_clean_prefix_wavefront_enabled)
        self.assertEqual(model.action_expert.freqs.dtype, torch.complex128)
        self.assertEqual(
            set(telemetry["wrappers"]),
            {"request_context_projections", *profiles.FAST_GRAPH_METHODS},
        )


if __name__ == "__main__":
    unittest.main()
