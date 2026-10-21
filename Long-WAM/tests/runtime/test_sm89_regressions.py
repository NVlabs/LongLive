# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.
"""Small cache/rounding regressions; GPU execution requires explicit opt-in."""

import copy
import os
import unittest
from types import SimpleNamespace

import torch

from longwam.runtime.optim.sm89.static_graph import (
    StaticGraphCallable,
    source_state,
    unchanged_source,
)


class SourceIdentityTests(unittest.TestCase):
    def test_mutation_and_graph_epoch_invalidate_cached_sources(self):
        value = torch.ones(4)
        previous = source_state(value)
        self.assertTrue(unchanged_source(previous, source_state(value)))
        value.add_(1)
        self.assertFalse(unchanged_source(previous, source_state(value)))
        previous = source_state(value)
        value._wam4090_graph_epoch = (123, 1)
        self.assertFalse(unchanged_source(previous, source_state(value)))
        self.assertFalse(
            unchanged_source(source_state(value), source_state(value.clone()))
        )


class RequestContextRegressionTests(unittest.TestCase):
    def test_new_context_and_unmerged_adapter_values_refresh_each_request(self):
        from longwam.runtime.optim.sm89.context_cache import RequestContextCache
        from torch import nn
        import torch.nn.functional as functional

        class UnmergedLinear(nn.Linear):
            def __init__(self):
                super().__init__(4, 6)
                self.lora_A = nn.Parameter(torch.randn(2, 4, dtype=torch.float32))
                self.lora_B = nn.Parameter(torch.randn(6, 2, dtype=torch.float32) * 0.2)

            def forward(self, value):
                base = functional.linear(value, self.weight, self.bias)
                with torch.autocast(device_type=value.device.type, enabled=False):
                    delta = functional.linear(
                        functional.linear(value.float(), self.lora_A), self.lora_B
                    )
                return base + delta.to(base.dtype)

        class Cross(nn.Module):
            def __init__(self):
                super().__init__()
                self.k, self.v, self.norm_k = (
                    UnmergedLinear(),
                    nn.Linear(4, 6),
                    nn.LayerNorm(6),
                )

            def forward(self, value):
                return self.norm_k(self.k(value)) + self.v(value)

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_embedding = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
                block = nn.Module()
                block.cross_attn = Cross()
                self.blocks = nn.ModuleList([block])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.video_expert, self.action_expert = Expert(), Expert()

            def _append_proprio_to_context(self, context, mask, proprio):
                return torch.cat([context, proprio.reshape(1, 1, 4)], dim=1), mask

            @torch.no_grad()
            def infer_joint_ar(self, *, context, proprio, text_cfg_scale=1):
                context, _ = self._append_proprio_to_context(context, None, proprio)
                result = 0
                for _ in range(3):
                    for expert in (self.video_expert, self.action_expert):
                        result = result + expert.blocks[0].cross_attn(
                            expert.text_embedding(context)
                        )
                return result

        torch.manual_seed(37)
        reference = Model().eval().requires_grad_(False)
        candidate = copy.deepcopy(reference)
        keys = set(candidate.state_dict())
        cache = RequestContextCache(candidate)
        context, proprio = torch.randn(1, 3, 4), torch.randn(1, 4)
        outputs = []
        for index in range(3):
            if index == 1:
                context.add_(0.3)
            if index == 2:
                for model in (reference, candidate):
                    model.action_expert.blocks[0].cross_attn.k.lora_B[0, 0].add_(0.4)
            expected = reference.infer_joint_ar(context=context, proprio=proprio)
            actual = candidate.infer_joint_ar(context=context, proprio=proprio)
            self.assertTrue(torch.equal(expected, actual))
            outputs.append(actual.clone())
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[1], outputs[2]))
        self.assertEqual(cache.preparations, 3)
        self.assertEqual(set(candidate.state_dict()), keys)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]))


@unittest.skipUnless(
    os.environ.get("RUN_SM89_GPU_TESTS") == "1",
    "Set RUN_SM89_GPU_TESTS=1 only on an authorized idle SM89 GPU",
)
class SM89GPURegressions(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    def test_graph_epoch_refresh_and_live_fp32_parameters(self):
        torch.manual_seed(23)
        layer = torch.nn.Linear(8, 8).cuda().eval().requires_grad_(False)
        producer = StaticGraphCallable(layer, name="regression_live_fp32")
        consumer = StaticGraphCallable(
            lambda x: x + 0.25, name="regression_epoch_consumer"
        )
        value = torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for index in range(3):
                value.fill_(index + 1)
                layer.weight.add_(0.125)
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    expected = (layer(value) + 0.25).clone()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = consumer(producer(value))
                self.assertTrue(
                    torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
                )
        self.assertEqual(producer.captures, 1)
        self.assertEqual(consumer.captures, 1)
        self.assertEqual(producer.replays, 3)
        self.assertEqual(consumer.replays, 3)
        self.assertEqual(layer.weight.dtype, torch.float32)

    def test_all_finite_bf16_operand_patterns_keep_original_rounding(self):
        from longwam.runtime.optim.sm89.rounded_pointwise import (
            original_gate,
            original_modulate,
            rounded_gate,
            rounded_modulate,
        )

        bits = torch.arange(65536, dtype=torch.int32)
        finite = bits[(bits & 0x7F80) != 0x7F80].to(torch.int16).view(torch.bfloat16)
        self.assertEqual(finite.numel(), 65280)
        value = finite.reshape(1, 255, 256).cuda()
        residual = finite.roll(541).reshape_as(value).cuda()
        table = torch.empty(1, 255, 6, 256, device="cuda", dtype=torch.bfloat16)
        for index in range(6):
            table[:, :, index].copy_(finite.roll(137 * (index + 1)).reshape_as(value))
        with torch.no_grad():
            for channel_only in (False, True):
                gate = table[:, : 1 if channel_only else 255, 2]
                shift = table[:, :, 0]
                scale = table[:, : 1 if channel_only else 255, 1]
                for actual, expected in (
                    (
                        rounded_gate(value, gate, residual),
                        original_gate(value, gate, residual),
                    ),
                    (
                        rounded_modulate(value, shift, scale),
                        original_modulate(value, shift, scale),
                    ),
                ):
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.uint8), expected.view(torch.uint8)
                        )
                    )
        # Operand-pattern coverage, not every combination of three operands.

    def test_shape_preserving_gemm_keeps_original_fp32_cast_live(self):
        from longwam.runtime.optim.sm89.gemm_shapes import VideoGemmShapes
        import torch.nn.functional as functional

        torch.manual_seed(29)
        expert = torch.nn.Module()
        expert.patch_size = (1, 1, 1)
        layer = torch.nn.Linear(8, 8).cuda().eval().requires_grad_(False)
        expert.time_embedding = torch.nn.Sequential(layer)
        original = layer.forward
        model = SimpleNamespace(
            video_expert=expert,
            num_clean_frames=4,
            num_imagine_frames=2,
            _prefill_video_clean_prefix=lambda clean, value: expert.time_embedding(
                value
            ),
        )
        helper = VideoGemmShapes(model, ("time",))
        graph = StaticGraphCallable(
            model._prefill_video_clean_prefix, name="regression_gemm"
        )
        clean = torch.zeros(1, 1, 4, 2, 2, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for index in range(3):
                layer.weight.add_(0.125)
                value = torch.full(
                    (1, 16, 8), float(index + 1), device="cuda", dtype=torch.bfloat16
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    expected = original(functional.pad(value, (0, 0, 0, 8)))[
                        :, :16
                    ].clone()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = graph(clean, value)
                self.assertTrue(
                    torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
                )
        self.assertEqual(helper.geometry, (16, 8, 24))
        self.assertEqual(graph.captures, 1)


if __name__ == "__main__":
    unittest.main()
