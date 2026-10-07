# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot integration imported from the author video-training branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/tests/test_padding_aware_error_recycling.py
# Changes: Robot-training/validation additions; existing upstream notices are retained where present.
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

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

from utils.error_buffer import ErrorBuffer


class _StubBaseModel(torch.nn.Module):
    pass


def _load_causal_diffusion_with_stubs():
    """Load the ER implementation without constructing the Wan backbone."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "model" / "diffusion.py"
    saved = {
        name: sys.modules.get(name)
        for name in (
            "model",
            "model.base",
            "pipeline",
            "utils.wan_5b_wrapper",
        )
    }

    fake_model = types.ModuleType("model")
    fake_model.__path__ = []
    fake_base = types.ModuleType("model.base")
    fake_base.BaseModel = _StubBaseModel

    fake_pipeline = types.ModuleType("pipeline")
    fake_pipeline.CausalDiffusionInferencePipeline = object

    fake_wrapper = types.ModuleType("utils.wan_5b_wrapper")
    fake_wrapper.WanDiffusionWrapper = object
    fake_wrapper.WanTextEncoder = object
    fake_wrapper.WanVAEWrapper = object

    sys.modules["model"] = fake_model
    sys.modules["model.base"] = fake_base
    sys.modules["pipeline"] = fake_pipeline
    sys.modules["utils.wan_5b_wrapper"] = fake_wrapper
    try:
        spec = importlib.util.spec_from_file_location(
            "_padding_aware_diffusion_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CausalDiffusion
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _BufferLayout:
    def __init__(self, num_blocks=0):
        self.num_blocks = num_blocks


class _ConstantBuffer:
    num_blocks = 0

    def __init__(self, block_size, value):
        self.entry = torch.full((block_size, 1, 1, 1), float(value))
        self.sample_calls = []

    def sample(self, timestep_index, device, dtype, block_pos=None, valid=True):
        self.sample_calls.append(("matched", block_pos, bool(valid)))
        if not valid:
            return None
        return self.entry.to(device=device, dtype=dtype)

    def sample_global(self, device, dtype, valid=True):
        self.sample_calls.append(("global", None, bool(valid)))
        if not valid:
            return None
        return self.entry.to(device=device, dtype=dtype)


class PaddingAwareErrorRecyclingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.CausalDiffusion = _load_causal_diffusion_with_stubs()

    def setUp(self):
        self.model = self.CausalDiffusion.__new__(self.CausalDiffusion)
        torch.nn.Module.__init__(self.model)
        self.model.device = torch.device("cpu")
        self.model.num_frame_per_block = 8
        self.model.er_num_blocks = 0
        self.model.er_block_offset = 0
        self.model.er_skip_block_0 = False

    def _local_mask(self, global_mask, sp_rank, context_frames):
        local = global_mask.chunk(4, dim=1)[sp_rank].contiguous()
        return self.model._build_er_block_valid_mask(
            loss_mask=local,
            batch_size=1,
            num_frame=16,
            context_frames=context_frames,
            device=torch.device("cpu"),
        )

    def test_sp4_block8_masks_every_mixed_length(self):
        # Every short length is padded to the fixed F64 carrier. Frame zero is
        # a valid I2V context frame even though it is excluded from loss.
        expected_by_length = {
            32: ([True, True], [True, True], [False, False], [False, False]),
            40: ([True, True], [True, True], [True, False], [False, False]),
            48: ([True, True], [True, True], [True, True], [False, False]),
            56: ([True, True], [True, True], [True, True], [True, False]),
            64: ([True, True], [True, True], [True, True], [True, True]),
        }
        for valid_frames, expected_ranks in expected_by_length.items():
            with self.subTest(valid_frames=valid_frames):
                global_mask = torch.zeros(1, 64)
                global_mask[:, 1:valid_frames] = 1
                for sp_rank in range(4):
                    context_frames = 1 if sp_rank == 0 else 0
                    actual = self._local_mask(
                        global_mask, sp_rank, context_frames
                    ).tolist()
                    self.assertEqual(actual, [expected_ranks[sp_rank]])
                self.assertEqual(
                    int(global_mask.sum().item()), valid_frames - 1
                )

    def test_local_collection_keeps_only_valid_blocks_without_reindexing(self):
        error = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1, 1, 1)
        index = torch.arange(8, dtype=torch.long).repeat_interleave(8).unsqueeze(0)
        f32_valid = torch.tensor(
            [[True, True, True, True, False, False, False, False]]
        )
        f64_valid = torch.ones((1, 8), dtype=torch.bool)

        f32_items = self.model._collect_local_items(
            _BufferLayout(), error, index, 1, 64,
            block_valid_mask=f32_valid,
        )
        f64_items = self.model._collect_local_items(
            _BufferLayout(), error, index, 1, 64,
            block_valid_mask=f64_valid,
        )

        self.assertEqual(len(f32_items), 4)
        self.assertEqual([item[1] for item in f32_items], [0, 1, 2, 3])
        self.assertEqual([item[0][0].item() for item in f32_items], [0, 8, 16, 24])
        self.assertEqual(len(f64_items), 8)
        self.assertEqual([item[1] for item in f64_items], list(range(8)))

    def test_all_injection_paths_leave_invalid_tail_untouched(self):
        valid = torch.tensor([[True, False]])
        value = torch.zeros(1, 16, 1, 1, 1)
        index = torch.zeros(1, 16, dtype=torch.long)
        error_buffer = _ConstantBuffer(block_size=8, value=3)
        noise_buffer = _ConstantBuffer(block_size=8, value=5)
        self.model.error_buffer = error_buffer
        self.model.noise_error_buffer = noise_buffer

        context_out = self.model._inject_error_buffer(
            value, index, 1, 16, block_valid_mask=valid
        )
        latent_out = self.model._inject_latent_error_buffer(
            value, index, 1, 16, block_valid_mask=valid
        )
        noise_out = self.model._inject_noise_error_buffer(
            value, index, 1, 16, block_valid_mask=valid
        )

        self.assertTrue(torch.equal(context_out[:, :8], torch.full_like(value[:, :8], 3)))
        self.assertTrue(torch.equal(latent_out[:, :8], torch.full_like(value[:, :8], 3)))
        self.assertTrue(torch.equal(noise_out[:, :8], torch.full_like(value[:, :8], 5)))
        self.assertFalse(context_out[:, 8:].any())
        self.assertFalse(latent_out[:, 8:].any())
        self.assertFalse(noise_out[:, 8:].any())
        self.assertEqual(len(error_buffer.sample_calls), 2)
        self.assertEqual(len(noise_buffer.sample_calls), 1)

    def test_i2v_context_error_slot_is_zero_when_replayed_cross_position(self):
        loss_mask = torch.ones(1, 16)
        loss_mask[:, 0] = 0
        block_valid = self.model._build_er_block_valid_mask(
            loss_mask,
            batch_size=1,
            num_frame=16,
            context_frames=1,
            device=torch.device("cpu"),
        )
        store_frames = self.model._build_er_store_frame_mask(
            loss_mask,
            batch_size=1,
            num_frame=16,
            context_frames=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(block_valid.tolist(), [[True, True]])
        self.assertFalse(store_frames[0, 0])
        self.assertTrue(store_frames[0, 1:].all())

        index = torch.zeros(1, 16, dtype=torch.long)
        source_blocks = torch.tensor([[True, False]])
        target_blocks = torch.tensor([[False, True]])
        frame_mask = store_frames.reshape(1, 16, 1, 1, 1)

        error_buffer = ErrorBuffer(
            num_buckets=1,
            max_size_per_bucket=2,
            num_train_timesteps=1000,
            modulate_factor=0,
        )
        noise_buffer = ErrorBuffer(
            num_buckets=1,
            max_size_per_bucket=2,
            num_train_timesteps=1000,
            modulate_factor=0,
        )
        latent_error = torch.ones(1, 16, 1, 1, 1) * frame_mask
        noise_error = torch.full_like(latent_error, 2.0) * frame_mask
        self.model._apply_gathered_items(
            error_buffer,
            self.model._collect_local_items(
                error_buffer,
                latent_error,
                index,
                1,
                16,
                block_valid_mask=source_blocks,
            ),
        )
        self.model._apply_gathered_items(
            noise_buffer,
            self.model._collect_local_items(
                noise_buffer,
                noise_error,
                index,
                1,
                16,
                block_valid_mask=source_blocks,
            ),
        )
        self.model.error_buffer = error_buffer
        self.model.noise_error_buffer = noise_buffer

        context_out = self.model._inject_error_buffer(
            torch.zeros_like(latent_error),
            index,
            1,
            16,
            block_valid_mask=target_blocks,
        )
        noise_out = self.model._inject_noise_error_buffer(
            torch.zeros_like(noise_error),
            index,
            1,
            16,
            block_valid_mask=target_blocks,
        )

        # The context source occupied slot zero of block 0.  When its error is
        # replayed into block 1, that slot must remain zero rather than
        # corrupting frame 8; supervised slots 9..15 still carry the error.
        self.assertEqual(context_out[0, 8].item(), 0.0)
        self.assertTrue(torch.equal(context_out[0, 9:], torch.ones_like(context_out[0, 9:])))
        self.assertEqual(noise_out[0, 8].item(), 0.0)
        self.assertTrue(torch.equal(noise_out[0, 9:], torch.full_like(noise_out[0, 9:], 2.0)))

    def test_warmup_gather_propagates_and_filters_block_validity(self):
        local_error = torch.cat(
            [torch.full((1, 8, 1, 1, 1), 1.0), torch.full((1, 8, 1, 1, 1), 2.0)],
            dim=1,
        )
        remote_error = torch.cat(
            [torch.full((1, 8, 1, 1, 1), 3.0), torch.full((1, 8, 1, 1, 1), 4.0)],
            dim=1,
        )
        local_index = torch.zeros(1, 16, dtype=torch.long)
        remote_index = torch.full((1, 16), 8, dtype=torch.long)
        local_valid = torch.tensor([[True, False]])
        remote_valid = torch.tensor([[False, True]])

        def fake_all_gather(outputs, source, group=None):
            del group
            if source.dtype == torch.uint8:
                sources = [local_valid.to(torch.uint8), remote_valid.to(torch.uint8)]
            elif source.dtype == torch.long:
                sources = [local_index, remote_index]
            else:
                sources = [local_error, remote_error]
            for output, gathered in zip(outputs, sources):
                output.copy_(gathered)

        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            filtered = self.model._gather_errors_for_buffer(
                _BufferLayout(),
                local_error,
                local_index,
                1,
                16,
                block_valid_mask=local_valid,
            )

            # Fixed F64/all-valid behavior remains the original four gathered
            # items in rank-major, then block-major order.
            local_valid.fill_(True)
            remote_valid.fill_(True)
            all_valid = self.model._gather_errors_for_buffer(
                _BufferLayout(),
                local_error,
                local_index,
                1,
                16,
                block_valid_mask=local_valid,
            )

        self.assertEqual([item[0][0].item() for item in filtered], [1, 4])
        self.assertEqual([item[0][0].item() for item in all_valid], [1, 2, 3, 4])

    def test_2d_position_buffer_does_not_shift_around_padding(self):
        self.model.er_num_blocks = 2
        buffer = ErrorBuffer(
            num_buckets=1,
            max_size_per_bucket=2,
            num_train_timesteps=1000,
            num_blocks=2,
            modulate_factor=0,
        )
        self.model.error_buffer = buffer

        error = torch.cat(
            [torch.ones(1, 8, 1, 1, 1), torch.full((1, 8, 1, 1, 1), 2.0)],
            dim=1,
        )
        index = torch.zeros(1, 16, dtype=torch.long)
        items = self.model._collect_local_items(
            buffer,
            error,
            index,
            1,
            16,
            block_valid_mask=torch.tensor([[False, True]]),
        )
        self.model._apply_gathered_items(buffer, items)

        self.assertEqual(len(buffer.buckets[(0, 0)]), 0)
        self.assertEqual(len(buffer.buckets[(1, 0)]), 1)

        # Position 1 has an entry, but an invalid target at position 1 must not
        # be modified and that entry must not be shifted into valid position 0.
        output = self.model._inject_latent_error_buffer(
            torch.zeros_like(error),
            index,
            1,
            16,
            block_valid_mask=torch.tensor([[True, False]]),
        )
        self.assertFalse(output.any())

    def test_error_buffer_invalid_add_cannot_replace_valid_entry(self):
        buffer = ErrorBuffer(
            num_buckets=1,
            max_size_per_bucket=1,
            num_train_timesteps=1000,
            modulate_factor=0,
            replacement_strategy="random",
        )
        original = torch.ones(8, 1, 1, 1)
        padded = torch.full((8, 1, 1, 1), 9.0)
        buffer.add(original, 0, valid=True)
        buffer.add(padded, 0, valid=False)

        self.assertEqual(buffer.stats()["total_added"], 1)
        self.assertEqual(buffer.stats()["total_entries"], 1)
        self.assertTrue(
            torch.equal(buffer.sample(0, torch.device("cpu"), torch.float32), original)
        )
        self.assertIsNone(
            buffer.sample(0, torch.device("cpu"), torch.float32, valid=False)
        )

    def test_strict_same_stage_error_buffer_round_trip(self):
        source = ErrorBuffer(
            num_buckets=4,
            max_size_per_bucket=2,
            num_train_timesteps=1000,
            modulate_factor=0.3,
            replacement_strategy="random",
            num_blocks=2,
            global_block_offset=4,
            shard_rank=1,
            shard_size=2,
        )
        source.add(torch.ones(8, 1, 1, 1), 300, block_pos=1)
        target = ErrorBuffer(
            num_buckets=4,
            max_size_per_bucket=2,
            num_train_timesteps=1000,
            modulate_factor=0.3,
            replacement_strategy="random",
            num_blocks=2,
            global_block_offset=4,
            shard_rank=1,
            shard_size=2,
        )
        target.load_state_dict(source.state_dict(), strict_layout=True)
        self.assertEqual(target.stats(), source.stats())
        self.assertEqual(set(target.buckets), set(source.buckets))

    def test_strict_error_buffer_layout_mismatch_fails_before_mutation(self):
        source = ErrorBuffer(num_buckets=2, max_size_per_bucket=1)
        source.add(torch.ones(8, 1, 1, 1), 0)
        state = source.state_dict()
        state["max_size"] = 2
        target = ErrorBuffer(num_buckets=2, max_size_per_bucket=1)
        with self.assertRaisesRegex(RuntimeError, "strict layout mismatch"):
            target.load_state_dict(state, strict_layout=True)
        self.assertEqual(target.stats()["total_added"], 0)
        self.assertEqual(target.stats()["total_entries"], 0)


if __name__ == "__main__":
    unittest.main()
