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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: tests/test_longwam_multiview_latents.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from longwam.models.wan22.longwam_base import LongWAMBase


class _CausalFakeVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 16

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[object] = []
        self.model = SimpleNamespace(z_dim=2)

    def encode(
        self,
        videos,
        *,
        device,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ) -> torch.Tensor:
        del device, tiled, tile_size, tile_stride
        self.calls.append(videos)
        if isinstance(videos, list):
            videos = torch.stack(videos)

        # Match the causal temporal grouping contract: frame 0 is latent 0 and
        # every subsequent group of four frames produces one more latent.
        signal = videos[:, 0, :, 0, 0].float()
        latent_steps = [signal[:, 0]]
        for start in range(1, signal.shape[1], self.temporal_downsample_factor):
            latent_steps.append(
                signal[:, start : start + self.temporal_downsample_factor].mean(dim=1)
            )
        latent = torch.stack(latent_steps, dim=1)
        latent = latent[:, None, :, None, None].expand(-1, 2, -1, 16, 16).clone()
        latent[:, 1] += 100.0
        return latent


def _model(vae: _CausalFakeVAE | None = None) -> LongWAMBase:
    model = LongWAMBase.__new__(LongWAMBase)
    nn.Module.__init__(model)
    model.vae = vae or _CausalFakeVAE()
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.video_expert = SimpleNamespace(
        action_conditioned=False,
        fuse_vae_embedding_in_latents=False,
    )
    model.proprio_encoder = None
    model.proprio_dim = None
    model.num_clean_frames = 4
    model.current_obs_idx = 0
    return model


def _three_view_video(num_frames: int) -> torch.Tensor:
    time = torch.arange(num_frames, dtype=torch.float32).reshape(1, 1, 1, num_frames, 1, 1)
    view = torch.tensor([10.0, 20.0, 30.0]).reshape(1, 3, 1, 1, 1, 1)
    return (time + view).expand(1, 3, 3, num_frames, 256, 256)


def test_three_views_are_encoded_independently_then_concatenated_in_slot_order():
    vae = _CausalFakeVAE()
    model = _model(vae)

    latents = model._encode_video_latents(_three_view_video(num_frames=5))

    assert len(vae.calls) == 1
    assert isinstance(vae.calls[0], torch.Tensor)
    assert vae.calls[0].shape == (3, 3, 5, 256, 256)
    assert latents.shape == (1, 2, 2, 16, 48)
    for view_index, expected in enumerate((10.0, 20.0, 30.0)):
        slot = latents[0, 0, 0, :, view_index * 16 : (view_index + 1) * 16]
        torch.testing.assert_close(slot, torch.full_like(slot, expected))


def test_three_view_current_image_uses_the_same_per_view_encoder():
    vae = _CausalFakeVAE()
    model = _model(vae)
    image = _three_view_video(num_frames=1).squeeze(3)

    latents = model._encode_input_image_latents_tensor(image)

    assert isinstance(vae.calls[0], torch.Tensor)
    assert vae.calls[0].shape == (3, 3, 1, 256, 256)
    assert latents.shape == (1, 2, 1, 16, 48)
    torch.testing.assert_close(
        latents[:, 0, ..., :16],
        torch.full_like(latents[:, 0, ..., :16], 10.0),
    )
    torch.testing.assert_close(
        latents[:, 0, ..., 16:32],
        torch.full_like(latents[:, 0, ..., 16:32], 20.0),
    )
    torch.testing.assert_close(
        latents[:, 0, ..., 32:],
        torch.full_like(latents[:, 0, ..., 32:], 30.0),
    )


def test_three_view_causal_prefix_matches_full_training_encode():
    model = _model()
    full_latents = model._encode_video_latents(_three_view_video(num_frames=21))
    history_latents = model._encode_video_latents(_three_view_video(num_frames=13))

    assert full_latents.shape[2] == 6
    assert history_latents.shape[2] == 4
    torch.testing.assert_close(full_latents[:, :, :4], history_latents)


def test_build_inputs_accepts_three_view_video_and_preserves_temporal_masks():
    model = _model()
    sample = {
        "video": _three_view_video(num_frames=5),
        "action": torch.zeros(1, 32, 12),
        "action_is_pad": torch.zeros(1, 32, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, 5, dtype=torch.bool),
        "context": torch.zeros(1, 2, 8),
        "context_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    inputs = model.build_inputs(sample)

    assert inputs["input_latents"].shape == (1, 2, 2, 16, 48)
    assert inputs["image_is_pad"].shape == (1, 5)
    assert inputs["action_is_pad"].shape == (1, 32)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 2, 3, 5, 256, 256), "V=3 and C=3"),
        ((1, 3, 1, 5, 256, 256), "V=3 and C=3"),
        ((1, 3, 3, 5, 256, 240), "exactly 256x256"),
    ],
)
def test_three_view_video_contract_fails_fast(shape, message):
    model = _model()
    video = torch.empty(shape, device="meta")

    with pytest.raises(ValueError, match=message):
        model._encode_video_latents(video)


def test_inference_contract_accepts_three_view_current_and_history_and_rejects_mixing():
    image = torch.empty((1, 3, 3, 256, 256), device="meta")
    history = torch.empty((1, 3, 3, 13, 256, 256), device="meta")
    prepared_image, height, width, is_three_view = LongWAMBase._prepare_inference_image(image)
    prepared_history = LongWAMBase._prepare_inference_obs_window(
        history,
        height=height,
        width=width,
        expect_three_view=is_three_view,
    )

    assert prepared_image is image
    assert prepared_history is history
    assert (height, width, is_three_view) == (256, 256, True)

    legacy_history = torch.empty((1, 3, 13, 256, 256), device="meta")
    with pytest.raises(ValueError, match="same view representation"):
        LongWAMBase._prepare_inference_obs_window(
            legacy_history,
            height=height,
            width=width,
            expect_three_view=True,
        )


def test_legacy_video_and_image_calls_keep_the_original_vae_boundaries():
    vae = _CausalFakeVAE()
    model = _model(vae)
    video = torch.randn(1, 3, 5, 16, 32)
    image = torch.randn(1, 3, 16, 32)

    video_latents = model._encode_video_latents(video)
    assert vae.calls[-1] is video
    assert video_latents.shape == (1, 2, 2, 16, 16)

    image_latents = model._encode_input_image_latents_tensor(image)
    image_call = vae.calls[-1]
    assert isinstance(image_call, list)
    assert len(image_call) == 1
    assert image_call[0].shape == (3, 1, 16, 32)
    torch.testing.assert_close(image_call[0][:, 0], image[0])
    assert image_latents.shape == (1, 2, 1, 16, 16)


def test_multiview_encoding_adds_no_state_dict_keys_or_shapes():
    model = _model()
    model.anchor = nn.Linear(2, 3)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    shapes_before = {key: tuple(value.shape) for key, value in state_before.items()}

    model._encode_video_latents(_three_view_video(num_frames=5))

    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == shapes_before
    clone = _model()
    clone.anchor = nn.Linear(2, 3)
    clone.load_state_dict(state_before, strict=True)


@pytest.mark.parametrize("cached", [False, True])
def test_joint_ar_cache_install_preserves_action_context(cached):
    model = _model()
    model.num_clean_frames = model.num_imagine_frames = 1
    model.joint_denoise = False
    model.imagine_infer_steps = 1
    model.imagine_timestep = 0.5
    model._action_cross_kv_cache_enabled = cached
    model.action_expert = SimpleNamespace(action_dim=2)
    model.video_expert.video_attention_mask_mode = "per_frame_causal"
    context = torch.randn(1, 3, 4)
    installed = []
    model.video_expert.pre_dit = lambda **kw: {
        "tokens": torch.zeros(1, 2, 4), "freqs": None, "t_mod": None,
        "context": kw["context"], "context_mask": kw["context_mask"],
        "meta": {"tokens_per_frame": 1},
    }
    model.mot = SimpleNamespace(
        prefill_video_cache=lambda **kw: [],
        prefill_action_cross_kv_cache=lambda value: [{"context": value}],
        install_action_cross_kv_cache=lambda value: installed.append(value) or len(value),
    )
    model._prepare_inference_image = lambda value: (value, 16, 16, False)
    model._imagine_future_latents = lambda **kw: torch.zeros(1, 2, 1, 1, 1)
    model._build_mot_attention_mask = lambda **kw: torch.ones(4, 4, dtype=torch.bool)
    model._stage_action_video_kv_cache = lambda value: value
    model.infer_action_scheduler = SimpleNamespace(
        build_inference_schedule=lambda **kw: (torch.ones(1), torch.ones(1)),
        step=lambda prediction, delta, value: value - prediction,
    )

    def predict(**kw):
        torch.testing.assert_close(kw["context"], context)
        assert kw["use_precomputed_action_cross_kv"] == cached
        return torch.zeros_like(kw["latents_action"])

    model._predict_action_noise_with_cache = predict
    result = model.infer_joint_ar(
        prompt=None, input_image=torch.zeros(1, 3, 16, 16), action_horizon=2,
        context=context, context_mask=torch.ones(1, 3, dtype=torch.bool),
        clean_latents=torch.zeros(1, 2, 1, 1, 1), num_inference_steps=1,
    )
    assert result["action"].shape == (2, 2)
    assert len(installed) == int(cached)


def test_action_denoiser_consumes_resident_cross_cache():
    model = _model()
    cache = [{"k": torch.ones(1), "v": torch.ones(1)}]
    calls = []
    model.action_expert = SimpleNamespace(
        pre_dit=lambda **kw: {
            "tokens": kw["action_tokens"], "freqs": None, "t_mod": None,
            "context": kw["context"], "context_mask": kw["context_mask"],
        },
        post_dit=lambda value, pre: value,
    )

    def forward(**kw):
        calls.append(kw["action_cross_kv_cache"])
        return kw["action_tokens"]

    model.mot = SimpleNamespace(
        action_cross_kv_cache=lambda: cache, forward_action_with_video_cache=forward,
    )
    for enabled in (False, True):
        model._predict_action_noise_with_cache(
            latents_action=torch.zeros(1, 2, 2), timestep_action=torch.ones(1),
            context=torch.zeros(1, 3, 4), context_mask=torch.ones(1, 3, dtype=torch.bool),
            video_kv_cache=[], attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_seq_len=2, use_precomputed_action_cross_kv=enabled,
        )
    assert calls[0] is None and calls[1] is cache
