# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot integration imported from the author video-training branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/tests/test_wan_backbone_initialization.py
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


class _FakeBackbone:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class _RecordingModel:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def load_config(cls, path, **kwargs):
        cls.calls.append(("load_config", path, kwargs))
        return {"_class_name": "RecordingModel", "dim": 1}

    @classmethod
    def from_config(cls, config, **kwargs):
        cls.calls.append(("from_config", config, kwargs))
        return _FakeBackbone()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append(("from_pretrained", path, kwargs))
        return _FakeBackbone()


class _FakeFlowMatchScheduler:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def set_timesteps(self, count, training=False):
        self.set_timesteps_call = (count, training)


class _FakeSchedulerInterface:
    @staticmethod
    def convert_x0_to_noise(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def convert_noise_to_x0(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def convert_velocity_to_x0(self, *args, **kwargs):
        raise NotImplementedError


def _load_wrapper_with_stubs():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils" / "wan_5b_wrapper.py"
    stub_names = (
        "utils",
        "utils.scheduler",
        "wan_5b",
        "wan_5b.modules",
        "wan_5b.modules.tokenizers",
        "wan_5b.modules.model",
        "wan_5b.modules.vae2_2",
        "wan_5b.modules.t5",
        "wan_5b.modules.causal_model",
    )
    saved = {name: sys.modules.get(name) for name in stub_names}

    fake_utils = types.ModuleType("utils")
    fake_utils.__path__ = []
    fake_scheduler = types.ModuleType("utils.scheduler")
    fake_scheduler.SchedulerInterface = _FakeSchedulerInterface
    fake_scheduler.FlowMatchScheduler = _FakeFlowMatchScheduler

    fake_wan = types.ModuleType("wan_5b")
    fake_wan.__path__ = []
    fake_modules = types.ModuleType("wan_5b.modules")
    fake_modules.__path__ = []
    fake_tokenizers = types.ModuleType("wan_5b.modules.tokenizers")
    fake_tokenizers.HuggingfaceTokenizer = object
    fake_model = types.ModuleType("wan_5b.modules.model")
    fake_model.WanModel = _RecordingModel
    fake_vae = types.ModuleType("wan_5b.modules.vae2_2")
    fake_vae._video_vae = object
    fake_t5 = types.ModuleType("wan_5b.modules.t5")
    fake_t5.umt5_xxl = object
    fake_causal = types.ModuleType("wan_5b.modules.causal_model")
    fake_causal.CausalWanModel = _RecordingModel

    replacements = {
        "utils": fake_utils,
        "utils.scheduler": fake_scheduler,
        "wan_5b": fake_wan,
        "wan_5b.modules": fake_modules,
        "wan_5b.modules.tokenizers": fake_tokenizers,
        "wan_5b.modules.model": fake_model,
        "wan_5b.modules.vae2_2": fake_vae,
        "wan_5b.modules.t5": fake_t5,
        "wan_5b.modules.causal_model": fake_causal,
    }
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "_wan_5b_wrapper_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class WanBackboneInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper_module = _load_wrapper_with_stubs()

    def setUp(self):
        _RecordingModel.reset()

    def test_default_path_loads_pretrained_weights(self):
        wrapper = self.wrapper_module.WanDiffusionWrapper(
            model_name="Wan2.2-TI2V-5B",
            is_causal=True,
            local_attn_size=-1,
            sink_size=8,
            num_frame_per_block=8,
        )

        self.assertTrue(wrapper.model.eval_called)
        self.assertEqual(
            _RecordingModel.calls,
            [
                (
                    "from_pretrained",
                    "wan_models/Wan2.2-TI2V-5B/",
                    {
                        "local_attn_size": -1,
                        "sink_size": 8,
                        "num_frame_per_block": 8,
                    },
                )
            ],
        )

    def test_explicit_config_path_never_loads_pretrained_weights(self):
        wrapper = self.wrapper_module.WanDiffusionWrapper(
            model_name="Wan2.2-TI2V-5B",
            is_causal=True,
            local_attn_size=-1,
            sink_size=0,
            num_frame_per_block=8,
            initialize_from_config=True,
        )

        self.assertTrue(wrapper.model.eval_called)
        self.assertEqual(_RecordingModel.calls[0][0], "load_config")
        self.assertEqual(
            _RecordingModel.calls[0][1:],
            (
                "wan_models/Wan2.2-TI2V-5B/",
                {"local_files_only": True},
            ),
        )
        self.assertEqual(
            _RecordingModel.calls[1],
            (
                "from_config",
                {"_class_name": "RecordingModel", "dim": 1},
                {
                    "local_attn_size": -1,
                    "sink_size": 0,
                    "num_frame_per_block": 8,
                },
            ),
        )
        self.assertNotIn(
            "from_pretrained",
            [call[0] for call in _RecordingModel.calls],
        )


if __name__ == "__main__":
    unittest.main()
