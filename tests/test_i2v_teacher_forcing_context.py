import unittest
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import torch


class _FakeScheduler:
    def __init__(self):
        self.num_train_timesteps = 1000
        self.timesteps = torch.arange(1000, dtype=torch.float32)
        self.sigmas = torch.linspace(1.0, 0.0, 1000)

    def add_noise(self, clean, noise, timestep):
        return clean + 10.0

    def training_target(self, clean, noise, timestep):
        return torch.zeros_like(clean)

    def training_weight(self, timestep):
        return torch.ones(timestep.numel(), device=timestep.device, dtype=torch.float32)


class _FakeGenerator:
    def __init__(self):
        self.recorded = None

    def __call__(
        self,
        *,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        clean_x,
        aug_t,
    ):
        self.recorded = {
            "noisy_image_or_video": noisy_image_or_video.detach().clone(),
            "timestep": timestep.detach().clone(),
            "clean_x": clean_x.detach().clone(),
            "aug_t": aug_t.detach().clone() if aug_t is not None else None,
        }
        return torch.zeros_like(noisy_image_or_video), torch.zeros_like(noisy_image_or_video)


class _FakeBuffer:
    num_blocks = 0

    def is_empty(self):
        return False

    def add(self, error_block, timestep_index, block_pos=None):
        pass

    def stats(self):
        return {
            "total_added": 0,
            "filled_buckets": "0/0",
            "total_entries": 0,
        }


class _StubBaseModel(torch.nn.Module):
    def _get_timestep(
        self,
        min_timestep,
        max_timestep,
        batch_size,
        num_frame,
        num_frame_per_block,
        uniform_timestep=False,
    ):
        if uniform_timestep:
            return torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long,
            ).repeat(1, num_frame)
        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.long,
        )
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        return timestep.reshape(timestep.shape[0], -1)


def _load_causal_diffusion_with_stubs():
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
            "_diffusion_under_test", module_path
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


class I2VTeacherForcingContextTest(unittest.TestCase):
    def test_teacher_forcing_keeps_i2v_context_clean_after_augmentation(self):
        CausalDiffusion = _load_causal_diffusion_with_stubs()
        model = CausalDiffusion.__new__(CausalDiffusion)
        torch.nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.args = SimpleNamespace(i2v=True)
        model.independent_first_frame = True
        model.num_frame_per_block = 2
        model.scheduler = _FakeScheduler()
        model.teacher_forcing = True
        model.noise_augmentation_max_timestep = 10
        model.generator = _FakeGenerator()
        model.error_buffer = _FakeBuffer()
        model.noise_error_buffer = _FakeBuffer()
        model.er_start_step = 0
        model.er_clean_prob = 0.0
        model.er_latent_inject_prob = 0.0
        model.er_noise_inject_prob = 0.0
        model.er_context_inject_prob = 1.0
        model.er_buffer_warmup_iter = -1

        def inject_context_error(clean_latent_aug, index, batch_size, num_frame):
            return clean_latent_aug + 100.0

        model._inject_error_buffer = inject_context_error

        clean_latent = torch.zeros(1, 4, 1, 1, 1)
        initial_latent = torch.full((1, 1, 1, 1, 1), 7.0)

        model.generator_loss(
            image_or_video_shape=[1, 4, 1, 1, 1],
            conditional_dict={"prompt_embeds": torch.zeros(1, 1)},
            unconditional_dict={},
            clean_latent=clean_latent,
            initial_latent=initial_latent,
            global_step=0,
        )

        recorded = model.generator.recorded
        self.assertTrue(torch.equal(recorded["noisy_image_or_video"][:, :1], initial_latent))
        self.assertTrue(torch.equal(recorded["clean_x"][:, :1], initial_latent))
        self.assertTrue((recorded["timestep"][:, :1] == 0).all())
        self.assertTrue((recorded["aug_t"][:, :1] == 0).all())

        # Later frames still receive clean-side augmentation and context error.
        self.assertTrue(torch.equal(recorded["clean_x"][:, 1:], torch.full((1, 3, 1, 1, 1), 110.0)))


if __name__ == "__main__":
    unittest.main()
