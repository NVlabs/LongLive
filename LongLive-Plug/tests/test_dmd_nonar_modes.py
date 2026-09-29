import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf

from utils.config import normalize_config


class _FakeInnerModel(torch.nn.Module):
    num_heads = 1
    dim = 1
    num_layers = 1

    def __init__(self):
        super().__init__()
        self.requires_grad_called = None

    def requires_grad_(self, requires_grad=True):
        self.requires_grad_called = requires_grad
        return self


class _FakeScheduler:
    def __init__(self):
        self.num_train_timesteps = 1000
        self.shift = 1.0
        self.timesteps = torch.arange(1000, dtype=torch.float32)
        self.sigmas = self.timesteps / 1000.0


class _CriticFakeScheduler(_FakeScheduler):
    alphas_cumprod = None

    def add_noise(self, clean, noise, timestep):
        sigma = timestep.to(dtype=clean.dtype).reshape(-1, 1, 1, 1) / 1000.0
        return clean * (1.0 - sigma) + noise * sigma

    def convert_x0_to_noise(self, x0, xt, timestep):
        return xt - x0


class _FakeWanDiffusionWrapper(torch.nn.Module):
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.is_causal = kwargs.pop("is_causal")
        self.model = _FakeInnerModel()
        self.scheduler = _FakeScheduler()
        _FakeWanDiffusionWrapper.instances.append(self)

    def get_scheduler(self):
        return self.scheduler

    def enable_gradient_checkpointing(self):
        pass


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def requires_grad_(self, requires_grad=True):
        return self


class _FakeVAE(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def requires_grad_(self, requires_grad=True):
        return self


def _load_base_with_stubs():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "model" / "base.py"
    saved = {
        name: sys.modules.get(name)
        for name in (
            "pipeline",
            "utils.wan_5b_wrapper",
        )
    }

    fake_pipeline = types.ModuleType("pipeline")
    fake_pipeline.SelfForcingTrainingPipeline = object

    fake_wrapper = types.ModuleType("utils.wan_5b_wrapper")
    fake_wrapper.WanDiffusionWrapper = _FakeWanDiffusionWrapper
    fake_wrapper.WanTextEncoder = _FakeTextEncoder
    fake_wrapper.WanVAEWrapper = _FakeVAE

    sys.modules["pipeline"] = fake_pipeline
    sys.modules["utils.wan_5b_wrapper"] = fake_wrapper
    try:
        spec = importlib.util.spec_from_file_location(
            "_base_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_wan_wrapper_with_stubs():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils" / "wan_5b_wrapper.py"
    module_names = (
        "wan_5b",
        "wan_5b.modules",
        "wan_5b.modules.tokenizers",
        "wan_5b.modules.model",
        "wan_5b.modules.vae2_1",
        "wan_5b.modules.vae2_2",
        "wan_5b.modules.t5",
        "wan_5b.modules.causal_model",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    fake_wan = types.ModuleType("wan_5b")
    fake_modules = types.ModuleType("wan_5b.modules")
    fake_tokenizers = types.ModuleType("wan_5b.modules.tokenizers")
    fake_model = types.ModuleType("wan_5b.modules.model")
    fake_vae_2_1 = types.ModuleType("wan_5b.modules.vae2_1")
    fake_vae = types.ModuleType("wan_5b.modules.vae2_2")
    fake_t5 = types.ModuleType("wan_5b.modules.t5")
    fake_causal = types.ModuleType("wan_5b.modules.causal_model")
    fake_tokenizers.HuggingfaceTokenizer = object
    fake_model.WanModel = object
    fake_vae_2_1._video_vae = lambda *args, **kwargs: None
    fake_vae._video_vae = lambda *args, **kwargs: None
    fake_t5.umt5_xxl = lambda *args, **kwargs: None
    fake_causal.CausalWanModel = object

    sys.modules["wan_5b"] = fake_wan
    sys.modules["wan_5b.modules"] = fake_modules
    sys.modules["wan_5b.modules.tokenizers"] = fake_tokenizers
    sys.modules["wan_5b.modules.model"] = fake_model
    sys.modules["wan_5b.modules.vae2_1"] = fake_vae_2_1
    sys.modules["wan_5b.modules.vae2_2"] = fake_vae
    sys.modules["wan_5b.modules.t5"] = fake_t5
    sys.modules["wan_5b.modules.causal_model"] = fake_causal
    try:
        spec = importlib.util.spec_from_file_location(
            "_wan_wrapper_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_trainer_with_stubs():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "trainer" / "distillation.py"
    module_names = (
        "utils.dataset",
        "utils.distributed",
        "utils.misc",
        "model",
        "pipeline",
        "peft",
        "wandb",
        "torchvision",
        "torchvision.io",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    fake_dataset = types.ModuleType("utils.dataset")
    fake_dataset.cycle = lambda dataloader: dataloader
    fake_dataset.MultiVideoConcatDataset = object
    fake_dataset.TextPromptDataset = object
    fake_dataset.multi_video_collate_fn = lambda batch: batch
    fake_dataset.eval_collate_fn = lambda batch: batch
    fake_dataset.DEFAULT_SCENE_CUT_PREFIX = "scene_cut"

    fake_distributed = types.ModuleType("utils.distributed")
    fake_distributed.EMA_FSDP = object
    fake_distributed.fsdp_wrap = lambda *args, **kwargs: args[0] if args else None
    fake_distributed.launch_distributed_job = lambda: None

    fake_misc = types.ModuleType("utils.misc")
    fake_misc.set_seed = lambda seed: None
    fake_misc.merge_dict_list = lambda values: values[0] if values else {}

    fake_model = types.ModuleType("model")
    fake_model.DMD = object

    fake_pipeline = types.ModuleType("pipeline")
    fake_pipeline.CausalDiffusionInferencePipeline = object

    fake_peft = types.ModuleType("peft")
    fake_peft.get_peft_model_state_dict = lambda *args, **kwargs: {}
    fake_peft.get_peft_model = lambda model, config: model
    fake_peft.LoraConfig = lambda *args, **kwargs: SimpleNamespace(**kwargs)

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.login = lambda *args, **kwargs: None
    fake_wandb.init = lambda *args, **kwargs: None
    fake_wandb.log = lambda *args, **kwargs: None

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision_io = types.ModuleType("torchvision.io")
    fake_torchvision_io.write_video = lambda *args, **kwargs: None

    sys.modules["utils.dataset"] = fake_dataset
    sys.modules["utils.distributed"] = fake_distributed
    sys.modules["utils.misc"] = fake_misc
    sys.modules["model"] = fake_model
    sys.modules["pipeline"] = fake_pipeline
    sys.modules["peft"] = fake_peft
    sys.modules["wandb"] = fake_wandb
    sys.modules["torchvision"] = fake_torchvision
    sys.modules["torchvision.io"] = fake_torchvision_io
    try:
        spec = importlib.util.spec_from_file_location(
            "_trainer_distillation_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_dmd_with_stubs():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "model" / "dmd.py"
    module_names = (
        "model.base",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    fake_base = types.ModuleType("model.base")
    fake_base.SelfForcingModel = torch.nn.Module
    sys.modules["model.base"] = fake_base
    try:
        spec = importlib.util.spec_from_file_location(
            "_dmd_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _FakeUniPCScheduler:
    def __init__(self, num_train_timesteps, shift, use_dynamic_shifting):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None

    def set_timesteps(self, sampling_steps, device, shift):
        self.timesteps = torch.tensor([900.0, 500.0, 100.0], device=device)[:sampling_steps]

    def step(self, flow_pred, timestep, latents, return_dict=False):
        return (latents - 0.25 * flow_pred,)


class _PatchedUniPC:
    def __enter__(self):
        self.saved = {
            name: sys.modules.get(name)
            for name in (
                "wan_5b",
                "wan_5b.utils",
                "wan_5b.utils.fm_solvers_unipc",
            )
        }
        fake_wan = types.ModuleType("wan_5b")
        fake_utils = types.ModuleType("wan_5b.utils")
        fake_unipc = types.ModuleType("wan_5b.utils.fm_solvers_unipc")
        fake_unipc.FlowUniPCMultistepScheduler = _FakeUniPCScheduler
        sys.modules["wan_5b"] = fake_wan
        sys.modules["wan_5b.utils"] = fake_utils
        sys.modules["wan_5b.utils.fm_solvers_unipc"] = fake_unipc

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _RecordingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, *, noisy_image_or_video, conditional_dict, timestep, **kwargs):
        self.calls.append({
            "noisy_image_or_video": noisy_image_or_video.detach().clone(),
            "prompt_embeds": conditional_dict["prompt_embeds"].detach().clone(),
            "timestep": timestep.detach().clone(),
            "kwargs": kwargs,
        })
        assert "kv_cache" not in kwargs
        assert "crossattn_cache" not in kwargs
        timestep_term = timestep.reshape(*timestep.shape, 1, 1, 1).to(noisy_image_or_video.dtype) / 1000.0
        flow_pred = noisy_image_or_video + timestep_term
        denoised = noisy_image_or_video - flow_pred
        return flow_pred, denoised


class _TrainableRecordingGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, **kwargs):
        self.calls.append({
            "grad_enabled": torch.is_grad_enabled(),
            "input_requires_grad": noisy_image_or_video.requires_grad,
            "timestep": timestep.detach().clone(),
            "kwargs": kwargs,
        })
        timestep_term = timestep.reshape(
            *timestep.shape, 1, 1, 1
        ).to(noisy_image_or_video.dtype) / 1000.0
        flow_pred = noisy_image_or_video * self.scale + timestep_term
        denoised = noisy_image_or_video - flow_pred
        return flow_pred, denoised


class _CheckpointedTrainableGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, **kwargs):
        self.calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "input_requires_grad": noisy_image_or_video.requires_grad,
            }
        )

        def checkpointed_scale(latent):
            return latent * self.scale

        scaled = torch.utils.checkpoint.checkpoint(
            checkpointed_scale,
            noisy_image_or_video,
            use_reentrant=False,
        )
        timestep_term = timestep.reshape(
            *timestep.shape, 1, 1, 1
        ).to(noisy_image_or_video.dtype) / 1000.0
        flow_pred = scaled + timestep_term
        return flow_pred, noisy_image_or_video - flow_pred


class _CheckpointedTrainableScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(
        self,
        *,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        clean_x=None,
    ):
        self.calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "input_requires_grad": noisy_image_or_video.requires_grad,
            }
        )

        def checkpointed_scale(latent):
            return latent * self.scale

        prediction = torch.utils.checkpoint.checkpoint(
            checkpointed_scale,
            noisy_image_or_video,
            use_reentrant=False,
        )
        return prediction, prediction


class _RecordingScore:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, *, noisy_image_or_video, conditional_dict, timestep, clean_x=None):
        self.calls.append({
            "shape": tuple(noisy_image_or_video.shape),
            "prompt_embeds": conditional_dict["prompt_embeds"].detach().clone(),
            "timestep": timestep.detach().clone(),
            "clean_x": None if clean_x is None else clean_x.detach().clone(),
        })
        pred = torch.full_like(noisy_image_or_video, self.value)
        return pred, pred


class _FakeBidirectionalTextEncoder:
    def __call__(self, *, text_prompts):
        values = [
            0.0 if prompt == "negative" else 2.0
            for prompt in text_prompts
        ]
        return {"prompt_embeds": torch.tensor(values, dtype=torch.float32).reshape(-1, 1)}


class _FakeBidirectionalGenerator:
    def __init__(self):
        self.calls = []
        self.scheduler = _FakeScheduler()

    def get_scheduler(self):
        return self.scheduler

    def __call__(self, *, noisy_image_or_video, conditional_dict, timestep):
        prompt_value = conditional_dict["prompt_embeds"].reshape(-1, 1, 1, 1, 1)
        flow_pred = torch.zeros_like(noisy_image_or_video) + prompt_value
        self.calls.append({
            "prompt_value": prompt_value.detach().clone(),
            "latent_dtype": noisy_image_or_video.dtype,
        })
        return flow_pred, noisy_image_or_video - flow_pred

    def _convert_flow_pred_to_x0(self, *, flow_pred, xt, timestep):
        return xt - flow_pred


class _MinimalTrainerWrapper:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)


class _MinimalTrainerDMD:
    """Small model graph that reaches the real Trainer LoRA-init branch."""

    def __init__(self, *_args, **_kwargs):
        self.generator = _MinimalTrainerWrapper()
        self.fake_score = _MinimalTrainerWrapper()
        self.real_score = None
        self.text_encoder = _MinimalTrainerWrapper()
        self.vae = None


def _minimal_trainer_lora_config(*, auto_resume=False, logdir=None):
    return OmegaConf.create(
        {
            "mixed_precision": False,
            "disable_wandb": True,
            "seed": 17,
            "logdir": logdir,
            "distribution_loss": "dmd",
            "sequence_parallel_size": 1,
            "auto_resume": auto_resume,
            "sharding_strategy": "hybrid_full",
            "generator_fsdp_wrap_strategy": "size",
            "adapter": {
                "type": "lora",
                "rank": 1,
                "alpha": 1,
                "dropout": 0.0,
                "apply_to_critic": True,
            },
        }
    )


def _minimal_lora_resume_checkpoint():
    return {
        "checkpoint_format_version": 3,
        "generator_lora": {"generator.lora_A.weight": torch.ones(1)},
        "critic_lora": {"critic.lora_A.weight": torch.ones(1)},
        "adapter_contract": {
            "type": "lora",
            "rank": 1,
            "alpha": 1.0,
            "dropout": 0.0,
            "apply_to_critic": True,
            "expected_target_modules": None,
        },
        "step": 250,
    }


class _LegacyGradientCheckpointModel:
    def __init__(self):
        self.gradient_checkpointing = False
        self.set_args = None

    def enable_gradient_checkpointing(self):
        raise TypeError(
            "WanModel._set_gradient_checkpointing() got an unexpected keyword argument 'enable'"
        )

    def _set_gradient_checkpointing(self, module, value=False):
        self.set_args = (module, value)
        self.gradient_checkpointing = value


class _RecordingDecodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(
            torch.empty((), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.input_dtype = None
        self.scale_dtypes = None

    def decode(self, z, scale):
        self.input_dtype = z.dtype
        self.scale_dtypes = tuple(item.dtype for item in scale)
        return torch.zeros(
            z.shape[0],
            3,
            z.shape[2],
            z.shape[3],
            z.shape[4],
            dtype=z.dtype,
            device=z.device,
        )


class DMDNonARModeTest(unittest.TestCase):
    def test_trainer_executes_common_seed_adapter_order_rank_seed_and_digest(self):
        trainer_module = _load_trainer_with_stubs()
        events = []

        class StopAfterDigest(Exception):
            pass

        def configure_adapter(_trainer, transformer, role):
            events.append(("adapter", role))
            return transformer

        def verify_digest(models, device):
            events.append(("digest", tuple(models), device))
            raise StopAfterDigest

        with (
            mock.patch.object(trainer_module, "DMD", _MinimalTrainerDMD),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=3),
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=8),
            mock.patch.object(trainer_module.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                trainer_module,
                "set_seed",
                side_effect=lambda seed: events.append(("seed", seed)),
            ),
            mock.patch.object(
                trainer_module.Trainer,
                "_configure_lora_for_model",
                configure_adapter,
            ),
            mock.patch.object(
                trainer_module,
                "assert_trainable_lora_parameters_synced",
                side_effect=verify_digest,
            ),
            # If the production digest call disappears, initialization must not
            # accidentally make this test pass by continuing into FSDP setup.
            mock.patch.object(
                trainer_module,
                "fsdp_wrap",
                side_effect=AssertionError("production digest call was skipped"),
            ),
        ):
            with self.assertRaises(StopAfterDigest):
                trainer_module.Trainer(_minimal_trainer_lora_config())

        self.assertEqual(
            events,
            [
                ("seed", 20),  # Trainer's initial rank-local stream.
                ("seed", 17),  # Common adapter-construction stream.
                ("adapter", "generator"),
                ("adapter", "fake_score"),
                ("seed", 20),  # Restored rank-local training stream.
                (
                    "digest",
                    ("generator", "critic"),
                    torch.device("cuda", 0),
                ),
            ],
        )

    def test_trainer_restores_rank_seed_when_critic_adapter_construction_fails(self):
        trainer_module = _load_trainer_with_stubs()
        events = []

        class AdapterConstructionFailure(Exception):
            pass

        def configure_adapter(_trainer, transformer, role):
            events.append(("adapter", role))
            if role == "fake_score":
                raise AdapterConstructionFailure
            return transformer

        digest = mock.Mock(
            side_effect=AssertionError("digest must not run after adapter failure")
        )
        with (
            mock.patch.object(trainer_module, "DMD", _MinimalTrainerDMD),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=3),
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=8),
            mock.patch.object(trainer_module.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                trainer_module,
                "set_seed",
                side_effect=lambda seed: events.append(("seed", seed)),
            ),
            mock.patch.object(
                trainer_module.Trainer,
                "_configure_lora_for_model",
                configure_adapter,
            ),
            mock.patch.object(
                trainer_module,
                "assert_trainable_lora_parameters_synced",
                digest,
            ),
        ):
            with self.assertRaises(AdapterConstructionFailure):
                trainer_module.Trainer(_minimal_trainer_lora_config())

        self.assertEqual(
            events,
            [
                ("seed", 20),
                ("seed", 17),
                ("adapter", "generator"),
                ("adapter", "fake_score"),
                ("seed", 20),
            ],
        )
        digest.assert_not_called()

    def test_resume_with_stale_fresh_pins_fails_closed_after_production_digest(self):
        trainer_module = _load_trainer_with_stubs()
        checkpoint_path = "/unused/checkpoint_model_000250/model.pt"
        summaries = {
            "generator": {
                "sha256": "evolved-generator",
                "parameter_count": 2,
                "element_count": 3,
            },
            "critic": {
                "sha256": "evolved-critic",
                "parameter_count": 2,
                "element_count": 3,
            },
        }
        digest = mock.Mock(return_value=summaries)
        find_latest = mock.Mock(return_value=checkpoint_path)

        with (
            mock.patch.object(trainer_module, "DMD", _MinimalTrainerDMD),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=0),
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=1),
            mock.patch.object(trainer_module.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                trainer_module.Trainer,
                "_configure_lora_for_model",
                lambda _trainer, transformer, _role: transformer,
            ),
            mock.patch.object(
                trainer_module.Trainer,
                "find_latest_checkpoint",
                find_latest,
            ),
            mock.patch.object(
                trainer_module.Trainer,
                "_load_lora_state_dict_compat",
                return_value=None,
            ),
            mock.patch.object(
                trainer_module.torch,
                "load",
                side_effect=lambda *_args, **_kwargs: _minimal_lora_resume_checkpoint(),
            ),
            mock.patch.object(
                trainer_module,
                "assert_trainable_lora_parameters_synced",
                digest,
            ),
            mock.patch.object(
                trainer_module,
                "fsdp_wrap",
                side_effect=AssertionError("stale fresh pin was not rejected"),
            ),
            mock.patch.dict(
                trainer_module.os.environ,
                {
                    "DMD_EXPECTED_FRESH_GENERATOR_LORA_SHA256": "fresh-generator",
                    "DMD_EXPECTED_FRESH_CRITIC_LORA_SHA256": "fresh-critic",
                },
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "generator LoRA does not match the pinned initialization",
            ):
                trainer_module.Trainer(
                    _minimal_trainer_lora_config(
                        auto_resume=True,
                        logdir="/unused",
                    )
                )

        digest.assert_called_once()
        self.assertEqual(tuple(digest.call_args.args[0]), ("generator", "critic"))
        find_latest.assert_called_once_with(
            "/unused",
            required_keys=("generator_lora", "adapter_contract", "optimizer_state_files", "step"),
            require_lora_optimizer_state=True,
        )

    def test_launcher_cleared_fresh_pins_allows_resume_past_production_digest(self):
        trainer_module = _load_trainer_with_stubs()
        checkpoint_path = "/unused/checkpoint_model_000250/model.pt"
        summaries = {
            "generator": {
                "sha256": "evolved-generator",
                "parameter_count": 2,
                "element_count": 3,
            },
            "critic": {
                "sha256": "evolved-critic",
                "parameter_count": 2,
                "element_count": 3,
            },
        }
        digest = mock.Mock(return_value=summaries)

        class StopAtFSDP(Exception):
            pass

        with (
            mock.patch.object(trainer_module, "DMD", _MinimalTrainerDMD),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=0),
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=1),
            mock.patch.object(trainer_module.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                trainer_module.Trainer,
                "_configure_lora_for_model",
                lambda _trainer, transformer, _role: transformer,
            ),
            mock.patch.object(
                trainer_module.Trainer,
                "find_latest_checkpoint",
                return_value=checkpoint_path,
            ),
            mock.patch.object(
                trainer_module.Trainer,
                "_load_lora_state_dict_compat",
                return_value=None,
            ),
            mock.patch.object(
                trainer_module.torch,
                "load",
                side_effect=lambda *_args, **_kwargs: _minimal_lora_resume_checkpoint(),
            ),
            mock.patch.object(
                trainer_module,
                "assert_trainable_lora_parameters_synced",
                digest,
            ),
            mock.patch.object(
                trainer_module,
                "fsdp_wrap",
                side_effect=StopAtFSDP,
            ),
            mock.patch.dict(trainer_module.os.environ, {}, clear=False),
        ):
            trainer_module.os.environ.pop(
                "DMD_EXPECTED_FRESH_GENERATOR_LORA_SHA256", None
            )
            trainer_module.os.environ.pop(
                "DMD_EXPECTED_FRESH_CRITIC_LORA_SHA256", None
            )
            with self.assertRaises(StopAtFSDP):
                trainer_module.Trainer(
                    _minimal_trainer_lora_config(
                        auto_resume=True,
                        logdir="/unused",
                    )
                )

        digest.assert_called_once()
        self.assertEqual(tuple(digest.call_args.args[0]), ("generator", "critic"))

    def test_release_rejects_causal_teacher(self):
        config = OmegaConf.create({
            "model_kwargs": {
                "model_name": "Wan2.2-TI2V-5B",
                "timestep_shift": 5.0,
                "num_frame_per_block": 8,
            },
            "algorithm": {
                "trainer": "score_distillation",
                "all_causal": False,
                "generator_is_causal": False,
                "fake_score_is_causal": False,
                "real_score_is_causal": True,
            },
        })

        with self.assertRaisesRegex(ValueError, "real_score_is_causal"):
            normalize_config(config)

    def test_release_rejects_all_causal(self):
        config = OmegaConf.create({
            "model_kwargs": {"model_name": "Wan2.2-TI2V-5B"},
            "algorithm": {"trainer": "score_distillation", "all_causal": True},
        })

        with self.assertRaisesRegex(ValueError, "all_causal"):
            normalize_config(config)

    def test_all_causal_false_defaults_score_models_noncausal(self):
        config = OmegaConf.create({
            "model_kwargs": {"model_name": "Wan2.2-TI2V-5B"},
            "algorithm": {"trainer": "score_distillation", "all_causal": False},
            "adapter": {"type": "lora", "apply_to_critic": True},
        })

        normalized = normalize_config(config)

        self.assertFalse(normalized.generator_is_causal)
        self.assertFalse(normalized.fake_score_is_causal)
        self.assertFalse(normalized.real_score_is_causal)

    def test_base_model_uses_bidirectional_attention_for_all_roles(self):
        base_module = _load_base_with_stubs()
        _FakeWanDiffusionWrapper.instances = []
        args = OmegaConf.create({
            "model_kwargs": {"model_name": "Wan2.2-TI2V-5B", "timestep_shift": 5.0},
            "real_model_kwargs": {"model_name": "Wan2.2-TI2V-5B"},
            "fake_model_kwargs": {"model_name": "Wan2.2-TI2V-5B"},
            "all_causal": True,
            "generator_is_causal": False,
            "real_score_is_causal": True,
            "fake_score_is_causal": False,
            "mixed_precision": False,
        })

        model = base_module.BaseModel(args, torch.device("cpu"))

        self.assertFalse(model.generator.is_causal)
        self.assertFalse(model.real_score.is_causal)
        self.assertFalse(model.fake_score.is_causal)
        self.assertEqual(
            model.text_encoder.kwargs,
            {
                "model_name": "Wan2.2-TI2V-5B",
                "model_dir": None,
            },
        )
        self.assertEqual(
            model.vae.kwargs,
            {
                "model_name": "Wan2.2-TI2V-5B",
                "model_dir": None,
                "vae_type": None,
            },
        )

    def test_wan_wrapper_handles_legacy_gradient_checkpointing_signature(self):
        wrapper_module = _load_wan_wrapper_with_stubs()
        wrapper = wrapper_module.WanDiffusionWrapper.__new__(
            wrapper_module.WanDiffusionWrapper
        )
        fake_model = _LegacyGradientCheckpointModel()
        wrapper.model = fake_model

        wrapper.enable_gradient_checkpointing()

        self.assertTrue(fake_model.gradient_checkpointing)
        self.assertEqual(fake_model.set_args, (fake_model, True))

    def test_wan_wrapper_infers_seq_len_from_current_latent_shape(self):
        wrapper_module = _load_wan_wrapper_with_stubs()
        wrapper = wrapper_module.WanDiffusionWrapper.__new__(
            wrapper_module.WanDiffusionWrapper
        )
        wrapper.model = SimpleNamespace(patch_size=(1, 2, 2))

        smoke_latent = torch.zeros(1, 32, 48, 12, 20)
        full_latent = torch.zeros(1, 32, 48, 44, 80)

        self.assertEqual(wrapper._infer_seq_len(smoke_latent), 32 * 6 * 10)
        self.assertEqual(wrapper._infer_seq_len(full_latent), 32 * 22 * 40)

    def test_vae_decode_casts_latents_to_vae_dtype(self):
        wrapper_module = _load_wan_wrapper_with_stubs()
        wrapper = wrapper_module.WanVAEWrapper.__new__(
            wrapper_module.WanVAEWrapper
        )
        torch.nn.Module.__init__(wrapper)
        wrapper.mean = torch.zeros(48, dtype=torch.float32)
        wrapper.std = torch.ones(48, dtype=torch.float32)
        fake_model = _RecordingDecodeModel()
        wrapper.model = fake_model

        latent = torch.zeros(1, 1, 48, 1, 1, dtype=torch.float32)
        decoded = wrapper.decode_to_pixel(latent)

        self.assertEqual(fake_model.input_dtype, torch.bfloat16)
        self.assertEqual(fake_model.scale_dtypes, (torch.bfloat16, torch.bfloat16))
        self.assertEqual(decoded.dtype, torch.float32)

    def test_train_returns_immediately_when_resume_reached_max_iters(self):
        Trainer = _load_trainer_with_stubs().Trainer
        trainer = Trainer.__new__(Trainer)
        trainer.step = 2000
        trainer.is_main_process = True
        trainer.config = OmegaConf.create({
            "max_iters": 2000,
            "evaluation": {"before_train": False},
        })
        trainer._visualize = mock.Mock()
        trainer.dataloader = iter([{"prompts": [["unused"]]}])
        trainer.fwdbwd_one_step = mock.Mock()

        trainer.train()

        trainer._visualize.assert_not_called()
        trainer.fwdbwd_one_step.assert_not_called()
        self.assertEqual(trainer.step, 2000)

    def test_bidirectional_visualization_uses_cfg_and_returns_scheduler_latents(self):
        Trainer = _load_trainer_with_stubs().Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.config = OmegaConf.create({
            "image_or_video_shape": [1, 4, 1, 1, 1],
            "inference": {"sampling_steps": 3, "guidance_scale": 3.0},
            "negative_prompt": "negative",
        })
        trainer.device = torch.device("cpu")
        trainer.dtype = torch.float32
        generator = _FakeBidirectionalGenerator()
        trainer.model = SimpleNamespace(
            text_encoder=_FakeBidirectionalTextEncoder(),
            generator=generator,
        )

        with _PatchedUniPC(), mock.patch(
            "torch.randn",
            return_value=torch.zeros(1, 4, 1, 1, 1),
        ):
            output = trainer._generate_bidirectional(4, [["positive"]])

        self.assertEqual(len(generator.calls), 6)
        self.assertEqual(generator.calls[0]["latent_dtype"], torch.float32)
        # Guided flow is 0 + 3 * (2 - 0) = 6. The fake scheduler subtracts
        # 0.25 * flow on every sampled step, matching the official Wan TI2V
        # path that decodes the final scheduler latent.
        self.assertTrue(torch.equal(output, torch.full((1, 4, 1, 1, 1), -4.5)))

    def test_noncausal_student_rollout_matches_full_sequence_manual_loop(self):
        base_module = _load_base_with_stubs()
        model = base_module.SelfForcingModel.__new__(base_module.SelfForcingModel)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(generator_is_causal=False, sampling_steps=3)
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.scheduler = _FakeScheduler()
        model.generator = _RecordingGenerator()
        noise = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
        cond = {"prompt_embeds": torch.tensor([[2.0], [2.0]])}

        with _PatchedUniPC(), mock.patch(
            "torch.randint",
            return_value=torch.tensor([1], dtype=torch.long),
        ):
            output, timestep_from, timestep_to = model._consistency_backward_simulation_noncausal(
                noise=noise,
                **cond,
            )

        first_timestep = torch.full((1, 4), 900.0)
        first_flow = noise + first_timestep.reshape(1, 4, 1, 1, 1) / 1000.0
        latents_after_first = noise - 0.25 * first_flow
        final_timestep = torch.full((1, 4), 500.0)
        expected = latents_after_first - (
            latents_after_first + final_timestep.reshape(1, 4, 1, 1, 1) / 1000.0
        )

        self.assertTrue(torch.allclose(output, expected))
        self.assertEqual(len(model.generator.calls), 2)
        self.assertTrue(torch.equal(model.generator.calls[0]["prompt_embeds"], torch.tensor([[2.0]])))
        for call in model.generator.calls:
            self.assertTrue((call["timestep"] == call["timestep"][:, :1]).all())
            self.assertNotIn("kv_cache", call["kwargs"])
            self.assertNotIn("crossattn_cache", call["kwargs"])
        self.assertEqual(timestep_from, 500)
        self.assertEqual(timestep_to, 900)

    def test_noncausal_last_step_only_runs_full_rollout_and_keeps_gradient(self):
        base_module = _load_base_with_stubs()
        model = base_module.SelfForcingModel.__new__(base_module.SelfForcingModel)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(
            generator_is_causal=False,
            sampling_steps=3,
            last_step_only=True,
        )
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.scheduler = _FakeScheduler()
        model.generator = _TrainableRecordingGenerator()
        noise = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1, 1, 1)

        with _PatchedUniPC(), mock.patch(
            "torch.randint",
            side_effect=AssertionError(
                "last_step_only must not sample an early exit index"
            ),
        ):
            output, timestep_from, timestep_to = (
                model._consistency_backward_simulation_noncausal(
                    noise=noise,
                    prompt_embeds=torch.ones(1, 1),
                )
            )

        self.assertEqual(len(model.generator.calls), 3)
        self.assertEqual(
            [call["grad_enabled"] for call in model.generator.calls],
            [False, False, True],
        )
        self.assertFalse(model.generator.calls[0]["input_requires_grad"])
        # Non-reentrant activation checkpointing preserves parameter gradients
        # without making the rollout latent itself require gradients.
        self.assertFalse(model.generator.calls[-1]["input_requires_grad"])
        self.assertTrue(output.requires_grad)
        output.sum().backward()
        self.assertIsNotNone(model.generator.scale.grad)
        self.assertTrue(torch.isfinite(model.generator.scale.grad))
        self.assertNotEqual(model.generator.scale.grad.item(), 0.0)
        self.assertEqual(timestep_from, 900)
        self.assertEqual(timestep_to, 0)

    def test_noncausal_rollout_preserves_outer_no_grad_for_critic(self):
        base_module = _load_base_with_stubs()
        model = base_module.SelfForcingModel.__new__(base_module.SelfForcingModel)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(
            generator_is_causal=False,
            sampling_steps=3,
            last_step_only=True,
        )
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.scheduler = _FakeScheduler()
        model.generator = _TrainableRecordingGenerator()
        noise = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1, 1, 1)

        with _PatchedUniPC(), torch.no_grad():
            output, _, _ = model._consistency_backward_simulation_noncausal(
                noise=noise,
                prompt_embeds=torch.ones(1, 1),
            )

        self.assertEqual(len(model.generator.calls), 3)
        self.assertEqual(
            [call["grad_enabled"] for call in model.generator.calls],
            [False, False, False],
        )
        self.assertTrue(
            all(not call["input_requires_grad"] for call in model.generator.calls)
        )
        self.assertFalse(output.requires_grad)
        self.assertIsNone(model.generator.scale.grad)


    def test_checkpointed_critic_backward_does_not_change_generator_gradient(self):
        base_module = _load_base_with_stubs()
        dmd_module = _load_dmd_with_stubs()
        model = base_module.SelfForcingModel.__new__(
            base_module.SelfForcingModel
        )
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(
            generator_is_causal=False,
            sampling_steps=3,
            last_step_only=True,
            backward_simulation=True,
            slice_last_frames=4,
            teacher_forcing=False,
            denoising_loss_type="flow",
            i2v=False,
            sequence_parallel_size=1,
        )
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.scheduler = _CriticFakeScheduler()
        model.generator = _CheckpointedTrainableGenerator()
        model.fake_score = _CheckpointedTrainableScore()
        model._slice_block_cond_dict = dmd_module.DMD._slice_block_cond_dict
        model.independent_first_frame = False
        model.min_num_training_frames = 4
        model.num_training_frames = 4
        model.num_frame_per_block = 1
        model.min_score_timestep = 0
        model.num_train_timestep = 1000
        model.min_step = 20
        model.max_step = 980
        model.timestep_shift = 1.0
        model.ts_schedule = False
        model.ts_schedule_max = False
        model.denoising_loss_func = lambda **kwargs: (
            (kwargs["x"] - kwargs["x_pred"]) ** 2
        ).mean()

        conditional_dict = {"prompt_embeds": torch.ones(1, 1)}
        noise = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
        with _PatchedUniPC():
            generator_output, _, _ = (
                model._consistency_backward_simulation_noncausal(
                    noise=noise,
                    **conditional_dict,
                )
            )
            generator_output.square().mean().backward()
            generator_grad_before_critic = model.generator.scale.grad.detach().clone()
            generator_calls_before_critic = len(model.generator.calls)

            torch.manual_seed(123)
            critic_loss, _ = dmd_module.DMD.critic_loss(
                model,
                image_or_video_shape=[1, 4, 1, 1, 1],
                conditional_dict=conditional_dict,
                unconditional_dict={"prompt_embeds": torch.zeros(1, 1)},
            )
            self.assertTrue(critic_loss.requires_grad)
            critic_loss.backward()

        critic_rollout_calls = model.generator.calls[
            generator_calls_before_critic:
        ]
        self.assertEqual(len(critic_rollout_calls), 3)
        self.assertTrue(
            all(not call["grad_enabled"] for call in critic_rollout_calls)
        )
        self.assertTrue(
            all(
                not call["input_requires_grad"]
                for call in critic_rollout_calls
            )
        )
        torch.testing.assert_close(
            model.generator.scale.grad,
            generator_grad_before_critic,
            rtol=0,
            atol=0,
        )
        self.assertIsNotNone(model.fake_score.scale.grad)
        self.assertTrue(torch.isfinite(model.fake_score.scale.grad))
        self.assertNotEqual(model.fake_score.scale.grad.item(), 0.0)
        self.assertEqual(
            model.fake_score.calls,
            [{"grad_enabled": True, "input_requires_grad": False}],
        )


    def test_noncausal_critic_and_teacher_receive_single_prompt_full_clip(self):
        DMD = _load_dmd_with_stubs().DMD

        model = DMD.__new__(DMD)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(
            fake_score_is_causal=False,
            real_score_is_causal=False,
            all_causal=True,
        )
        model.fake_score = _RecordingScore(0.0)
        model.real_score = _RecordingScore(1.0)
        model.fake_guidance_scale = 1.0
        model.real_guidance_scale = 3.0

        noisy = torch.zeros(1, 4, 1, 1, 1)
        estimated = torch.zeros_like(noisy)
        timestep = torch.ones(1, 4)
        cond = {"prompt_embeds": torch.tensor([[2.0], [2.0]])}
        uncond = {"prompt_embeds": torch.tensor([[0.0]])}

        model._compute_kl_grad(
            noisy_image_or_video=noisy,
            estimated_clean_image_or_video=estimated,
            timestep=timestep,
            conditional_dict=cond,
            unconditional_dict=uncond,
            normalization=False,
        )

        self.assertEqual(len(model.fake_score.calls), 2)
        self.assertEqual(len(model.real_score.calls), 2)
        for call in model.fake_score.calls + model.real_score.calls:
            self.assertEqual(call["shape"], (1, 4, 1, 1, 1))
            self.assertEqual(call["prompt_embeds"].shape[0], 1)
        self.assertTrue(torch.equal(model.fake_score.calls[0]["prompt_embeds"], torch.tensor([[2.0]])))
        self.assertTrue(torch.equal(model.real_score.calls[0]["prompt_embeds"], torch.tensor([[2.0]])))

    def test_noncausal_full_clip_rejects_different_prompt_segments(self):
        base_module = _load_base_with_stubs()
        model = base_module.SelfForcingModel.__new__(base_module.SelfForcingModel)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(generator_is_causal=False, sampling_steps=3)
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.scheduler = _FakeScheduler()
        model.generator = _RecordingGenerator()

        with _PatchedUniPC(), mock.patch(
            "torch.randint",
            return_value=torch.tensor([1], dtype=torch.long),
        ):
            with self.assertRaisesRegex(ValueError, "multiple different prompt segments"):
                model._consistency_backward_simulation_noncausal(
                    noise=torch.zeros(1, 4, 1, 1, 1),
                    prompt_embeds=torch.tensor([[1.0], [2.0]]),
                )


if __name__ == "__main__":
    unittest.main()
