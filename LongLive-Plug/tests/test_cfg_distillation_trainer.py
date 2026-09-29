import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf


def _load_trainer_with_existing_stubs():
    helper_path = pathlib.Path(__file__).with_name("test_dmd_nonar_modes.py")
    spec = importlib.util.spec_from_file_location(
        "_dmd_test_helpers_for_cfg_trainer",
        helper_path,
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper._load_trainer_with_stubs()


def test_cfg_training_step_updates_generator_without_touching_critic():
    trainer_module = _load_trainer_with_existing_stubs()
    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    batch = {"prompts": [["test prompt"]]}
    generator_optimizer = mock.Mock()
    generator = SimpleNamespace(
        clip_grad_norm_=mock.Mock(return_value=torch.tensor(0.25))
    )

    trainer.config = OmegaConf.create(
        {
            "max_iters": 1,
            "evaluation": {"before_train": False},
            "no_save": True,
            "log_iters": 1,
            "ema_start_step": 100,
            "ema_weight": 0.0,
            "gc_interval": 100,
        }
    )
    trainer.step = 0
    trainer.uses_critic = False
    trainer.is_main_process = True
    trainer.disable_wandb = True
    trainer.gradient_accumulation_steps = 1
    trainer.max_grad_norm_generator = 1.0
    trainer.generator_optimizer = generator_optimizer
    trainer.critic_optimizer = None
    trainer.generator_ema = None
    trainer.model = SimpleNamespace(generator=generator)
    trainer.dataloader = iter([batch])
    trainer.previous_time = None
    trainer.vis_interval = -1
    trainer.fwdbwd_one_step = mock.Mock(
        return_value={
            "generator_loss": torch.tensor(1.0),
            "cfg_distill_loss": torch.tensor(1.0),
            "teacher_guidance_delta_norm": torch.tensor(2.0),
            "student_teacher_rmse": torch.tensor(1.0),
            "timestep_mean": torch.tensor(500.0),
        }
    )

    trainer.train()

    trainer.fwdbwd_one_step.assert_called_once_with(batch, True)
    generator_optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    generator_optimizer.step.assert_called_once_with()
    generator.clip_grad_norm_.assert_called_once_with(1.0)
    assert trainer.step == 1


def test_cfg_rejects_a_critic_forward_backward_step():
    Trainer = _load_trainer_with_existing_stubs().Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.uses_critic = False

    try:
        trainer.fwdbwd_one_step({}, False)
    except RuntimeError as exc:
        assert "no critic" in str(exc)
    else:
        raise AssertionError("CFG-only trainer unexpectedly accepted a critic step")


def test_cfg_lora_checkpoint_contains_generator_only(tmp_path):
    trainer_module = _load_trainer_with_existing_stubs()
    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    generator_model = object()
    gather = mock.Mock(return_value={"adapter.weight": torch.tensor([1.0])})
    gather_optimizer = mock.Mock(
        return_value={"state": {0: {"step": torch.tensor(1.0)}}, "param_groups": []}
    )

    trainer.is_lora_enabled = True
    trainer.uses_critic = False
    trainer.apply_lora_to_critic = False
    trainer.is_main_process = True
    trainer.model = SimpleNamespace(
        generator=SimpleNamespace(model=generator_model),
        fake_score=None,
    )
    trainer._gather_lora_state_dict = gather
    trainer._gather_fsdp_optimizer_state = gather_optimizer
    trainer._capture_rank_training_state = mock.Mock(
        return_value={"rank": 0, "step": 7, "data": {"batches_consumed": 7}}
    )
    trainer.generator_optimizer = object()
    trainer.output_path = str(tmp_path)
    trainer.world_size = 1
    trainer.step = 7
    trainer.config = OmegaConf.create(
        {
            "max_checkpoints": 0,
            "adapter": {
                "type": "lora",
                "rank": 8,
                "alpha": 8,
                "dropout": 0.0,
                "apply_to_critic": False,
                "expected_target_modules": 1,
            },
        }
    )

    trainer.save()

    checkpoint_path = tmp_path / "checkpoint_model_000007" / "model.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert set(checkpoint) == {
        "checkpoint_format_version",
        "generator_lora",
        "adapter",
        "adapter_contract",
        "optimizer_state_files",
        "step",
    }
    assert checkpoint["step"] == 7
    assert checkpoint["optimizer_state_files"] == {
        "generator": "generator_optimizer.pt",
        "training": "training_state.pt",
    }
    assert (checkpoint_path.parent / "generator_optimizer.pt").is_file()
    assert (checkpoint_path.parent / "training_state.pt").is_file()
    assert (checkpoint_path.parent / "_SUCCESS").read_text() == "step=7\n"
    assert not (checkpoint_path.parent / "critic_optimizer.pt").exists()
    gather.assert_called_once_with(generator_model)
    gather_optimizer.assert_called_once_with(
        trainer.model.generator,
        trainer.generator_optimizer,
    )
    assert not list(checkpoint_path.parent.glob(".model.pt.tmp.*"))


def test_lora_resume_skips_invalid_newest_checkpoint(tmp_path):
    Trainer = _load_trainer_with_existing_stubs().Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.is_main_process = True
    trainer.uses_critic = False
    trainer.world_size = 1

    valid_dir = tmp_path / "checkpoint_model_000001"
    valid_dir.mkdir()
    valid_path = valid_dir / "model.pt"
    torch.save(
        {
            "checkpoint_format_version": 3,
            "generator_lora": {},
            "adapter_contract": {},
            "optimizer_state_files": {
                "generator": "generator_optimizer.pt",
                "training": "training_state.pt",
            },
            "step": 1,
        },
        valid_path,
    )
    torch.save({"state": {}}, valid_dir / "generator_optimizer.pt")
    torch.save(
        {
            "checkpoint_format_version": 1,
            "step": 1,
            "world_size": 1,
            "rank_states": [{}],
        },
        valid_dir / "training_state.pt",
    )
    (valid_dir / "_SUCCESS").write_text("step=1\n", encoding="ascii")

    invalid_dir = tmp_path / "checkpoint_model_000002"
    invalid_dir.mkdir()
    (invalid_dir / "model.pt").write_bytes(b"incomplete")

    selected = trainer.find_latest_checkpoint(
        str(tmp_path),
        required_keys=(
            "generator_lora",
            "adapter_contract",
            "optimizer_state_files",
            "step",
        ),
        require_lora_optimizer_state=True,
    )

    assert selected == str(valid_path)




def test_runtime_stop_uses_rank_zero_broadcast():
    trainer_module = _load_trainer_with_existing_stubs()
    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.is_main_process = False
    trainer.device = torch.device("cpu")

    def broadcast_rank_zero_stop(stop_flag, src):
        assert src == 0
        stop_flag.fill_(1)

    with mock.patch.object(
        trainer_module.dist,
        "is_initialized",
        return_value=True,
    ), mock.patch.object(
        trainer_module.dist,
        "broadcast",
        side_effect=broadcast_rank_zero_stop,
    ):
        should_stop = trainer._synchronized_runtime_stop_requested(
            elapsed_runtime=1.0,
            max_runtime_seconds=100.0,
        )

    assert should_stop
