from pathlib import Path

import pytest
import torch

from trainer.distillation import (
    Trainer,
    canonical_lora_adapter_config,
    validate_lora_adapter_config,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_complete_checkpoint(path, *, step, uses_critic):
    path.mkdir(parents=True, exist_ok=True)
    optimizer_files = {
        "generator": "generator_optimizer.pt",
        "training": "training_state.pt",
    }
    if uses_critic:
        optimizer_files["critic"] = "critic_optimizer.pt"
    torch.save(
        {
            "checkpoint_format_version": 3,
            "generator_lora": {},
            "adapter_contract": {},
            "optimizer_state_files": optimizer_files,
            "step": step,
        },
        path / "model.pt",
    )
    torch.save({"state": {}}, path / "generator_optimizer.pt")
    if uses_critic:
        torch.save({"state": {}}, path / "critic_optimizer.pt")
    torch.save(
        {
            "checkpoint_format_version": 1,
            "step": step,
            "world_size": 1,
            "rank_states": [{}],
        },
        path / "training_state.pt",
    )
    (path / "_SUCCESS").write_text(f"step={step}\n", encoding="ascii")


def test_find_latest_complete_lora_checkpoint(tmp_path):
    incomplete_weights = tmp_path / "checkpoint_model_000100"
    _touch(incomplete_weights / "model.pt")

    incomplete_optimizers = tmp_path / "checkpoint_model_000200"
    _touch(incomplete_optimizers / "model.pt")
    _touch(incomplete_optimizers / "generator_optimizer.pt")
    _touch(incomplete_optimizers / "critic_optimizer.pt")

    complete = tmp_path / "checkpoint_model_000300"
    _write_complete_checkpoint(complete, step=300, uses_critic=True)

    interrupted_newer = tmp_path / "checkpoint_model_000400"
    _touch(interrupted_newer / "model.pt")

    trainer = Trainer.__new__(Trainer)
    trainer.uses_critic = True
    trainer.world_size = 1
    assert trainer.find_latest_checkpoint(str(tmp_path)) == str(
        interrupted_newer / "model.pt"
    )
    assert trainer.find_latest_checkpoint(
        str(tmp_path), require_lora_optimizer_state=True
    ) == str(complete / "model.pt")


def test_find_latest_exact_resume_fails_instead_of_restarting_from_legacy(tmp_path):
    legacy = tmp_path / "checkpoint_model_000100"
    _touch(legacy / "model.pt")
    _touch(legacy / "generator_optimizer.pt")
    _touch(legacy / "critic_optimizer.pt")
    _touch(legacy / "_SUCCESS")

    trainer = Trainer.__new__(Trainer)
    trainer.uses_critic = True
    trainer.world_size = 1
    with pytest.raises(RuntimeError, match="none can be exactly resumed"):
        trainer.find_latest_checkpoint(
            str(tmp_path), require_lora_optimizer_state=True
        )


def test_optimizer_step_range_reports_adam_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, betas=(0.0, 0.999))
    parameter.grad = torch.tensor([0.5])
    optimizer.step()

    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    count, minimum, maximum = trainer._optimizer_step_range(optimizer)

    assert count == 1
    assert minimum == 1
    assert maximum == 1


def test_atomic_torch_save_replaces_target(tmp_path):
    target = Path(tmp_path) / "state.pt"
    Trainer._atomic_torch_save({"step": 1}, str(target))
    Trainer._atomic_torch_save({"step": 2}, str(target))

    assert torch.load(target, map_location="cpu")["step"] == 2
    assert not list(tmp_path.glob("*.tmp.*"))


def test_adapter_resume_contract_accepts_identical_config():
    adapter = {
        "type": "lora",
        "rank": 128,
        "alpha": 128,
        "dropout": 0.0,
        "apply_to_critic": True,
        "expected_target_modules": 400,
        "verbose": False,
    }

    assert canonical_lora_adapter_config(adapter) == {
        "type": "lora",
        "rank": 128,
        "alpha": 128.0,
        "dropout": 0.0,
        "apply_to_critic": True,
        "expected_target_modules": 400,
    }
    assert validate_lora_adapter_config(
        adapter,
        dict(adapter),
        checkpoint_path="checkpoint/model.pt",
        require_metadata=True,
    )


@pytest.mark.parametrize(
    ("field", "new_value"),
    [("rank", 64), ("alpha", 64), ("dropout", 0.1), ("apply_to_critic", False)],
)
def test_adapter_resume_contract_rejects_behavior_changes(field, new_value):
    saved = {
        "type": "lora",
        "rank": 128,
        "alpha": 128,
        "dropout": 0.0,
        "apply_to_critic": True,
    }
    current = dict(saved)
    current[field] = new_value

    with pytest.raises(RuntimeError, match="adapter config changed"):
        validate_lora_adapter_config(
            saved,
            current,
            checkpoint_path="checkpoint/model.pt",
            require_metadata=True,
        )


def test_exact_resume_requires_adapter_metadata():
    with pytest.raises(RuntimeError, match="has no adapter metadata"):
        validate_lora_adapter_config(
            None,
            {"type": "lora", "rank": 128},
            checkpoint_path="checkpoint/model.pt",
            require_metadata=True,
        )


def test_integrated_sfp_steps_generator_before_critic_rollout():
    events = []

    class RecordingOptimizer:
        def __init__(self, label):
            self.label = label

        def zero_grad(self, set_to_none=True):
            events.append(f"{self.label}_zero")

        def step(self):
            events.append(f"{self.label}_step")

    class RecordingModel:
        def __init__(self, label):
            self.label = label

        def clip_grad_norm_(self, value):
            events.append(f"{self.label}_clip")
            return torch.tensor(value)

    trainer = Trainer.__new__(Trainer)
    trainer.dataloader = iter(["generator_batch", "critic_batch"])
    trainer.gradient_accumulation_steps = 1
    trainer.max_grad_norm_generator = 10.0
    trainer.max_grad_norm_critic = 10.0
    trainer.generator_optimizer = RecordingOptimizer("generator")
    trainer.critic_optimizer = RecordingOptimizer("critic")
    trainer.generator_ema = None
    trainer.model = type(
        "ModelPair",
        (),
        {
            "generator": RecordingModel("generator"),
            "fake_score": RecordingModel("critic"),
        },
    )()

    def fwdbwd(batch, train_generator):
        role = "generator" if train_generator else "critic"
        events.append(f"{role}_fwdbwd:{batch}")
        if train_generator:
            return {
                "generator_loss": torch.tensor(1.0),
                "generator_grad_norm": torch.tensor(0.0),
                "dmdtrain_gradient_norm": torch.tensor(1.0),
            }
        return {
            "critic_loss": torch.tensor(1.0),
            "critic_grad_norm": torch.tensor(0.0),
        }

    trainer.fwdbwd_one_step = fwdbwd

    trainer._run_sfp_optimization_step(True)

    assert events.index("generator_fwdbwd:generator_batch") < events.index(
        "generator_step"
    )
    assert events.index("generator_step") < events.index(
        "critic_fwdbwd:critic_batch"
    )
    assert events.index("critic_fwdbwd:critic_batch") < events.index(
        "critic_step"
    )
