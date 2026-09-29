import multiprocessing as mp
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist

from utils.lora_utils import (
    assert_trainable_lora_parameters_synced,
    configure_lora_for_model,
    insert_peft_adapter_name,
    load_lora_checkpoint,
    trainable_parameter_digest,
)
from utils.misc import set_seed


class _LoraPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 3, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(3, 2, bias=False)}
        )


class _PeftLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.layer = _LoraPair()
        self.active_adapter = "default"


def _distributed_lora_seed_worker(
    rank,
    world_size,
    init_method,
    perturb_critic_rank,
    invalidate_critic_rank,
    result_queue,
):
    """Exercise the pre-FSDP seed/digest contract in a real process group."""

    try:
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world_size,
            init_method=init_method,
        )

        common_seed = 2718
        set_seed(common_seed)
        generator = _PeftLikeModel()
        critic = _PeftLikeModel()
        if perturb_critic_rank is not None and rank == perturb_critic_rank:
            with torch.no_grad():
                critic.base_model.model.layer.lora_A["default"].weight[0, 0] += 1
        if invalidate_critic_rank is not None and rank == invalidate_critic_rank:
            # Exercise the production local-error sentinel: this trainable
            # parameter deliberately violates the pre-FSDP LoRA-only contract.
            critic.register_parameter(
                "unexpected_trainable_weight",
                torch.nn.Parameter(torch.ones(1)),
            )

        # This is the rank-local stream that training should see after all
        # adapters have been constructed from the common stream.
        set_seed(common_seed + rank)
        post_construction_draw = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        )
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.get_rng_state().clone()

        try:
            summaries = assert_trainable_lora_parameters_synced(
                {"generator": generator, "critic": critic},
                torch.device("cpu"),
            )
        except RuntimeError as exc:
            rng_unchanged = (
                random.getstate() == python_rng_before
                and np.array_equal(np.random.get_state()[1], numpy_rng_before[1])
                and np.random.get_state()[0] == numpy_rng_before[0]
                and np.random.get_state()[2:] == numpy_rng_before[2:]
                and torch.equal(torch.get_rng_state(), torch_rng_before)
            )
            result_queue.put(
                {
                    "rank": rank,
                    "status": "rejected",
                    "message": str(exc),
                    "post_construction_draw": post_construction_draw,
                    "rng_unchanged": rng_unchanged,
                }
            )
        else:
            rng_unchanged = (
                random.getstate() == python_rng_before
                and np.array_equal(np.random.get_state()[1], numpy_rng_before[1])
                and np.random.get_state()[0] == numpy_rng_before[0]
                and np.random.get_state()[2:] == numpy_rng_before[2:]
                and torch.equal(torch.get_rng_state(), torch_rng_before)
            )
            result_queue.put(
                {
                    "rank": rank,
                    "status": "ok",
                    "generator_sha256": summaries["generator"]["sha256"],
                    "critic_sha256": summaries["critic"]["sha256"],
                    "post_construction_draw": post_construction_draw,
                    "rng_unchanged": rng_unchanged,
                }
            )
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        result_queue.put(
            {
                "rank": rank,
                "status": "unexpected_error",
                "message": repr(exc),
            }
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_lora_seed_case(
    tmp_path,
    name,
    perturb_critic_rank=None,
    invalidate_critic_rank=None,
):
    world_size = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / f"{name}.store"
    init_method = f"file://{init_path}"
    processes = [
        context.Process(
            target=_distributed_lora_seed_worker,
            args=(
                rank,
                world_size,
                init_method,
                perturb_critic_rank,
                invalidate_critic_rank,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    try:
        results = [result_queue.get(timeout=30) for _ in range(world_size)]
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():  # pragma: no cover - defensive cleanup
                process.terminate()
                process.join(timeout=5)
    for process in processes:
        assert process.exitcode == 0
    return sorted(results, key=lambda item: item["rank"])


class CausalWanAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1, bias=False)


class WanAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1, bias=False)


class _TransformerWithBothAttentionTypes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.causal = CausalWanAttentionBlock()
        self.bidirectional = WanAttentionBlock()


def test_insert_peft_adapter_name_for_checkpoint_keys():
    state = {
        "base_model.model.layer.lora_A.weight": torch.ones(3, 2),
        "base_model.model.layer.lora_B.weight": torch.ones(2, 3),
        "base_model.model.layer.lora_A.default.weight": torch.zeros(3, 2),
    }

    remapped = insert_peft_adapter_name(state)

    assert "base_model.model.layer.lora_A.default.weight" in remapped
    assert "base_model.model.layer.lora_B.default.weight" in remapped
    assert "base_model.model.layer.lora_A.weight" not in remapped
    assert "base_model.model.layer.lora_B.weight" not in remapped


def test_load_lora_state_dict_compat_loads_checkpoint_keys():
    model = _PeftLikeModel()
    checkpoint_state = {
        "base_model.model.layer.lora_A.weight": torch.full((3, 2), 2.0),
        "base_model.model.layer.lora_B.weight": torch.full((2, 3), 3.0),
    }

    load_lora_checkpoint(
        model,
        checkpoint_state,
        "generator",
        is_main_process=False,
    )

    assert torch.allclose(
        model.base_model.model.layer.lora_A["default"].weight,
        torch.full((3, 2), 2.0),
    )
    assert torch.allclose(
        model.base_model.model.layer.lora_B["default"].weight,
        torch.full((2, 3), 3.0),
    )


def test_configure_lora_for_model_targets_bidirectional_generator(monkeypatch):
    captured = {}

    def fake_lora_config(**kwargs):
        captured["target_modules"] = set(kwargs["target_modules"])
        return kwargs

    def fake_get_peft_model(model, peft_config):
        captured["peft_config"] = peft_config
        return model

    monkeypatch.setattr("utils.lora_utils.peft.LoraConfig", fake_lora_config)
    monkeypatch.setattr("utils.lora_utils.peft.get_peft_model", fake_get_peft_model)

    model = _TransformerWithBothAttentionTypes()
    wrapped = configure_lora_for_model(
        model,
        model_name="generator",
        lora_config={"type": "lora", "rank": 1, "alpha": 1, "dropout": 0.0},
        is_main_process=False,
    )

    assert wrapped is model
    assert captured["target_modules"] == {"bidirectional.proj"}


def test_configure_lora_for_model_targets_bidirectional_critic(monkeypatch):
    captured = {}

    def fake_lora_config(**kwargs):
        captured["target_modules"] = set(kwargs["target_modules"])
        return kwargs

    monkeypatch.setattr("utils.lora_utils.peft.LoraConfig", fake_lora_config)
    monkeypatch.setattr("utils.lora_utils.peft.get_peft_model", lambda model, _: model)

    configure_lora_for_model(
        _TransformerWithBothAttentionTypes(),
        model_name="fake_score",
        lora_config={"type": "lora", "rank": 1, "alpha": 1, "dropout": 0.0},
        is_main_process=False,
    )

    assert captured["target_modules"] == {"bidirectional.proj"}


def test_trainable_parameter_digest_is_stable_and_detects_changes():
    model = _PeftLikeModel()
    first, parameter_count, element_count = trainable_parameter_digest(model)
    second, _, _ = trainable_parameter_digest(model)

    assert first == second
    assert parameter_count == 2
    assert element_count == 12

    with torch.no_grad():
        model.base_model.model.layer.lora_A["default"].weight[0, 0] += 1
    changed, _, _ = trainable_parameter_digest(model)
    assert changed != first


def test_cross_rank_digest_check_rejects_mismatch(monkeypatch):
    model = _PeftLikeModel()
    monkeypatch.setattr("utils.lora_utils.dist.is_initialized", lambda: True)
    monkeypatch.setattr("utils.lora_utils.dist.get_world_size", lambda: 2)

    def fake_all_gather(outputs, local):
        outputs[0].copy_(local)
        outputs[1].copy_(local)
        outputs[1][0] ^= 1

    monkeypatch.setattr("utils.lora_utils.dist.all_gather", fake_all_gather)
    with pytest.raises(RuntimeError, match="differ across ranks"):
        assert_trainable_lora_parameters_synced(
            {"generator": model}, torch.device("cpu")
        )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_real_gloo_common_seed_sync_and_rank_local_rng_streams(tmp_path):
    synced = _run_distributed_lora_seed_case(tmp_path, "synced")

    assert [result["status"] for result in synced] == ["ok", "ok"]
    assert len({result["generator_sha256"] for result in synced}) == 1
    assert len({result["critic_sha256"] for result in synced}) == 1
    assert (
        synced[0]["post_construction_draw"]
        != synced[1]["post_construction_draw"]
    )
    assert all(result["rng_unchanged"] for result in synced)

    mismatched = _run_distributed_lora_seed_case(
        tmp_path,
        "mismatched",
        perturb_critic_rank=1,
    )
    assert [result["status"] for result in mismatched] == [
        "rejected",
        "rejected",
    ]
    assert all(
        "critic LoRA parameters differ across ranks" in result["message"]
        for result in mismatched
    )
    assert all(result["rng_unchanged"] for result in mismatched)

    locally_invalid = _run_distributed_lora_seed_case(
        tmp_path,
        "locally_invalid",
        invalidate_critic_rank=1,
    )
    assert [result["status"] for result in locally_invalid] == [
        "rejected",
        "rejected",
    ]
    assert all(
        "critic LoRA validation failed on ranks [1]" in result["message"]
        for result in locally_invalid
    )
    assert all(result["rng_unchanged"] for result in locally_invalid)
