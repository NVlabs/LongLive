# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot integration imported from the author video-training branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/tests/test_generator_checkpoint_loading.py
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

import importlib.machinery
import importlib.util
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


def _load_diffusion_trainer_helpers():
    module_path = Path(__file__).resolve().parents[1] / "trainer" / "diffusion.py"
    saved = dict(sys.modules)
    try:
        model = types.ModuleType("model")
        model.CausalDiffusion = object
        sys.modules["model"] = model

        wan = types.ModuleType("wan_5b")
        wan.__path__ = []
        wan_dist = types.ModuleType("wan_5b.distributed")
        wan_dist.__path__ = []
        sp_training = types.ModuleType("wan_5b.distributed.sp_training")
        sp_training.SequenceParallelHelper = object
        sys.modules.update(
            {
                "wan_5b": wan,
                "wan_5b.distributed": wan_dist,
                "wan_5b.distributed.sp_training": sp_training,
            }
        )

        utils = types.ModuleType("utils")
        utils.__path__ = []
        dataset = types.ModuleType("utils.dataset")
        for name in (
            "MultiTextConcatDataset",
            "MultiVideoConcatDataset",
            "build_distributed_sampler",
            "cycle",
            "eval_collate_fn",
            "multi_video_collate_fn",
            "resolve_resume_data_cursor",
        ):
            setattr(dataset, name, object)
        config = types.ModuleType("utils.config")
        config.section_get = lambda *args, **kwargs: None
        config.wan_default_config = {}
        evaluation = types.ModuleType("utils.evaluation")
        for name in (
            "deterministic_evaluation_seed",
            "evaluation_artifact_run_modes",
            "evaluation_group_specs",
            "evaluation_run_modes",
            "fixed_evaluation_log_key",
            "fixed_evaluation_writer",
            "require_equal_evaluation_value",
            "select_fixed_evaluation_subset",
            "synchronize_evaluation_failure",
        ):
            setattr(evaluation, name, lambda *args, **kwargs: None)
        misc = types.ModuleType("utils.misc")
        misc.set_seed = lambda seed: None
        distributed = types.ModuleType("utils.distributed")
        from datetime import timedelta
        distributed.TRAINING_PROCESS_GROUP_TIMEOUT = timedelta(minutes=60)
        for name in ("EMA_FSDP", "FSDP"):
            setattr(distributed, name, object)
        for name in ("barrier", "fsdp_wrap", "launch_distributed_job"):
            setattr(distributed, name, lambda *args, **kwargs: None)
        sys.modules.update(
            {
                "utils": utils,
                "utils.dataset": dataset,
                "utils.config": config,
                "utils.evaluation": evaluation,
                "utils.misc": misc,
                "utils.distributed": distributed,
            }
        )

        omegaconf = types.ModuleType("omegaconf")
        omegaconf.__spec__ = importlib.machinery.ModuleSpec(
            "omegaconf", loader=None
        )
        omegaconf.OmegaConf = object
        sys.modules["omegaconf"] = omegaconf
        sys.modules["wandb"] = types.ModuleType("wandb")

        torchvision = types.ModuleType("torchvision")
        torchvision.__path__ = []
        torchvision_io = types.ModuleType("torchvision.io")
        torchvision_io.write_video = lambda *args, **kwargs: None
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.io"] = torchvision_io

        spec = importlib.util.spec_from_file_location(
            "_trainer_diffusion_checkpoint_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        current = set(sys.modules)
        original = set(saved)
        for name in current - original:
            sys.modules.pop(name, None)
        for name, value in saved.items():
            sys.modules[name] = value


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)


class _Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.scale = torch.nn.Parameter(torch.ones(()))


class GeneratorCheckpointLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_diffusion_trainer_helpers()

    def _resume_contract_body(self, step, world_size=2):
        return self.module._resume_contract_body(
            step=step,
            topology={
                "world_size": world_size,
                "sequence_parallel_size": 1,
                "data_parallel_size": world_size,
            },
            training={
                "batch_size_per_data_parallel_rank": 1,
                "gradient_accumulation_steps": 1,
                "global_batch_size": world_size,
                "max_iters": 20,
                "model_name": "fixture",
                "num_frame_per_block": 8,
                "image_or_video_shape": [1, 8, 1, 1, 1],
                "algorithm": {
                    "causal": True,
                    "teacher_forcing": True,
                    "i2v": True,
                    "independent_first_frame": True,
                    "num_train_timestep": 1000,
                    "denoising_loss_type": "flow",
                    "noise_augmentation_max_timestep": 0,
                },
                "attention_and_schedule": {
                    "local_attn_size": 8,
                    "sink_size": 0,
                    "timestep_shift": 5.0,
                    "t_scale": 1.0,
                    "rope_method": "linear",
                },
                "distributed_precision": {
                    "mixed_precision": True,
                    "sharding_strategy": "hybrid_full",
                    "gradient_checkpointing": True,
                    "vae_halo_latents": 28,
                },
                "inference_conditioning": {
                    "negative_prompt": "fixture negative prompt",
                    "guidance_scale": 3.0,
                },
            },
            code={
                relative_path: "a" * 64
                for relative_path in self.module.RESUME_CRITICAL_CODE_PATHS
            },
            dataset={"dataset_class": "Fixture", "index_contract": {"count": 4}},
            sampler={
                "sampler_class": "FixtureSampler",
                "seed": 7,
                "shuffle": False,
                "drop_last": False,
                "num_replicas": world_size,
                "num_samples_per_replica": 2,
                "total_size": 4,
                "dataset_size": 4,
                "batches_per_epoch": 2,
            },
            data_cursor={
                "step": step,
                "epoch": step // 2,
                "batch_in_epoch": step % 2,
                "consumed_microbatches": step,
            },
            ema_required=False,
            error_buffer_required=True,
            noise_error_buffer_required=True,
        )

    def test_resume_critical_code_inventory_is_complete_and_hashed(self):
        root = Path(__file__).resolve().parents[1]
        identity = self.module._training_code_identity(root)
        self.assertEqual(
            tuple(identity), self.module.RESUME_CRITICAL_CODE_PATHS
        )
        for required in (
            "train.py",
            "trainer/sp_helper.py",
            "utils/wan_5b_wrapper.py",
            "utils/i2v_conditioning.py",
            "utils/config.py",
        ):
            self.assertIn(required, identity)
        self.assertTrue(
            all(
                len(digest) == 64
                and set(digest).issubset(set("0123456789abcdef"))
                for digest in identity.values()
            )
        )

    def _publish_fixture(self, root, step, world_size=2):
        staging, final = self.module._checkpoint_directory_paths(root, step)
        self.module._prepare_checkpoint_staging_directory(staging, final)
        staging = Path(staging)
        (staging / "model.pt").write_bytes(b"closed-model")
        for rank in range(world_size):
            (staging / self.module._rank_state_filename(rank)).write_bytes(
                f"closed-rank-{rank}".encode()
            )
        for stem in ("error_buffer", "noise_error_buffer"):
            (staging / self.module._canonical_buffer_filename(stem, 0)).write_bytes(
                f"closed-{stem}".encode()
            )
        body = self._resume_contract_body(step, world_size)
        contract = self.module._write_sealed_json(
            staging / self.module.RESUME_CONTRACT_FILENAME, body
        )
        self.module._write_completion_marker(staging, contract)
        self.module._publish_checkpoint_directory(
            staging, final, expected_step=step
        )
        return Path(final), contract

    def test_raw_inner_state_loads_only_exact_inner_model(self):
        source = _Wrapper()
        target = _Wrapper()
        state = {key: value.clone() for key, value in source.model.state_dict().items()}

        loaded_target = self.module._strict_load_generator_state_dict(target, state)

        self.assertEqual(loaded_target, "generator.model")
        for key, value in source.model.state_dict().items():
            self.assertTrue(torch.equal(value, target.model.state_dict()[key]))

    def test_persisted_wandb_id_enables_allow_resume(self):
        self.assertEqual(self.module._wandb_resume_mode("phase-run-id"), "allow")
        self.assertIsNone(self.module._wandb_resume_mode(None))
        self.assertIsNone(self.module._wandb_resume_mode(""))

    def test_allocation_stop_preserves_global_step_boundary(self):
        self.assertFalse(self.module._allocation_stop_reached(299, 300))
        self.assertTrue(self.module._allocation_stop_reached(300, 300))
        self.assertTrue(self.module._allocation_stop_reached(301, 300))
        self.assertFalse(self.module._allocation_stop_reached(600, None))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.module._allocation_stop_reached(0, 0)

    def test_evaluation_cadence_is_global_and_includes_exact_final_step(self):
        due = self.module._evaluation_due
        self.assertFalse(due(499, 500, 4352, evaluate_at_end=True))
        self.assertTrue(due(500, 500, 4352, evaluate_at_end=True))
        self.assertTrue(due(4000, 500, 4352, evaluate_at_end=True))
        self.assertFalse(due(4351, 500, 4352, evaluate_at_end=True))
        self.assertTrue(due(4352, 500, 4352, evaluate_at_end=True))
        self.assertFalse(due(4352, 500, 4352, evaluate_at_end=False))
        with self.assertRaisesRegex(ValueError, "step must be at least 1"):
            due(0, 500, 4352, evaluate_at_end=True)

    def test_checkpoint_cadence_always_covers_stop_and_final_boundaries(self):
        due = self.module._checkpoint_due
        self.assertTrue(due(500, 500, 4352))
        self.assertFalse(due(700, 500, 4352))
        self.assertTrue(due(800, 500, 4352, stop_after_step=800))
        self.assertFalse(due(4351, 500, 4352))
        self.assertTrue(due(4352, 500, 4352))
        with self.assertRaisesRegex(ValueError, "interval must be at least 1"):
            due(1, 0, 4352)

    def test_prompt_and_tensor_artifacts_publish_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path = root / "prompt.txt"
            self.module.save_prompts_to_txt(
                ["move the block", "move the block", "release the block"],
                str(prompt_path),
                True,
            )
            self.assertEqual(
                prompt_path.read_text(encoding="utf-8"),
                "[0,1] move the block\n[2] release the block\n",
            )

            tensor_path = root / "latent.pt"
            expected = torch.arange(4)
            self.module._atomic_torch_save(expected, tensor_path)
            self.assertTrue(torch.equal(torch.load(tensor_path), expected))
            self.assertEqual(
                [path for path in root.iterdir() if ".tmp." in path.name], []
            )

    def test_video_artifact_is_decoded_before_atomic_publication(self):
        import av

        def write_video_with_av(path, frames, fps):
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=fps)
                stream.width = int(frames.shape[2])
                stream.height = int(frames.shape[1])
                stream.pix_fmt = "yuv420p"
                for value in frames.cpu().numpy():
                    frame = av.VideoFrame.from_ndarray(value, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

        original_write_video = self.module.write_video
        self.module.write_video = write_video_with_av
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "validation.mp4"
                frames = torch.randint(
                    0, 256, (3, 16, 16, 3), dtype=torch.uint8
                )
                metadata = self.module._atomic_write_video_artifact(
                    path,
                    frames,
                    fps=24,
                    expected_frames=3,
                    expected_width=16,
                    expected_height=16,
                )
                self.assertEqual(metadata["decoded_frames"], 3)
                self.assertTrue(path.is_file())
                self.assertEqual(
                    [item for item in path.parent.iterdir() if ".tmp." in item.name],
                    [],
                )
                with self.assertRaisesRegex(RuntimeError, "frame-count mismatch"):
                    self.module._validate_mp4_artifact(
                        path,
                        expected_frames=4,
                        expected_fps=24,
                        expected_width=16,
                        expected_height=16,
                    )
        finally:
            self.module.write_video = original_write_video

    def test_partial_raw_state_is_rejected(self):
        wrapper = _Wrapper()
        state = dict(wrapper.model.state_dict())
        state.pop(next(iter(state)))
        with self.assertRaisesRegex(RuntimeError, "Refusing a partial load"):
            self.module._strict_load_generator_state_dict(wrapper, state)

    def test_raw_checkpoint_has_no_resume_payload(self):
        source = _Wrapper()
        target = _Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.pt"
            torch.save(source.model.state_dict(), path)
            resume, loaded_target = self.module._load_generator_checkpoint_strict(
                path, target
            )
        self.assertIsNone(resume)
        self.assertEqual(loaded_target, "generator.model")

    def test_full_checkpoint_retains_only_auxiliary_resume_state(self):
        source = _Wrapper()
        target = _Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.pt"
            torch.save(
                {
                    "generator": source.state_dict(),
                    "step": 17,
                    "generator_optimizer": {"state": {}},
                },
                path,
            )
            resume, loaded_target = self.module._load_generator_checkpoint_strict(
                path, target
            )
        self.assertEqual(loaded_target, "generator")
        self.assertEqual(resume["step"], 17)
        self.assertNotIn("generator", resume)

    def test_full_checkpoint_generator_only_discards_resume_state(self):
        source = _Wrapper()
        target = _Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.pt"
            torch.save(
                {
                    "generator": source.state_dict(),
                    "generator_ema": source.state_dict(),
                    "generator_optimizer": {"state": {1: {"step": 17}}},
                    "step": 17,
                },
                path,
            )
            resume, loaded_target = self.module._load_generator_checkpoint_strict(
                path,
                target,
                retain_auxiliary_state=False,
            )

        self.assertIsNone(resume)
        self.assertEqual(loaded_target, "generator")
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_generator_checkpoint_mode_validation_fails_before_resolution(self):
        called = False

        def find_latest(_):
            nonlocal called
            called = True
            return None

        with self.assertRaisesRegex(ValueError, "generator_ckpt_load_mode"):
            self.module._resolve_checkpoint_load_plan(
                auto_resume=True,
                output_path="/unused",
                generator_ckpt="/unused/base.pt",
                generator_ckpt_load_mode="not-a-mode",
                find_latest_checkpoint=find_latest,
            )
        self.assertFalse(called)

    def test_auto_resume_forces_full_state_over_generator_only(self):
        latest = "/run/checkpoint_model_000123/model.pt"
        plan = self.module._resolve_checkpoint_load_plan(
            auto_resume=True,
            output_path="/run",
            generator_ckpt="/base/checkpoint_model_000600/model.pt",
            generator_ckpt_load_mode="generator_only",
            find_latest_checkpoint=lambda _: latest,
        )
        self.assertEqual(plan, (latest, "auto_resume", "resume"))

    def test_rng_state_restores_python_numpy_and_torch_cpu(self):
        random.seed(11)
        np.random.seed(12)
        state = self.module._capture_rng_state()
        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )
        for _ in range(5):
            random.random()
            np.random.random()
            torch.rand(4)
        self.module._restore_rng_state(state)
        observed = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )
        self.assertEqual(observed[0], expected[0])
        self.assertEqual(observed[1], expected[1])
        self.assertTrue(torch.equal(observed[2], expected[2]))

    def test_empty_logdir_uses_explicit_generator_only_phase_init(self):
        base = "/base/checkpoint_model_000600/model.pt"
        plan = self.module._resolve_checkpoint_load_plan(
            auto_resume=True,
            output_path="/new-run",
            generator_ckpt=base,
            generator_ckpt_load_mode="generator_only",
            find_latest_checkpoint=lambda _: None,
        )
        self.assertEqual(plan, (base, "generator_ckpt", "generator_only"))

    def test_explicit_checkpoint_defaults_to_resume_mode(self):
        base = "/base/checkpoint_model_000600/model.pt"
        plan = self.module._resolve_checkpoint_load_plan(
            auto_resume=False,
            output_path="/unused",
            generator_ckpt=base,
            generator_ckpt_load_mode=None,
            find_latest_checkpoint=lambda _: self.fail(
                "disabled auto-resume must not inspect the logdir"
            ),
        )
        self.assertEqual(plan, (base, "generator_ckpt", "resume"))

    def test_legacy_checkpoint_falls_back_when_mmap_is_unavailable(self):
        source = _Wrapper()
        target = _Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(
                source.model.state_dict(),
                path,
                _use_new_zipfile_serialization=False,
            )
            resume, loaded_target = self.module._load_generator_checkpoint_strict(
                path,
                target,
                retain_auxiliary_state=False,
            )
        self.assertIsNone(resume)
        self.assertEqual(loaded_target, "generator.model")

    def test_incomplete_staging_is_ignored_until_atomic_publication(self):
        trainer = self.module.Trainer.__new__(self.module.Trainer)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous, _ = self._publish_fixture(root, 100)

            staging, final = self.module._checkpoint_directory_paths(root, 200)
            self.module._prepare_checkpoint_staging_directory(staging, final)
            staging = Path(staging)
            final = Path(final)
            (staging / "model.pt").write_bytes(b"model")

            # Even a non-empty model file in the hidden staging directory is
            # not a resumable checkpoint, and missing ER shards block publish.
            self.assertEqual(
                trainer.find_latest_checkpoint(root),
                str((previous / "model.pt").resolve()),
            )
            required = self.module._checkpoint_required_filenames(
                2,
                1,
                error_buffer_required=True,
                noise_error_buffer_required=True,
            )
            self.assertEqual(len(required), 5)
            with self.assertRaisesRegex(RuntimeError, "resume contract"):
                self.module._publish_checkpoint_directory(
                    staging, final, expected_step=200
                )

            for filename in required:
                path = staging / filename
                if not path.exists():
                    path.write_bytes(b"closed")
            body = self._resume_contract_body(200)
            contract = self.module._write_sealed_json(
                staging / self.module.RESUME_CONTRACT_FILENAME, body
            )
            self.module._write_completion_marker(staging, contract)
            self.module._publish_checkpoint_directory(
                staging, final, expected_step=200
            )

            self.assertFalse(staging.exists())
            self.assertTrue(final.is_dir())
            self.assertEqual(
                trainer.find_latest_checkpoint(root),
                str((final / "model.pt").resolve()),
            )

    def test_visible_incomplete_latest_checkpoint_fails_closed(self):
        trainer = self.module.Trainer.__new__(self.module.Trainer)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._publish_fixture(root, 100)
            broken = root / "checkpoint_model_000200"
            broken.mkdir()
            (broken / "model.pt").write_bytes(b"partial")
            with self.assertRaisesRegex(
                self.module.ResumeContractError, "resume contract"
            ):
                trainer.find_latest_checkpoint(root)

    def test_resume_model_envelope_missing_optimizer_fails_closed(self):
        body = self._resume_contract_body(5)
        contract = self.module._sealed_json_document(body)
        with self.assertRaisesRegex(
            self.module.ResumeContractError, "generator_optimizer"
        ):
            self.module._validate_resume_model_envelope(
                {
                    "checkpoint_contract_version": self.module.RESUME_CONTRACT_VERSION,
                    "resume_contract_sha256": contract["contract_sha256"],
                    "step": 5,
                },
                contract,
            )

    def test_resume_rejects_recipe_or_code_drift(self):
        body = self._resume_contract_body(5)
        contract = self.module._sealed_json_document(body)
        common = {
            "topology": body["topology"],
            "training": body["training"],
            "code": body["code"],
            "dataset": body["dataset"],
            "sampler": body["sampler"],
            "data_cursor": body["data_cursor"],
            "ema_required": False,
            "error_buffer_required": True,
            "noise_error_buffer_required": True,
        }
        self.module._validate_resume_runtime_contract(contract, **common)

        changed_training = dict(body["training"], max_iters=21)
        with self.assertRaisesRegex(
            self.module.ResumeContractError, "training"
        ):
            self.module._validate_resume_runtime_contract(
                contract, **dict(common, training=changed_training)
            )
        with self.assertRaisesRegex(self.module.ResumeContractError, "code"):
            self.module._validate_resume_runtime_contract(
                contract,
                **dict(
                    common,
                    code={"trainer/diffusion.py": "b" * 64},
                ),
            )


if __name__ == "__main__":
    unittest.main()
