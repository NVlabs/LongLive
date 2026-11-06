# SPDX-FileCopyrightText: Copyright (c) 2026 AuricTW
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM dependency regression tests.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

"""RoboCasa serving must not require the optional training data stack."""

import subprocess
import sys
import textwrap

import pytest


def run_without_training_packages(code, *args):
    # Use a fresh interpreter so a full training install cannot hide an import.
    blocked = "datasets", "pyarrow", "pandas", "git", "boto3", "wandb", "jsonlines", "lerobot"
    result = subprocess.run(
        [sys.executable, "-B", "-c", (
            "import sys; "
            f"sys.modules.update(dict.fromkeys({blocked!r}));\n"
            + textwrap.dedent(code)
        ), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_text_cache_reads_training_format_without_training_packages(tmp_path):
    run_without_training_packages(
        """
        import hashlib
        from pathlib import Path
        import sys
        import torch
        from longwam.inference import TextCache

        directory = Path(sys.argv[1])
        prompt = "Pour tea 茶"
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = directory / f"{digest}.t5_len128.wan22ti2v5b.pt"
        context = torch.ones(128, 4096, dtype=torch.bfloat16)
        mask = torch.arange(128) < 7
        torch.save({"context": context, "mask": mask}, path)
        actual, actual_mask = TextCache(directory)(prompt)
        assert actual.dtype == context.dtype
        torch.testing.assert_close(actual[:7], context[:7])
        assert not bool(actual[7:].any())
        assert actual_mask.dtype == torch.bool and bool(actual_mask.all())
        """,
        tmp_path,
    )


def test_statistics_loader_does_not_require_gitpython(tmp_path):
    run_without_training_packages(
        """
        import json
        from pathlib import Path
        import sys
        import torch
        from longwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from longwam.datasets.lerobot.processors.longwam_processor import LongWAMProcessor

        path = Path(sys.argv[1]) / "stats.json"
        path.write_text(json.dumps({"action": {"min": [-1, 0], "max": [1, 2]}}))
        stats = load_dataset_stats_from_json(str(path))
        torch.testing.assert_close(stats["action"]["min"], torch.tensor([-1., 0.]))
        torch.testing.assert_close(stats["action"]["max"], torch.tensor([1., 2.]))
        assert callable(LongWAMProcessor)
        """,
        tmp_path,
    )


@pytest.mark.parametrize("benchmark", ["robocasa_gr1", "robocasa365"])
def test_socket_policy_reaches_config_loading_without_training_packages(benchmark):
    run_without_training_packages(
        """
        import sys
        from unittest.mock import patch
        from omegaconf import OmegaConf
        from longwam.evaluation import serve

        class ReachedConfig(Exception):
            pass

        settings = OmegaConf.create({"benchmark": sys.argv[1], "model_config": "bundle.yaml"})
        # Stop before any bundle, model, GPU or socket is needed. Exercise the
        # real server dispatch and policy loader's imports up to that boundary.
        with patch("longwam.inference.load_run_config", side_effect=ReachedConfig) as load:
            try:
                serve(settings)
            except ReachedConfig:
                load.assert_called_once_with("bundle.yaml")
            else:
                raise AssertionError("Policy server did not reach config loading")
        """,
        benchmark,
    )
