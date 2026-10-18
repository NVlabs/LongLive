# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

"""Keep Stage 2 training and inference on the same validated NumPy series."""

from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, supported by the public package.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def requirements_for(name, extra=None):
    requirements = list(PROJECT["dependencies"])
    if extra:
        requirements.extend(PROJECT["optional-dependencies"][extra])
    return [req for value in requirements if (req := Requirement(value)).name == name]


@pytest.mark.parametrize("extra", [None, "infra", "train"])
@pytest.mark.parametrize("version,allowed", [
    ("1.26.4", False),
    ("2.0.0", False),
    ("2.2.5", False),
    ("2.2.6", True),
    ("2.3.0", False),
    ("3.0.0", False),
])
def test_numpy_constraints_match_environment(extra, version, allowed):
    requirements = requirements_for("numpy", extra)
    assert len(requirements) == 1  # Extras must inherit the shared base requirement.
    assert all(version in req.specifier for req in requirements) is allowed


def test_lerobot_and_training_data_pins_remain_separate():
    assert not requirements_for("lerobot")
    assert not requirements_for("lerobot", "train")
    assert str(requirements_for("lerobot", "infra")[0].specifier) == "==0.4.1"
    assert str(requirements_for("datasets", "train")[0].specifier) == "==3.6.0"


def test_lerobot_headless_opencv_matches_base_series():
    opencv = requirements_for("opencv-python")[0]
    headless = requirements_for("opencv-python-headless", "infra")[0]
    assert headless.specifier == opencv.specifier


def test_installed_numpy_matches_shared_requirement():
    import numpy as np

    assert np.__version__ in requirements_for("numpy")[0].specifier


def test_numpy_arrow_training_tensor_round_trip(tmp_path):
    import numpy as np
    import torch

    datasets = pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")
    from longwam.datasets.lerobot.lerobot.datasets.utils import (
        get_hf_features_from_features,
        hf_transform_to_torch,
    )

    actions = np.arange(48, dtype=np.float32).reshape(16, 3) / 16
    features = get_hf_features_from_features({"action": {"dtype": "float32", "shape": (3,)}})
    path = str(tmp_path / "actions.parquet")
    datasets.Dataset.from_dict({"action": actions}, features=features).to_parquet(path)
    dataset = datasets.Dataset.from_parquet(path, cache_dir=str(tmp_path / "cache"))
    dataset.set_transform(hf_transform_to_torch)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=16)))["action"]
    assert batch.dtype == torch.float32
    np.testing.assert_array_equal(batch.numpy(), actions)

    # A tiny backward pass checks the data boundary, not full Long-WAM training.
    weight = torch.ones(3, 1, requires_grad=True)
    loss = (batch @ weight).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(weight.grad).all()


def test_runtime_import_does_not_require_training_boto3():
    subprocess.run(
        [sys.executable, "-B", "-c", (
            "import sys; sys.modules['boto3'] = None; "
            "from longwam.runtime import create_policy; "
            "assert callable(create_policy)"
        )],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
