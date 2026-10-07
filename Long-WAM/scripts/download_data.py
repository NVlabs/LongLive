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
# End Long-WAM attribution.

"""Download benchmark dataset files; never delete or automatically extract archives."""

import argparse
from pathlib import Path

DATASETS = {
    "libero": ("yuanty/LIBERO-fastwam", None),
    "robotwin2": ("yuanty/robotwin2.0-fastwam", None),
    "domino": ("h-embodvis/DOMINO", None),
    "robocasa_gr1": (
        "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim",
        "09c6de8af50168090e7e9cc01e1ec3bce788de24",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=DATASETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", help="Optional pinned dataset commit")
    parser.add_argument("--include", action="append", help="HF allow-pattern; may be repeated")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    repo, revision = DATASETS[args.benchmark]
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=args.revision or revision,
        allow_patterns=args.include,
        local_dir=str(args.output.expanduser()),
    )


if __name__ == "__main__":
    main()
