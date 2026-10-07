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

"""Download a published ELM checkpoint bundle at its pinned revision."""

import argparse
from pathlib import Path
import yaml

from longwam.paths import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", help="Optional explicit release repository")
    parser.add_argument("--revision", help="Prefer an immutable commit")
    args = parser.parse_args()
    registry = yaml.safe_load((repository_root() / "configs/checkpoints.yaml").read_text())
    if args.benchmark not in registry:
        parser.error(f"Unknown checkpoint: {args.benchmark}")
    entry = registry[args.benchmark]
    repo_id = args.repo_id or entry["repo_id"]
    if not repo_id:
        parser.error("Checkpoint address is intentionally TBD; supply --repo-id when released.")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=args.revision or entry["revision"],
        local_dir=str(args.output.expanduser()),
    )


if __name__ == "__main__":
    main()
