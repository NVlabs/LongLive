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

"""Prune only explicitly committed weight/state pairs in this training output."""

import json
from pathlib import Path
import re
import shutil


def prune_checkpoint_pairs(weights_dir, state_dir, keep_last, protected_steps=()):
    if keep_last is None:
        return []
    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
        raise ValueError("checkpoint_keep_last must be null or a positive integer")
    weights_dir, state_dir = Path(weights_dir), Path(state_dir)
    protected = {int(step) for step in protected_steps}
    pairs = []
    for state in state_dir.iterdir():
        match = re.fullmatch(r"step_(\d+)", state.name)
        if not match or state.is_symlink() or not state.is_dir():
            continue
        step = int(match[1])
        weight = weights_dir / f"{state.name}.pt"
        marker = state / ".longwam-complete"
        if (
            not marker.is_file()
            or marker.is_symlink()
            or weight.is_symlink()
            or not weight.is_file()
        ):
            continue
        try:
            if json.loads((state / "trainer_state.json").read_text())["global_step"] != step:
                continue
        except (OSError, ValueError, KeyError):
            continue
        if (state / ".keep").exists():
            protected.add(step)
        pairs.append((step, weight, state))
    pairs.sort()
    removed = []
    for step, weight, state in pairs[:-keep_last]:
        if step in protected:
            continue
        shutil.rmtree(state)
        weight.unlink()
        removed.append(step)
    return removed
