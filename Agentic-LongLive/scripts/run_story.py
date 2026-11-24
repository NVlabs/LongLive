# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate one story with the live system: prompts are revealed one at a time, the agent writes memory
asynchronously and every shot starts from the newest committed memory.

    python scripts/run_story.py --config configs/longlive.yaml \
        --story stories/longstory_online/collie_muddy_day.json output_dir=outputs/collie

Extra ``key=value`` arguments override the config; ``--ablation`` adds overlay files (configs/ablations/*.yaml).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # MP4 writers fork after the text encoder ran
# memory K/V change size every version: growable segments keep the prefill GPU from fragmenting (same results)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from agentic_longlive.config import load_config  # noqa: E402
from agentic_longlive.runtime.session import LiveSession  # noqa: E402
from agentic_longlive.story import Story  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config, e.g. configs/longlive.yaml")
    parser.add_argument("--story", help="story JSON (overrides config.story)")
    parser.add_argument("--ablation", action="append", default=[], help="overlay YAML, may repeat")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    overrides = list(args.overrides) + ([f"story={args.story}"] if args.story else [])
    cfg = load_config(args.config, overrides, args.ablation)
    summary = LiveSession(cfg, Story.load(cfg.story)).run()
    keys = ("story", "shots", "backend", "memory_mode", "runtime_mode", "setup_seconds", "mean", "lag_histogram",
            "agent", "video")
    print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
