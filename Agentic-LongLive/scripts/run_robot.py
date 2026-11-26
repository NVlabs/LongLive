# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a long-horizon robot episode (64 s, eight 8 s instructions) with and without the memory agent.

The video model runs out of process: start a generator worker that serves ``--queue`` first (e.g. the FastH3 worker,
``agentic_longlive/workers/fasth3.py``). The agent needs an OpenAI-compatible VLM endpoint (the paper: Qwen3.6-35B-A3B
on SGLang, thinking off). See docs/robotics.md.

    python scripts/run_robot.py --episode stories/robot/05_open_drawer_ego.json \\
        --image-root data/robot_first_frames --queue /path/to/fasth3_queue \\
        --agent-url http://127.0.0.1:31001/v1 --arm both --policy v4 --out outputs/robot/05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_longlive.robotics.agent import POLICIES  # noqa: E402
from agentic_longlive.robotics.episode import load_episode, resolve_first_frame  # noqa: E402
from agentic_longlive.robotics.llm import DEFAULT_EXTRA_BODY, DEFAULT_MODEL, AgentLLM  # noqa: E402
from agentic_longlive.robotics.runner import SEED0, Seeds, run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, type=Path, help="episode script (stories/robot/*.json)")
    ap.add_argument("--image", type=Path, help="the episode's first frame (overrides --image-root)")
    ap.add_argument("--image-root", type=Path,
                    help="folder holding the first frames under the episode's 'first_frame' name")
    ap.add_argument("--queue", required=True, type=Path, help="file queue served by the generator worker")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--arm", choices=("base", "ours", "both"), default="both",
                    help="base = generator alone, ours = generator + memory agent (default: both, same seeds)")
    ap.add_argument("--policy", choices=POLICIES, default="v4",
                    help="memory policy of the agent (default: v4, the paper's)")
    ap.add_argument("--grounding-crops", action="store_true",
                    help="v4: crop objects by name with Grounding DINO instead of the observer's boxes "
                         "(hybrid2 always does)")
    ap.add_argument("--agent-url", help="OpenAI-compatible base URL of the agent VLM, e.g. http://127.0.0.1:31001/v1")
    ap.add_argument("--agent-model", default=DEFAULT_MODEL, help=f"served model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--agent-api-key-env", help="environment variable holding an API key for the agent endpoint")
    ap.add_argument("--agent-extra-body", default=json.dumps(DEFAULT_EXTRA_BODY),
                    help="JSON merged into every chat request (default turns Qwen3 thinking off; '{}' for other APIs)")
    ap.add_argument("--seed", type=int, default=SEED0, help="segment i >= 1 uses seed + i (default: %(default)s)")
    ap.add_argument("--first-seed", type=int, default=SEED0, help="seed of segment 0, shared by both arms")
    ap.add_argument("--job-timeout", type=float, default=1800.0, help="seconds to wait for one segment")
    a = ap.parse_args()

    ep = load_episode(a.episode)
    image = resolve_first_frame(ep, a.image, a.image_root)
    arms = ("base", "ours") if a.arm == "both" else (a.arm,)
    llm = None
    if "ours" in arms:
        if not a.agent_url:
            ap.error("--agent-url is required for the ours arm")
        llm = AgentLLM(a.agent_url, a.agent_model, api_key_env=a.agent_api_key_env,
                       extra_body=json.loads(a.agent_extra_body))
    crop_mode = "grounding" if (a.grounding_crops and a.policy == "v4") else None
    videos = run_episode(ep, image, a.queue, a.out, arms=arms, policy=a.policy, llm=llm, crop_mode=crop_mode,
                         seeds=Seeds(base=a.seed, first=a.first_seed), job_timeout_s=a.job_timeout)
    for v in videos:
        print(v)


if __name__ == "__main__":
    main()
