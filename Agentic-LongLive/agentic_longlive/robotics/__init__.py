# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Long-horizon robot videos with an observation-based memory agent (paper appendix, "Robot Videos").

Input: one first-frame image of a robot workspace + a script of eight 8 s instructions in which things leave the view
and come back. A video model renders each segment continuing from the previous one; the memory agent observes finished
segments, keeps a ledger and a bank of screenshots of the stream, and decides which pictures enter the generator's
sink and how the next instruction is worded.

* ``agent``     observer / router / planner, policies ``v4`` (the paper's) and ``hybrid2``
* ``prompts``   the agent's prompts and the generator prompt's fixed texts
* ``episode``   episode scripts and generator prompts
* ``runner``    the real-time (lag-1) schedule, arms ``base`` and ``ours``, file-queue generator client
* ``llm``       OpenAI-compatible VLM client of the agent
* ``grounding`` Grounding DINO object boxes (hybrid2 crops)
* ``video``     ffmpeg frame extraction and joining

Entry point: ``scripts/run_robot.py``; documentation: ``docs/robotics.md``.
"""

from .agent import POLICIES, HideAgent, RobotAgent, make_agent  # noqa: F401
from .episode import base_prompt, load_episode, ours_prompt  # noqa: F401
from .runner import Seeds, run_episode  # noqa: F401
