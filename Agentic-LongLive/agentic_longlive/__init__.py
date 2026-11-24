# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic LongLive: asynchronous agentic memory for long video generation.

* ``runtime``   - the live session: prompts revealed one at a time, the memory agent running beside the video loop,
  memory prefilled ahead of time, a video loop that never waits for the agent.
* ``memory``    - entity memory (text state + images per entity), composed G / M snapshots, entity cards.
* ``agent``     - the memory agent: planner, member policy, painter (draw + check), scene router.
* ``backends``  - video generators: LongLive-2.0 in process, MiniMax-H3 generators through ``workers``.
* ``workers``   - LongLive-H3 and FastH3-Omni worker processes and their file-queue protocol.
* ``models``    - swappable LLM / VLM / image providers (self-hosted OpenAI-compatible servers or APIs).
* ``rl``        - training the memory planner (Qwen3-8B LoRA) with SFT and GRPO.
* ``robotics``  - the robot-video agent (observer, router, planner) and its real-time segment loop.
* ``eval``      - the LongStory-Online judge and metrics.
"""

__version__ = "1.0.0"
