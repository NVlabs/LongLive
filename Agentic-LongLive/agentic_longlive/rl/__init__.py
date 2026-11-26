# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training the memory planner ("learning what to remember").

The planner LLM (Qwen3-8B + LoRA) writes one JSON memory update per revealed shot prompt with the deployed planner
prompt (``agent.prompts.PLAN_SYSTEM`` / ``plan_user``), parser (``memory.update.parse_plan``), entity store and G/M
member selection. Training is text-only: the agent is open loop, so its inputs never depend on the video.

* ``storygen``   – training stories with exact hidden per-entity state (full and terse prompt versions).
* ``textenv``    – text-only rollouts of the planner through a story; applying a candidate output to a stored state.
* ``targets``    – hindsight targets of a decision from the hidden annotations (needed facts, slots, retention).
* ``judge``      – the LLM judge that labels needed facts in a memory (training: Qwen3.6, evaluation: Gemini).
* ``imagelayer`` – renders G / M with the agent's image model and audits them with a VLM (image reward term).
* ``reward``     – reward versions 1 (best-of-K selection), 3 (text-only ablation) and 4 (paper).
* ``score``      – scoring candidate outputs (judge + image layer + reward) in a thread pool.
* ``collect``    – teacher rollouts (GRPO state pool) and best-of-K SFT data.
* ``train``      – SFT warm start and GRPO with LoRA (torchrun).
* ``merge``      – merges the LoRA into the base weights for EAGLE-3 serving.
* ``evaluate``   – offline text-layer evaluation; ``imgeval`` – image-layer evaluation.

Modules with heavy dependencies (``train``: torch / transformers / peft; ``merge``: safetensors) are imported only
when used.
"""
