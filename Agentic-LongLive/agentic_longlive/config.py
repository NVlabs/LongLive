# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed run configuration.

A run is described by one YAML file (``configs/*.yaml``). A config may list ``defaults``
(other YAML files, merged first, relative to the including file) and may be overridden
from the command line with ``key.subkey=value`` pairs. The defaults below are the full system of the paper;
the ablation switches are documented next to each field and collected as overlays in ``configs/ablations``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class ModelSpec:
    """A swappable model provider (LLM, VLM checker or image generator)."""

    # gemini | openai_compat (LLM) | openai_image (image) | mock
    provider: str = "mock"
    model: str = ""
    base_url: str = ""
    # Name of the environment variable that holds an API key (never the key itself).
    api_key_env: str = ""
    # Name of the environment variable that holds a credentials *file path*
    # (e.g. a Google service-account JSON for Vertex AI).
    credentials_env: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    max_concurrency: int = 4
    min_interval_s: float = 0.0
    timeout_s: float = 180.0
    max_retries: int = 5


@dataclass
class AgentConfig:
    planner: ModelSpec = field(default_factory=ModelSpec)  # writes state updates and all memory text
    checker: ModelSpec = field(default_factory=ModelSpec)  # VLM that checks generated images
    image: ModelSpec = field(default_factory=ModelSpec)  # image generation / editing
    # Member selection: llm | recent | frequency | all_active (rules are ablation baselines).
    policy: str = "llm"
    # Anytime refinement of the composed memory images (Method 1): at most this many rounds.
    refine_rounds: int = 3
    # Repair rounds for each entity image after a failed check.
    entity_repair_rounds: int = 1
    check_images: bool = True
    # Commit rule. True (default): the first composed M of every update (round 0) is committed whatever the check
    # says, so memory never falls behind because an image failed its check (no lag from checking); later rounds
    # replace it only when they pass (see commit_min_score) and are not worse. G is the global identity anchor and
    # is replaced only by a checked image (the previous G stays meanwhile); without M slots the rule applies to G.
    # False: every version must pass its check (the earlier rule; a failed update leaves the old memory).
    commit_first_round: bool = True
    # Checker calibration: an image that fails its check but scores at least this much (0-10) may still be
    # committed (memory or entity image) when nothing better exists, so a minor flaw does not leave memory
    # stale; a passing image always wins. None: only passing images are committed. Checker dependent.
    commit_min_score: Optional[float] = None
    # Abort further refinement when a newer prompt is revealed (keeps memory lag low).
    stop_refine_on_new_prompt: bool = True
    # When the agent falls behind, one planner call integrates every prompt revealed since its last
    # update (False: each pending prompt is processed in full, in order).
    batch_catch_up: bool = True
    # Round 0 composes G/M right away from the current entity images + the new state text while changed
    # entities are redrawn in the background; later rounds recompose with the fresh entity images.
    # False: wait for all entity redraws before round 0.
    pipeline_entities: bool = True
    # Number of most recent revealed prompts shown to the planner besides the entity table.
    context_prompts: int = 3
    max_g_members: int = 3
    max_m_members: int = 4
    # Image size requested from the image model; images are fitted to the backend resolution.
    image_size: str = "1536x1024"
    # G shows its members on a plain studio backdrop (no place); False: G is composed in the planner's scene.
    neutral_g: bool = True
    # Ablation (closed loop): let the agent read the last frame of generated shots. The full system never does, so
    # video errors are never written back into memory.
    read_generated_video: bool = False
    # Upper bound only: the agent may read the *next* prompt. Breaks the online protocol.
    oracle_next_prompt: bool = False


@dataclass
class MemoryConfig:
    # none: no external memory (backend default behaviour)
    # anchor: only the cast identity anchors as G
    # text: G/M drawn directly from text (no entity memory; the R5-style ablation)
    # entity: full two-layer memory (entity memory -> composed G/M)
    mode: str = "entity"
    use_g: bool = True
    m_slots: int = 2
    persistent_transient_split: bool = True
    revision: bool = True
    multi_view: bool = True
    # Split each cast card at session setup (one planner call) into an unchanging identity description and the
    # card's changeable appearance (clothing, facial hair, accessories, ...) stored as initial facts, so later prompts
    # can change it. False: the whole card is the entity's immutable description.
    cast_split: bool = True
    # Entity images listed with every snapshot (protagonists, M members, then the current location); the scene router
    # reads the location descriptions from this list.
    max_reference_images: int = 9


@dataclass
class RuntimeConfig:
    # async: video never waits for the agent; sync: shot i waits until memory built from
    # prompts <= i-1 has finished all refinement rounds (the synchronous-agent baseline).
    mode: str = "async"
    # Prefill memory K/V on the auxiliary device as soon as a version is committed.
    aux_prefill: bool = True
    # Only use memory whose watermark is at most i-1-lag (lag ablation; 0 keeps the default d=1).
    extra_lag: int = 0
    # Seconds to wait for the bootstrap memory before the first shot (session setup time).
    bootstrap_timeout_s: float = 1800.0
    # Fail fast if the agent thread raises.
    stop_on_agent_error: bool = True
    # Replay the memory another run used at every shot (its output_dir) instead of running the agent. The agent is
    # open loop (it never reads the video), so injection ablations can share one memory stream; memory.use_g and
    # memory.m_slots still select which blocks are injected. Prefill runs inline.
    replay_memory: str = ""
    # Resume an interrupted run in the same output_dir: shots listed in <output_dir>/progress.jsonl whose video exists
    # are kept and not generated again (preemptible nodes). Only without a live agent (replayed or no memory), where a
    # shot depends on nothing but its prompt, its memory snapshot and its seed.
    resume_shots: bool = False
    # Prompt pacing: prompt i arrives prompt_interval_s * i seconds after prompt 0 (playback pace, e.g. 253 frames /
    # 24 fps = 10.54 s); the boundary of shot i waits for it. 0 reveals every prompt as soon as the video loop is ready.
    prompt_interval_s: float = 0.0
    # Entity cards: append "<name>: <description> Current state: ..." for every memory-store entity named in the
    # revealed prompt (whole-word string match against the store at the injected snapshot's watermark, no model
    # call after the prompt) to the generator's text prompt.
    entity_cards: bool = True


@dataclass
class BackendConfig:
    # longlive (LongLive-2.0, in process) | h3 (MiniMax-H3 family through a worker queue) | mock (CPU test stand-in)
    name: str = "mock"
    # sink | none (none disables memory injection even if the agent runs)
    injection: str = "sink"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    story: str = ""
    output_dir: str = "outputs/run"
    seed: int = 0
    # Limit the number of shots (None: the whole story).
    max_shots: Optional[int] = None
    backend: BackendConfig = field(default_factory=BackendConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # Write a JSONL trace of every agent decision (inputs, outputs, timing) for audit and RL.
    trace: bool = True


def _load_with_defaults(path: Path, seen: Optional[set] = None):
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config defaults at {path}")
    seen.add(path)
    raw = OmegaConf.load(path)
    defaults: List[str] = list(raw.pop("defaults", []) or [])
    merged = OmegaConf.create()
    for item in defaults:
        merged = OmegaConf.merge(merged, _load_with_defaults(path.parent / item, seen))
    return OmegaConf.merge(merged, raw)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None,
                overlays: Optional[List[str]] = None) -> RunConfig:
    """Load a run config: structured defaults <- YAML (+defaults) <- overlays <- CLI overrides."""
    cfg = OmegaConf.structured(RunConfig)
    if path:
        cfg = OmegaConf.merge(cfg, _load_with_defaults(Path(path)))
    for overlay in overlays or []:
        cfg = OmegaConf.merge(cfg, _load_with_defaults(Path(overlay)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    run: RunConfig = OmegaConf.to_object(cfg)
    validate(run)
    return run


def validate(cfg: RunConfig) -> None:
    allowed = {
        "memory.mode": (cfg.memory.mode, {"none", "anchor", "text", "entity"}),
        "runtime.mode": (cfg.runtime.mode, {"async", "sync"}),
        "backend.injection": (cfg.backend.injection, {"sink", "none"}),
        "agent.policy": (cfg.agent.policy, {"llm", "recent", "frequency", "all_active"}),
    }
    for key, (value, options) in allowed.items():
        if value not in options:
            raise ValueError(f"{key}={value!r} is not one of {sorted(options)}")
    if cfg.agent.refine_rounds < 1:
        raise ValueError("agent.refine_rounds must be >= 1 (round 0 is the first draft)")
    if cfg.memory.m_slots < 0:
        raise ValueError("memory.m_slots must be >= 0")


def to_dict(cfg: RunConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)
