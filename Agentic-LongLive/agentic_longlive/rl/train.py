# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SFT and GRPO for the memory planner (Qwen3-8B + LoRA), single- or multi-GPU with torchrun.

  torchrun --nproc_per_node K -m agentic_longlive.rl.train --config configs/rl/grpo.json [key=value ...]

Overrides are ``key=value`` pairs (values parsed as JSON when possible); nested reward fields use dots, e.g.
``reward.image=false``.

SFT (``mode=sft``, the RFT warm start): next-token NLL of the best-of-K teacher outputs (``sft_data``) whose best
reward is at least ``sft_min_reward``, with the deployed chat template (Qwen3 thinking off).

GRPO (``mode=grpo``), one optimizer update per batch:
  rank 0: sample ``batch_states`` decision states (teacher-trajectory states; every ``refresh_every`` steps the current
          policy also rolls through training stories and its own states join the pool with probability
          ``refresh_frac``), draw ``group_size`` completions per state from the SGLang rollout server with the current
          LoRA adapter (JSON mode and chat template as deployed), score each completion with the hindsight reward,
          compute group advantages A = (r - mean) / max(std, std_floor) and drop groups whose reward std is below
          ``min_group_std`` (no signal);
  all ranks: DAPO token-level policy gradient  L = -sum_i A_i sum_t log pi(y_it) / N_tokens  (no KL term) over their
          share of the completions, gradients all-reduced, gradient clipping, AdamW step on the LoRA parameters (the
          step is skipped when the pre-clip norm exceeds ``grad_guard``); rank 0 saves the adapter and hot-loads it
          into SGLang for the next step.
With ``pipeline`` the next batch is sampled and scored in the background while the current update runs, i.e. with
the adapter from before the update (one-step off-policy). The importance ratio is taken as 1 (no clipping).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .common import Endpoints, LLMAdapter, chat, http_post, read_jsonl

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Cfg:
    mode: str = "grpo"  # sft | grpo
    run: str = "grpo"  # run name (prefix of the adapter names loaded into SGLang)
    out: str = "outputs/rl/grpo"
    model: str = "Qwen/Qwen3-8B"  # HF id (resolved through HF_HOME) and the SGLang served base-model name
    init_adapter: str = ""  # LoRA to start from (GRPO: the SFT adapter)
    lora_r: int = 32
    lora_alpha: int = 64
    lr: float = 2e-5
    warmup: int = 5  # linear warm-up steps
    steps: int = 120  # GRPO steps
    batch_states: int = 8
    group_size: int = 8
    temperature: float = 1.0  # GRPO sampling temperature
    max_new_tokens: int = 2048
    max_prompt_tokens: int = 7500  # prompts are left-truncated to this many tokens
    states: List[str] = field(default_factory=list)  # GRPO state pool files (collect rollouts)
    sft_data: str = ""  # best-of-K rows (collect bestofk)
    sft_epochs: float = 1.0
    sft_batch: int = 32
    sft_min_reward: float = -9.0
    rollout_urls: str = "http://127.0.0.1:30100/v1"  # SGLang Qwen3-8B server(s) with --enable-lora
    judge_urls: str = "http://127.0.0.1:31000/v1,http://127.0.0.1:31001/v1"
    judge_model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    reward: Dict[str, Any] = field(default_factory=dict)  # RewardConfig fields
    std_floor: float = 0.1
    min_group_std: float = 0.02
    save_every: int = 20
    seed: int = 0
    grad_clip: float = 1.0
    grad_guard: float = 0.0  # skip the update when the pre-clip gradient norm exceeds this (0 = off)
    refresh_every: int = 0  # on-policy state refresh period in steps (0 = off)
    refresh_stories: Any = field(default_factory=list)  # story files (list, or comma-separated glob patterns)
    refresh_k: int = 8  # stories per refresh
    refresh_frac: float = 0.5  # probability that a sampled state comes from the on-policy pool
    judge_workers: int = 96
    pipeline: bool = True  # one-step off-policy sampling while the update runs
    micro_tokens: int = 24000  # padded tokens per forward micro-batch
    image_urls: str = ""  # reward version 4 image term: FLUX.2-klein servers (OpenAI image API, .../v1)
    image_model: str = "black-forest-labs/FLUX.2-klein-4B"
    vlm_urls: str = ""  # image audit (Qwen3.6 vision); default: judge_urls
    vlm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    resume: bool = True  # GRPO: continue from the newest adapter_stepNNNN (+ optimizer state) in ``out``


def _parse_value(v: str) -> Any:
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def build_cfg(data: Dict[str, Any], overrides: Sequence[str] = ()) -> Cfg:
    """Cfg from a config dict plus ``key=value`` / ``reward.key=value`` overrides."""
    data = json.loads(json.dumps(data))
    for ov in overrides:
        k, v = ov.split("=", 1)
        node, keys = data, k.split(".")
        for part in keys[:-1]:
            node = node.setdefault(part, {})
        node[keys[-1]] = _parse_value(v)
    known = {f.name for f in fields(Cfg)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise SystemExit(f"unknown config keys: {unknown}")
    cfg = Cfg(**data)
    if isinstance(cfg.refresh_stories, str):
        cfg.refresh_stories = sorted(f for pat in cfg.refresh_stories.split(",") if pat for f in glob.glob(pat))
    return cfg


def load_cfg(argv: Optional[Sequence[str]] = None) -> Cfg:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    return build_cfg(json.loads(Path(args.config).read_text()), args.overrides)


# ----------------------------------------------------------------------------------------------- model / tokens

def load_model(cfg: Cfg, device):
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(cfg.model, attn_implementation="sdpa", **{dkey: torch.bfloat16})
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if cfg.init_adapter:
        model = PeftModel.from_pretrained(model, cfg.init_adapter, is_trainable=True)
    else:
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=LORA_TARGETS)
        model = get_peft_model(model, lcfg)
    model.train()
    return model, tok


class Tok:
    """Tokenisation identical to deployment: chat template with thinking off; completions end with <|im_end|>."""

    def __init__(self, tok, max_prompt: int):
        self.tok = tok
        self.max_prompt = max_prompt
        self.end = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.end

    def prompt(self, system: str, user: str) -> List[int]:
        text = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids[-self.max_prompt:]

    def completion(self, text: str, finished: bool = True) -> List[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids + ([self.end] if finished else [])


def batch_logp(model, rows: Sequence[Tuple[List[int], List[int]]], device, pad_id: int) -> List[torch.Tensor]:
    """Per-token log-probs for several (prompt_ids, completion_ids) rows in one right-padded forward. Causal attention
    means real tokens never attend to the padding that follows them, so no attention mask is needed (flash SDPA)."""
    causal = model.get_base_model()
    width = max(len(p) + len(c) for p, c in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    for i, (p, c) in enumerate(rows):
        ids[i, : len(p) + len(c)] = torch.tensor(p + c, device=device)
    hidden = causal.model(input_ids=ids).last_hidden_state
    out = []
    for i, (p, c) in enumerate(rows):
        h = hidden[i, len(p) - 1: len(p) - 1 + len(c)]
        logits = causal.lm_head(h).float()
        target = torch.tensor(c, device=device)
        out.append(torch.log_softmax(logits, dim=-1).gather(-1, target[:, None]).squeeze(-1))
    return out


def micro_batches(rows: Sequence[tuple], max_tokens: int) -> Iterator[List[tuple]]:
    """Group rows ``(prompt_ids, completion_ids, ...)`` (longest first) so that rows * padded width <= max_tokens."""
    rows = sorted(rows, key=lambda r: -(len(r[0]) + len(r[1])))
    batch, width = [], 0
    for r in rows:
        w = max(width, len(r[0]) + len(r[1]))
        if batch and w * (len(batch) + 1) > max_tokens:
            yield batch
            batch, w = [], len(r[0]) + len(r[1])
        batch.append(r)
        width = w
    if batch:
        yield batch


def allreduce_grads(params, world: int) -> None:
    if world == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def lr_at(cfg: Cfg, step: int) -> float:
    return cfg.lr * min(1.0, step / max(1, cfg.warmup))


def group_advantages(totals: Sequence[float], std_floor: float, min_group_std: float) -> Optional[List[float]]:
    """GRPO advantages (r - mean) / max(std, std_floor) of one group; None when std < min_group_std (no signal)."""
    r = torch.tensor(list(totals))
    std = float(r.std(unbiased=False))
    if std < min_group_std:
        return None
    return ((r - r.mean()) / max(std, std_floor)).tolist()


# ----------------------------------------------------------------------------------------------- SGLang adapters

def _base(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


def sgl_load(urls: str, name: str, path: str) -> None:
    """Load (or replace) a LoRA adapter on every rollout server; an existing adapter of that name is unloaded first."""
    import urllib.error

    for u in urls.split(","):
        for _ in range(3):
            try:
                out = http_post(_base(u) + "/load_lora_adapter", {"lora_name": name, "lora_path": path}, timeout=600)
                if isinstance(out, dict) and out.get("success") is False:
                    raise RuntimeError(f"load_lora_adapter failed: {out}")
                break
            except (urllib.error.HTTPError, RuntimeError):
                sgl_unload(u, name)
                time.sleep(2)
        else:
            raise RuntimeError(f"could not load adapter {name} on {u}")


def sgl_unload(urls: str, name: str) -> None:
    for u in urls.split(","):
        try:
            http_post(_base(u) + "/unload_lora_adapter", {"lora_name": name}, timeout=120)
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------- SFT

def run_sft(cfg: Cfg, model, tk: Tok, rank: int, world: int, device, log) -> None:
    rows = [r for r in read_jsonl(cfg.sft_data) if max(r.get("rewards") or [0]) >= cfg.sft_min_reward]
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    n_steps = max(1, int(len(rows) * cfg.sft_epochs) // cfg.sft_batch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    if rank == 0:
        log(f"SFT rows={len(rows)} steps={n_steps} world={world}")
    for step in range(1, n_steps + 1):
        t0 = time.time()
        batch = [rows[k % len(rows)] for k in range((step - 1) * cfg.sft_batch, step * cfg.sft_batch)]
        enc = [(tk.prompt(r["system"], r["user"]), tk.completion(r["output"])) for r in batch]
        total = sum(len(c) for _, c in enc)
        mine = enc[rank::world]
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, step)
        opt.zero_grad(set_to_none=True)
        nll = 0.0
        for mb in micro_batches(mine, cfg.micro_tokens):
            lps = batch_logp(model, mb, device, tk.pad)
            loss = -sum(lp.sum() for lp in lps) / total
            loss.backward()
            nll += float(loss.item()) * total
        allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        t = torch.tensor([nll], device=device)
        if world > 1:
            dist.all_reduce(t)
        if rank == 0:
            log(json.dumps({"step": step, "nll": float(t.item()) / total, "tokens": total, "grad_norm": float(gn),
                            "sec": round(time.time() - t0, 1)}), jsonl=True)
            if step % cfg.save_every == 0 or step == n_steps:
                model.save_pretrained(str(Path(cfg.out) / f"adapter_step{step:04d}"))
    if rank == 0:
        model.save_pretrained(str(Path(cfg.out) / "adapter_final"))


# ----------------------------------------------------------------------------------------------- GRPO

class StatePool:
    """Teacher-trajectory states plus a bounded pool of on-policy states; only states with needed facts are used."""

    def __init__(self, cfg: Cfg, log):
        self.cfg = cfg
        self.base = []
        for p in cfg.states:
            self.base += [s for s in read_jsonl(p) if s["targets"]["F"]]
        if not self.base:
            raise SystemExit(f"no usable GRPO states in {cfg.states}")
        self.fresh: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.rng = random.Random(cfg.seed + 7)
        log(f"state pool: {len(self.base)} base states from {len(cfg.states)} files")

    def sample(self, k: int) -> List[Dict[str, Any]]:
        out = []
        with self.lock:
            for _ in range(k):
                if self.fresh and self.rng.random() < self.cfg.refresh_frac:
                    out.append(self.rng.choice(self.fresh))
                else:
                    out.append(self.rng.choice(self.base))
        return out

    def add(self, rows: List[Dict[str, Any]], cap: int = 2000) -> None:
        with self.lock:
            self.fresh = (self.fresh + [r for r in rows if r["targets"]["F"]])[-cap:]


def refresh_worker(cfg: Cfg, pool: StatePool, adapter_name: str, log) -> None:
    """Roll the current policy through training stories (deployment temperature) and add its own states."""
    from .collect import state_rows
    from .textenv import load_story, rollout, story_meta

    files = list(cfg.refresh_stories)
    rng = random.Random(hash(adapter_name) & 0xFFFF)
    rng.shuffle(files)
    eps = Endpoints(cfg.rollout_urls)

    def one(path):
        meta = story_meta(path)
        llm = LLMAdapter(eps, f"{cfg.model}:{adapter_name}", temperature=0.2, max_tokens=cfg.max_new_tokens)
        recs = rollout(load_story(path), llm)
        return state_rows(recs, meta, "terse" if "terse" in path else "full", f"refresh_{adapter_name}")

    rows = []
    with ThreadPoolExecutor(cfg.refresh_k) as ex:
        for r in ex.map(one, files[: cfg.refresh_k]):
            rows += r
    pool.add(rows)
    log(f"refresh {adapter_name}: +{len(rows)} on-policy states (fresh pool {len(pool.fresh)})")


def latest_checkpoint(out: Path) -> Optional[Tuple[int, Path]]:
    """(step, dir) of the newest complete adapter_stepNNNN in ``out``, or None."""
    best = None
    for d in out.glob("adapter_step*"):
        try:
            k = int(d.name.removeprefix("adapter_step"))
        except ValueError:
            continue
        if (d / "adapter_model.safetensors").is_file() and (best is None or k > best[0]):
            best = (k, d)
    return best


STAT_KEYS = ("belief", "support", "contradict", "visual", "slots", "retain", "n_false", "n_facts", "n_entities",
             "n_tokens", "false_pen", "len_pen", "image", "img_g", "img_m", "n_m_off", "n_m", "n_g", "m_recall",
             "g_recall", "img_fact_s", "img_fact_c", "img_qg", "img_qm")


def run_grpo(cfg: Cfg, model, tk: Tok, rank: int, world: int, device, log, start_step: int = 1) -> None:
    from .reward import RewardConfig
    from .score import Scorer, openai_judge

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    out = Path(cfg.out)
    pool = StatePool(cfg, log) if rank == 0 else None
    if start_step > 1:
        opt_file = out / f"opt_step{start_step - 1:04d}.pt"
        if opt_file.is_file():
            opt.load_state_dict(torch.load(opt_file, map_location=device))
        if rank == 0:
            log(f"resume at step {start_step}: adapter {cfg.init_adapter}, optimizer "
                f"{'restored' if opt_file.is_file() else 'reset (no state file)'}")
            fresh_file = out / "fresh_pool.json"
            if fresh_file.is_file():
                pool.add(json.loads(fresh_file.read_text()))
                log(f"resume: {len(pool.fresh)} on-policy states restored")
    rcfg = RewardConfig(**cfg.reward)
    image_layer = None
    if rank == 0 and rcfg.version >= 4 and rcfg.image:
        from .imagelayer import ImageLayer
        image_layer = ImageLayer(cfg.image_urls, cfg.vlm_urls or cfg.judge_urls, vlm_model=cfg.vlm_model,
                                 image_model=cfg.image_model, save_dir=str(out / "renders"))
    scorer = Scorer(openai_judge(cfg.judge_urls, cfg.judge_model), rcfg, workers=cfg.judge_workers,
                    tokenizer=tk.tok, image_layer=image_layer) if rank == 0 else None
    eps = Endpoints(cfg.rollout_urls)
    adapter, prev_adapter = None, None
    init_name = f"{cfg.run}_init"
    tag = f"{cfg.run}_{int(time.time()) % 100000}"  # unique adapter names per launch
    if rank == 0 and cfg.init_adapter:
        adapter = init_name
        sgl_load(cfg.rollout_urls, adapter, cfg.init_adapter)
    refresh_thread: Optional[threading.Thread] = None
    pinned: List[str] = []
    sample_pool = ThreadPoolExecutor(cfg.batch_states)
    prefetcher = ThreadPoolExecutor(1)

    def produce(step: int, adapter_name: Optional[str]) -> Dict[str, Any]:
        """Sample groups with ``adapter_name`` and score them (rank 0; runs in the background when pipelined)."""
        t0 = time.time()
        states = pool.sample(cfg.batch_states)
        model_name = f"{cfg.model}:{adapter_name}" if adapter_name else cfg.model

        def sample(st):
            return chat(eps, model_name, st["system"], st["user"], temperature=cfg.temperature,
                        max_tokens=cfg.max_new_tokens, n=cfg.group_size, timeout=1200)

        infos = list(sample_pool.map(sample, states))
        t_roll = time.time() - t0
        jobs, index = [], []
        for gi, (st, info) in enumerate(zip(states, infos)):
            for ci, text in enumerate(info["texts"]):
                jobs.append((st, text))
                index.append((gi, ci))
        scores = scorer.score_many(jobs)
        t_score = time.time() - t0 - t_roll
        groups, stats_rows = [], []
        for gi, (st, info) in enumerate(zip(states, infos)):
            rs = [(ci, scores[k]) for k, (g, ci) in enumerate(index) if g == gi]
            valid = [(ci, s) for ci, s in rs if s.get("total") is not None]
            stats_rows += [s for _, s in rs]
            if len(valid) < 2:
                continue
            adv = group_advantages([s["total"] for _, s in valid], cfg.std_floor, cfg.min_group_std)
            if adv is None:
                groups.append(None)
                continue
            p_ids = tk.prompt(st["system"], st["user"])
            comps = [tk.completion(info["texts"][ci], info["finish"][ci] != "length") for ci, _ in valid]
            groups.append({"prompt": p_ids, "comps": comps, "adv": adv})
        live = [g for g in groups if g is not None]
        totals = [s["total"] for s in stats_rows if s.get("total") is not None]

        def mean(key):
            v = [s[key] for s in stats_rows if isinstance(s.get(key), (int, float))]
            return round(sum(v) / len(v), 4) if v else None
        stat = {"step": step, "sampler": adapter_name or "base", "reward": round(sum(totals) / max(1, len(totals)), 4),
                "reward_std": round(float(torch.tensor(totals).std(unbiased=False)) if totals else 0.0, 4),
                "format_fail": round(sum(1 for s in stats_rows if s.get("format") == 0.0) / max(1, len(stats_rows)), 4),
                "judge_fail": sum(1 for s in stats_rows if s.get("judge_failed")),
                "groups_live": len(live), "groups_flat": sum(1 for g in groups if g is None),
                **{k: mean(k) for k in STAT_KEYS},
                "img_fail": sum(1 for s in stats_rows if s.get("image_failed")),
                "t_rollout": round(t_roll, 1), "t_score": round(t_score, 1), "fresh_states": len(pool.fresh)}
        if image_layer:
            rs_ = image_layer.renderer.stats
            stat.update({"renders": rs_["renders"], "render_hits": rs_["hits"],
                         "render_sec": round(rs_["sec"] / max(1, rs_["renders"]), 2)})
        return {"groups": live, "stat": stat}

    future = None
    for step in range(start_step, cfg.steps + 1):
        t0 = time.time()
        payload = None
        if rank == 0:
            if cfg.refresh_every and (step == start_step or step % cfg.refresh_every == 0) and \
                    (refresh_thread is None or not refresh_thread.is_alive()):
                pin = f"{tag}_pin{step}"
                src = str(out / "adapter_live") if adapter and adapter != init_name else cfg.init_adapter
                if src:
                    pin_dir = out / f"pin_{step:04d}"
                    if Path(src).is_dir() and src != str(pin_dir):
                        shutil.copytree(src, pin_dir, dirs_exist_ok=True)
                    sgl_load(cfg.rollout_urls, pin, str(pin_dir))
                    pinned.append(pin)
                    for old in pinned[:-2]:
                        sgl_unload(cfg.rollout_urls, old)
                    pinned = pinned[-2:]
                    refresh_thread = threading.Thread(target=refresh_worker, args=(cfg, pool, pin, log), daemon=True)
                    refresh_thread.start()
            payload = future.result() if future is not None else produce(step, adapter)
            payload["stat"]["step"] = step
            if cfg.pipeline and step < cfg.steps:
                # one-step off-policy: the next batch is sampled with the adapter from before this step's update
                future = prefetcher.submit(produce, step + 1, adapter)
            else:
                future = None
        obj = [payload]
        if world > 1:
            dist.broadcast_object_list(obj, src=0)
        payload = obj[0]
        groups = payload["groups"]
        total_tok = sum(len(c) for g in groups for c in g["comps"]) or 1
        seqs = [(g["prompt"], c, a) for g in groups for c, a in zip(g["comps"], g["adv"]) if abs(a) > 1e-8]
        seqs.sort(key=lambda x: -(len(x[0]) + len(x[1])))
        mine = seqs[rank::world]
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, step)
        opt.zero_grad(set_to_none=True)
        neg_logp, ntok = 0.0, 0
        for mb in micro_batches(mine, cfg.micro_tokens):
            lps = batch_logp(model, [(p, c) for p, c, _ in mb], device, tk.pad)
            loss = -sum(a * lp.sum() for (_, _, a), lp in zip(mb, lps)) / total_tok
            loss.backward()
            neg_logp += float(-sum(lp.detach().sum() for lp in lps).item())
            ntok += sum(len(c) for _, c, _ in mb)
        allreduce_grads(params, world)
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)) if seqs else 0.0
        skipped = bool(cfg.grad_guard and gn > cfg.grad_guard)
        if seqs and not skipped:
            opt.step()
        opt.zero_grad(set_to_none=True)
        t = torch.tensor([neg_logp, float(ntok)], device=device)
        if world > 1:
            dist.all_reduce(t)
        if rank == 0:
            live_dir = out / "adapter_live"
            model.save_pretrained(str(live_dir))
            new = f"{tag}_s{step}"
            sgl_load(cfg.rollout_urls, new, str(live_dir))
            if prev_adapter and prev_adapter not in pinned and prev_adapter != init_name:
                sgl_unload(cfg.rollout_urls, prev_adapter)
            prev_adapter, adapter = adapter, new
            if step % cfg.save_every == 0 or step == cfg.steps:
                tmp = out / f".tmp_adapter_step{step:04d}"
                shutil.copytree(live_dir, tmp, dirs_exist_ok=True)
                torch.save(opt.state_dict(), out / f"opt_step{step:04d}.pt")
                for old in sorted(out.glob("opt_step*.pt"))[:-2]:
                    old.unlink()
                (out / "fresh_pool.tmp").write_text(json.dumps(pool.fresh))
                (out / "fresh_pool.tmp").rename(out / "fresh_pool.json")
                final = out / f"adapter_step{step:04d}"
                if final.exists():
                    shutil.rmtree(final)
                tmp.rename(final)  # the adapter dir appears only when complete (resume relies on it)
            stat = payload["stat"]
            stat.update({"skipped": skipped, "grad_norm": round(gn, 4),
                         "nll_tokens": round(float(t[0]) / max(1.0, float(t[1])), 4), "seqs": len(seqs),
                         "lr": lr_at(cfg, step), "sec": round(time.time() - t0, 1)})
            log(json.dumps(stat), jsonl=True)
        if world > 1:
            dist.barrier()


def main() -> None:
    cfg = load_cfg()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))

    def log(msg: str, jsonl: bool = False) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(out / ("train_log.jsonl" if jsonl else "train.log"), "a") as fh:
            fh.write((msg if jsonl else line) + "\n")

    start_step = 1
    if cfg.mode == "grpo" and cfg.resume:
        ck = latest_checkpoint(out)
        if ck is not None:
            if ck[0] >= cfg.steps:
                if rank == 0:
                    log(f"already finished ({ck[1]}); nothing to do")
                    (out / "DONE").touch()
                os._exit(0)
            cfg.init_adapter, start_step = str(ck[1]), ck[0] + 1
    model, tok = load_model(cfg, device)
    tk = Tok(tok, cfg.max_prompt_tokens)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"model {cfg.model} adapter={cfg.init_adapter or 'new'} trainable={n_train} world={world}")
    if cfg.mode == "sft":
        run_sft(cfg, model, tk, rank, world, device, log)
    elif cfg.mode == "grpo":
        run_grpo(cfg, model, tk, rank, world, device, log, start_step)
    else:
        raise SystemExit(f"unknown mode {cfg.mode!r} (sft | grpo)")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        (out / "DONE").touch()
        log("training finished")
    os._exit(0)  # background sampler / refresh threads would otherwise keep rank 0 alive


if __name__ == "__main__":
    main()
