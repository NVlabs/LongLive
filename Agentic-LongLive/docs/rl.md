# Training the RL memory planner

The memory planner is Qwen3-8B with a LoRA adapter (r = 32, alpha = 64, all linear layers). For every revealed shot
prompt it writes one JSON memory update with the deployed planner prompt (`agent/prompts.py`), parser
(`memory/update.py`), entity store and G / M member selection (`agent/policy.py`). Training is text only: the agent
never reads the video, so its inputs do not depend on the generator. The code is in `agentic_longlive/rl/`.

**Reward (version 4).** r = 0.35 r_text + 0.3 r_img + 0.15 r_prec + 0.05 r_ret − r_pen.

- `r_text`: an LLM judge (Qwen3.6-35B-A3B) labels every persistent state that the next six shots need as held
  (+1), missing (0) or wrong (−1.5), using hidden annotations that the policy never sees.
- `r_img`: G and M are rendered with the agent's image model (FLUX.2-klein-4B, fixed seed), and a VLM checks each
  entity of the next shot. A recognisable entity earns 0.51 in G and 0.49 in M (down to half if drawn in a wrong
  state). Each M member absent from the next shot costs 0.11. The weights come from a regression on real videos.
- `r_prec` and `r_ret` reward precise facts and keeping returning entities. `r_pen` penalises unmatched entities,
  false facts and length.

`reward.py` documents all reward versions.

**Recipe.** A teacher (Qwen3.6-35B-A3B) plays the 56 training stories. Supervised fine-tuning (SFT) on the best of 4
outputs per state (2,865 of 3,138 states pass the 0.2 reward filter) warm-starts the policy. GRPO then runs
80 steps of 8 states × 8 samples with these settings:

- DAPO token-level loss, no KL term;
- zero-variance groups dropped, group std floor 0.1;
- temperature 0.9, lr 2e-5, gradient clipping 0.5;
- one-step off-policy pipelining.

Rollouts come from an SGLang Qwen3-8B server that hot-loads each new adapter. The deployed checkpoint (GRPO seed 0,
step 80) is merged into Qwen3-8B and served with EAGLE-3.

## Data layout

All paths are relative to `Agentic-LongLive/`; `data/` and `outputs/` are git-ignored.

```
data/rl/
  train/terse/trNNN_xxxxxx.json   56 training stories (24-32 shots), terse prompts       } 112 files, 11 MB
  train/full/trNNN_xxxxxx.json    the same stories, full prompts (restate everything)    }
  dev/terse/tr05N_xxxxxx.json     8 development stories                                  8 files, 0.9 MB
  rollouts/states_teacher.jsonl   3,138 teacher decision states, the GRPO state pool     125 MB
  sft/bestof.jsonl                best of 4 per state with rewards, the SFT data         51 MB
outputs/rl/
  sft/adapter_final               SFT adapter (GRPO init)
  grpo_s0/adapter_step0080        final GRPO adapter (checkpoints every 10 steps, train_log.jsonl)
  planner_merged/                 merged planner served by scripts/serve_agent.sh
```

Each story file uses the unified story format (`story.py`). Every shot also carries a hidden `annotation` (`present`,
`entity_state`, ...), and the `bible` lists the ground-truth entities. Only rewards and evaluation read them.
Each state row holds the planner's system and user prompt, the entity store before the decision, the teacher output
and the hindsight `targets` (`targets.py`).

**Download** (coming soon): the training data (about 190 MB) from the Hugging Face dataset
`Efficient-Large-Model/Agentic-LongLive-RL-Data`, and the merged planner from
`Efficient-Large-Model/Agentic-LongLive-Planner-Qwen3-8B` (16 GB, `scripts/download_weights.sh`).

```bash
hf download Efficient-Large-Model/Agentic-LongLive-RL-Data --repo-type dataset --local-dir data/rl   # coming soon
```

To write new stories instead, run
`python -m agentic_longlive.rl.storygen --urls http://127.0.0.1:31100/v1 --out data/rl/stories`. This needs the
judge servers below. With the default seed it reproduces the story ids above. Move `tr050`-`tr057` (terse) to `dev/`
and the other 56 to `train/{terse,full}`. `--preset hard` writes an entity-rich held-out set with Gemini.

## Requirements

- **Trainer environment:** `pip install torch transformers peft` (safetensors and huggingface_hub come with
  transformers), plus `requirements.txt`.
- **Services:** SGLang ≥ 0.5 (`sglang serve`), which may live in a separate environment.
- **Evaluation and `--preset hard` only:** Gemini credentials (`GEMINI_SERVICE_ACCOUNT`, the path to a Vertex AI
  service-account JSON).

## Steps

GPU ids and ports are set in `scripts/rl/env.sh` (defaults: one 8-GPU node per seed). Every script accepts
environment overrides.

| GPUs | Service |
| --- | --- |
| 0, 1 | Qwen3.6-35B-A3B-FP8 (teacher, judge, image auditor) |
| 2 | Qwen3-8B rollout server with LoRA |
| 3-6 | FLUX.2-klein-4B |
| 7 | trainer |

```bash
bash scripts/rl/serve.sh            # 0. services: judges :31100/:31101, rollout :30100, image :32100-32103
bash scripts/rl/collect.sh          # 1. teacher rollouts -> states_teacher.jsonl, best of 4 -> bestof.jsonl (skip with released data)
bash scripts/rl/sft.sh              # 2. SFT warm start -> outputs/rl/sft/adapter_final (no servers needed)
bash scripts/rl/grpo.sh             # 3. GRPO, 80 steps -> outputs/rl/grpo_s0 (SEED=1, 2 for more seeds)
bash scripts/rl/serve_planner.sh    # 4. merge + serve with EAGLE-3 as the agent planner (:30000, configs/agents/oss.yaml)
```

Step 1 needs only `bash scripts/rl/serve.sh judge`. Stop the RL services with `bash scripts/rl/serve.sh stop` before
step 4 if the GPUs are shared. The scripts wrap these module CLIs:

```bash
python -m agentic_longlive.rl.collect rollouts --stories data/rl/train/terse data/rl/train/full \
    --planner-urls $JUDGE_URLS --planner-model Qwen/Qwen3.6-35B-A3B-FP8 --tag teacher --out data/rl/rollouts
python -m agentic_longlive.rl.collect bestofk --states data/rl/rollouts/states_teacher.jsonl \
    --planner-urls $JUDGE_URLS --judge-urls $JUDGE_URLS --k 3 --out data/rl/sft/bestof.jsonl
torchrun --nproc_per_node 1 -m agentic_longlive.rl.train --config configs/rl/sft.json
torchrun --nproc_per_node 1 -m agentic_longlive.rl.train --config configs/rl/grpo.json seed=0 out=outputs/rl/grpo_s0
python -m agentic_longlive.rl.merge --adapter outputs/rl/grpo_s0/adapter_step0080 --out outputs/rl/planner_merged
```

Config values can be overridden as `key=value`, including nested reward fields (`reward.image=false`). GRPO resumes
from the newest checkpoint in `out`. The paper's ablations are separate configs:

- `configs/rl/ablations/grpo_text_reward.json`: "RL, text reward only" (reward version 3).
- `configs/rl/ablations/grpo_not_rendered.json`: "calibrated, not rendered" (the image weights applied to membership
  only).

Run them with `CONFIG=<config> bash scripts/rl/grpo.sh`.

**GPU budget.** SFT is 89 steps, about 75 minutes on one A100 80GB. In the paper, each GRPO seed trained on one H100.
The three seeds shared the rollout, judge and image services on two 8×H100 nodes and took about 2-2.5 hours in
parallel. The single-node layout above runs one seed. More image servers or judges shorten the scoring time. Each
step logs `t_rollout` and `t_score` in `train_log.jsonl`.

## Evaluation

Each planner rolls through held-out stories online. The memory after every decision is judged against the hidden
annotations, by Gemini by default, a different model family from the training judge.

```bash
python -m agentic_longlive.rl.evaluate rollout --name rl --urls http://127.0.0.1:30000/v1 \
    --model agentic-longlive-planner --stories STORIES/terse --out outputs/rl/eval
python -m agentic_longlive.rl.evaluate judge --name rl --stories STORIES/terse --out outputs/rl/eval
python -m agentic_longlive.rl.evaluate summary --out outputs/rl/eval
```

`--judge openai` uses the self-hosted Qwen3.6 judge instead, and `--rule recent --src NAME` scores a rule-based member
selection on another planner's states. `evaluate from-runs` turns live runs of the agent into records.
`agentic_longlive.rl.imgeval` renders and audits the G / M images of the same decisions.
