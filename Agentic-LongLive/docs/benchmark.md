# LongStory-Online: benchmark and evaluation

LongStory-Online (`stories/longstory_online/`) has 25 stories and 857 shots: 9 hand-written stories (211 shots) and 16
LongStory-Hard stories (646 shots). A method receives one prompt per shot. Each shot has a hidden annotation (place,
planned camera and content, persistent states, entities), and only the evaluation reads it. The main protocol uses
**terse** prompts. `stories/longstory_online/full/` holds the full-prompt control. The story list, statistics and file
format are in [`stories/longstory_online/README.md`](../stories/longstory_online/README.md).

## Protocol

* Every method sees **one prompt per shot** and never reads a future prompt. Before shot 0 it may read only the story
  `setup` (title, style, cast cards).
* Each shot is **generated once**. Regeneration and best-of-*N* are not allowed, but a method may keep its own
  closed-loop mechanisms.
* Methods never read `annotation`, `bible` or `terse` fields.
* Report the terse set as the main result and use the same seeds for all methods. The paper uses 3 seeds on
  LongLive-2.0 and averages them.

## Generate all stories

`scripts/run_story.py` runs one story through the live system and writes a run directory. Run it once per story:

```bash
for s in stories/longstory_online/*.json; do python scripts/run_story.py --config configs/longlive.yaml --story "$s" seed=0 output_dir=outputs/ours_s0/$(basename "$s" .json); done
```

Use the same loop with an ablation (`--ablation configs/ablations/no_memory.yaml`) or another config for each
compared method. Another generator can be scored too if it writes the same layout, one directory per story:

```text
outputs/<method>/<story>/
  config.json    {"story": "stories/longstory_online/<story>.json"}   (the file stem identifies the story)
  summary.json   {"per_shot": [{"shot": 0, "video_path": "shots/shot_000.mp4"}, ...]}
  shots/shot_000.mp4 ...
```

## Score

```bash
pip install google-auth                       # Gemini judge via Vertex AI
export GEMINI_SERVICE_ACCOUNT=/path/to/service_account.json
python scripts/evaluate.py --runs outputs/ours_s0 outputs/base_s0 --stories stories/longstory_online \
    --judge gemini --imaging --out results
```

Several seeds of one method are given as `NAME=DIR,DIR,...`. `--compare A B` adds paired differences A - B, with seed
*k* of A paired with seed *k* of B:

```bash
python scripts/evaluate.py --runs ours=outputs/ours_s0,outputs/ours_s1,outputs/ours_s2 \
    base=outputs/base_s0,outputs/base_s1,outputs/base_s2 --compare ours base --judge gemini --imaging --out results
```

Outputs:

* `results/table.md`: all 25 stories, the hand-written stories and LongStory-Hard;
* `results/scores.json`: every method, subset and story;
* `results/compare.{md,json}`: the paired differences;
* `<run>/eval/`: the judge records (`suite.json`, `identity.json`), the judge frames and the run's `scores.json`.

Records are cached, so an interrupted evaluation resumes where it stopped and `--judge none` only re-aggregates.
One seed of one method costs about 1,350 judge calls: 857 shots, 140 revisit pairs and up to 349 protagonist pairs.

**Judges.** The paper uses Gemini 3.8 Flash at temperature 0 with JSON output (`--judge gemini`). Credentials come
from `GEMINI_SERVICE_ACCOUNT` (Vertex AI) or, with `--judge-backend studio`, from `GEMINI_API_KEY`. Any
OpenAI-compatible VLM endpoint works as well, for example a Qwen VL model served by vLLM or SGLang:

```bash
python scripts/evaluate.py --runs outputs/ours_s0 --judge openai --judge-url http://127.0.0.1:8000/v1 \
    --judge-model Qwen/Qwen3-VL-32B-Instruct --cache eval_qwen --out results_qwen
```

Absolute numbers depend on the judge, so compare methods only under the same judge. Use `--cache` to keep the
records of different judges apart. The judge should differ from the models that produced the video (the agent and the
RL reward models).

## Metrics

| Metric | Definition |
|---|---|
| **State** | Shot judge on four frames (4/35/65/97% of the shot): each expected persistent fact is confirmed, contradicted or not visible. Score: confirmed / (confirmed + contradicted) per shot |
| **Identity** | On shots where the judge sees the protagonist, whether it matches the cast card (face, species, body, age; clothing may change) |
| **Same prot.** | The shots that count are fixed by the plan (`eval/expected_visible.json`, 349 shots). Each scores (rating - 1) / 4 of the protagonist pair judge (1-5) against the method's first shot of the protagonist, and 0 if the shot judge does not see the protagonist. The reference shot is excluded |
| **Location** | On scene changes, the shot is judged at its planned place |
| **Revisit** | On the 140 returns to an earlier place, the place pair judge (two frames per shot, 1-5) compares the shot with the last earlier shot there. Score: (rating - 1) / 4 if both shots are judged at their planned place, 0 otherwise |
| **Imaging** | VBench imaging quality (MUSIQ-SPAQ) on 16 frames per shot (`--imaging`, needs `torch torchvision pyiqa`) |
| **Artifacts** | Share of shots the judge flags with severe artifacts (ghosted blends, melting anatomy, frozen or corrupted frames). Lower is better |
| Framing | Supplementary: camera distance, angle and subject agree with the planned camera and content |
| Imaging drift | Supplementary: imaging quality of the last minute of a story minus the first minute (stories longer than 2 minutes) |
| MIR | Supplementary: ArcFace face identity in the StoryEngine protocol (`--mir`, needs `insightface onnxruntime-gpu`) |

Per-shot metrics are pooled over all shots of all stories of one seed. Story-level metrics (drift, MIR) are averaged
over stories, and seeds are averaged. A paired difference uses the shots present in both methods, with a 95% interval
from a story-clustered bootstrap (2000 draws, seeded) that counts as significant when it excludes zero. The judge
system prompts are in `agentic_longlive/eval/judge.py` and the paper's supplement. Do not edit them: a changed prompt
is a different benchmark version.
