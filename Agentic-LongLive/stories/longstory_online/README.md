# LongStory-Online

LongStory-Online is a benchmark for **online** long multi-shot video generation. A method receives the prompts of a
story one shot at a time and has to keep characters, places and persistent changes consistent over minutes of video.
It has 25 stories and 857 shots, about 150 minutes of video:

* **9 hand-written stories** (211 shots, 14-40 shots each);
* **16 LongStory-Hard stories** (`hd00`-`hd15`, 646 shots, 38-42 shots each). An LLM designed them and code
  accumulated their hidden states from initial attributes and per-shot changes. Each has 13-14 entities, 2-4 of them
  per shot, and 7-36 returns after 5 or more absent shots.

Stories have more entities than a memory has slots, long absences and revisits, persistent changes (clothing, dirt,
damage, assembly), and prompts that overturn earlier inferences. Each shot has a **hidden annotation** (place,
planned camera and content, expected persistent states, and for the Hard stories the entities present). Only the
evaluation reads it. The annotations were frozen before any method was run.

## Files

| Path | Content |
|---|---|
| `<story>.json` | the 25 stories with **terse** prompts (main protocol) |
| `full/<story>.json` | the same stories with full prompts (control). Setup and annotations are identical, only the prompts differ |
| `eval/expected_visible.json` | plan visibility for *Same protagonist*: whether each shot's hidden plan shows the protagonist recognisably (349 of 857 shots). It was judged once from the plan alone (Gemini 3.8 Flash, text only) and is method independent |
| `SHA256SUMS` | checksums of all the files above (`shasum -a 256 -c SHA256SUMS`) |

Terse prompts read like real user input. The first shot is unchanged and a new place keeps its description, but a
continued or revisited place becomes one phrase ("Still on the open beach.", "Back in the kitchen."). The protagonist
and earlier entities are named without appearance or state, changes that happen in the shot stay as actions, and
camera motion is kept. Averaged over stories, a terse prompt has 93 words and a full prompt 164.

## Online protocol

1. Before shot 0 a method may read only the session setup (`setup`: title, style, cast cards).
2. Prompts are revealed one at a time. Shot *i* is generated after prompt *i* is revealed, and the method never reads
   a later prompt.
3. Each shot is generated **once**: no regeneration and no best-of-*N*. A method may keep its own closed-loop
   mechanisms (for example, reading its own past frames).
4. A method never reads `annotation` (or `bible`, `terse`) fields. They exist only for scoring.

`scripts/run_story.py` enforces this: the agent only sees prompts that `PromptStream` has revealed. See
[`docs/benchmark.md`](../../docs/benchmark.md) for how to generate all stories and score them.

## Stories

Places: scene changes between consecutive shots (new place / continued place / revisit of an earlier place).
Visible: shots whose plan shows the protagonist (the *Same protagonist* denominator). Words: mean prompt words per
shot (terse / full; the total row averages over stories). Every shot also carries its prompt in MiniMax-H3's audio-video format (`variants.h3`, written from that shot's prompt
only), used by the MiniMax-H3 generators.

| Story | Title | Protagonist | Shots | Places | New / cont. / revisit | Visible | Words |
|---|---|---|---|---|---|---|---|
| `astronaut_tomatoes` | Tomatoes in Orbit | Commander Imani | 15 | 7 | 6 / 6 / 2 | 9 | 171 / 244 |
| `chef_mei_long_day` | Chef Mei's Longest Day | Chef Mei | 16 | 6 | 5 / 9 / 1 | 13 | 147 / 237 |
| `collie_muddy_day` | Pip's Muddy Day | Pip (a border collie) | 14 | 6 | 5 / 8 / 0 | 12 | 146 / 234 |
| `detective_silver_key` | The Silver Key | Detective Sol | 16 | 10 | 9 / 6 / 0 | 7 | 165 / 236 |
| `gpu_build_pc` | Build Night | Alex | 40 | 3 | 2 / 37 / 0 | 1 | 150 / 241 |
| `gpu_from_sketch` | From a Pencil Sketch to a Frame of Game | Lin | 39 | 14 | 13 / 21 / 4 | 22 | 158 / 242 |
| `gpu_from_sketch_v2` | From a Pencil Sketch to a Frame of Game (v2) | Daniel | 40 | 13 | 12 / 24 / 3 | 2 | 173 / 240 |
| `kintsugi_robot_k7` | Unit K7 and the Golden Crack | Unit K7 (a robot) | 16 | 4 | 3 / 10 / 2 | 12 | 144 / 240 |
| `kite_runner_seasons` | The Red Kite | Leo | 15 | 5 | 4 / 10 / 0 | 10 | 152 / 233 |
| `hd00_a8b881` | Tangled Saturday at Salon Solstice | Clara Ruiz | 42 | 5 | 4 / 19 / 18 | 21 | 58 / 118 |
| `hd01_d6dd3f` | A New Coat of Paint | Maya | 38 | 5 | 4 / 29 / 4 | 8 | 56 / 114 |
| `hd02_1dd2cc` | Stitches of the High Seas | Maren Sterling | 42 | 5 | 4 / 26 / 11 | 18 | 58 / 123 |
| `hd03_528390` | Crossing the Fjord | Erik Lindqvist | 40 | 5 | 4 / 25 / 10 | 19 | 56 / 115 |
| `hd04_ff1cf1` | The Shutter Synchrony | Arthur Finch | 40 | 5 | 4 / 16 / 19 | 20 | 63 / 123 |
| `hd05_7c16e7` | A Day of Care at the Savanna Reserve | Clara | 38 | 4 | 3 / 32 / 2 | 18 | 54 / 117 |
| `hd06_84f4af` | Whiteout Extraction | Elena Rostova | 42 | 4 | 3 / 36 / 2 | 13 | 58 / 131 |
| `hd07_16f5f1` | The Joymaker Express | Elias Thorne | 42 | 5 | 4 / 35 / 2 | 13 | 58 / 127 |
| `hd08_bbc480` | Downpour at Timber Ridge | Maya Lin | 42 | 5 | 4 / 35 / 2 | 13 | 64 / 134 |
| `hd09_8184b6` | The Detailer's Touch | Milo Vance | 40 | 5 | 4 / 26 / 9 | 13 | 61 / 132 |
| `hd10_29d8a2` | Circuit and Crater | Leo Miller | 42 | 5 | 4 / 32 / 5 | 21 | 53 / 121 |
| `hd11_6d02cc` | Core Retrieval in the Whiteout | Dr. Elena Vance | 40 | 5 | 4 / 32 / 3 | 14 | 58 / 121 |
| `hd12_6a0e20` | The Dead Letter Drift | Clara Brooks | 38 | 5 | 4 / 14 / 19 | 16 | 64 / 128 |
| `hd13_bba9f7` | Blood on the Brocade | Clara Vance | 40 | 5 | 4 / 29 / 6 | 17 | 52 / 107 |
| `hd14_f8b61a` | The Master Groom's Trial | Silas Vance | 40 | 5 | 4 / 27 / 8 | 20 | 56 / 118 |
| `hd15_464363` | The Master of Strings | Clara | 40 | 5 | 4 / 27 / 8 | 17 | 58 / 122 |
| **Total** | | | **857** | | **121 / 571 / 140** | **349** | 93 / 164 |

## Story format

Each file is one JSON object, loaded by `agentic_longlive.story.Story.load`:

```text
id            story id ("<name>_terse"; the full-prompt files use the plain name, Hard stories "<name>_full")
setup         known before shot 0
  title, style
  cast        [{id, name, description}]; cast[0] is the protagonist. The description is the identity card the
              judges compare against
  extras      optional session-level direction for audio generators (voice, music, speech, framing)
bible         Hard stories only, hidden: entities {id: {kind (character|object), name, identity, initial
              attributes, protagonist}} and locations {id: {name, look}}
shots[]
  prompt      the text revealed for this shot
  annotation  hidden, evaluation only (see below)
  variants    optional per-shot prompts for other generators: h3 (MiniMax-H3 sectioned prompt), h3mem (the same
              with the entity descriptions written into the text)
  terse       hand-written stories only: metadata of the terse rewrite (location_status new|same|revisit, words
              [full, terse])
```

The hidden `annotation` of a shot has these fields:

| Field | Stories | Meaning |
|---|---|---|
| `scene` | all | planned place (key). The location judge chooses among these keys |
| `camera` | all | planned camera (shot size, angle, motion) |
| `beat` | all | planned content: subject and action |
| `state` | all | expected persistent facts visible in this shot (`key: value`, e.g. `coat_condition: soaking wet coat`, or `clara.outfit: ...` on Hard stories). The shot judge checks each key |
| `title_en` | all | short shot title |
| `present` | Hard | entities in the shot |
| `first_appearance` | Hard | entities first seen in this shot |
| `changes` | Hard | persistent changes made in this shot `[{entity, attr, value}]` |
| `start_state` / `entity_state` | Hard | attributes of the present entities before the shot / of all entities after its changes |
| `location_visit` | Hard | `new`, `same` or `revisit` |
| `momentary` | Hard | a transient action that must not persist |

The judges use `scene`, `camera`, `beat`, `state` and the protagonist card. The plan visibility also uses `present`.
