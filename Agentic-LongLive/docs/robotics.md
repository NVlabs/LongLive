# Robot videos: long-horizon tasks with an observing memory agent

Generated robot data needs long videos in which the robot works through many steps while parts of the scene leave the
view and come back. This application renders such a video from **one first-frame image** of a robot workspace and a
**script of eight instructions** (8 s each, 64 s of continuous video). A video model renders each 8 s segment starting
from the previous segment's last frame (a keyframe at frame 0, no cut). Without memory, that frame is all the generator
knows about the scene, so a drawer reopened after half a minute shows different contents, or the camera turns back to a
different fan.

The memory agent fixes this without ever making the generator wait. Unlike the open-loop agent of the main method, it
**observes the video**: in one continuous scene, where an object ends up or whether a drawer is still open depends on
what was generated, not on the script. Its memory pictures are **screenshots of the stream** (frames or crops), so no
image model is needed. This is the paper's appendix section "Robot Videos" (long-horizon robot tasks).

Code: `agentic_longlive/robotics/`, entry point `scripts/run_robot.py`, scripts in `stories/robot/`.

## What you need

| Component | Paper setup | Notes |
|---|---|---|
| Episode script | `stories/robot/*.json` | 30 benchmark episodes + 2 project-page demos ([stories/robot/README.md](../stories/robot/README.md)) |
| First frame | RoVid-X validation clips, frame 0 | not distributed; see [stories/robot/README.md](../stories/robot/README.md) for clip ids and file names |
| Agent VLM | Qwen3.6-35B-A3B (FP8) on SGLang, thinking off | any OpenAI-compatible chat endpoint with image input |
| Video generator | FastH3-Omni (8 steps) | runs out of process as a file-queue worker |
| Tools | `ffmpeg` on `PATH` | frame extraction and joining |
| hybrid2 only | `torch`, `transformers` | Grounding DINO (`IDEA-Research/grounding-dino-base`, downloaded on first use) |

**Agent server.** Serve the VLM with an OpenAI-compatible API, for example with SGLang:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B-FP8 --port 31001   # add --tp N for your GPUs
```

The client turns Qwen3 thinking off by sending `chat_template_kwargs: {enable_thinking: false}`. For other endpoints
set `--agent-model`, `--agent-extra-body '{}'` and, if needed, `--agent-api-key-env NAME` (the name of an environment
variable that holds the key).

**Generator worker.** `scripts/run_robot.py` never talks to a video server. It writes one job per segment into a queue
directory (`agentic_longlive/workers/protocol.py`) and waits for the worker's MP4. Start the FastH3-Omni worker
(`agentic_longlive/workers/fasth3.py`) on a queue directory first. Each job carries:

| Field | Value |
|---|---|
| `prompt` | the final generator prompt (below) |
| `refs` | the pinned memory pictures, then the latest frame (absolute paths) |
| `ref_seconds` | `0.25 * k` for picture k (the sink layout) |
| `video_start_seconds` | `2.0` (the target video starts 2 s after the pictures, after a cut) |
| `keyframe` | the latest frame; the input image for segment 0 |
| `ref_scale` | `0.5` (pictures are cover-resized to half the 1344×768 canvas) |
| `duration_seconds` | `8` |
| `seed` | `1000 + i` for segment i (both arms) |

Any worker that implements these fields and returns a 24 fps MP4 can be used.

## Run

```bash
python scripts/run_robot.py \
    --episode stories/robot/05_open_drawer_ego.json \
    --image-root data/robot_first_frames \
    --queue /path/to/fasth3_queue \
    --agent-url http://127.0.0.1:31001/v1 \
    --arm both --policy v4 \
    --out outputs/robot/05
```

- `--arm base` renders the generator alone, `--arm ours` the generator with the memory agent, `--arm both` (default)
  both with the same seeds, sharing segment 0 (nothing to remember yet).
- `--policy v4` (default) is the paper's agent, `--policy hybrid2` the variant behind two of the project-page demos.
- `--image frame.png` instead of `--image-root DIR` gives the first frame directly.
- `--seed 2000` reproduces the paper's second-seed configuration (segments 1-7 use 2000 + i, segment 0 keeps 1000).

Outputs in `--out`:

```text
base.mp4, ours.mp4                 joined 64 s videos (the keyframe at the start of each later segment is dropped)
<arm>/seg<k>.mp4                   the segments as rendered
<arm>/seg<k>/last.png              hand-off frames (keyframe of the next segment)
<arm>/episode.json                 per segment: prompt, pictures, keyframe, seed, timing; ours also the plans
                                   and the agent's windows (what it observed and planned while each segment rendered)
ours/agent.json                    the agent trace (every observer / router / planner reply) and its final memory
ours/agent/mem/                    the memory pictures (screenshots and crops)
```

Tests (CPU, fake VLM, fake worker, ffmpeg mocked): `python -m pytest tests/test_robotics.py -q`.

## How it works

**Generator prompt.** The pinned pictures and then the latest frame enter the generator's attention sink. Without
pinned pictures the prompt is

```text
Picture 1 is the latest frame of this continuous robot video, shown before the target video; the target video
continues from it without a cut.

integrated_multimodal_description: [Shot 1] <camera / robot / setting sentence> <instruction>

overall_soundscape: Quiet room tone with faint robot motor whirs.
```

With pinned pictures, a fixed preamble says that Pictures 1..k-1 were recorded earlier in this same video, names what
each one shows (its caption), states that Picture k is the latest frame from which the target continues without a cut,
and that places and objects look exactly as in the earlier pictures when they come back.

**Agent** (`agentic_longlive/robotics/agent.py`, prompts in `prompts.py`).

- *Observer.* After each segment it looks at three frames (40, 110, 191) and the instruction that was rendered. It
  describes the view, lists the tracked objects with their position and look, records whether each container is open
  or closed and what it holds, and picks a frame to keep, a frame that shows the inside of an open container, and boxes
  for up to two objects. Code turns this into a *ledger* (an object last seen in a container that is now closed stays
  "inside it (hidden)") and a *bank* of pictures: one view per segment, container interiors and the newest crop of
  each object, each with a caption and its segment. The input image is observed before the video starts.
- *Router.* Before each segment it answers one narrow question about the next instruction: does something come back
  into view (the camera turns back, a container is reopened, a cover is lifted, an object is taken out)? On the 30
  benchmark scripts it agrees with the annotations in 188 of 210 segments.
- *Planner.* Only when something comes back, it sees the bank as thumbnails and lists every picture of the returning
  place, container or object. Code pins the newest matching view and the newest close view (at most two), and the
  instruction is rewritten to name what comes back and how it looked when last seen. Otherwise nothing is pinned and the
  instruction is only rewritten to name the objects with their look. Pinning only on returns matters: pictures pinned
  in almost every segment make the generator restore old states (a tube put into a drawer reappears on the cabinet).

**Real-time schedule.** The agent never blocks the generator. While segment i-1 is generated, the agent observes
segment i-2 and plans segment i (lag 1); the input image is observed while segment 0 is generated. `episode.json`
records each agent window and how long, if at all, the generator waited for it.

**Policies.**

| | `v4` (paper) | `hybrid2` |
|---|---|---|
| object crops | the observer's boxes | Grounding DINO boxes for the object's name, robot boxes excluded |
| extra memory | none | a snapshot per *hide event* (a container closed, a cover put on, the camera turning away): the last of 12 frames in which the hidden thing is fully visible, never overwritten; container snapshots also get a zoomed crop |
| container / cover reopened | newest matching view + close view, rewrite | the zoomed pre-hide snapshot, strict rewrite (closed containers stay closed, hidden contents are not mentioned) |
| camera returns to an area | newest matching view + close view, rewrite | the area's pre-hide snapshot + the newest matching object crop; the user's instruction verbatim + "<area> comes back into view exactly as it looked before the camera turned away" |
| nothing comes back | no picture, rewrite | no picture, rewrite |

A second router (`ROUTE6`) decides which hide event comes back and how ("camera" can only bring back an area, "open"
only a container's contents). On the 30 benchmark episodes the two policies were statistically tied (two seeds on
FastH3-Omni, one on H3 Ref2VA); the paper reports `v4`.

## Paper results (for context)

30 episodes, whole 64 s videos, blind Gemini 3.8 Flash judgments at the 41 return moments (*Score*: consistency with
the last appearance, 0-10; *Correct*: back in view with identity and position both at least 7), instruction following
over 210 segments (*Progress*) and robot identity drift (*Drift*: segment ends with a robot identity score of at most 5).

| Generator | Memory | Score | Δ [95% CI] | Visible (%) | Correct (%) | Progress | Drift (%) |
|---|---|---|---|---|---|---|---|
| FastH3-Omni (real time) | none | 3.98 | – | 43.9 | 19.5 | 5.67 | 30.5 |
| | ours (v4) | 5.59 | +1.61 [0.89, 2.26] | 75.6 | 56.1 | 5.84 | 21.9 |
| | ours (v4), seed 2 | 5.63 | +1.66 [0.84, 2.39] | 80.5 | 48.8 | 5.88 | 27.1 |
| H3 Ref2VA (offline) | none | 3.54 | – | 29.3 | 4.9 | 5.62 | 39.9 |
| | ours (v4) | 5.51 | +1.98 [1.40, 2.51] | 65.9 | 43.9 | 5.42 | 28.4 |

In real time, the hidden thing comes back with the same identity and position at 56% instead of 20% of the return
moments. Camera returns gain the most (+2.55 and +2.35), reopened containers less (+0.71, not significant, and +1.62):
in several reopenings the generator never opens the lid or takes the object out, which memory cannot fix. Task progress stays
level. Timed on six episodes with FastH3-Omni on eight B200 GPUs and the agent on two A100 GPUs, the agent took 2.3 s
on average and never made the generator wait, and each 8 s segment took 7.6 s with the last decoded frame passed
straight to the next request.

## Notes

- **Hand-off frame.** The next segment starts from `video.last_frame`, which seeks 0.1 s before the end of the MP4 and
  keeps the last decoded frame. For a 192-frame segment this is frame 190 (1/24 s before the end), as in the paper's
  runs.
- **Latency.** This runner hands each segment over through its MP4 file (ffmpeg extracts the frame). The paper's
  real-time timing passed the decoded frame directly; with the MP4 hand-off a segment took about 0.5 s longer on the
  same B200 setup, so real time is then borderline.
- **Offline system.** The paper's offline arm (H3 Ref2VA, 50 steps) used the same agent with a Gemini model, observed
  segment i-1 before rendering segment i (lag 0) and cropped with Grounding DINO. This runner implements the real-time
  schedule; `--grounding-crops` switches v4 to Grounding DINO crops.
- **Reproducibility.** The agent runs at temperature 0, but neither the VLM server nor the generator is bit-exact
  across hardware and versions, so runs give videos of the same kind rather than the paper's exact videos.
