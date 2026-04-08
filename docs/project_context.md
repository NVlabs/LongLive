# Project Context: Reference-Subject Controlled Autoregressive Video Generation

## 1. Background

This project is based on [LongLive](https://github.com/YIYANGCAI/LongLive-dev) (ICLR 2026), a frame-level autoregressive (AR) framework for real-time interactive long video generation. The base model is a 1.3B parameter student model distilled from a bidirectional video generation teacher (DMD). It supports:

- Autoregressive video generation (up to 240s, 20.7 FPS on H100)
- Interactive prompt switching during generation via KV-recache mechanism
- Short window attention + frame sink for long-range consistency

Core technical components:
- **KV-Recache**: refreshes cached KV states at prompt switch boundaries for smooth transitions
- **Streaming training**: aligns training with inference by training on sequential chunks
- **Frame sink**: maintains long-range temporal consistency with minimal compute

## 2. Codebase Structure

```
LongLive-dev/
├── trainer/
│   └── distillation.py          # Trainer class, training loop entry
├── model/
│   └── streaming_training.py    # StreamingTrainingModel, chunk generation logic
├── pipeline/
│   ├── streaming_training.py    # StreamingTrainingPipeline, base denoising pipeline
│   └── streaming_switch_training.py  # StreamingSwitchTrainingPipeline, prompt switch denoising
├── configs/
│   ├── longlive_train_init.yaml
│   └── longlive_train_long.yaml
├── dataset_meta/
│   └── meta.jsonl               # Dataset metadata (reference_image, video_clip, prompt)
├── feature_extraction.py        # Offline VAE feature extraction (video + ref images)
├── feature_extraction.sh        # Multi-GPU launch script for feature extraction
├── train.py                     # Training entry point
├── train_init.sh / train_long.sh
├── inference.py / inference.sh
├── interactive_inference.py / interactive_inference.sh
├── model/                       # DMD, DMDSwitch model definitions
├── utils/                       # Dataset, distributed, memory utilities
└── wan/                         # WAN model (VAE, configs)
```

## 3. Training Call Chain

```
distillation.py: Trainer.fwdbwd_one_step_streaming()
    # Single training step. Manages sequence lifecycle, calls generate_next_chunk.
    ↓
model/streaming_training.py: StreamingTrainingModel.generate_next_chunk()
    # Generates a chunk of video frames with overlap handling and random chunk sizing.
    ↓
model/streaming_training.py: StreamingTrainingModel._generate_chunk()
    # Core generation: resolves current conditional_dict, determines if prompt switch
    # occurs, and dispatches to the appropriate pipeline.
    ↓
pipeline/streaming_switch_training.py: StreamingSwitchTrainingPipeline.generate_chunk_with_cache()
    # The actual denoising process. Handles prompt switching via KV-recache within a chunk.
    # Falls back to parent (no-switch) pipeline when no switch is needed.
```

Key data flow:
- `conditional_dict` carries prompt embeddings and conditioning info
- KV cache is maintained across chunks for temporal continuity
- Prompt switch is determined by `switch_frame_index` within `_generate_chunk`

## 4. Development Goal

Build a **dynamic reference insert** framework on top of the existing DMD distillation pipeline — enabling reference-based subject control that is additive (not replacive), trained in two stages:

### What already exists
- **Feature extraction pipeline** (`feature_extraction.py` + `feature_extraction.sh`): encodes video clips and reference images into VAE latents using WAN's 3D-VAE
  - Video → `latents`: `[1, T, 16, h, w]`
  - Reference images → `latents_ref`: `[1, num_refs, 16, h, w]`
  - Output: `.pt` files + `data.jsonl` index

- **Dataset format** (`meta.jsonl`):
  ```json
  {
    "reference_image": ["path/to/ref1.png", "path/to/ref2.png"],
    "video_clip": "path/to/video.mp4",
    "prompt": "two girls are sitting in a classroom."
  }
  ```

### What needs to be built

The goal is to build a **dynamic reference insert** framework — analogous to prompt switching, but with a key difference: reference control is **additive**, not replacive. When a new reference subject is introduced, previous reference information is preserved and interacts with the new one, rather than being overwritten.

This will be achieved in two training stages:

**Stage 1: Single-reference video generation (SFT)**
- Train the model to generate video conditioned on a single reference image
- Introduce the **ID Memory Bank** as a persistent memory of the single subject's identity, ensuring ID fidelity does not degrade over long-duration autoregressive generation
- Data: single-reference video pairs (one reference image + video clip)
- Integration: add a supervised training path alongside the existing DMD distillation, loading pre-extracted `.pt` features (video latents + reference latents + prompt)

**Stage 2: Multi-reference dynamic insert**
- Extend training with multi-reference video data where multiple subjects appear
- The ID Memory Bank now manages multiple reference identities simultaneously, supporting dynamic insertion of new references at arbitrary time points during generation
- New references are accumulated (not replaced) — the bank maintains all active identities and enables proper interaction between them via attention

### Core mechanism: ID Memory Bank

The ID Memory Bank serves as a dynamic, accumulative store of reference identity information injected into the video generation process.

**How it works:**
- Reference image → 3D-VAE → latent tokens
- Latent tokens are stored in the ID Memory Bank
- Bank tokens are concatenated with video tokens and participate in self-attention / cross-attention, injecting identity information into the generated frames

**Stage 1 role:** The bank holds a single identity. Its purpose is to act as persistent ID memory — as autoregressive generation progresses over many frames, the bank prevents identity drift by continuously providing the original reference signal.

**Stage 2 role:** The bank accumulates multiple identities over time. When a new reference is dynamically inserted (similar to a prompt switch), its latent tokens are added to the bank without removing existing ones. All stored identities interact through attention, enabling the model to generate video with multiple consistent subjects.
