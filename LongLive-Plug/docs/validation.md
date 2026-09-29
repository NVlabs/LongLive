# Release validation

Native-resolution comparisons on 16 prompts per recipe found broadly comparable visual quality between retained intermediate checkpoints (CFG: 300; Wan2.1 Few-Step: 706; Wan2.2 Few-Step: 1,150) and final checkpoints (500, 1,000 and 1,500, respectively). Prompts, initial noise and UniPC sampling settings were matched; review used chronological contact sheets, with denser temporal sampling for flagged cases. Subject duplication and prompt-adherence failures remain in some intermediate and final samples. The default budgets of 250 / 600 / 1,000 iterations were not directly tested at those exact checkpoints.

The four-recipe cleanup was checked against the preceding release (`7553e15`), using PyTorch 2.9.1+cu128, BF16 transformer weights, native fused SDPA and FP32 VAE decoding.

## CPU and interface checks

- 83 regression tests passed, covering CFG cache validation, non-AR rollout, critic/generator gradient separation, PEFT loading, optimizer restoration, RNG/data-cursor restoration, complete checkpoint selection and rejection of unsupported modes.
- All five training, inference, cache-generation and asset-download CLI help commands imported successfully. All 34 production Python modules are reachable from these entry points.
- Prompt preparation preserves the pinned source, deterministic split and cache provenance contract.
- Adapter settings and checked training parameters match the preceding release except for the subsequently shortened `max_iters` limits; see [`config_equivalence.json`](config_equivalence.json).
- Separate before/after runs of the actual loss and rollout code with small synthetic backbones produced bitwise-equal losses, gradients and post-step RNG state for all four recipes. This checks objective logic, not full-size model gradient equivalence.

## Real-checkpoint GPU comparison

Both implementations loaded the same production adapter for each recipe, then used identical deterministic noise and synthetic UMT5-shaped conditions with a small latent of `[1, 3, C, 8, 12]`. They ran the full recipe sampling count (50 CFG / 4 DMD steps), followed by VAE decoding.

| Compared output | Result, all four checkpoints |
| --- | --- |
| Transformer flow prediction | Bitwise equal |
| Predicted clean latent | Bitwise equal |
| Final sampled latent | Bitwise equal |
| Decoded floating-point pixels | Bitwise equal |
| Gradient statistics | Finite; maximum relative L2 difference 0.1163% |

Gradient statistics contain the mean, mean square and maximum magnitude of every trainable adapter tensor. They were not bitwise equal. Raw comparisons are in [`gpu_parity.json`](gpu_parity.json). These small-latent checks exercise real weights and sampling code; they do not replace native-resolution visual assessment or establish bitwise training equivalence.

## Distributed training and continuation

All four recipes completed four-GPU initialization, one optimizer step, full checkpoint writing, and restoration into a second step. Both DMD optimizers were saved and restored. See [`training_resume.json`](training_resume.json).

CFG checks used production cached latent dimensions with microbatch one. DMD checks used smaller latents (one temporal block and 4×4 spatial dimensions) with real backbone weights. Exact continuation requires the same code version, world size and training configuration.
