# Reproduced recipes

The four YAML files below are the training configurations for the corresponding experiments.

| Recipe | Training video size (W×H×frames) | Teacher guidance | Generator LR | Critic LR | Weight decay | Seed |
| --- | --- | --- | --- | --- | --- | --- |
| wan21_cfg | 832×480×81 | conventional CFG=5 | 1e-5 | — | 0.01 | 0, fixed |
| wan22_cfg | 640×352×125 | conventional CFG=5 | 1e-5 | — | 0 | 20260724 |
| wan21_dmd | 832×480×81 | `real_guidance_scale=4`, equivalent to CFG=5 | 1e-5 | 2e-6 | 0.01 | 0, fixed |
| wan22_dmd | 1280×704×125 | `real_guidance_scale=3`, equivalent to CFG=4 | 1e-5 | 2e-6 | 0 | 2 |

The DMD teacher uses `conditional + real_guidance_scale × (conditional − unconditional)`. The inference CLI uses conventional CFG, so these numeric settings differ by one. All distilled adapters are sampled with CFG=1. Training shapes in YAML are **latent** `[batch, time, channels, height, width]`, not pixel dimensions.

## Launch training

The README uses single-machine examples. Those examples require the stated GPU count on that machine. To use four machines with four GPUs each, run the following command on each machine, setting `NODE_RANK` to a different value from 0 to 3 and `MASTER_ADDR` to the reachable IP address of machine 0:

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" --master_port=29500 train.py \
  --config_path configs/wan22_dmd.yaml \
  --logdir runs/wan22_dmd/train --disable-wandb --no_visualize
```

Use the same code, environment and configuration on every machine, with shared paths for assets, caches and the training log directory. The same launch options apply to teacher-cache generation. The Wan2.1 DMD recipe uses 32 GPUs; with four GPUs per machine, set `--nnodes=8` and use ranks 0 through 7.

## CFG-only

The teacher integrates a native 50-step UniPC trajectory with CFG=5. The cache stores noisy states, guided flow targets, and per-frame guidance energy for 64 fixed prompts. Low-noise indices 35–49 repeat twice in the sampling manifest; tensors are stored once. Relative-guidance loss weighting uses epsilon 1e-6 and maximum weight 16. No critic is allocated.

`cache_teacher.py` uses cache batch size one without changing the training recipe's batch size. World size must divide 64; the reproduction used 16 GPUs, four prompts per rank and noise seed 20260927. A completed manifest is written only after every shard passes validation. Existing shards/manifests are never silently overwritten. Cache generation is a separate step and does not support mid-trajectory resume; use a fresh cache directory after investigating an interrupted cache run.

Training verifies cache shard checksums across ranks before allocating the model. Each rank then validates its view of the same manifest. CFG-only training uses a cached target and a generator LoRA; the 50-step sampling schedule remains intact.

## Few-Step

These experiments use CFG during few-step training, which gives better results than training without CFG. Guidance-scale adjustment is handled by the separate CFG LoRA branch.

DMD performs full-sequence four-step backward simulation with a randomly selected gradient-carrying denoising step. The teacher backbone remains frozen; the generator and critic each have a rank-128 adapter. Critic updates must not backpropagate into generator adapters. Existing no-grad and optimizer-resume regressions cover this behavior.

Wan2.1 uses the integrated `sfp_training` update-order path with a **non-AR** generator; the flag does not enable an AR model. Wan2.2 uses the standard DMD update loop. EMA was disabled by the LoRA training path in all four runs; its inactive configuration fields have been removed.

The production 14B DMD run included an early, explicit 16→32 GPU continuation. The released recipe starts at the final 32-GPU/global-batch-64 setting. It preserves the final method and parameters; a fresh run is not a bitwise replay of that topology transition. Exact automatic resume requires the same world size, data and behavior-affecting configuration.

## Inference options

Inference requires a GPU. `--checkpoint` accepts the exported safetensors adapter or a training `model.pt` containing `generator_lora`; it does not accept a merged backbone. Omitting it runs the base model. The default resolution is 832×480 for Wan2.1 and 1280×704 for Wan2.2. Defaults are 81 frames at 16 fps for Wan2.1; 121 frames for Wan2.2 CFG and 125 frames for Wan2.2 DMD, at 24 fps.

For matched comparisons, use the same prompts and seed for two adapters:

```bash
torchrun --standalone --nproc_per_node=4 inference.py \
  --config configs/wan22_cfg.yaml \
  --checkpoint adapters/wan22_cfg/adapter_model.safetensors \
  --prompt-file data/heldout.txt --count 16 --seed 20261000 \
  --output outputs/wan22_cfg
```

Each case has an MP4 and JSON containing its prompt, seed, sampling settings and initial-noise SHA-256. `--save-latents` additionally saves the final latent. `--width`, `--height`, `--frames`, `--steps` and `--guidance` override inference settings; published comparisons must state these overrides.

See [release validation](validation.md) for implementation checks and their scope.
