# LongLive2.0 Usage

This document contains the release commands for installation, training, inference, and utilities. The root README keeps the project overview and paper figures.

## Installation

Create a Python 3.10 environment and install the required packages:

```bash
conda create -n longlive2 python=3.10 -y
conda activate longlive2
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

TensorRT is not required for the default training or inference path. If a
TensorRT utility is needed, install it separately after the base requirements:

```bash
pip install nvidia-pyindex
pip install nvidia-tensorrt
pip install pycuda
```

Download the Wan2.2-TI2V-5B components and replace the `/path/to/longlive2.0/...` placeholders in the config files before running training or inference.

If you set `inference.vae_type` to `mg_lightvae` or `mg_lightvae_v2`, download
the corresponding VAE checkpoints from the Hugging Face repository
`Skywork/Matrix-Game-3.0` and place them under `wan_models/Matrix-Game-3.0/`:

```text
wan_models/Matrix-Game-3.0/MG-LightVAE.pth
wan_models/Matrix-Game-3.0/MG-LightVAE_v2.pth
```

### NVFP4 Environment

The default installation above is the clean BF16 release setup. NVFP4 training
and inference use local CUDA extensions and are more version-sensitive, so keep
them in a separate environment.

Known-good NVFP4 baseline inherited from the Sage branch:

```text
Python:          3.12.12
PyTorch:         2.10.0+cu128
TorchVision:     0.25.0+cu128
CUDA target:     12.8
FlashAttention:  2.8.3, built from source
```

Create or activate the NVFP4 environment:

```bash
conda create -n longlive2_nvfp4 python=3.12 -y
conda activate longlive2_nvfp4

conda install -c nvidia cuda-toolkit=12.8 -y

pip install -r requirements.txt

pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0
pip install --upgrade torchao==0.16.0
```

If you already have a working `qlive` environment from LongLive_Sage, you can
activate it instead of creating `longlive2_nvfp4`.

Verify the Torch/CUDA pair:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda); print(torchvision.__version__)"
```

Build the modified local `fouroversix` package:

```bash
cd fouroversix
pip install ninja packaging psutil "setuptools>=77.0.3"

# Optional: limit compile targets.
export CUDA_ARCHS=100   # B200 / GB200 / GB300
# export CUDA_ARCHS=120 # RTX 50/60 series, if needed

pip install --no-build-isolation -e .
cd ..
```

Build FlashAttention from source, rather than relying on a prebuilt wheel:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3
pip install -U pip setuptools wheel ninja packaging
pip install --no-build-isolation -e .
cd ..
```

Install TransformerEngine if `model_quant_use_transformer_engine: true` will be
used:

```bash
python -m pip install --no-build-isolation "transformer-engine[pytorch]"
```

Build the fused LongLive FP4 KV-cache dequant extension:

```bash
cd utils/kernel
python setup.py build_ext --inplace
cd ../..
```

Quick NVFP4 checks:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import fouroversix; from utils.quant import LongLiveQuantizationConfig, quantize_to_fp4"
python -c "from utils.kernel.kv_dequant import dequantize_kv_cache_fp4"
```

The release NVFP4 configs and direct run commands are summarized below. See
`README_NVFP4.md` for lower-level implementation notes.

## Configs

The release keeps three main configs:

```text
configs/train_ar.yaml       # AR diffusion training
configs/train_dmd.yaml      # DMD distillation
configs/inference.yaml      # inference
```

TorchAO FP8 PTQ inference has a separate config:

```text
configs/fp8/inference_fp8.yaml
```

The NVFP4 path keeps its configs separate from the default BF16 release path:

```text
configs/nvfp4/train_ar_nvfp4.yaml          # stage 1 AR teacher-forcing training
configs/nvfp4/train_dmd_nvfp4_step4.yaml   # stage 2 DMD LoRA distillation, 4-step rollout
configs/nvfp4/inference_nvfp4.yaml         # NVFP4 inference with optional KV quantization
```

The configs use a shared organization:

- `model_kwargs`: arguments passed to `WanDiffusionWrapper`.
- `infra`: distributed training/runtime settings.
- `algorithm`: AR or DMD objective settings.
- `training`: optimizer, batch size, checkpoint cadence, and loop settings.
- `data`: training or prompt data paths.
- `inference`: sampling and cache settings.
- `checkpoints`: model and LoRA checkpoint paths.
- `adapter`: optional LoRA settings. Remove this section to disable LoRA.

## Training

### AR Diffusion Training

Edit `configs/train_ar.yaml` to set the dataset path, evaluation prompt path, logging path, and distributed runtime settings. Then run:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_ar.yaml \
  --logdir logs/test_train_ar \
  --wandb-save-dir wandb \
  --disable-wandb
```

Notes:

- `infra.sequence_parallel_size` controls the SP group size.
- `infra.vae_halo_latents` controls chunk-halo VAE preparation.
- `model_kwargs.local_attn_size` is a model construction setting.
- `inference.sink_size`, `inference.multi_shot_sink`, and `inference.multi_shot_rope_offset` control evaluation-time generation during training.

### DMD Distillation

Edit `configs/train_dmd.yaml` to set the dataset path and initialization checkpoints. Then run:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_dmd.yaml \
  --logdir logs/test_train_dmd \
  --wandb-save-dir wandb \
  --disable-wandb
```

Notes:

- `algorithm.real_guidance_scale` and `algorithm.fake_guidance_scale` are used by score distillation.
- `inference.sampling_steps` controls the distillation rollout sampling steps.
- If `adapter` is present, LoRA distillation is enabled. Otherwise the generator is fully fine-tuned.
- Auto-resume is enabled by default unless `--no-auto-resume` is passed.

### NVFP4 Training

Use the `longlive2_nvfp4` environment and build the NVFP4 extensions before
running these commands. Replace the `/path/to/...` placeholders in the configs
first.

Stage 1 trains the NVFP4 AR teacher-forcing model:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --config_path configs/nvfp4/train_ar_nvfp4.yaml \
  --logdir logs/nvfp4_ar \
  --wandb-save-dir wandb \
  --disable-wandb
```

Stage 2 runs NVFP4 DMD LoRA distillation from the AR checkpoint:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --config_path configs/nvfp4/train_dmd_nvfp4_step4.yaml \
  --logdir logs/nvfp4_dmd_step4 \
  --wandb-save-dir wandb \
  --disable-wandb
```

Notes:

- `--nproc_per_node` controls the per-node GPU count. The NVFP4 examples use 4
GPUs; set it to 8 or another value for your machine.
- `infra.model_quant` enables NVFP4 generator training for stage 1.
- `infra.generator_quant`, `infra.real_score_quant`, and
`infra.fake_score_quant` choose which DMD networks use NVFP4 in stage 2.

After stage 1 and stage 2 are complete, you can pre-merge the AR generator and
DMD LoRA weights for inference. The export script reads `generator_ckpt`,
`lora_ckpt`, `adapter`, and `model_quant_*` from the NVFP4 inference config.

To save a compact FourOverSix materialized NVFP4 generator checkpoint:

```bash
python scripts/save_merged_nvfp4_generator.py \
  --config_path configs/nvfp4/inference_nvfp4.yaml \
  --output_path /path/to/model_4o6.pt \
  --backend fouroversix \
  --device cuda:0
```

To save merged BF16 weights for TransformerEngine runtime quantization:

```bash
python scripts/save_merged_nvfp4_generator.py \
  --config_path configs/nvfp4/inference_nvfp4.yaml \
  --output_path /path/to/model_te.pt \
  --backend transformer_engine \
  --device cuda:0
```

The `fouroversix` export is the small packed/materialized NVFP4 checkpoint. The
`transformer_engine` export intentionally saves merged BF16 weights, because a
TransformerEngine module `state_dict` is not a compact packed NVFP4 storage
format; TE quantization is applied again when inference loads the BF16 weights.

### Merge Generator and LoRA Weights

For the regular BF16 release path, you can pre-merge the AR generator checkpoint
and DMD LoRA checkpoint into one reusable generator checkpoint. This keeps
quick-start inference simple: inference only loads `checkpoints.generator_ckpt`
and does not need to construct or load LoRA adapters at runtime.

```bash
python scripts/merge_lora_generator.py \
  --config_path configs/inference.yaml \
  --output_path /path/to/longlive2_merged_generator.pt \
  --device cuda:0
```

After the merge, set `checkpoints.generator_ckpt` in `configs/inference.yaml` to
the merged checkpoint. If you run the full `inference.py` entry point, remove or
unset `checkpoints.lora_ckpt` and the `adapter` section so LoRA is not applied a
second time.

## Inference

Edit `configs/inference.yaml` to set:

- `data.data_path`: prompt folder.
- `checkpoints.generator_ckpt`: merged generator checkpoint.
- `output_folder`: output video directory.
- `num_samples`: number of sampled videos per prompt.

Run:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 inference.py \
  --config_path configs/inference.yaml
```

Inference notes:

- `inference.sampling_steps` controls the number of denoising steps.
- `inference.guidance_scale` controls inference CFG.
- `inference.sink_size` controls the standard attention sink size.
- `inference.multi_shot_sink` enables the multi-shot attention sink.
- `inference.multi_shot_rope_offset` controls the multi-shot RoPE offset.

### FP8 PTQ Inference

Set `checkpoints.generator_ckpt` in `configs/fp8/inference_fp8.yaml` to the
downloaded merged BF16 `model_bf16.pt`, then run:

```bash
python inference.py --config_path configs/fp8/inference_fp8.yaml
```

`fp8_quant: true` applies TorchAO row-wise dynamic W8A8 quantization after the
generator has been loaded and converted to BF16, and before `torch.compile`.
It cannot be combined with `model_quant: true`, which selects the NVFP4 path.
With the provided 5B model, 300 eligible core Linear layers use FP8 while six
small conditioning/output projections remain BF16 for stability and to avoid
FP8 overhead.

The validated stack is Python 3.10, PyTorch 2.8.0+cu128, and TorchAO 0.13.0 on
H100 (SM90); compute capability 8.9 or newer is required. The supplied config
uses `torch_compile: auto`: it skips compilation when `inference_iter`
explicitly limits the run to fewer than three samples, and enables it when all
prompts are requested. Its `max-autotune` warm-up can take several minutes
while guard/shape variants are compiled. Use repeated inference and discard all
compile/warm-up samples when measuring steady-state performance; set
`torch_compile: false` for a short eager-mode smoke test.

The supplied config uses the single 8-latent-frame block validated on H100.
Longer generation introduces additional KV-cache shapes and may trigger more
compilation or eager fallback; validate the intended frame count before
benchmarking or deployment.

The initial FP8 path targets `inference.py`; `inference_sp.py` rejects the flag
until TorchAO tensor-subclass behavior is validated with Ulysses collectives.

### NVFP4 Inference

Edit `configs/nvfp4/inference_nvfp4.yaml` to set:

- `data.data_path`: prompt folder.
- `checkpoints.generator_ckpt`: AR or base generator checkpoint.
- `checkpoints.lora_ckpt`: optional DMD LoRA checkpoint.
- `output_folder`: output video directory.
- `num_samples`: number of sampled videos per prompt.

Run:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 inference.py \
  --config_path configs/nvfp4/inference_nvfp4.yaml
```

For single-GPU inference, use `python` directly:

```bash
python inference.py --config_path configs/nvfp4/inference_nvfp4.yaml
```

There are two recommended checkpoint styles for NVFP4 inference:

FourOverSix compact/materialized NVFP4 checkpoint:

```yaml
checkpoints:
  generator_ckpt: /path/to/model_4o6.pt

merge_lora: false
model_quant: true
model_quant_use_transformer_engine: false
```

TransformerEngine runtime quantization from merged BF16 weights:

```yaml
checkpoints:
  generator_ckpt: /path/to/model_te.pt

merge_lora: false
model_quant: true
model_quant_use_transformer_engine: true
```

Do not set `model_quant_use_transformer_engine: true` when loading a FourOverSix
materialized checkpoint. FourOverSix checkpoints store `quantized_weight_*`
buffers and can only be loaded by the FourOverSix path. TransformerEngine
inference should load merged BF16 weights and quantize them at runtime.

NVFP4 inference notes:

- `model_quant` enables generator NVFP4 inference. For regular BF16
checkpoints, it quantizes/materializes weights during startup; for pre-saved
FourOverSix checkpoints, the checkpoint already contains materialized weights.
- `merge_lora` merges the LoRA checkpoint into the base generator before
quantized materialization. Set it to `false` when `generator_ckpt` already
points to a merged export from `scripts/save_merged_nvfp4_generator.py`.
- `inference.kv_quant` enables FP4 KV-cache storage; the fused dequant extension
from `utils/kernel` must be built first.
- `inference.streaming_vae`, `inference.async_vae`, `inference.vae_type`, and
`inference.vae_device` control streaming or asynchronous VAE decode.
- `torch_compile` can be set to `auto`, `true`, or `false`; the default config
uses `auto` with safe error suppression.

### Sequence-parallel (SP) inference

`inference_sp.py` drives **Ulysses sequence-parallel** sampling for WAN (see `configs/inference_sp.yaml` for `sp_size`, `dp_size`, prompts, checkpoints, and the usual `inference.*` knobs). Launch one process per GPU with **`--nproc_per_node` equal to `sp_size × dp_size`** (the shipped example sets `sp_size: 4` and `dp_size: 1`, so four ranks).

```bash
torchrun --nproc_per_node=4 inference_sp.py --config_path configs/inference_sp.yaml
```

## Utilities

Inspect SP VAE halo windows:

```bash
python scripts/compute_sp_vae_chunk_halo.py --config configs/train_ar.yaml
```

Decode saved VAE latents:

```bash
python scripts/decode_vae_latents.py --help
python scripts/decode_lightvae_latents.py --help
```
