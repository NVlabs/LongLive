# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from torchvision import transforms  # noqa: F401
from einops import rearrange

from utils.misc import set_seed
from utils.distributed import barrier
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.single_prompt_causal_inference import (
    SinglePromptCausalInferencePipeline,
)
from utils.dataset import TextDataset

# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("Single prompt causal inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--use_quantized", action="store_true",
                    help="Use quantized models from ../longlive_models/")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ======================== LOAD QUANTIZED MODELS TO CPU FIRST ========================
quantized_base_state = None
quantized_lora_state = None

if args.use_quantized:
    print("\n" + "="*70)
    print("🔧 Pre-loading QUANTIZED models to CPU")
    print("="*70)

    import sys
    sys.path.insert(0, '..')

    base_path = '../longlive_models/longlive_base_bfloat16.pt'
    lora_path = '../longlive_models/lora_bfloat16.pt'

    print(f"📥 Loading base checkpoint to CPU from: {base_path}")
    base_checkpoint = torch.load(base_path, map_location='cpu', weights_only=False)

    print(f"📥 Loading LoRA checkpoint to CPU from: {lora_path}")
    lora_checkpoint = torch.load(lora_path, map_location='cpu', weights_only=False)

    # Extract the models
    quantized_base_state = base_checkpoint['generator']
    quantized_lora_state = lora_checkpoint['generator_lora']
    quantized_critic_lora = lora_checkpoint.get('critic_lora', {})

    print(f"✅ Loaded to CPU - will transfer after pipeline init")
    print(f"   Base params: {sum(p.numel() for p in quantized_base_state.values() if isinstance(p, torch.Tensor)):,}")
    print(f"   LoRA params: {sum(p.numel() for p in quantized_lora_state.values() if isinstance(p, torch.Tensor)):,}")
    print("="*70 + "\n")

    # Don't load from checkpoint files
    config.generator_ckpt = None
    config.lora_ckpt = None

# ======================== END PRE-LOADING ========================

# ----------------------------- Distributed setup -----------------------------
if "LOCAL_RANK" in os.environ:  # Multi-GPU via torchrun
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )

    set_seed(config.seed + local_rank)
    print(f"[Rank {rank}] Distributed mode on GPU {local_rank}")

else:  # Single-GPU mode
    assert torch.cuda.is_available(), "CUDA is required but not available"

    local_rank = 0
    rank = 0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    set_seed(config.seed)
    print("Single GPU mode on cuda:0")

low_memory = get_cuda_free_memory_gb(device) < 40
torch.set_grad_enabled(False)

# ======================== INITIALIZE PIPELINE ========================
pipeline = SinglePromptCausalInferencePipeline(config, device=device)
print("Generator device:", next(pipeline.generator.parameters()).device)
print("VAE device:", next(pipeline.vae.parameters()).device)
print("Text encoder device:", next(pipeline.text_encoder.parameters()).device)

# ======================== LOAD QUANTIZED WEIGHTS ========================
if args.use_quantized and quantized_base_state is not None:
    print("\n" + "="*70)
    print("🔧 Loading quantized base model into pipeline")
    print("="*70)

    missing, unexpected = pipeline.generator.load_state_dict(quantized_base_state, strict=False)
    if local_rank == 0:
        if missing:
            print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
        if unexpected:
            print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
        print("✅ Quantized base model loaded")

    print("🔧 Converting ALL generator parameters to bfloat16...")
    for name, param in pipeline.generator.named_parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    for name, buffer in pipeline.generator.named_buffers():
        if buffer.dtype != torch.bfloat16 and buffer.dtype == torch.float32:
            buffer.data = buffer.data.to(torch.bfloat16)
    print("✅ All generator parameters converted to bfloat16")

    # Clear the CPU checkpoint to free memory
    del base_checkpoint
    del quantized_base_state
    import gc
    gc.collect()

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"\n🔧 LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")

    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load quantized LoRA weights
    if args.use_quantized and quantized_lora_state is not None:
        if local_rank == 0:
            print(f"Loading QUANTIZED LoRA weights from CPU")
        peft.set_peft_model_state_dict(pipeline.generator.model, quantized_lora_state)
        if local_rank == 0:
            print("✅ Quantized LoRA weights loaded")

        # Clear LoRA checkpoint
        del lora_checkpoint
        del quantized_lora_state
        gc.collect()
    elif not args.use_quantized:
        # Original LoRA loading for non-quantized
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            if local_rank == 0:
                print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
            if local_rank == 0:
                print("LoRA weights loaded")

    pipeline.is_lora_enabled = True

# ======================== MODEL PARALLEL GPU SETUP ========================
print("\n🚚 Setting up model parallelism (Text → GPU0, Gen+VAE → GPU2)")

gpu_text = torch.device("cuda:0")
gpu_main = torch.device("cuda:2")

torch.cuda.empty_cache()

# 1️⃣ Move TEXT ENCODER to GPU 0 (prompt encoding only)
print("🧠 Moving text encoder to GPU 0...")
pipeline.text_encoder = pipeline.text_encoder.to(gpu_text)
print(f"Text encoder device: {next(pipeline.text_encoder.parameters()).device}")

torch.cuda.empty_cache()

# 2️⃣ Move GENERATOR to GPU 2
print("🎬 Moving generator to GPU 2 in bfloat16...")
pipeline.generator = pipeline.generator.to(gpu_main, dtype=torch.bfloat16)
print(f"Generator device: {next(pipeline.generator.parameters()).device}")

torch.cuda.empty_cache()

# 3️⃣ Move VAE to GPU 2
print("🖼️ Moving VAE to GPU 2 in bfloat16...")
pipeline.vae = pipeline.vae.to(gpu_main, dtype=torch.bfloat16)
print(f"VAE device: {next(pipeline.vae.parameters()).device}")

torch.cuda.empty_cache()

print("\n✅ Model parallel setup complete")
print(f"   Text Encoder → {next(pipeline.text_encoder.parameters()).device}")
print(f"   Generator    → {next(pipeline.generator.parameters()).device}")
print(f"   VAE          → {next(pipeline.vae.parameters()).device}")

# Set main device for noise generation & diffusion
device = gpu_main
# ======================== END GPU SETUP ========================

# ======================== OPTIONAL: USE SPECIFIC GPU FOR INFERENCE ========================
# Uncomment and modify if you want to use a specific GPU (e.g., GPU 2)
# inference_device = torch.device("cuda:2")
# print(f"\n🚚 Moving models to {inference_device} for inference...")
# pipeline.generator.to(inference_device)
# pipeline.vae.to(inference_device)
# torch.cuda.empty_cache()
# device = inference_device
# ======================== END GPU SETUP ========================

# ----------------------------- Build dataset -----------------------------
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=config.data_path)

num_prompts_total = len(dataset)
print(f"Number of prompt lines: {num_prompts_total}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ----------------------------- Inference loop -----------------------------
print("\n" + "="*70)
print("🚀 Starting video generation...")
print("="*70)

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    
    # Get the prompt from the batch
    prompt = batch_data['prompts'][0]
    extended_prompt = batch_data.get('extended_prompts', [prompt])[0]
    
    # Use extended prompt if available, otherwise use regular prompt
    text_prompt = extended_prompt if extended_prompt else prompt
    prompts = [text_prompt] * config.num_samples

    # Get the actual device where generator is located
    gen_device = next(pipeline.generator.parameters()).device
    
    sampled_noise = torch.randn(
        [
            config.num_samples,
            config.num_output_frames,
            16,
            60,
            104,
        ],
        device=gen_device,  # Use generator's device
        dtype=torch.bfloat16,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=False,
        )

    current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    model_type = "quantized" if args.use_quantized else "regular"

    for seed_idx in range(config.num_samples):
        if config.save_with_index:
            output_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4")
        else:
            short_name = text_prompt[:100].replace("/", "_")
            output_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")

        write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)

        if local_rank == 0:
            print(f"✅ Saved: {output_path}")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

print("\n" + "="*70)
print("🎉 Video generation complete!")
print("="*70)

if dist.is_initialized():
    dist.destroy_process_group()
