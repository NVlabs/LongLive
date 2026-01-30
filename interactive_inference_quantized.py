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

from pipeline.interactive_causal_inference import (
    InteractiveCausalInferencePipeline,
)
from utils.dataset import MultiTextDataset

# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("Interactive causal inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--use_quantized", action="store_true",
                    help="Use quantized models from ../longlive_models/")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)
# config.model_kwargs.local_attn_size = 32
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
    config.lora_ckpt = None  # We'll load this manually later

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
pipeline = InteractiveCausalInferencePipeline(config, device=device)
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

# Move pipeline to appropriate dtype and device
# print("\n🔧 Moving pipeline to bfloat16...")
# pipeline = pipeline.to(dtype=torch.bfloat16)

if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)

# ======================== USE GPU 1 or 2 FOR INFERENCE ========================

# ======================== USE GPU 2 WITH DYNAMIC SWAP ========================
# inference_device = torch.device('cuda:2')
#
# print(f"\n🎬 Loading all components to GPU 2")
# print(f"   GPU 2 Free VRAM before: {get_cuda_free_memory_gb(inference_device):.2f} GB")
#
# torch.cuda.empty_cache()
#
# print(f"   GPU 2 Free VRAM after clearing GPU 0: {get_cuda_free_memory_gb(inference_device):.2f} GB")
#
# # Now move everything to GPU 2
# print("   Moving generator to GPU 2...")
# pipeline.generator.to(device=inference_device)
#
# print("   Converting VAE to bfloat16...")
# pipeline.vae = pipeline.vae.to(dtype=torch.bfloat16)
#
# # print("   Moving text_encoder from CPU to GPU 2...")
# # pipeline.text_encoder = pipeline.text_encoder.to(device=inference_device)
#
# print(f"✅ All components on GPU 2")
# print(f"   GPU 2 VRAM used: {(torch.cuda.memory_allocated(2) / 1024**3):.2f} GB")

print("\n🚚 Moving models to GPU 2 for inference...")

inference_device = torch.device("cuda:2")

pipeline.generator.to(inference_device)
pipeline.vae.to(inference_device)

# Optional but recommended for speed
pipeline.generator = pipeline.generator.to(dtype=torch.bfloat16)
pipeline.vae = pipeline.vae.to(dtype=torch.bfloat16)

torch.cuda.empty_cache()

print("Generator device:", next(pipeline.generator.parameters()).device)
print("VAE device:", next(pipeline.vae.parameters()).device)
print("Text encoder device:", next(pipeline.text_encoder.parameters()).device)

device = inference_device

# ======================== END GPU SETUP ========================

# ----------------------------- Build dataset -----------------------------
if isinstance(config.switch_frame_indices, int):
    switch_frame_indices: List[int] = [int(config.switch_frame_indices)]
else:
    switch_frame_indices: List[int] = [
        int(x) for x in str(config.switch_frame_indices).split(",") if str(x).strip()
    ]

dataset = MultiTextDataset(config.data_path)

num_segments = len(dataset[0]["prompts_list"])
assert len(switch_frame_indices) == num_segments - 1, (
    "The number of switch_frame_indices should be the number of prompt segments minus 1"
)

print("Number of segments:", num_segments)
print("Switch frame indices:", switch_frame_indices)

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
    prompts_list: List[str] = batch_data["prompts_list"]

    sampled_noise = torch.randn(
        [
            config.num_samples,
            config.num_output_frames,
            16,
            60,
            104,
        ],
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
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
            short_name = prompts_list[0][0][:100].replace("/", "_")
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
