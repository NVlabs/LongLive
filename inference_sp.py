# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from math import gcd

import peft
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from pipeline.causal_diffusion_inference_sp import CausalDiffusionInferencePipelineSP
from utils.config import normalize_config, section_get
from utils.dataset import MultiTextConcatDataset, eval_collate_fn
from utils.lora_utils import configure_lora_for_model
from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from utils.misc import set_seed
from utils.nvfp4_checkpoint import (
    clean_fsdp_state_dict_keys,
    drop_fouroversix_master_weights,
    is_nvfp4_state_dict,
    is_te_nvfp4_checkpoint,
    quantize_model_for_fouroversix_nvfp4,
    unwrap_generator_state_dict,
)


def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return
            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                prompt = prompts_for_sample[seg_idx]
                if prompt == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    f.write(f"[{','.join(str(i) for i in current_indices)}] {current_prompt}\n")
                    current_prompt = prompt
                    current_indices = [seg_idx]
            f.write(f"[{','.join(str(i) for i in current_indices)}] {current_prompt}\n")
    except Exception as exc:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {exc}")


def compute_group_specs(world_size, sp_size, dp_size, num_heads, num_frame_per_block,
                        auto_sp_remainder=False):
    """Compute DP groups whose ranks each form a Ulysses SP group."""
    valid_base = gcd(num_heads, num_frame_per_block)
    valid_sp_sizes = sorted(size for size in range(1, valid_base + 1) if valid_base % size == 0)
    if sp_size not in valid_sp_sizes:
        raise ValueError(
            f"sp_size={sp_size} must divide gcd(num_heads={num_heads}, "
            f"num_frame_per_block={num_frame_per_block})={valid_base}"
        )

    full_groups = min(world_size // max(sp_size, 1), dp_size)
    groups = [
        (sp_size, list(range(i * sp_size, (i + 1) * sp_size)))
        for i in range(full_groups)
    ]
    ranks_used = full_groups * sp_size
    if auto_sp_remainder:
        remaining_gpus = world_size - ranks_used
        remaining_quota = dp_size - full_groups
        while remaining_gpus > 0 and remaining_quota > 0:
            size = max((v for v in valid_sp_sizes if v <= remaining_gpus), default=1)
            groups.append((size, list(range(ranks_used, ranks_used + size))))
            ranks_used += size
            remaining_gpus -= size
            remaining_quota -= 1
    return groups


def _maybe_to_dict(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _expected_inference_samples(config):
    inference_iter = int(getattr(config, "inference_iter", -1))
    return inference_iter + 1 if inference_iter >= 0 else None


def _resolve_torch_compile(config):
    setting = getattr(config, "torch_compile", False)
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        if not bool(getattr(config, "model_quant", False)):
            return False, "auto disabled because model_quant is false"
        min_samples = int(getattr(config, "torch_compile_min_samples", 2))
        expected_samples = _expected_inference_samples(config)
        if expected_samples is not None and expected_samples < min_samples:
            return False, f"auto disabled because expected samples ({expected_samples}) < {min_samples}"
        return True, "auto enabled for repeated quantized inference"
    return _config_bool(setting, default=False), "explicit setting"


def quantize_generator_model(model, config, keep_master_weights, is_main_process):
    from utils.quant import (
        ModelQuantizationConfig,
        _materialize_mixed_quantized_weights_for_inference,
        _materialize_quantized_weights_for_inference,
        _materialize_transformer_engine_weights_for_inference,
        quantize_model_with_filter,
    )

    use_transformer_engine = bool(getattr(config, "model_quant_use_transformer_engine", False))
    te_inference_only = bool(getattr(config, "model_quant_te_inference_only", use_transformer_engine))
    te_low_precision_weights = bool(getattr(config, "model_quant_te_low_precision_weights", te_inference_only))
    te_fallback_to_fouroversix = bool(getattr(config, "model_quant_te_fallback_to_fouroversix", False))
    quant_cfg = ModelQuantizationConfig(
        scale_rule=getattr(config, "model_quant_scale_rule", "static_6"),
        quantize_backend=getattr(config, "model_quant_backend", None),
        activation_scale_rule=getattr(
            config, "model_quant_activation_scale_rule",
            getattr(config, "model_quant_scale_rule", "static_6"),
        ),
        weight_scale_rule=getattr(config, "model_quant_weight_scale_rule", None),
        gradient_scale_rule=getattr(config, "model_quant_gradient_scale_rule", None),
    )
    quant_cfg.keep_master_weights = keep_master_weights
    model, matched_modules = quantize_model_with_filter(
        model,
        quant_config=quant_cfg,
        filtered_modules=getattr(config, "model_quant_filtered_modules", None),
        use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
        cast_model_to_bf16=True,
        materialize_for_inference=False,
        use_transformer_engine=use_transformer_engine,
        te_inference_only=te_inference_only,
        te_low_precision_weights=te_low_precision_weights,
        te_recipe_kwargs=_maybe_to_dict(getattr(config, "model_quant_te_recipe_kwargs", None)),
        te_module_kwargs=_maybe_to_dict(getattr(config, "model_quant_te_module_kwargs", None)),
        te_fallback_to_fouroversix=te_fallback_to_fouroversix,
        verbose=is_main_process,
    )
    materialize_fn = _materialize_quantized_weights_for_inference
    if use_transformer_engine and te_fallback_to_fouroversix:
        materialize_fn = _materialize_mixed_quantized_weights_for_inference
    elif use_transformer_engine:
        materialize_fn = _materialize_transformer_engine_weights_for_inference
    if is_main_process:
        print(f"[NVFP4] Generator quantized; {len(matched_modules)} modules excluded")
    return model, materialize_fn


def materialize_quantized_generator(model, device, materialize_fn, stage_desc, is_main_process):
    mat_modules, master_bytes, quantized_bytes = materialize_fn(model, target_device=device)
    if is_main_process:
        print(
            f"[NVFP4] Materialized quantized generator weights {stage_desc}: "
            f"{len(mat_modules)} modules, master_weight={master_bytes / (1024 ** 3):.3f} GiB, "
            f"quantized_weight={quantized_bytes / (1024 ** 3):.3f} GiB"
        )


def configure_generator_torch_compile(pipeline, config, is_main_process):
    compile_enabled, reason = _resolve_torch_compile(config)
    if not compile_enabled:
        if is_main_process and str(getattr(config, "torch_compile", "false")).lower() == "auto":
            print(f"[torch.compile] skipped: {reason}")
        return
    if not hasattr(pipeline.generator, "configure_torch_compile"):
        if is_main_process:
            print("[torch.compile][warn] Current generator does not expose configure_torch_compile; skipping")
        return
    compiled = pipeline.generator.configure_torch_compile(
        backend=str(getattr(config, "torch_compile_backend", "inductor")),
        mode=getattr(config, "torch_compile_mode", "max-autotune-no-cudagraphs"),
        fullgraph=_config_bool(getattr(config, "torch_compile_fullgraph", False)),
        dynamic=_config_bool(getattr(config, "torch_compile_dynamic", False)),
        options=_maybe_to_dict(getattr(config, "torch_compile_options", None)),
        suppress_errors=_config_bool(getattr(config, "torch_compile_suppress_errors", True), default=True),
    )
    if is_main_process:
        print(f"[torch.compile] {'enabled' if compiled else 'not enabled'}: target=generator_model")


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config YAML file")
te_quant_group = parser.add_mutually_exclusive_group()
te_quant_group.add_argument("--use_te_quant", dest="use_te_quant", action="store_true")
te_quant_group.add_argument("--no_use_te_quant", dest="use_te_quant", action="store_false")
parser.set_defaults(use_te_quant=None)
args = parser.parse_args()

config = normalize_config(OmegaConf.load(args.config_path))
if args.use_te_quant is not None:
    config.model_quant_use_transformer_engine = args.use_te_quant
if not hasattr(config, "sampling_steps") or config.sampling_steps is None:
    raise ValueError("sampling_steps must be defined in the SP inference config")
if not hasattr(config, "guidance_scale") or config.guidance_scale is None:
    config.guidance_scale = 1.0

config.use_ema = section_get(config, "inference", "use_ema", getattr(config, "use_ema", False))
config.output_folder = section_get(config, "inference", "output_folder", getattr(config, "output_folder", "videos/longlive2_sp"))
config.num_samples = section_get(config, "inference", "num_samples", getattr(config, "num_samples", 1))
config.num_output_frames = getattr(config, "num_output_frames", config.image_or_video_shape[1])
config.save_with_index = getattr(config, "save_with_index", False)
config.inference_iter = getattr(config, "inference_iter", -1)
if getattr(config, "i2v", False):
    raise NotImplementedError("I2V inference is not included in this SP release path.")
if getattr(config, "kv_quant", False):
    print("[SP][warn] kv_quant is not supported in Ulysses SP inference; disabling it.")
    config.kv_quant = False

sp_size = int(getattr(config, "sp_size", 1))
dp_size = int(getattr(config, "dp_size", 1))
auto_sp_remainder = bool(getattr(config, "auto_sp_remainder", False))
model_num_heads = int(getattr(config, "model_num_heads", 24))

sp_group = None
dp_rank = 0
sp_rank = 0
effective_sp_size = sp_size
total_dp_groups = 1

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    os.environ.setdefault("NCCL_CROSS_NIC", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    set_seed(config.seed + rank)
    config.distributed = True

    group_specs = compute_group_specs(
        world_size=world_size,
        sp_size=sp_size,
        dp_size=dp_size,
        num_heads=model_num_heads,
        num_frame_per_block=int(getattr(config, "num_frame_per_block", 8)),
        auto_sp_remainder=auto_sp_remainder,
    )
    assigned_count = sum(len(ranks) for _, ranks in group_specs)
    if assigned_count != world_size:
        raise ValueError(
            f"SP group layout assigns {assigned_count}/{world_size} ranks. "
            "Increase dp_size, lower sp_size, or enable auto_sp_remainder."
        )
    total_dp_groups = len(group_specs)
    sp_groups_all = [dist.new_group(ranks=ranks) for _, ranks in group_specs]
    for dp_i, (eff_sp, ranks) in enumerate(group_specs):
        if rank in ranks:
            dp_rank = dp_i
            sp_rank = ranks.index(rank)
            effective_sp_size = eff_sp
            sp_group = sp_groups_all[dp_i]
            break
    if rank == 0:
        print(
            f"[SP] Parallelism: {total_dp_groups} DP group(s), "
            f"sp_sizes={[size for size, _ in group_specs]}, assigned={assigned_count}/{world_size}"
        )
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False
    effective_sp_size = 1

is_main_process = rank == 0
use_effective_sp = effective_sp_size > 1 and dist.is_initialized()
use_multi_dp = total_dp_groups > 1

if use_effective_sp:
    from wan_5b.distributed.sp_ulysses_inference import init_sequence_parallel
    init_sequence_parallel(group=sp_group)
    if is_main_process:
        print(f"[SP] Ulysses mode enabled: sp_sizes={[size for size, _ in group_specs]}")
elif is_main_process:
    print("[SP] Running SP model with world_size=1")

torch.set_grad_enabled(False)
free_vram = get_cuda_free_memory_gb(device)
low_memory = free_vram < 40
if is_main_process:
    print(f"[SP] Free VRAM: {free_vram:.1f} GB, low_memory={low_memory}")

pipeline = CausalDiffusionInferencePipelineSP(
    config,
    device=device,
    sp_group=sp_group,
    dp_rank=dp_rank,
)

merge_lora = bool(getattr(config, "merge_lora", False))
has_lora_adapter = bool(getattr(config, "adapter", None) and configure_lora_for_model is not None)
materialize_quantized_weights_for_inference = None
generator_checkpoint = None
generator_lora_state = None
generator_ckpt_path = getattr(config, "generator_ckpt", None)
loaded_prequantized_generator = False
prequantized_generator_backend = None

if generator_ckpt_path:
    if is_main_process:
        print(f"[SP] Loading generator checkpoint: {generator_ckpt_path}")
    generator_checkpoint = torch.load(generator_ckpt_path, map_location="cpu", mmap=True)
    is_lora_only_checkpoint = (
        isinstance(generator_checkpoint, dict)
        and "generator_lora" in generator_checkpoint
        and not any(key in generator_checkpoint for key in ("generator", "generator_ema", "model"))
    )
    if is_lora_only_checkpoint:
        generator_lora_state = generator_checkpoint["generator_lora"]
    else:
        raw_gen_state_dict = unwrap_generator_state_dict(generator_checkpoint, use_ema=config.use_ema)
        if config.use_ema:
            raw_gen_state_dict = clean_fsdp_state_dict_keys(raw_gen_state_dict)
        if is_te_nvfp4_checkpoint(generator_checkpoint):
            raise ValueError("TransformerEngine module state_dict checkpoints are not supported here.")
        if is_nvfp4_state_dict(raw_gen_state_dict):
            if not getattr(config, "model_quant", False):
                raise ValueError("generator_ckpt is materialized NVFP4 but model_quant is false.")
            if getattr(config, "model_quant_use_transformer_engine", False):
                raise ValueError("Materialized NVFP4 checkpoints require model_quant_use_transformer_engine=false.")
            pipeline.generator.model, matched_modules = quantize_model_for_fouroversix_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=is_main_process,
            )
            dropped_modules = drop_fouroversix_master_weights(pipeline.generator.model)
            pipeline.generator.load_state_dict(raw_gen_state_dict, strict=True)
            loaded_prequantized_generator = True
            prequantized_generator_backend = "fouroversix"
            if is_main_process:
                print(
                    f"[NVFP4] Prepared SP generator: {len(dropped_modules)} materialized modules, "
                    f"{len(matched_modules)} modules excluded"
                )
        elif config.use_ema:
            missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
            if is_main_process and (missing or unexpected):
                print(f"[SP][warn] missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict, strict=True)

pipeline.is_lora_enabled = False
pipeline.is_lora_merged = False
if loaded_prequantized_generator:
    has_lora_adapter = False
    merge_lora = False
    config.merge_lora = False

if getattr(config, "model_quant", False) and not merge_lora and not loaded_prequantized_generator:
    pipeline.generator.model, materialize_quantized_weights_for_inference = quantize_generator_model(
        pipeline.generator.model,
        config=config,
        keep_master_weights=has_lora_adapter,
        is_main_process=is_main_process,
    )

if has_lora_adapter:
    if is_main_process:
        print(f"[SP] Applying LoRA config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=is_main_process,
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu", mmap=True)
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
    elif generator_lora_state is not None:
        peft.set_peft_model_state_dict(pipeline.generator.model, generator_lora_state)
    if merge_lora:
        pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
        pipeline.is_lora_merged = True
    else:
        pipeline.is_lora_enabled = True
elif merge_lora and is_main_process:
    print("merge_lora=True requested but no adapter config was found; continuing without LoRA merge")

del generator_checkpoint

if loaded_prequantized_generator:
    pipeline.text_encoder.to(dtype=torch.bfloat16)
    pipeline.vae.to(dtype=torch.bfloat16)
else:
    pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)

if getattr(config, "model_quant", False) and not loaded_prequantized_generator:
    if merge_lora:
        pipeline.generator.model, materialize_quantized_weights_for_inference = quantize_generator_model(
            pipeline.generator.model,
            config=config,
            keep_master_weights=False,
            is_main_process=is_main_process,
        )
        stage_desc = "after LoRA merge" if pipeline.is_lora_merged else "for inference"
    else:
        stage_desc = "after LoRA wrapping" if pipeline.is_lora_enabled else "for inference"
    materialize_quantized_generator(
        pipeline.generator.model,
        device=device,
        materialize_fn=materialize_quantized_weights_for_inference,
        stage_desc=stage_desc,
        is_main_process=is_main_process,
    )
elif loaded_prequantized_generator and is_main_process:
    print(f"[NVFP4] Using pre-saved {prequantized_generator_backend} generator weights")

pipeline.generator.model.eval().requires_grad_(False)
configure_generator_torch_compile(pipeline, config, is_main_process)

vae_device_str = getattr(config, "vae_device", None)
use_dedicated_vae_device = bool(getattr(config, "streaming_vae", False)) and bool(vae_device_str)
if use_dedicated_vae_device and sp_rank == 0:
    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    if is_main_process:
        print(f"[SP] VAE on {vae_device}, diffusion on {device}")
else:
    pipeline.vae.to(device=device)

nfpb = getattr(config, "num_frame_per_block", 8)
num_blocks = config.num_output_frames // nfpb
dataset = MultiTextConcatDataset(
    data_path=config.data_path,
    num_blocks=num_blocks,
    chunks_per_shot=getattr(config, "chunks_per_shot", 0),
    scene_cut_prefix=getattr(config, "scene_cut_prefix", "The scene transitions. "),
    deterministic=True,
)
if is_main_process:
    print(f"[data] data_path={config.data_path}, mode={dataset._mode}, num_blocks={num_blocks}")
num_prompts = len(dataset)
if use_multi_dp:
    sampler = DistributedSampler(
        dataset,
        num_replicas=total_dp_groups,
        rank=dp_rank,
        shuffle=False,
        drop_last=True,
    )
elif dist.is_initialized():
    sampler = SequentialSampler(dataset)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(
    dataset, batch_size=1, sampler=sampler, num_workers=0,
    drop_last=False, collate_fn=eval_collate_fn,
)

if is_main_process:
    os.makedirs(config.output_folder, exist_ok=True)
if dist.is_initialized():
    dist.barrier()

save_latents_only = section_get(
    config,
    "inference",
    "save_latents_only",
    getattr(config, "save_latents_only", getattr(config, "save_latent_only", False)),
    aliases=("save_latent_only", "return_latents"),
)

for i, batch_data in tqdm(enumerate(dataloader), disable=not is_main_process):
    idx = batch_data["idx"].item()
    block_prompts = list(batch_data["prompts"][0])
    if len(block_prompts) < num_blocks:
        block_prompts += [block_prompts[-1]] * (num_blocks - len(block_prompts))
    elif len(block_prompts) > num_blocks:
        block_prompts = block_prompts[:num_blocks]
    prompt = block_prompts[0]
    prompts = [block_prompts] * config.num_samples

    shape = config.image_or_video_shape
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, shape[2], shape[3], shape[4]],
        device=device,
        dtype=torch.bfloat16,
    )
    if use_effective_sp:
        src = dist.get_global_rank(sp_group, 0)
        dist.broadcast(sampled_noise, src=src, group=sp_group)

    if is_main_process:
        print(f"\n[SP] Generating video {idx}: {prompt[:60]}...")
    generated = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=save_latents_only,
    )

    should_save = (sp_rank == 0) if use_effective_sp else True
    if idx < num_prompts and should_save:
        if getattr(pipeline, "is_lora_merged", False):
            model_type = "merged_lora"
        elif getattr(pipeline, "is_lora_enabled", False):
            model_type = "lora"
        elif getattr(config, "use_ema", False):
            model_type = "ema"
        else:
            model_type = "regular"
        mode = f"dp{dp_rank}_sp{effective_sp_size}" if use_multi_dp else f"sp{effective_sp_size}"
        if save_latents_only:
            latents = generated
        else:
            current_video = rearrange(generated, "b t c h w -> b t h w c").cpu()
            video = 255.0 * current_video
            if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()

        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                base_name = f"rank{rank}-{idx}-{seed_idx}_{model_type}_{mode}"
            else:
                base_name = f"rank{rank}-{prompt[:100]}-{seed_idx}_{model_type}_{mode}"
            if save_latents_only:
                torch.save(latents[seed_idx].cpu(), os.path.join(config.output_folder, f"{base_name}.pt"))
            else:
                output_path = os.path.join(config.output_folder, f"{base_name}.mp4")
                fps = 24 if "5B" in config.model_kwargs.model_name else 16
                write_video(output_path, video[seed_idx], fps=fps)
                if is_main_process:
                    print(f"[SP] Saved: {output_path}")
            save_prompts_to_txt(
                prompts[seed_idx] if isinstance(prompts[seed_idx], list) else [prompts[seed_idx]],
                os.path.join(config.output_folder, f"{base_name}_prompts.txt"),
                is_main_process=is_main_process,
            )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
