# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalDiffusionInferencePipeline
from utils.dataset import MultiTextConcatDataset, eval_collate_fn
from utils.misc import set_seed
from utils.config import normalize_config, section_get
from utils.nvfp4_checkpoint import (
    clean_fsdp_state_dict_keys,
    drop_fouroversix_master_weights,
    is_nvfp4_state_dict,
    is_te_nvfp4_checkpoint,
    quantize_model_for_fouroversix_nvfp4,
    unwrap_generator_state_dict,
)

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    """Save per-block prompts alongside the video.

    Consecutive identical prompts are merged, e.g.:
        [0] a, [1] a, [2] b  =>  [0,1] a\\n[2] b\\n
    """
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return
            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                p = prompts_for_sample[seg_idx]
                if p == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    indices_str = ",".join(str(i) for i in current_indices)
                    f.write(f"[{indices_str}] {current_prompt}\n")
                    current_prompt = p
                    current_indices = [seg_idx]
            indices_str = ",".join(str(i) for i in current_indices)
            f.write(f"[{indices_str}] {current_prompt}\n")
    except Exception as e:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
te_quant_group = parser.add_mutually_exclusive_group()
te_quant_group.add_argument(
    "--use_te_quant",
    dest="use_te_quant",
    action="store_true",
    help="Override config and enable TransformerEngine quantization",
)
te_quant_group.add_argument(
    "--no_use_te_quant",
    dest="use_te_quant",
    action="store_false",
    help="Override config and disable TransformerEngine quantization",
)
parser.set_defaults(use_te_quant=None)
args = parser.parse_args()

config = normalize_config(OmegaConf.load(args.config_path))
if args.use_te_quant is not None:
    config.model_quant_use_transformer_engine = args.use_te_quant

if not hasattr(config, "sampling_steps") or config.sampling_steps is None:
    raise ValueError("sampling_steps must be defined in the inference config")

if not hasattr(config, "guidance_scale") or config.guidance_scale is None:
    config.guidance_scale = 1.0

config.use_ema = section_get(config, "inference", "use_ema", getattr(config, "use_ema", False))
config.output_folder = section_get(config, "inference", "output_folder", getattr(config, "output_folder", "videos/longlive2"))
config.num_samples = section_get(config, "inference", "num_samples", getattr(config, "num_samples", 1))
config.num_output_frames = getattr(config, "num_output_frames", config.image_or_video_shape[1])
config.save_with_index = getattr(config, "save_with_index", False)
config.inference_iter = getattr(config, "inference_iter", -1)
if getattr(config, "i2v", False):
    raise NotImplementedError("I2V inference is not included in this release path.")


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
    if inference_iter >= 0:
        return inference_iter + 1
    return None


def _resolve_torch_compile(config):
    setting = getattr(config, "torch_compile", False)
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        if not bool(getattr(config, "model_quant", False)):
            return False, "auto disabled because model_quant is false"
        min_samples = int(getattr(config, "torch_compile_min_samples", 2))
        expected_samples = _expected_inference_samples(config)
        if expected_samples is not None and expected_samples < min_samples:
            return (
                False,
                "auto disabled because expected samples "
                f"({expected_samples}) < torch_compile_min_samples ({min_samples})",
            )
        return True, "auto enabled for repeated quantized inference"
    return _config_bool(setting, default=False), "explicit setting"


def quantize_generator_model(model, config, keep_master_weights):
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
            config,
            "model_quant_activation_scale_rule",
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
        verbose=True,
    )
    materialize_fn = _materialize_quantized_weights_for_inference
    if use_transformer_engine and te_fallback_to_fouroversix:
        materialize_fn = _materialize_mixed_quantized_weights_for_inference
    elif use_transformer_engine:
        materialize_fn = _materialize_transformer_engine_weights_for_inference
    if local_rank == 0:
        print(f"[NVFP4] Generator quantized; {len(matched_modules)} modules excluded")
    return model, materialize_fn


def materialize_quantized_generator(model, device, materialize_fn, stage_desc):
    mat_modules, master_bytes, quantized_bytes = materialize_fn(
        model,
        target_device=device,
    )
    if local_rank == 0:
        print(
            f"[NVFP4] Materialized quantized generator weights {stage_desc}: "
            f"{len(mat_modules)} modules, "
            f"master_weight={master_bytes / (1024 ** 3):.3f} GiB, "
            f"quantized_weight={quantized_bytes / (1024 ** 3):.3f} GiB"
        )


def configure_generator_torch_compile(pipeline, config):
    compile_enabled, reason = _resolve_torch_compile(config)
    if not compile_enabled:
        if local_rank == 0 and str(getattr(config, "torch_compile", "false")).lower() == "auto":
            print(f"[torch.compile] skipped: {reason}")
        return
    target = str(getattr(config, "torch_compile_target", "generator_model")).lower()
    if target not in {"generator_model", "model"}:
        if local_rank == 0:
            print(f"[torch.compile][warn] Unsupported target={target}; expected generator_model")
        return
    if not hasattr(pipeline.generator, "configure_torch_compile"):
        if local_rank == 0:
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
    if local_rank == 0:
        status = "enabled" if compiled else "not enabled"
        print(f"[torch.compile] {status}: target={target}")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
else:
    local_rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)


# Initialize pipeline
pipeline = CausalDiffusionInferencePipeline(config, device=device)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

merge_lora = bool(getattr(config, "merge_lora", False))
has_lora_adapter = bool(getattr(config, "adapter", None) and configure_lora_for_model is not None)
materialize_quantized_weights_for_inference = None
generator_checkpoint = None
generator_lora_state = None
generator_ckpt_path = getattr(config, "generator_ckpt", None)
loaded_prequantized_generator = False
prequantized_generator_backend = None
if generator_ckpt_path:
    generator_checkpoint = torch.load(generator_ckpt_path, map_location="cpu")
    is_lora_only_checkpoint = (
        isinstance(generator_checkpoint, dict)
        and "generator_lora" in generator_checkpoint
        and not any(key in generator_checkpoint for key in ("generator", "generator_ema", "model"))
    )
    if is_lora_only_checkpoint:
        generator_lora_state = generator_checkpoint["generator_lora"]
        if local_rank == 0:
            print(f"Found LoRA generator weights in {generator_ckpt_path}")
    else:
        raw_gen_state_dict = unwrap_generator_state_dict(generator_checkpoint, use_ema=config.use_ema)
        if config.use_ema:
            raw_gen_state_dict = clean_fsdp_state_dict_keys(raw_gen_state_dict)
        if is_te_nvfp4_checkpoint(generator_checkpoint):
            raise ValueError(
                "This checkpoint was saved as a TransformerEngine module state_dict, "
                "which is not packed NVFP4 and is no longer a supported export format. "
                "Regenerate with `--backend transformer_engine` to save merged bf16 weights "
                "for TE runtime quantization, or use `--backend fouroversix` for a compact "
                "materialized NVFP4 checkpoint."
            )
        elif is_nvfp4_state_dict(raw_gen_state_dict):
            if not getattr(config, "model_quant", False):
                raise ValueError(
                    "generator_ckpt is a materialized NVFP4 checkpoint, but model_quant is false. "
                    "Set model_quant: true in the inference yaml."
                )
            if getattr(config, "model_quant_use_transformer_engine", False):
                raise ValueError(
                    "Materialized NVFP4 generator checkpoints use FourOverSix modules. "
                    "Set model_quant_use_transformer_engine: false when loading this checkpoint."
                )
            if local_rank == 0:
                print(f"[NVFP4] Loading materialized generator checkpoint from {generator_ckpt_path}")
            pipeline.generator.model, matched_modules = quantize_model_for_fouroversix_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=(local_rank == 0),
            )
            dropped_modules = drop_fouroversix_master_weights(pipeline.generator.model)
            pipeline.generator.load_state_dict(raw_gen_state_dict, strict=True)
            loaded_prequantized_generator = True
            prequantized_generator_backend = "fouroversix"
            if local_rank == 0:
                print(
                    "[NVFP4] Prepared quantized generator architecture: "
                    f"{len(dropped_modules)} materialized modules, "
                    f"{len(matched_modules)} modules excluded"
                )
        elif config.use_ema:
            missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
            if local_rank == 0:
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
                if len(unexpected) > 0:
                    print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
        else:
            print(f"Loading generator from {generator_ckpt_path}")
            pipeline.generator.load_state_dict(raw_gen_state_dict, strict=True)

pipeline.is_lora_enabled = False
pipeline.is_lora_merged = False

if loaded_prequantized_generator:
    if has_lora_adapter or merge_lora or getattr(config, "lora_ckpt", None):
        if local_rank == 0:
            print("[NVFP4] Pre-quantized generator checkpoint is already saved with merged weights; skipping LoRA setup")
    has_lora_adapter = False
    merge_lora = False
    config.merge_lora = False

if getattr(config, "model_quant", False) and not merge_lora and not loaded_prequantized_generator:
    pipeline.generator.model, materialize_quantized_weights_for_inference = quantize_generator_model(
        pipeline.generator.model,
        config=config,
        keep_master_weights=has_lora_adapter,
    )

if has_lora_adapter:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
        if merge_lora:
            print("LoRA weights will be merged into the base model before inference")
    # Apply LoRA to the generator transformer after loading base weights.
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights from lora_ckpt. If omitted, fall back to generator_ckpt
    # when it directly contains generator_lora.
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA weights from lora_ckpt: {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    elif generator_lora_state is not None:
        if local_rank == 0:
            print(f"Loading LoRA weights from generator_ckpt: {generator_ckpt_path}")
        peft.set_peft_model_state_dict(pipeline.generator.model, generator_lora_state)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint configured; using initialized LoRA adapters")

    if merge_lora:
        if local_rank == 0:
            print("Merging LoRA weights into generator before quantization / inference...")
        pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
        pipeline.is_lora_merged = True
    else:
        pipeline.is_lora_enabled = True
elif merge_lora and local_rank == 0:
    print("merge_lora=True requested but no adapter config was found; continuing without LoRA merge")

del generator_checkpoint


# Move pipeline to appropriate dtype and device
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
        )
        stage_desc = "after LoRA merge" if pipeline.is_lora_merged else "for inference"
    else:
        stage_desc = "after LoRA wrapping" if pipeline.is_lora_enabled else "for inference"
    materialize_quantized_generator(
        pipeline.generator.model,
        device=device,
        materialize_fn=materialize_quantized_weights_for_inference,
        stage_desc=stage_desc,
    )
elif loaded_prequantized_generator and local_rank == 0:
    print(f"[NVFP4] Using pre-saved {prequantized_generator_backend} generator weights from checkpoint")

pipeline.generator.model.eval().requires_grad_(False)
configure_generator_torch_compile(pipeline, config)

vae_device_str = getattr(config, "vae_device", None)
use_dedicated_vae_device = bool(getattr(config, "streaming_vae", False)) and bool(vae_device_str)
if use_dedicated_vae_device:
    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    if local_rank == 0:
        print(f"[inference] VAE on {vae_device}, diffusion on {device}")
else:
    pipeline.vae.to(device=device)
    if vae_device_str and local_rank == 0:
        print(f"[inference] Ignoring vae_device={vae_device_str} because streaming_vae is false")

# Create dataset
nfpb = getattr(config, 'num_frame_per_block', 8)
num_blocks = config.num_output_frames // nfpb

data_path = config.data_path
chunks_per_shot = getattr(config, 'chunks_per_shot', 0)
scene_cut_prefix = getattr(config, 'scene_cut_prefix', "The scene transitions. ")
dataset = MultiTextConcatDataset(
    data_path=data_path,
    num_blocks=num_blocks,
    chunks_per_shot=chunks_per_shot,
    scene_cut_prefix=scene_cut_prefix,
    deterministic=True,
)
collate_fn = eval_collate_fn
if local_rank == 0:
    print(f"[data] data_path={data_path}, mode={dataset._mode}, num_blocks={num_blocks}")
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0,
                        drop_last=False, collate_fn=collate_fn)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []

    # MultiTextConcatDataset + eval_collate_fn: prompts[0] is List[str].
    block_prompts = list(batch['prompts'][0])
    prompt = block_prompts[0]  # for filename
    prompts = [block_prompts] * config.num_samples

    shape = config.image_or_video_shape
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, shape[2], shape[3], shape[4]], device=device, dtype=torch.bfloat16
    )
    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)
    print('sampled_noise.shape', sampled_noise.shape, 'prompts', prompts)
    save_latents_only = section_get(
        config,
        "inference",
        "save_latents_only",
        getattr(config, "save_latents_only", getattr(config, "save_latent_only", False)),
        aliases=("save_latent_only", "return_latents"),
    )
    inference_kwargs = dict(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=save_latents_only,
    )
    generated = pipeline.inference(**inference_kwargs)

    if not save_latents_only:
        current_video = rearrange(generated, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()
    else:
        latents = generated

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                base_name = f'rank{rank}-{idx}-{seed_idx}_{model_type}'
            else:
                base_name = f'rank{rank}-{prompt[:100]}-{seed_idx}_{model_type}'

            if save_latents_only:
                latent_path = os.path.join(config.output_folder, f'{base_name}.pt')
                torch.save(latents[seed_idx].cpu(), latent_path)
            else:
                output_path = os.path.join(config.output_folder, f'{base_name}.mp4')
                fps = 24 if '5B' in config.model_kwargs.model_name else 16
                write_video(output_path, video[seed_idx], fps=fps)

            prompt_txt_path = os.path.join(config.output_folder, f'{base_name}_prompts.txt')
            save_prompts_to_txt(
                prompts[seed_idx] if isinstance(prompts[seed_idx], list) else [prompts[seed_idx]],
                prompt_txt_path,
                is_main_process=(rank == 0),
            )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
