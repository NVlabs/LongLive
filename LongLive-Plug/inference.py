"""Generate full-sequence videos with a LongLive-Plug generator LoRA."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import torch
from omegaconf import OmegaConf
from utils.config import normalize_config, wan_default_config
from utils.lora_utils import configure_lora_for_model, load_lora_checkpoint
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


@torch.no_grad()
def sample(model, noise, cond, uncond, steps, guidance):
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=noise.device, shift=model.scheduler.shift)
    latent = noise.clone()
    for timestep in scheduler.timesteps:
        ts = timestep * torch.ones(latent.shape[:2], device=latent.device, dtype=torch.float32)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            flow, _ = model(noisy_image_or_video=latent, conditional_dict=cond, timestep=ts)
            if guidance != 1:
                negative, _ = model(noisy_image_or_video=latent, conditional_dict=uncond, timestep=ts)
                flow = negative.float() + guidance * (flow.float() - negative.float())
            else:
                flow = flow.float()
        latent = scheduler.step(flow, timestep, latent, return_dict=False)[0]
        if not torch.isfinite(latent).all():
            raise FloatingPointError('Nonfinite sample')
    return latent


def load_adapter(model, path, config):
    model.model = configure_lora_for_model(model.model, 'generator', config.adapter,
                                           is_main_process=False)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(str(path))
    else:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        weights = payload.get('generator_lora', payload)
    load_lora_checkpoint(model.model, weights, 'generator', is_main_process=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', type=Path, help='Generator LoRA .safetensors or model.pt; omit for the base model')
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/videos'))
    p.add_argument('--model-dir')
    p.add_argument('--seed', type=int, default=20261000)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--steps', type=int)
    p.add_argument('--guidance', type=float, default=1.)
    p.add_argument('--width', type=int)
    p.add_argument('--height', type=int)
    p.add_argument('--frames', type=int)
    p.add_argument('--save-latents', action='store_true')
    args = p.parse_args()
    cfg = normalize_config(OmegaConf.load(args.config))
    if args.model_dir:
        cfg.model_kwargs.model_dir = args.model_dir
    rank, world = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', rank))
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    device = torch.device('cuda', rank)
    defaults = wan_default_config[cfg.model_kwargs.model_name]
    width, height = args.width or defaults['resolution'][0], args.height or defaults['resolution'][1]
    frames = args.frames or (81 if 'Wan2.1' in cfg.model_kwargs.model_name else (121 if cfg.distribution_loss == 'cfg_guidance' else 125))
    stride = defaults['spatial_compression_ratio']
    if (frames-1) % 4 or width % (stride*2) or height % (stride*2):
        p.error('Frames must be 4n+1; width/height must be divisible by 2 × VAE spatial stride')
    prompt_list = [args.prompt] if args.prompt else [s.strip() for s in args.prompt_file.read_text().splitlines() if s.strip()][:args.count]
    mine = [(i, text) for i, text in enumerate(prompt_list) if i % world == global_rank]
    if not mine:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    text = WanTextEncoder(model_name=cfg.model_kwargs.model_name, model_dir=cfg.model_kwargs.model_dir).to(device=device, dtype=torch.bfloat16)
    conditions = [(i, prompt, text(text_prompts=[prompt])) for i, prompt in mine]
    uncond = text(text_prompts=[cfg.negative_prompt]) if args.guidance != 1 else None
    del text
    gc.collect(); torch.cuda.empty_cache()
    model = WanDiffusionWrapper(**dict(cfg.model_kwargs), is_causal=False).to(device=device, dtype=torch.bfloat16)
    if args.checkpoint:
        load_adapter(model, args.checkpoint, cfg)
    model.eval().requires_grad_(False)
    shape = [1, (frames-1)//4+1, defaults['latent_channels'], height//stride, width//stride]
    generated = []
    for i, prompt, cond in conditions:
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed+i), dtype=torch.float32)
        noise_sha = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
        latent = sample(model, noise.to(device), cond, uncond, args.steps or cfg.sampling_steps, args.guidance).cpu()
        generated.append((i, latent))
        metadata = {'prompt': prompt, 'seed': args.seed+i, 'noise_sha256': noise_sha, 'shape': shape,
                    'steps': args.steps or cfg.sampling_steps, 'guidance': args.guidance, 'scheduler': 'UniPC',
                    'checkpoint': str(args.checkpoint) if args.checkpoint else None}
        (args.output/f'case_{i:02d}.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if args.save_latents:
            torch.save(latent, args.output/f'case_{i:02d}.pt')
    del model, conditions, uncond
    gc.collect(); torch.cuda.empty_cache()
    vae = WanVAEWrapper(model_name=cfg.model_kwargs.model_name, model_dir=cfg.model_kwargs.model_dir).to(device=device, dtype=torch.float32)
    for i, latent in generated:
        video = vae.decode_to_pixel(latent.to(device))
        if not torch.isfinite(video).all():
            raise FloatingPointError('Nonfinite decoded video')
        pixels = ((video[0].float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
        imageio.mimwrite(args.output/f'case_{i:02d}.mp4', pixels, fps=defaults['fps'], codec='libx264', quality=8, macro_block_size=1)
        print(f'Saved case_{i:02d}.mp4', flush=True)


if __name__ == '__main__':
    main()
