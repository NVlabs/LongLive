"""Create the 64-prompt CFG teacher cache using a published training recipe."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from utils.config import normalize_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.data.image_or_video_shape[0] = 1
    normalized = normalize_config(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    if normalized.distribution_loss != 'cfg_guidance':
        p.error('Teacher caching applies only to CFG-only recipes')
    world = int(os.environ.get('WORLD_SIZE', 1))
    if 64 % world:
        p.error('World size must divide the 64 cache prompts')
    output = Path(normalized.data_path).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', 0))
    runtime = output/f'cache_config_rank_{rank}.yaml'
    tmp = runtime.with_suffix('.tmp')
    OmegaConf.save(cfg, tmp, resolve=True)
    tmp.replace(runtime)
    model = Path(normalized.model_kwargs.model_dir).resolve()
    data = Path(normalized.eval_data_path).resolve().parent
    sys.argv = ['cache_teacher', '--config-path', str(runtime), '--model-name', normalized.model_kwargs.model_name,
                '--model-dir', str(model), '--model-index', str(model/'diffusion_pytorch_model.safetensors.index.json'),
                '--text-encoder', str(model/'models_t5_umt5-xxl-enc-bf16.pth'),
                '--expected-latent-shape', *map(str, normalized.image_or_video_shape),
                '--expected-world-size', str(world), '--expected-prompt-count', '64',
                '--prompts-per-rank', str(64//world), '--seed', '20260927',
                '--prompt-file', str(data/'cfg_train_64.txt'),
                '--prompt-provenance', str(data/'cfg_train_64.provenance.json'), '--output-dir', str(output)]
    from scripts.cache_cfg_teacher_trajectories import main as cache
    cache()


if __name__ == '__main__':
    main()
