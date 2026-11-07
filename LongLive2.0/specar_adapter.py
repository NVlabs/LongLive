# SPDX-License-Identifier: Apache-2.0
"""Optional SpecAR decoding for inference.py: the 4-step generator generates `specar.k` blocks spread evenly over the
video and the 2-step generator (the merged-LoRA `model_te.pt`) generates the rest, through one shared KV cache.

Enabled by a `specar:` section in the inference config:

    specar:
      drafter_ckpt: /path/to/LongLive-2.0-5B-NVFP4-S2/model_te.pt   # the 2-step generator, BF16
      k: 2                       # blocks the 4-step generator generates; the rest are drafted
      kv_mode: shared            # shared (default) | replay | window, see the specar package

Without the section nothing changes. The decoding algorithm lives in the `specar` package
(https://github.com/statite-ai/SpecAR, `pip install "specar @ git+https://github.com/statite-ai/SpecAR"`); this file
only loads the drafter and wraps `pipeline.inference` calls. Text-to-video, one sample per call; other requests run
the stock path and the reason is printed.
"""
import contextlib
import types

import torch

STATE = {"enabled": False, "reason": ""}


def setup(pipeline, config, device, verbose=True):
    """Call after the generator is loaded and moved to `device`. Returns the engine, or None (stock path)."""
    section = getattr(config, "specar", None)
    if not section or not getattr(section, "drafter_ckpt", None):
        STATE["reason"] = "no specar section in the config"
        return None
    try:
        from specar.model_longlive import LongLiveSpecAR
    except ImportError:
        STATE["reason"] = "specar package not installed (pip install \"specar @ git+https://github.com/statite-ai/SpecAR\")"
        if verbose:
            print(f"[specar] disabled: {STATE['reason']}")
        return None
    from utils.inference_utils import load_generator_checkpoint
    from utils.wan_5b_wrapper import WanDiffusionWrapper
    drafter = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)
    load_generator_checkpoint(drafter, section.drafter_ckpt)
    drafter = drafter.to(device=device, dtype=torch.bfloat16)
    drafter.model.eval().requires_grad_(False)
    models = types.SimpleNamespace(pipe=pipeline, target=pipeline.generator, drafter=drafter, config=config, device=device)
    engine = LongLiveSpecAR(models, kv_mode=getattr(section, "kv_mode", "shared"))
    engine.k = int(getattr(section, "k", 2))
    STATE.update(enabled=True, reason="")
    if verbose:
        print(f"[specar] enabled: k={engine.k}, kv_mode={engine.kv_mode}, drafter {section.drafter_ckpt}")
    return engine


@contextlib.contextmanager
def maybe(engine, inference_kwargs, num_frame_per_block, verbose=True):
    """Wrap one `pipeline.inference(**inference_kwargs)` call. Falls back to the stock path for image-to-video and
    for more than one sample."""
    if engine is None:
        yield False
        return
    from specar.schedule import uniform_k
    noise = inference_kwargs["noise"]
    if inference_kwargs.get("initial_latent") is not None or noise.shape[0] != 1:
        if verbose:
            print("[specar] stock path for this video (image-to-video or num_samples > 1 are not supported yet)")
        yield False
        return
    num_blocks = noise.shape[1] // num_frame_per_block
    k = min(engine.k, num_blocks)
    with engine.wrap(uniform_k(k, num_blocks), num_blocks):
        yield True
    if verbose:
        bs = engine.block_seconds
        print(f"[specar] {num_blocks} blocks, k={k}: diffusion {sum(bs):.1f} s "
              f"({(num_frame_per_block * num_blocks - 1) * 4 + 1} frames, {((num_frame_per_block * num_blocks - 1) * 4 + 1) / sum(bs):.1f} fps)")
