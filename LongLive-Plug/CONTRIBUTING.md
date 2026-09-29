# Contributing to LongLive-Plug

Keep changes focused on the four non-AR recipes in `configs/`. Preserve upstream copyright and license notices. Do not commit model weights, training data, caches, credentials, logs or generated videos.

Install `requirements-dev.txt`, then run `python -m pytest tests`. Changes to distributed training, adapter loading, sampling or checkpoint formats also require GPU validation: check the two backbones, both objectives, generator/critic gradient separation, and complete optimizer/RNG/data resume. State the tested GPU architecture, precision and world size in the pull request.

Configuration-only changes should explain their effect on global batch, learning rates, guidance convention, update counts and sampling settings. Keep README examples runnable and asset paths portable. Do not claim visual or benchmark improvements from loss values alone.
