# Third-party acknowledgments

LongLive-Plug retains code and ideas from these projects. Original copyright and SPDX notices remain in the corresponding source files.

- [Wan](https://github.com/Wan-Video/Wan2.2): diffusion backbones, UMT5, VAEs, attention and flow samplers in `wan_5b/` (Alibaba Wan Team).
- [LongLive](https://github.com/NVlabs/LongLive): training infrastructure and model wrappers (NVIDIA Corporation & Affiliates).
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): distribution-matching objectives and distributed training structure, as identified by source-file notices.
- [PEFT](https://github.com/huggingface/peft), [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), and [PyTorch](https://github.com/pytorch/pytorch) are installed dependencies, not vendored copies.

See `LICENSE` for the code's Apache-2.0 license. Downloaded Wan model weights and VidProM prompt data are separate assets governed by their upstream terms; `docs/assets.json` records their exact repositories and revisions.
