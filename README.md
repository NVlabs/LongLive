<p align="center" style="border-radius: 10px">
  <img src="assets/longlive2/logo.png" width="100%" alt="LongLive2.0 logo"/>
</p>

# 🎬 LongLive 2.0: An NVFP4 Parallel Infrastructure for Long Video Generation

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2605.18739)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/NVlabs/LongLive)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=7oQALy32fiU)
[![Models](https://img.shields.io/badge/Model-BF16-yellow)](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B)
[![Models](https://img.shields.io/badge/Model-NVFP4-orange)](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4)
[![Demo](https://img.shields.io/badge/Demo-Page-brightgreen)](https://nvlabs.github.io/LongLive/LongLive2/)
[![Docs](https://img.shields.io/badge/Full-Documentation-green)](https://nvlabs.github.io/LongLive/LongLive2/docs/)

<div align="center">

<!-- TODO: replace this text block with the final project-page video/demo embed. -->

[![Watch the video](assets/longlive2/first-video-frame.png)](https://www.youtube.com/watch?v=7oQALy32fiU)

</div>

## 💡 TLDR: Infra with NVFP4 and parallelism for both training and inference

<p align="center" style="border-radius: 10px">
  <img src="assets/longlive2/teaser.jpg" width="100%" alt="LongLive2.0 teaser"/>
</p>

## News
- 🔥 [2026.05.13] We release **LongLive 2.0**, infra with NVFP4, parallelism and multi-shot for AR training, DMD distillation, and inference (⚡45.7 FPS). The original LongLive 1.0 is now in the [v1.0](https://github.com/NVlabs/LongLive/tree/v1.0) branch.
- 🔥 [2026.04.12] LongLive supports kv cache compression with [TriAttention](https://github.com/WeianMao/triattention), with 50% KV reduction and no quality drop. Check it [here](https://github.com/WeianMao/triattention/tree/main/longlive)
- 🎉 [2026.1.27] LongLive is accepted by **ICLR-2026**.
- 🔥 [2026.1.11] LongLive supports adapting LongLive's original RoPE into KV-cache relative RoPE and generates infinite long videos!
- 🔥 [2025.11.3] We implement LongLive on linear attention model [SANA-Video](https://nvlabs.github.io/Sana/Video/)! Now SANA-Video can generate 60s interactive videos in real-time.
- 🔥 [2025.9.29] We release [Paper](https://arxiv.org/abs/2509.22622), this GitHub repo [LongLive](https://github.com/NVlabs/LongLive) with all training and inference code, the model weight [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B), and demo page [Website](https://nvlabs.github.io/LongLive).

## Introduction

**LongLive 1.0**: Real-time Interactive Long Video Generation. [You can find it here](https://github.com/NVlabs/LongLive/tree/v1.0) in our V1.0 branch.

**LongLive 2.0**: an NVFP4 Parallel Infrastructure for Long Video Generation
- For training, it supports
  - [x] Balanced sequence parallel for AR training (teacher-forcing).
  - [x] AR training on multi-shot (or single-shot) videos. 
  - [x] NVFP4 (or BF16) for both AR training and few-step distillation.
- For inference, it supports
  - [x] NVFP4 inference (W4A4) and NVFP4 KV Cache.
  - [x] Multi-shot attention sink.
  - [x] Sequence parallel inference.
  - [x] Async decoding.


<p align="left" style="border-radius: 10px">
  <img src="assets/longlive2/fig_framework_overview.png" width="80%" alt="LongLive2.0 framework overview"/>
</p>


**LongLive 1.0**: Real-time Interactive Long Video Generation. It accepts sequential user prompts and generates corresponding videos in real time, enabling user-guided long video generation. The key insights are attention sink, KV-recache, and streaming long tuning. 


<p align="left" style="border-radius: 10px">
  <img src="assets/longlive2/LongLive1_teaser.png" width="80%" alt="LongLive1.0 framework overview"/>
</p>


## Getting Started
- [Full Documentation](https://nvlabs.github.io/LongLive/LongLive2/docs/)
- [Installation](https://nvlabs.github.io/LongLive/LongLive2/docs/#installation)
- [NVFP4 Setup](https://nvlabs.github.io/LongLive/LongLive2/docs/#nvfp4-installation)
- [Training](https://nvlabs.github.io/LongLive/LongLive2/docs/#training)
- [Inference](https://nvlabs.github.io/LongLive/LongLive2/docs/#inference)
- [Data Organization](https://nvlabs.github.io/LongLive/LongLive2/docs/#training-data)

### Quick Start

#### BF16

```python
import torch
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import (
    load_generator_checkpoint,
    place_vae_for_streaming,
    prepare_single_prompt_inputs,
    save_video,
)

prompt = "A compact silver robot walks through a clean robotics lab."
merged_checkpoint_path = "LongLive-2.0-5B/model_bf16.pt"

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
device = torch.device("cuda")

torch.set_grad_enabled(False)
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)  # honor streaming_vae + vae_device when set
pipe.generator.model.eval().requires_grad_(False)

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
video = pipe.inference(noise=noise, text_prompts=prompts)
save_video(video[0], "videos/quickstart/sample.mp4", fps=24)
```

`place_vae_for_streaming` is a no-op unless `inference.streaming_vae` is true and `inference.vae_device` is set, so toggling streaming-pipeline decode in your yaml is enough — the script does not need to change.

#### NVFP4

Point `checkpoints.generator_ckpt` in `configs/nvfp4/inference_nvfp4.yaml` at the downloaded checkpoint and set `model_quant_use_transformer_engine` according to the backend you are using:

- TransformerEngine checkpoint (`model_te.pt`): `model_quant_use_transformer_engine: true`
- FourOverSix checkpoint (`model_4o6.pt`): `model_quant_use_transformer_engine: false`

`setup_nvfp4_pipeline` handles checkpoint loading, NVFP4 module wrapping, weight materialization, dtype/device placement, and the streaming-pipeline VAE relocation for both backends — the bf16 `pipe.to(...)` shortcut is unsafe here because it would cast the quantized buffers.

```python
import torch
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import prepare_single_prompt_inputs, save_video, setup_nvfp4_pipeline

prompt = "A compact silver robot walks through a clean robotics lab."

config = normalize_config(OmegaConf.load("configs/nvfp4/inference_nvfp4.yaml"))
device = torch.device("cuda")

torch.set_grad_enabled(False)
pipe = CausalDiffusionInferencePipeline(config, device=device)
setup_nvfp4_pipeline(pipe, config, device)
pipe.generator.model.eval().requires_grad_(False)

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
video = pipe.inference(noise=noise, text_prompts=prompts)
save_video(video[0], "videos/quickstart/sample_nvfp4.mp4", fps=24)
```

## Models

| Model | FPS ↑ | Params | VBench ↑ | Multi-shot |
| --- | ---: | ---: | ---: | :---: |
| [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B) | 20.7 | 1.3B | 84.87 |  |
| [LongLive-2.0-5B](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B) | 24.8 | 5B | 85.06 | ✅ |
| [LongLive-2.0-5B-NVFP4-4Step](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4) | 29.7 | 5B | 84.51 | ✅ |
| [LongLive-2.0-5B-NVFP4-2Step](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2) | 45.7 | 5B | 83.14 | ✅ |

## License
This repository is released under the Apache 2.0 license. See [LICENSE](LICENSE) for details.

## Citation
Please consider citing our work if you find them useful:

```bibtex
@article{longlive_2.0,
  title={LongLive2.0: An NVFP4 Parallel Infrastructure for Long Video Generation},
  author={Chen, Yukang and Wang, Luozhou and Huang, Wei and Yang, Shuai and Zhang, Bohan and Xiao, Yicheng and Chu, Ruihang and Mao, Weian and Hu, Qixin and Liu, Shaoteng and Zhao, Yuyang and Mao, Huizi and Chen, Ying-Cong and Xie, Enze and Qi, Xiaojuan and Han, Song},
  journal={arXiv preprint arXiv},
  year={2026}
}
```

```bibtex
@inproceedings{longlive,
    title={Longlive: Real-time interactive long video generation}, 
    author={Yang, Shuai and Huang, Wei and Chu, Ruihang and Xiao, Yicheng and Zhao, Yuyang and Wang, Xianbang and Li, Muyang and Xie, Enze and Chen, Yingcong and Lu, Yao and others},
    booktitle={ICLR},
    year={2026},
}
```

## Acknowledgement
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): the AR training codebase and formulation we build upon.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2): the base video diffusion model components used in this release.
