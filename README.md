<p align="center" style="border-radius: 10px">
  <img src="assets/LongLive-logo.png" width="100%" alt="logo"/>
</p>

# 🎬 LongLive: Real-time Interactive Long Video Generation

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2509.22622)
[![Code](https://img.shields.io/badge/GitHub-LongLive-blue)](https://github.com/NVlabs/LongLive)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=CO1QC7BNvig)
[![Demo](https://img.shields.io/badge/Demo-Page-bron)](https://nvlabs.github.io/LongLive)

<div align="center">

[![Watch the video](assets/video-first-frame.png)](https://www.youtube.com/watch?v=CO1QC7BNvig)
[![Watch the video](assets/Comparison_with_Sora2.png)](https://x.com/yukangchen_/status/1973405662177529993)

</div>

## 💡 TLDR: Turn interactive prompts into long videos—instantly, as you type!

**LongLive: Real-time Interactive Long Video Generation [[Paper](https://arxiv.org/abs/2509.22622)]** <br />
[Shuai Yang](https://andysonys.github.io/), [Wei Huang](https://aaron-weihuang.com/), [Ruihang Chu](https://ruihang-chu.github.io/), [Yicheng Xiao](https://easonxiao-888.github.io/), [Yuyang Zhao](https://yuyangzhao.com/), [Xianbang Wang](https://peppaking8.github.io/), [Muyang Li](https://lmxyy.me/), [Enze Xie](https://xieenze.github.io/), [Yingcong Chen](https://www.yingcong.me/), [Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en), [Song Han](http://songhan.mit.edu/), [Yukang Chen](https://yukangchen.com/) <br />

We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss.

## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Introduction](#introduction)
4. [Installation](#installation)
5. [Inference](#inference)
6. [Training](#training)
7. [How to contribute](#how-to-contribute)
8. [Citation](#citation)
9. [License](#license)
10. [Acknowledgement](#acknowledgement)

## News
- [x] [2025.10.11] Many thanks to @yondonfu for building an interactive UI based on LongLive. Please check it here: https://github.com/daydreamlive/scope
- [x] [2025.10.1] We compare Sora2 (+ GPT-5 prompt engineering) with LongLive-1.3B in the interactive long video generation. See [here](https://x.com/yukangchen_/status/1973405662177529993) for details.
- [x] [2025.9.30] We release [example prompts](https://github.com/NVlabs/LongLive/tree/main/example) to reproduce our demo videos.
- [x] [2025.9.29] We release [Paper](https://arxiv.org/abs/2509.22622), this GitHub repo [LongLive](https://github.com/NVlabs/LongLive) with all training and inference code, the model weight [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B), and demo page [Website](https://nvlabs.github.io/LongLive).

## Highlights
1. **Long Video Gen**: LongLive supports up to 240s video generation, with visual consistency.
2. **Real-time Inference**: LongLive supports 20.7 FPS generation speed on a single H100 GPU, and 24.8 FPS with FP8 quantization with marginal quality loss.
3. **Efficient Fine-tuning**: LongLive extends a short-clip model to minute-long generation in 32 H100 GPU-days.

## Introduction
<p align="center" style="border-radius: 10px">
  <img src="assets/pipeline.jpg" width="100%" alt="logo"/>
<strong>LongLive accepts sequential user prompts and generates corresponding videos in real time, enabling user-guided long video generation.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/framework.png" width="100%" alt="logo"/>
<strong>The framework of LongLive. (Left) Frame Sink + Short window attention. (Right) KV-recache.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/streaming_long.jpg" width="100%" alt="logo"/>
<strong>The streaming long tuning pipeline. Our approach trains on long sequences by reusing the historical KV cache each iteration to generate the next 5s clip, then supervising it with the teacher.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/frame_sink.png" width="100%" alt="logo"/>
<strong>The effectiveness of Frame Sink.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/effects-KV-recache.png" width="100%" alt="logo"/>
<strong>The effectiveness of KV re-cache. Consistent transitions with new-prompt compliance.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/demo.png" width="100%" alt="logo"/>
<strong>Interactive 60s videos with 6 prompts. See our demo <a href="https://nvlabs.github.io/LongLive"><strong>Website</strong></a> for video examples.</strong>
</p>


## Installation
**Requirements**

We tested this repo on the following setup:
* Nvidia GPU with at least 40 GB memory (A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

**Environment**

Create a conda environment and install dependencies:
```
git clone https://github.com/NVlabs/LongLive
cd LongLive
conda create -n longlive python=3.10 -y
conda activate longlive
conda install nvidia/label/cuda-12.4.1::cuda
conda install -c nvidia/label/cuda-12.4.1 cudatoolkit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Inference
**Download checkpoints**

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models
```

**Single Prompt Video Generation**
```
bash inference.sh
```
**Interactive Long Video Generation**
```
bash interactive_inference.sh
```
**Hints for video prompt**

1. When building interactive prompts, include a brief subject (who/what) and background/setting (where) in every prompt. Re-stating these anchors at each step greatly improves global coherence during prompt switches.
See the `example` for the exact prompt set we used to produce some of our videos on the demo page.

2. LongLive supports diverse interaction—action changes, introducing/removing objects, background shifts, style changes, and more. But during large scene transitions the camera motion cannot be explicitly controlled. In another word, LongLive excels at cinematic long takes, but is less suited to rapid shot-by-shot edits or fast cutscenes.

## Training
**Download checkpoints**

Please follow [Self-Forcing](https://github.com/guandeh17/Self-Forcing) to download text prompts and ODE initialized checkpoint.

Download Wan2.1-T2V-14B as the teacher model.

```
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
```

**Step1: Self-Forcing Initialization for Short Window and Frame Sink**
```
bash train_init.sh
```
**Step2: Streaming Long Tuning**
```
bash train_long.sh
```

## How to contribute
- Make sure to have git installed.
- Create your own [fork](https://github.com/NVlabs/LongLive/fork) of the project.
- Clone the repository on your local machine, using git clone and pasting the url of this project.
- Read both the `Requirements` and `Installation and Quick Guide` sections below.
- Commit and push your changes.
- Make a pull request when finished modifying the project.


## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.
```bibtex
@article{yang2025longlive,
      title={LongLive: Real-time Interactive Long Video Generation},
      author={Shuai Yang and Wei Huang and Ruihang Chu and Yicheng Xiao and Yuyang Zhao and Xianbang Wang and Muyang Li and Enze Xie and Yingcong Chen and Yao Lu and Song Hanand Yukang Chen},
      year={2025},
      eprint={2509.22622},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
- LongLive code is under CC-BY-NC-SA 4.0 license.
- LongLive-1.3B model weight is under CC-BY-NC 4.0 license.

## Acknowledgement
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): the codebase and algorithm we built upon. Thanks for their wonderful work.
- [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon. Thanks for their wonderful work.
