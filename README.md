<p align="center" style="border-radius: 10px">
  <img src="assets/LongLive-logo.png" width="100%" alt="logo"/>
</p>

# 🎬 LongLive: Real-time Interactive Long Video Generation

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2509.22622)
[![Code](https://img.shields.io/badge/GitHub-LongLive-blue)](https://github.com/NVlabs/LongLive)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=CO1QC7BNvig)
[![vs-Sora2](https://img.shields.io/badge/VS-Sora2-red)](https://x.com/yukangchen_/status/1973405662177529993)
[![Docs](https://img.shields.io/badge/Docs-Online-brightgreen)](https://nvlabs.github.io/LongLive/docs)
[![Demo](https://img.shields.io/badge/Demo-Page-bron)](https://nvlabs.github.io/LongLive) 
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVlabs/LongLive)

<div align="center">

[![Watch the video](assets/video-first-frame.png)](https://www.youtube.com/watch?v=CO1QC7BNvig)

</div>

## 💡 TLDR: Turn interactive prompts into long videos—instantly, as you type!

**LongLive: Real-time Interactive Long Video Generation [[Paper](https://arxiv.org/abs/2509.22622)]** <br />
[Shuai Yang](https://andysonys.github.io/), [Wei Huang](https://aaron-weihuang.com/), [Ruihang Chu](https://ruihang-chu.github.io/), [Yicheng Xiao](https://easonxiao-888.github.io/), [Yuyang Zhao](https://yuyangzhao.com/), [Xianbang Wang](https://peppaking8.github.io/), [Muyang Li](https://lmxyy.me/), [Enze Xie](https://xieenze.github.io/), [Yingcong Chen](https://www.yingcong.me/), [Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en), [Song Han](http://songhan.mit.edu/), [Yukang Chen](https://yukangchen.com/) <br />


## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Introduction](#introduction)
4. [How to contribute](#how-to-contribute)
5. [Citation](#citation)
6. [License](#license)
7. [Acknowledgement](#acknowledgement)

## News
- [x] [2026.1.27] **LongLive is accepted by ICLR-2026.** 🎉🎉🎉
- [x] [2026.1.11] Many thanks @qixinhu11 for adapting LongLive's original RoPE into KV-cache relative RoPE. Now LongLive supports generating infinite long videos!
- [x] [2025.12.4] We fix a bug in `global_sink==False` mode. Now our model generate videos in higher quality.
- [x] [2025.11.3] We implement LongLive on linear attention model [SANA-Video](https://nvlabs.github.io/Sana/Video/)! Now SANA-Video can generate 60s interactive videos in real-time.
- [x] [2025.11.1] The license has been changed from CC-BY-NC-SA 4.0 to **Apache 2.0**.
- [x] [2025.10.11] Many thanks to @yondonfu for building an interactive UI based on LongLive. Please check it [here](https://github.com/daydreamlive/scope).
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

Please see our [docs](https://nvlabs.github.io/LongLive/docs) for Installation, Train, and Inference.

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

## Acknowledgement
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): the codebase and algorithm we built upon. Thanks for their wonderful work.
- [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon. Thanks for their wonderful work.
