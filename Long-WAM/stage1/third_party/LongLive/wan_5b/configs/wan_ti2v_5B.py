# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: wan_5b/configs/wan_ti2v_5B.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/wan_5b/configs/wan_ti2v_5B.py
# Source: https://github.com/Wan-Video/Wan2.2
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan TI2V 5B ------------------------#

ti2v_5B = EasyDict(__name__='Config: Wan TI2V 5B')
ti2v_5B.update(wan_shared_cfg)

# t5
ti2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
ti2v_5B.t5_tokenizer = 'google/umt5-xxl'

# vae
ti2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
ti2v_5B.vae_stride = (4, 16, 16)

# transformer
ti2v_5B.patch_size = (1, 2, 2)
ti2v_5B.dim = 3072
ti2v_5B.ffn_dim = 14336
ti2v_5B.freq_dim = 256
ti2v_5B.num_heads = 24
ti2v_5B.num_layers = 30
ti2v_5B.window_size = (-1, -1)
ti2v_5B.qk_norm = True
ti2v_5B.cross_attn_norm = True
ti2v_5B.eps = 1e-6

# inference
ti2v_5B.sample_fps = 24
ti2v_5B.sample_shift = 5.0
ti2v_5B.sample_steps = 50
ti2v_5B.sample_guide_scale = 5.0
ti2v_5B.frame_num = 121
