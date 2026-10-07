# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: wan_5b/configs/wan_t2v_A14B.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/wan_5b/configs/wan_t2v_A14B.py
# Source: https://github.com/Wan-Video/Wan2.2
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan T2V A14B ------------------------#

t2v_A14B = EasyDict(__name__='Config: Wan T2V A14B')
t2v_A14B.update(wan_shared_cfg)

# t5
t2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_A14B.vae_stride = (4, 8, 8)

# transformer
t2v_A14B.patch_size = (1, 2, 2)
t2v_A14B.dim = 5120
t2v_A14B.ffn_dim = 13824
t2v_A14B.freq_dim = 256
t2v_A14B.num_heads = 40
t2v_A14B.num_layers = 40
t2v_A14B.window_size = (-1, -1)
t2v_A14B.qk_norm = True
t2v_A14B.cross_attn_norm = True
t2v_A14B.eps = 1e-6
t2v_A14B.low_noise_checkpoint = 'low_noise_model'
t2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
t2v_A14B.sample_shift = 12.0
t2v_A14B.sample_steps = 40
t2v_A14B.boundary = 0.875
t2v_A14B.sample_guide_scale = (3.0, 4.0)  # low noise, high noise
