#!/usr/bin/env python
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""Map a LongLive stage-1 AR checkpoint (CausalWanModel EMA, `model.`-prefixed) into a
bare LongWAMBase `WanVideoDiT` state_dict, and verify it loads.

The two share byte-for-byte submodule names (patch_embedding / text_embedding /
time_embedding / time_projection / blocks.* / head); the only difference is the leading
`model.` wrapper prefix. We strip it and save a plain state_dict that the LongWAMBase video
expert can `load_state_dict(strict=False)`.

Usage:
  python tools/weights/map_longlive_ar_to_video_expert.py \
    --src checkpoints/stage1/model_ema.pt \
    --out checkpoints/initialization/longlive_ar_video_expert.pt
"""

import argparse
import torch

# Geometry of the Wan2.2-TI2V-5B video DiT (see configs/model/longwam_base.yaml).
VIDEO_DIT_CONFIG = dict(
    has_image_input=False,
    patch_size=[1, 2, 2],
    in_dim=48,
    hidden_dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    out_dim=48,
    num_heads=24,
    attn_head_dim=128,
    num_layers=30,
    eps=1.0e-06,
    seperated_timestep=True,
    require_clip_embedding=False,
    require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True,
    use_gradient_checkpointing=False,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="model.")
    args = ap.parse_args()

    print(f"[map] loading {args.src}")
    sd = torch.load(args.src, map_location="cpu", mmap=True, weights_only=False)
    assert isinstance(sd, dict), type(sd)
    n_pref = sum(k.startswith(args.prefix) for k in sd)
    print(f"[map] {len(sd)} tensors, {n_pref} start with {args.prefix!r}")
    assert n_pref == len(sd), "not all keys carry the expected prefix"

    out_sd = {k[len(args.prefix) :]: v for k, v in sd.items()}
    nparam = sum(v.numel() for v in out_sd.values())
    print(f"[map] stripped -> {len(out_sd)} keys, {nparam / 1e9:.3f}B params")

    # Verify it loads into a fresh WanVideoDiT.
    from longwam.models.wan22.wan_video_dit import WanVideoDiT

    dit = WanVideoDiT(**VIDEO_DIT_CONFIG)
    res = dit.load_state_dict(out_sd, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("freqs")]  # freqs is a buffer
    unexpected = list(res.unexpected_keys)
    print(
        f"[map] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print("  missing (first 20):", missing[:20])
    if unexpected:
        print("  unexpected (first 20):", unexpected[:20])

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_sd, args.out)
    print(f"[map] saved -> {args.out}")
    ok = len(missing) == 0
    print(
        f"[map] RESULT: {'OK (all video-expert params covered)' if ok else 'INCOMPLETE — see missing keys'}"
    )


if __name__ == "__main__":
    main()
