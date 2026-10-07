# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/quantize/cuda/ops.py
# Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/quantize/cuda/ops.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import torch
import fouroversix._C  # noqa: F401

def quantize_to_fp4(
        x: torch.Tensor,
        is_nvfp4: bool,  # noqa: FBT001
        is_rtn: bool,  # noqa: FBT001
        is_rht: bool,  # noqa: FBT001
        is_2d: bool,  # noqa: FBT001
        is_transpose: bool,  # noqa: FBT001
        selection_rule: int,
        rbits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return torch.ops.fouroversix.quantize_to_fp4.default(
            x,
            is_nvfp4,
            is_rtn,
            is_rht,
            is_2d,
            is_transpose,
            selection_rule,
            rbits,
        )

@torch.library.register_fake("fouroversix::quantize_to_fp4")
def _(
        x: torch.Tensor,
        is_nvfp4: bool,  # noqa: FBT001
        is_rtn: bool,  # noqa: ARG001, FBT001
        is_rht: bool,  # noqa: ARG001, FBT001
        is_2d: bool,  # noqa: ARG001, FBT001
        is_transpose: bool,  # noqa: ARG001, FBT001
        selection_rule: int,  # noqa: ARG001
        rbits: int,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        return (
            torch.empty(x.shape[0], x.shape[1] // 2, dtype=torch.uint8, device=x.device),
            torch.empty(
                x.shape[0] * x.shape[1] // (16 if is_nvfp4 else 32),
                dtype=torch.float8_e4m3fn if is_nvfp4 else torch.float8_e8m0fnu,
                device=x.device,
            ),
            torch.empty(1, dtype=torch.float32, device=x.device),
        )