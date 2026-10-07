# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: wan_5b/__init__.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/wan_5b/__init__.py
# Source: https://github.com/Wan-Video/Wan2.2
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

__all__ = ["WanI2V", "WanT2V", "WanTI2V"]


def __getattr__(name):
    if name == "WanI2V":
        from .image2video import WanI2V
        return WanI2V
    if name == "WanT2V":
        from .text2video import WanT2V
        return WanT2V
    if name == "WanTI2V":
        from .textimage2video import WanTI2V
        return WanTI2V
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
