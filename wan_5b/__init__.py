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
