# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/datasets/dataset_utils.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image


def obtain_image_size(data: torch.Tensor | Image.Image) -> tuple[int, int]:
    r"""Return spatial size from a PIL image or image/video tensor.

    Args:
        data (torch.Tensor | Image.Image): Input image or video tensor.
    Returns:
        width (int): Input width.
        height (int): Input height.
    """

    if isinstance(data, Image.Image):
        width, height = data.size
    elif isinstance(data, torch.Tensor):
        height, width = data.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


class ResizeSmallestSideAspectPreserving:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        r"""Resize while preserving aspect ratio.

        The output is scaled so both spatial dimensions are at least the
        requested target size.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor | Image.Image: Resized image or video tensor.
        """

        assert self.args is not None, "Please specify args in augmentations"

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(video)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] >= img_h and target_size[1] >= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {(img_w, img_h)} compute {target_size}"
        )

        return transforms_F.resize(
            video,
            size=target_size,  # type: ignore
            interpolation=self.args.get("interpolation", transforms_F.InterpolationMode.BICUBIC),
            antialias=True,
        )


class CenterCrop:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        r"""Center crop to the requested spatial size.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor | Image.Image: Center cropped image or video tensor.
        """
        assert (self.args is not None) and ("img_w" in self.args) and ("img_h" in self.args), (
            "Please specify size in args"
        )

        img_w, img_h = self.args["img_w"], self.args["img_h"]
        return transforms_F.center_crop(video, [img_h, img_w])


class Normalize:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor:
        r"""Convert to tensor if needed and normalize by mean/std.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor: Normalized image or video tensor.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        if isinstance(video, torch.Tensor):
            data = video.to(dtype=torch.float32)
            if video.dtype == torch.uint8:
                data = data / 255.0
            data = data.to(dtype=torch.get_default_dtype())
        else:
            data = transforms_F.to_tensor(video)  # division by 255 is applied in to_tensor()

        return transforms_F.normalize(tensor=data, mean=mean, std=std)
