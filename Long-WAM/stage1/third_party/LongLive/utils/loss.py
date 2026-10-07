# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: utils/loss.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/utils/loss.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

from abc import ABC, abstractmethod
import torch


class DenoisingLoss(ABC):
    @abstractmethod
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Base class for denoising loss.
        Input:
            - x: the clean data with shape [B, F, C, H, W]
            - x_pred: the predicted clean data with shape [B, F, C, H, W]
            - noise: the noise with shape [B, F, C, H, W]
            - noise_pred: the predicted noise with shape [B, F, C, H, W]
            - alphas_cumprod: the cumulative product of alphas (defining the noise schedule) with shape [T]
            - timestep: the current timestep with shape [B, F]
        """
        pass


class X0PredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        err = (x - x_pred) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


class VPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        weights = 1 / (1 - alphas_cumprod[timestep].reshape(*timestep.shape, 1, 1, 1))
        err = weights * (x - x_pred) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


class NoisePredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        err = (noise - noise_pred) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


class FlowPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        gradient_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        err = (kwargs["flow_pred"] - (noise - x)) ** 2
        if gradient_mask is not None:
            return err[gradient_mask].mean()
        return err.mean()


NAME_TO_CLASS = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss
}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]
