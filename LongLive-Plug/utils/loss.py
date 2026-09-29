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


NAME_TO_CLASS = {"flow": FlowPredLoss}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]
