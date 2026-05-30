import torch


def _get_i2v_context_frames(
    image_or_video: torch.Tensor,
    initial_latent: torch.Tensor | None,
) -> int:
    if initial_latent is None:
        return 0
    if image_or_video.ndim != initial_latent.ndim:
        raise ValueError(
            f"initial_latent rank {initial_latent.ndim} must match "
            f"image/video rank {image_or_video.ndim}."
        )
    if image_or_video.shape[0] != initial_latent.shape[0]:
        raise ValueError(
            f"initial_latent batch {initial_latent.shape[0]} must match "
            f"image/video batch {image_or_video.shape[0]}."
        )
    if image_or_video.shape[2:] != initial_latent.shape[2:]:
        raise ValueError(
            f"initial_latent shape after frames {tuple(initial_latent.shape[2:])} "
            f"must match image/video {tuple(image_or_video.shape[2:])}."
        )

    context_frames = int(initial_latent.shape[1])
    if context_frames <= 0:
        return 0
    if context_frames >= image_or_video.shape[1]:
        raise ValueError(
            f"initial_latent has {context_frames} frames but clip has "
            f"{image_or_video.shape[1]} frames."
        )
    return context_frames


def _overwrite_i2v_context(
    image_or_video: torch.Tensor,
    initial_latent: torch.Tensor | None,
    context_frames: int,
) -> torch.Tensor:
    if context_frames <= 0:
        return image_or_video
    output = image_or_video.clone()
    output[:, :context_frames] = initial_latent[:, :context_frames].to(
        device=output.device,
        dtype=output.dtype,
    )
    return output


def _zero_i2v_context_timestep(
    timestep: torch.Tensor,
    context_frames: int,
) -> torch.Tensor:
    if context_frames <= 0:
        return timestep
    output = timestep.clone()
    output[:, :context_frames] = 0
    return output


def _i2v_loss_mask_like(
    image_or_video: torch.Tensor,
    context_frames: int,
) -> torch.Tensor | None:
    if context_frames <= 0:
        return None
    mask = torch.ones_like(image_or_video, dtype=torch.bool)
    mask[:, :context_frames] = False
    return mask
