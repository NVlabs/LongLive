import functools
import math

import torch

"""
Credit: TransformerEngine
https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/nvfp4_tensor.py
"""

HADAMARD_DIMENSION = 16


def get_no_random_sign_vector(device: int) -> torch.Tensor:
    """Non-random sign vector for Hadamard transform."""
    return torch.tensor([1], dtype=torch.float32, device=device)


def get_wgrad_sign_vector(device: int) -> torch.Tensor:
    """
    Hard-coded random signs for Hadamard transform.

    https://xkcd.com/221/

    """
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=device,
    )


def get_hadamard_matrix(hadamard_dimension: int, device: int) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix."""

    if hadamard_dimension != HADAMARD_DIMENSION:
        msg = f"Only hadamard dimension {HADAMARD_DIMENSION} is supported."
        raise ValueError(msg)

    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        * hadamard_scale
    )


@functools.cache
def get_rht_matrix(
    *,
    with_random_sign_mask: bool = True,
    device: str | int = "cuda",
) -> torch.Tensor:
    """Construct matrix used in random Hadamard transform."""
    if with_random_sign_mask:
        signs = get_wgrad_sign_vector(device=device)
    else:
        signs = get_no_random_sign_vector(device=device)
    sign_matrix = signs * torch.eye(
        HADAMARD_DIMENSION,
        dtype=torch.float32,
        device=device,
    )
    rht_matrix = sign_matrix @ get_hadamard_matrix(HADAMARD_DIMENSION, device=device)
    return rht_matrix.to(dtype=torch.bfloat16)


def to_blocked(a: torch.Tensor) -> torch.Tensor:
    return (
        a.view(a.shape[0] // 128, 128, a.shape[1] // 4, 4)
        .transpose(1, 2)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )
