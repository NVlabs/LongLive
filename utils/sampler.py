import torch


def build_training_sampler(dataset, *, seed, rank=None, num_replicas=None):
    """Build a shuffled sampler with one seed shared by all DP replicas."""
    if (rank is None) != (num_replicas is None):
        raise ValueError("rank and num_replicas must be provided together.")

    sampler_kwargs = {}
    if rank is not None:
        sampler_kwargs.update(rank=rank, num_replicas=num_replicas)

    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
        seed=int(seed),
        **sampler_kwargs,
    )
