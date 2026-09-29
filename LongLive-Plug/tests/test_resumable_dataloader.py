import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from utils.resumable_dataloader import ResumableDistributedDataLoader


class IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def make_stream(rank=0, num_replicas=2, size=24):
    dataset = IndexDataset(size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        seed=17,
        drop_last=True,
    )
    generator = torch.Generator().manual_seed(17)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    return ResumableDistributedDataLoader(
        loader,
        sampler,
        sampler_seed=17,
        data_id="unit-test-prompts",
    )


def take(stream, count):
    return [tuple(next(stream).tolist()) for _ in range(count)]


def test_distributed_epochs_are_disjoint_and_reshuffled():
    rank0 = make_stream(rank=0)
    rank1 = make_stream(rank=1)
    batches_per_epoch = len(rank0.dataloader)

    rank0_epoch0 = take(rank0, batches_per_epoch)
    rank1_epoch0 = take(rank1, batches_per_epoch)
    rank0_epoch1 = take(rank0, batches_per_epoch)

    rank0_indices = {item for batch in rank0_epoch0 for item in batch}
    rank1_indices = {item for batch in rank1_epoch0 for item in batch}
    assert rank0_indices.isdisjoint(rank1_indices)
    assert rank0_indices | rank1_indices == set(range(24))
    assert rank0_epoch1 != rank0_epoch0


def test_resume_returns_the_exact_next_distributed_batches():
    uninterrupted = make_stream(rank=0)
    take(uninterrupted, 8)
    saved_state = uninterrupted.state_dict()
    expected = take(uninterrupted, 10)

    resumed = make_stream(rank=0)
    resumed.load_state_dict(saved_state)

    assert take(resumed, 10) == expected
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_resume_rejects_changed_data_parallel_contract():
    original = make_stream(rank=0, num_replicas=2)
    take(original, 3)

    changed = make_stream(rank=0, num_replicas=3)
    with pytest.raises(RuntimeError, match="data contract changed"):
        changed.load_state_dict(original.state_dict())
