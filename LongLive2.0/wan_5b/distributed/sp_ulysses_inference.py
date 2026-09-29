# Copyright 2024-2025 LongLive Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ulysses-style sequence-parallel primitives for Wan2.2-TI2V-5B inference.
"""

from typing import Optional
import time

import torch
import torch.distributed as dist


_SP_GROUP: Optional[dist.ProcessGroup] = None
_SP_WORLD_SIZE: int = 1
_SP_RANK: int = 0

_SP_COMM_STATS = {
    "all_gather_time": 0.0,
    "all_gather_count": 0,
    "all_gather_bytes": 0,
    "all_to_all_time": 0.0,
    "all_to_all_count": 0,
    "all_to_all_bytes": 0,
    "barrier_time": 0.0,
    "barrier_count": 0,
}
_SP_PROFILING_ENABLED: bool = False


def init_sequence_parallel(
    world_size: int | None = None,
    rank: int | None = None,
    group: Optional[dist.ProcessGroup] = None,
):
    """Initialize sequence-parallel state for the current process."""
    global _SP_GROUP, _SP_WORLD_SIZE, _SP_RANK
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before init_sequence_parallel")
    if group is not None:
        _SP_GROUP = group
        _SP_WORLD_SIZE = dist.get_world_size(group) if world_size is None else world_size
        _SP_RANK = dist.get_rank(group) if rank is None else rank
    else:
        _SP_WORLD_SIZE = world_size if world_size is not None else dist.get_world_size()
        _SP_RANK = rank if rank is not None else dist.get_rank()
        _SP_GROUP = dist.group.WORLD
    if _SP_RANK == 0:
        print(f"[SP-Ulysses-5B] Initialized: world_size={_SP_WORLD_SIZE}, rank={_SP_RANK}")
    return _SP_GROUP


def get_sp_group() -> dist.ProcessGroup:
    if _SP_GROUP is None:
        raise RuntimeError("SP group not initialized. Call init_sequence_parallel first.")
    return _SP_GROUP


def get_sp_world_size() -> int:
    return _SP_WORLD_SIZE


def get_sp_rank() -> int:
    return _SP_RANK


def is_sp_enabled() -> bool:
    return _SP_WORLD_SIZE > 1


def enable_sp_profiling():
    global _SP_PROFILING_ENABLED
    _SP_PROFILING_ENABLED = True
    reset_sp_comm_stats()


def disable_sp_profiling():
    global _SP_PROFILING_ENABLED
    _SP_PROFILING_ENABLED = False


def reset_sp_comm_stats():
    global _SP_COMM_STATS
    _SP_COMM_STATS = {
        key: 0.0 if "time" in key or "bytes" in key else 0
        for key in _SP_COMM_STATS
    }


def get_sp_comm_stats():
    return _SP_COMM_STATS.copy()


def sp_all_gather(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if not is_sp_enabled():
        return tensor
    global _SP_COMM_STATS, _SP_PROFILING_ENABLED
    world_size = get_sp_world_size()
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    if _SP_PROFILING_ENABLED:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    dist.all_gather(tensor_list, tensor, group=get_sp_group())
    if _SP_PROFILING_ENABLED:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        _SP_COMM_STATS["all_gather_time"] += elapsed
        _SP_COMM_STATS["all_gather_count"] += 1
        _SP_COMM_STATS["all_gather_bytes"] += (
            tensor.numel() * tensor.element_size() * (world_size - 1)
        )
    return torch.cat(tensor_list, dim=dim)


def sp_scatter(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if not is_sp_enabled():
        return tensor
    chunks = torch.chunk(tensor, get_sp_world_size(), dim=dim)
    return chunks[get_sp_rank()].contiguous()


def sp_all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    if not is_sp_enabled():
        return tensor
    global _SP_COMM_STATS, _SP_PROFILING_ENABLED
    world_size = get_sp_world_size()
    if _SP_PROFILING_ENABLED:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    scatter_chunks = [
        chunk.contiguous() for chunk in torch.chunk(tensor, world_size, dim=scatter_dim)
    ]
    recv_chunks = [torch.empty_like(scatter_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(recv_chunks, scatter_chunks, group=get_sp_group())
    output = torch.cat(recv_chunks, dim=gather_dim)
    if _SP_PROFILING_ENABLED:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        _SP_COMM_STATS["all_to_all_time"] += elapsed
        _SP_COMM_STATS["all_to_all_count"] += 1
        _SP_COMM_STATS["all_to_all_bytes"] += (
            scatter_chunks[0].numel() * tensor.element_size() * (world_size - 1) * 2
        )
    return output


def ulysses_seq_to_head(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, S_local, N, D] to [B, S_total, N_local, D]."""
    return sp_all_to_all(tensor, scatter_dim=2, gather_dim=1) if is_sp_enabled() else tensor


def ulysses_head_to_seq(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, S_total, N_local, D] to [B, S_local, N, D]."""
    return sp_all_to_all(tensor, scatter_dim=1, gather_dim=2) if is_sp_enabled() else tensor


def sp_barrier():
    if not is_sp_enabled():
        return
    global _SP_COMM_STATS, _SP_PROFILING_ENABLED
    if _SP_PROFILING_ENABLED:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
    dist.barrier(group=get_sp_group())
    if _SP_PROFILING_ENABLED:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        _SP_COMM_STATS["barrier_time"] += elapsed
        _SP_COMM_STATS["barrier_count"] += 1


def sp_print(msg: str, rank_only: int = 0):
    if get_sp_rank() == rank_only:
        print(f"[SP-5B Rank {get_sp_rank()}] {msg}")


def profile_sp_communication():
    if not is_sp_enabled():
        return
    rank = get_sp_rank()
    world_size = get_sp_world_size()
    test_size = (1, 880, 24, 128)
    test_tensor = torch.randn(test_size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        _ = sp_all_gather(test_tensor, dim=1)
        _ = ulysses_seq_to_head(test_tensor)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        _ = sp_all_gather(test_tensor, dim=1)
        torch.cuda.synchronize()
    all_gather_time = (time.perf_counter() - start) / 10 * 1000
    start = time.perf_counter()
    for _ in range(10):
        _ = ulysses_seq_to_head(test_tensor)
        torch.cuda.synchronize()
    all_to_all_time = (time.perf_counter() - start) / 10 * 1000
    if rank == 0:
        all_gather_bw = (
            test_tensor.numel() * test_tensor.element_size() * (world_size - 1) / 1e9
        ) / (all_gather_time / 1000)
        all_to_all_bw = (
            test_tensor.numel() * test_tensor.element_size() * (world_size - 1) / 1e9
        ) / (all_to_all_time / 1000)
        print("\n[SP-Ulysses-5B Profile]")
        print(f"  World Size: {world_size}")
        print(f"  Test Shape: {test_size}")
        print(f"  All-Gather: {all_gather_time:.2f} ms, Bandwidth: {all_gather_bw:.2f} GB/s")
        print(f"  All-to-All: {all_to_all_time:.2f} ms, Bandwidth: {all_to_all_bw:.2f} GB/s")
