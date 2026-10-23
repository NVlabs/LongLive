# SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM RTX 4090 contribution.

"""Replay original deterministic kernels; randomness stays outside capture.

This helper is an experiment, not a numerical-equivalence guarantee. Every
output is compared with an unmodified reference on real saved observations.
"""

from __future__ import annotations


def source_state(value):
    try:
        version = value._version
    except RuntimeError:
        version = None
    # CUDA replay writes do not increment Tensor._version. The producer marks
    # each graph output with an epoch so downstream graphs copy fresh contents.
    return (value, version, getattr(value, "_wam4090_graph_epoch", None))


def unchanged_source(previous, current):
    return (
        previous is not None
        and current[1] is not None
        and previous[0] is current[0]
        and previous[1:] == current[1:]
    )


def same_expression(value, expected):
    """Recognize only aliases or exact copies staged by this graph helper."""
    visited = set()
    for _ in range(8):
        if id(value) in visited:
            return False
        visited.add(id(value))
        if value.data_ptr() == expected.data_ptr():
            return True
        value = getattr(value, "_wam4090_input_origin", None)
        if value is None:
            return False
    return False


class StaticGraphCallable:
    def __init__(self, fn, *, name: str):
        self.fn = fn
        self.name = name
        self.entries = {}
        self.calls = self.captures = self.replays = self.copied_tensors = 0

    def __call__(self, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten

        if torch.is_grad_enabled():
            raise RuntimeError("Static graph replay is inference-only; enter no_grad")
        precision = (
            torch.is_autocast_enabled("cuda"),
            str(torch.get_autocast_dtype("cuda")),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        )
        flat, spec = tree_flatten((args, kwargs))
        key = (
            precision,
            str(spec),
            tuple(
                (tuple(x.shape), str(x.dtype), str(x.device), tuple(x.stride()))
                if isinstance(x, torch.Tensor)
                else (type(x).__name__, repr(x))
                for x in flat
            ),
        )
        self.calls += 1
        if key not in self.entries:
            if len(self.entries) >= 4:
                raise RuntimeError(
                    f"{self.name}: unexpected shape growth; graph allocation stopped"
                )
            static = []
            for value in flat:
                if isinstance(value, torch.Tensor):
                    if value.device.type != "cuda":
                        raise ValueError(f"{self.name}: CPU tensor cannot be captured")
                    staged = value.detach().clone(memory_format=torch.preserve_format)
                    staged._wam4090_input_origin = value
                    static.append(staged)
                else:
                    static.append(value)
            rebuilt_args, rebuilt_kwargs = tree_unflatten(static, spec)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            # Autocast's Python weight cache must not freeze an FP32 LoRA
            # boundary cast from warmup into the graph. Record the original
            # dtype conversion as a kernel so later adapter updates remain live.
            autocast_enabled = torch.is_autocast_enabled("cuda")
            autocast_dtype = torch.get_autocast_dtype("cuda")
            with (
                torch.cuda.stream(stream),
                torch.autocast(
                    "cuda", enabled=autocast_enabled, dtype=autocast_dtype, cache_enabled=False
                ),
            ):
                for _ in range(2):
                    self.fn(*rebuilt_args, **rebuilt_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with (
                torch.autocast(
                    "cuda", enabled=autocast_enabled, dtype=autocast_dtype, cache_enabled=False
                ),
                torch.cuda.graph(graph),
            ):
                output = self.fn(*rebuilt_args, **rebuilt_kwargs)
            self.entries[key] = {
                "graph": graph,
                "static": static,
                "output": output,
                "sources": [None] * len(flat),
            }
            self.captures += 1
        entry = self.entries[key]
        for index, value in enumerate(flat):
            if not isinstance(value, torch.Tensor):
                continue
            state = source_state(value)
            previous = entry["sources"][index]
            # Retain the source object, not only its address: allocator address
            # reuse must never turn a new observation into a stale cached copy.
            if not unchanged_source(previous, state):
                entry["static"][index].copy_(value)
                entry["sources"][index] = state
                self.copied_tensors += 1
            entry["static"][index]._wam4090_input_origin = value
        entry["graph"].replay()
        self.replays += 1
        for value in tree_flatten(entry["output"])[0]:
            if isinstance(value, torch.Tensor):
                value._wam4090_graph_epoch = (id(self), self.replays)
        return entry["output"]

    def telemetry(self):
        return {
            "calls": self.calls,
            "captures": self.captures,
            "replays": self.replays,
            "copied_tensors": self.copied_tensors,
            "graphs": len(self.entries),
        }
