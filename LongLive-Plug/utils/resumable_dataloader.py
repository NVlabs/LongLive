"""Deterministic, checkpointable iteration over a distributed DataLoader.

PyTorch's :class:`DistributedSampler` only changes its shuffle when
``set_epoch`` is called.  Re-wrapping a DataLoader in an infinite ``cycle``
therefore repeats epoch zero forever and a restarted job begins at the first
batch again.  This wrapper owns the epoch and next-batch cursor so a training
checkpoint can resume the exact distributed data order.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

class ResumableDistributedDataLoader(Iterator):
    """Turn a finite distributed DataLoader into a resumable infinite stream.

    ``state_dict`` always describes the *next* batch to return.  A dedicated
    DataLoader generator is reseeded per epoch, keeping worker seeding
    deterministic without consuming the model's global PyTorch RNG state.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        dataloader,
        sampler,
        *,
        sampler_seed: int,
        data_id: Optional[str] = None,
    ) -> None:
        if not hasattr(sampler, "set_epoch"):
            raise TypeError("A sampler with set_epoch() is required")
        if len(dataloader) <= 0:
            raise ValueError(
                "The distributed DataLoader has no complete batches; increase "
                "the dataset size or disable drop_last."
            )

        self.dataloader = dataloader
        self.sampler = sampler
        self.sampler_seed = int(sampler_seed)
        self.data_id = data_id
        self.epoch = 0
        self.batch_in_epoch = 0
        self.batches_consumed = 0
        self._iterator = None

    def __iter__(self):
        return self

    def prepare(self) -> None:
        """Materialize workers and skip to the saved cursor without yielding."""
        if self._iterator is None:
            self._start_epoch()

    def _contract(self) -> Dict[str, Any]:
        return {
            "dataset_size": len(self.dataloader.dataset),
            "batches_per_epoch": len(self.dataloader),
            "batch_size": self.dataloader.batch_size,
            "num_replicas": int(getattr(self.sampler, "num_replicas", 1)),
            "drop_last": bool(getattr(self.sampler, "drop_last", False)),
            "dataset_type": (
                f"{type(self.dataloader.dataset).__module__}."
                f"{type(self.dataloader.dataset).__qualname__}"
            ),
            "sampler_type": (
                f"{type(self.sampler).__module__}."
                f"{type(self.sampler).__qualname__}"
            ),
            "num_workers": int(self.dataloader.num_workers),
            "prefetch_factor": self.dataloader.prefetch_factor,
            "data_id": self.data_id,
        }

    def _start_epoch(self) -> None:
        self.sampler.seed = self.sampler_seed
        self.sampler.set_epoch(self.epoch)

        loader_generator = getattr(self.dataloader, "generator", None)
        if loader_generator is not None:
            loader_generator.manual_seed(self.sampler_seed + self.epoch)

        iterator = iter(self.dataloader)
        for _ in range(self.batch_in_epoch):
            try:
                next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(
                    "Saved data cursor exceeds the current DataLoader length"
                ) from exc
        self._iterator = iterator

    def __next__(self):
        self.prepare()

        try:
            batch = next(self._iterator)
        except StopIteration:  # pragma: no cover - canonical state avoids this
            self.epoch += 1
            self.batch_in_epoch = 0
            self._start_epoch()
            batch = next(self._iterator)

        self.batch_in_epoch += 1
        self.batches_consumed += 1
        if self.batch_in_epoch == len(self.dataloader):
            # Canonicalize an epoch boundary as the start of the next epoch.
            self.epoch += 1
            self.batch_in_epoch = 0
            self._iterator = None
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "batches_consumed": self.batches_consumed,
            "sampler_seed": self.sampler_seed,
            "contract": self._contract(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != self.FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported resumable DataLoader state version: "
                f"{state.get('format_version')}"
            )

        saved_contract = state.get("contract")
        current_contract = self._contract()
        if saved_contract != current_contract:
            raise RuntimeError(
                "Cannot exactly resume because the data contract changed: "
                f"saved={saved_contract}, current={current_contract}"
            )

        epoch = int(state["epoch"])
        batch_in_epoch = int(state["batch_in_epoch"])
        batches_consumed = int(state["batches_consumed"])
        if epoch < 0 or not 0 <= batch_in_epoch < len(self.dataloader):
            raise RuntimeError(
                f"Invalid saved data cursor: epoch={epoch}, "
                f"batch_in_epoch={batch_in_epoch}"
            )
        expected_consumed = epoch * len(self.dataloader) + batch_in_epoch
        if batches_consumed != expected_consumed:
            raise RuntimeError(
                "Inconsistent saved data cursor: "
                f"batches_consumed={batches_consumed}, expected={expected_consumed}"
            )

        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        self.batches_consumed = batches_consumed
        # The checkpoint is authoritative when a run originally used seed=0
        # and the launcher generated a fresh random seed before loading it.
        self.sampler_seed = int(state["sampler_seed"])
        self._iterator = None
