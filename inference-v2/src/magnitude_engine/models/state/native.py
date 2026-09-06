"""Library cache ownership with trim or committed-copy/replay reconciliation.

Cache implementation knowledge stays here. Generation sees the same prefix-commit
contract as paged execution. A construction-supplied capacity estimator accounts
for the selected family's cache geometry and allocation quantum before forward.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    ConcatenateKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_vlm.models.cache import ArraysCache as VLMArrayCache
from mlx_vlm.models.cache import KVCache as VLMKVCache
from mlx_vlm.models.cache import RotatingKVCache as VLMRotatingKVCache

from magnitude_engine.resources.budget import MemoryBudget, Reservation
from magnitude_engine.resources.retention import RetainedStorage

from ..inputs import ModelInputs
from .native_batch import BatchLease, DenseBatch, DenseKV

Cache = Any
_APPEND_ONLY = (KVCache, ConcatenateKVCache, QuantizedKVCache, ChunkedKVCache, VLMKVCache)
_SUPPORTED = (
    *_APPEND_ONLY,
    ArraysCache,
    RotatingKVCache,
    CacheList,
    VLMArrayCache,
    VLMRotatingKVCache,
)


def _arrays(value: Any) -> Iterator[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _arrays(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _arrays(child)
    elif type(value) in _SUPPORTED:
        yield from _arrays(vars(value))


def _detach(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_detach(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if type(value) in _SUPPORTED:
        result = copy(value)
        for key, child in vars(value).items():
            setattr(result, key, _detach(child))
        return result
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported library cache field: {type(value).__name__}")


def _can_trim(cache: Cache) -> bool:
    if type(cache) in _APPEND_ONLY:
        return True
    return type(cache) is CacheList and all(_can_trim(c) for c in cache.caches)


class CacheImage:
    """Detached cache graph with reserved storage, materialized only when consumed.

    MLX array copies preserve the pre-update value even when cache wrappers mutate.
    A fully accepted transaction discards its rollback image without submitting it.
    Persistent checkpoints and rollback explicitly complete the image before transfer.
    """

    def __init__(self, budget: MemoryBudget, caches: list[Cache]):
        self.reservation = budget.reserve(
            "library-state-image", sum(a.nbytes for a in _arrays(caches))
        )
        self.caches: list[Cache] = []
        try:
            self.caches = _detach(caches)
        except BaseException:
            self.close()
            raise

    def materialize(self) -> None:
        mx.eval(*_arrays(self.caches))

    def close(self) -> None:
        self.caches.clear()
        self.reservation.close()


@dataclass(eq=False)
class LibraryState:
    store: LibraryStateStore
    caches: list[Cache]
    position: int = 0
    capacity_bytes: int = 0
    allocated_bytes: int = 0
    charges: list[Reservation] = field(default_factory=list)
    closed: bool = False
    active: LibraryTransaction | None = None
    batch: DenseBatch | None = None
    staged_growth: Reservation | None = None


class LibraryCheckpoint:
    def __init__(self, store: LibraryStateStore, state: LibraryState):
        self.store = store
        self.length = state.position
        self.image = CacheImage(store.budget, state.caches)
        try:
            self.image.materialize()
        except BaseException:
            self.image.close()
            raise
        self.closed = False

    @property
    def reclaimable(self) -> bool:
        return True

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        return (RetainedStorage(self.image, self.image.reservation.size),)

    def close(self) -> None:
        if not self.closed:
            self.image.close()
            self.closed = True


class LibraryTransaction:
    def __init__(self, state: LibraryState, inputs: ModelInputs, *, committed_inputs: int = 0):
        if type(committed_inputs) is not int or not 0 <= committed_inputs <= inputs.count:
            raise ValueError("committed prefix is outside the state advance")
        self.committed_inputs = committed_inputs
        self.state = state
        self.inputs = inputs
        self.base = state.position
        self.closed = False
        self.reconciled = False
        self.snapshot: CacheImage | None = None
        self.batch_lease: BatchLease | None = None
        self.indices = tuple(i for i, c in enumerate(state.caches) if not _can_trim(c))
        self.growth_peak, state.staged_growth = state.staged_growth, None
        need = state.store.capacity(self.base + inputs.count)
        self.capacity_bytes = need
        if need < 0:
            raise ValueError("cache capacity estimate must be nonnegative")
        growth: Reservation | None = None
        try:
            if state.batch is not None:
                self.batch_lease = state.batch.acquire()
            if state.batch is None and need > state.allocated_bytes:
                self.growth_peak = state.store.budget.reserve(
                    "library-cache-growth", state.allocated_bytes
                )
            if state.batch is None and need > state.capacity_bytes:
                growth = state.store.budget.reserve("library-cache", need - state.capacity_bytes)
            if self.indices and committed_inputs < inputs.count:
                self.snapshot = CacheImage(
                    state.store.budget, [state.caches[i] for i in self.indices]
                )
        except BaseException:
            if growth is not None:
                growth.close()
            self.close()
            raise
        if growth is not None:
            state.charges.append(growth)
            state.capacity_bytes = need
        state.active = self

    def reconcile(self, accepted: int) -> ModelInputs | None:
        if (
            self.closed
            or self.reconciled
            or not self.committed_inputs <= accepted <= self.inputs.count
        ):
            raise ValueError("invalid library cache reconciliation")
        self.reconciled = True
        if accepted == self.inputs.count:
            return None
        replay = bool(self.indices)
        drop = self.inputs.count if replay else self.inputs.count - accepted
        for index, cache in enumerate(self.state.caches):
            if index not in self.indices and cache.trim(drop) != drop:
                raise RuntimeError("library attention cache could not trim the rejected inputs")
        if self.snapshot is not None:
            # Only rejected work consumes the rollback image. Complete its detached
            # pre-forward values before replaying exactly the accepted prefix.
            self.snapshot.materialize()
            for index, cache in zip(self.indices, self.snapshot.caches, strict=True):
                self.state.caches[index] = cache
        return self.inputs.prefix(accepted) if replay and accepted else None

    def finish(self, accepted: int) -> None:
        if not self.reconciled:
            raise RuntimeError("library transaction must be reconciled before commit")
        self.state.position = self.base + accepted
        self.state.allocated_bytes = max(self.state.allocated_bytes, self.capacity_bytes)
        if self.state.active is self:
            self.state.active = None

    def commit_all(self) -> None:
        """Publish the whole causal library advance without evaluating its arrays."""
        if self.closed or self.reconciled:
            raise RuntimeError("library transaction is already resolved")
        self.reconciled = True
        self.finish(self.inputs.count)

    def close(self) -> None:
        if self.closed:
            return
        if self.snapshot is not None:
            self.snapshot.close()
        if self.growth_peak is not None:
            self.growth_peak.close()
        if self.batch_lease is not None:
            self.batch_lease.close()
        if self.state.active is self:
            self.state.active = None
        self.closed = True


class LibraryStateStore:
    def __init__(
        self,
        make_cache: Callable[[], list[Cache]],
        budget: MemoryBudget,
        capacity: Callable[[int], int],
    ):
        self.make_cache = make_cache
        self.budget = budget
        self.capacity = capacity
        self._batches: list[DenseBatch] = []

    def _check(self, state: LibraryState) -> None:
        if state.store is not self or state.closed:
            raise ValueError("library state is closed or belongs to another model")

    def create(self, checkpoint: LibraryCheckpoint | None = None) -> LibraryState:
        if checkpoint is not None and (checkpoint.closed or checkpoint.store is not self):
            raise ValueError("library checkpoint is closed or belongs to another model")
        caches = self.make_cache()
        if any(type(c) not in _SUPPORTED for c in caches):
            raise TypeError("cache implementation requires an explicit state adapter")
        state = LibraryState(self, caches)
        if checkpoint is not None:
            if tuple(map(type, caches)) != tuple(map(type, checkpoint.image.caches)):
                raise ValueError("checkpoint cache layout differs from the model")
            need = max(
                self.capacity(checkpoint.length),
                sum(a.nbytes for a in _arrays(checkpoint.image.caches)),
            )
            reservation = self.budget.reserve("library-cache", need)
            try:
                state.caches = _detach(checkpoint.image.caches)
                mx.eval(*_arrays(state.caches))
            except BaseException:
                reservation.close()
                raise
            state.charges.append(reservation)
            state.position = checkpoint.length
            state.capacity_bytes = need
            state.allocated_bytes = need
        return state

    def reserve(self, state: LibraryState, input_capacity: int) -> None:
        # Reserve the declared continuation before admission or chained execution.
        # Storage is still allocated lazily. Replacement peaks use allocated_bytes,
        # independently of this prepaid capacity, and are charged by begin().
        self._check(state)
        if state.active:
            raise RuntimeError("capacity preparation requires an idle library state")
        need = self.capacity(state.position + input_capacity)
        if need < 0:
            raise ValueError("cache capacity estimate must be nonnegative")
        if state.batch is None and state.position == 0 and not state.charges:
            for batch in self._batches:
                if need <= batch.capacity and batch.attach(state):
                    return
        if state.batch is not None:
            state.batch.reserve(need)
            return
        if need > state.capacity_bytes:
            state.charges.append(self.budget.reserve("library-cache", need - state.capacity_bytes))
            state.capacity_bytes = need

    def prepare_batch(self, states: tuple[LibraryState, ...], width: int) -> None:
        for state in states:
            self._check(state)
        if len(states) == 1 and states[0].batch is None:
            return
        batch = states[0].batch
        if batch is None or any(state.batch is not batch for state in states):
            batch = DenseBatch(states)
        else:
            live = tuple(state for state in batch.states if state.batch is batch)
            if len(live) < batch.width and all(state.active is None for state in live):
                # Compact only on real departures, never on query-width splits.
                # If peers are resolving transactions they still own the cohort.
                try:
                    batch = DenseBatch(live)
                except MemoryError:
                    pass  # Existing storage remains valid if compaction cannot fit its peak.
        end = max(state.position for state in states) + width
        batch.reserve(self.capacity(end))
        if any(isinstance(layer, DenseKV) and layer.keys is not None
               and end > layer.keys.shape[2] for layer in batch.layers):
            states[0].staged_growth = self.budget.reserve(
                "native-batch-growth",
                sum(a.nbytes for layer in batch.layers for a in layer.arrays()),
            )

    def can_batch(self, states: tuple[LibraryState, ...]) -> bool:
        return DenseBatch.supports(states)

    def repair_group(self, state: LibraryState) -> object:
        return state if state.batch is None else state.batch

    def batch_caches(self, states: tuple[LibraryState, ...]) -> list[Cache]:
        batch = states[0].batch
        if batch is None or any(state.batch is not batch for state in states):
            raise RuntimeError("native batch state was not prepared")
        return batch.view(states)

    def publish_batch(self, states: tuple[LibraryState, ...]) -> None:
        if states[0].batch is not None:
            states[0].batch.publish()

    def begin(
        self, state: LibraryState, inputs: ModelInputs, *, committed_inputs: int = 0
    ) -> LibraryTransaction:
        self._check(state)
        if state.active:
            raise RuntimeError("library state already has a pending forward")
        return LibraryTransaction(state, inputs, committed_inputs=committed_inputs)

    def arrays(self, state: LibraryState) -> tuple[mx.array, ...]:
        self._check(state)
        return tuple(_arrays(state.caches))

    def layer_arrays(self, state: LibraryState, index: int) -> tuple[mx.array, ...]:
        self._check(state)
        return tuple(_arrays(state.caches[index]))

    def rewind(self, state: LibraryState, position: int) -> None:
        self._check(state)
        if state.active or not 0 <= position <= state.position:
            raise ValueError("rewind requires an idle state and an earlier position")
        if not all(_can_trim(c) for c in state.caches):
            raise ValueError("recurrent library state requires checkpoint restore, not trim")
        drop = state.position - position
        for cache in state.caches:
            if cache.trim(drop) != drop:
                raise RuntimeError("library state could not reach the rewind boundary")
        state.position = position

    def checkpoint(self, state: LibraryState) -> LibraryCheckpoint:
        self._check(state)
        if state.active:
            raise RuntimeError("cannot checkpoint uncommitted library state")
        return LibraryCheckpoint(self, state)

    def release(self, state: LibraryState) -> None:
        self._check(state)
        if state.active:
            raise RuntimeError("complete library work before releasing state")
        state.caches.clear()
        if state.batch is not None:
            state.batch.remove(state)
        for charge in state.charges:
            charge.close()
        state.charges.clear()
        state.closed = True
