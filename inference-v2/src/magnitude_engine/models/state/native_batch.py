"""Persistent dense KV storage with independent row cursors.

Physical batch storage outlives an execution group. Different proposal widths
can use subsets without rebuilding histories; acceptance only changes row cursors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_vlm.models.cache import ArraysCache as VLMArrayCache
from mlx_vlm.models.cache import KVCache as VLMKVCache

if TYPE_CHECKING:
    from .native import LibraryState


class DenseBatch:
    @staticmethod
    def supports(states: tuple[LibraryState, ...]) -> bool:
        return all(
            type(cache) in (KVCache, VLMKVCache, ArraysCache, VLMArrayCache)
            for state in states
            for cache in state.caches
        )

    def __init__(self, states: tuple[LibraryState, ...]):
        self.store = states[0].store
        self.states = list(states)
        self.width = len(states)
        self.capacity = max(state.capacity_bytes for state in states)
        self.charges = [self.store.budget.reserve("native-batch", self.width * self.capacity)]
        self.layers: list[DenseKV | DenseArrays] = []
        self.closed = False
        self._users = 0
        try:
            for index in range(len(states[0].caches)):
                caches = [state.caches[index] for state in states]
                if all(type(cache) in (KVCache, VLMKVCache) for cache in caches):
                    self.layers.append(DenseKV(self, index, caches))
                elif all(type(cache) in (ArraysCache, VLMArrayCache) for cache in caches):
                    self.layers.append(DenseArrays(self, index, caches))
                else:
                    raise ValueError("native batching requires a qualified cache adapter")
            # Membership changes copy existing state once. No such copy or fence
            # occurs when the same storage serves a subsequent execution group.
            mx.eval(*(a for layer in self.layers for a in layer.arrays()))
        except BaseException:
            self.close()
            raise
        for state in states:
            if state.batch is not None:
                state.batch.remove(state)
            for charge in state.charges:
                charge.close()
            state.charges.clear()
            state.batch = self
        self.store._batches.append(self)
        self.refresh()

    def attach(self, state: LibraryState) -> bool:
        """Reuse a departed row's reserved storage for a new, empty request."""
        index = next((i for i, row in enumerate(self.states) if row.batch is not self), None)
        if index is None:
            return False
        if state.position or state.charges or state.batch is not None:
            raise ValueError("vacant storage requires an empty unallocated state")
        self.states[index] = state
        state.batch = self
        state.capacity_bytes = self.capacity
        self.refresh()
        return True

    def reserve(self, capacity: int) -> None:
        if capacity > self.capacity:
            self.charges.append(
                self.store.budget.reserve("native-batch", self.width * (capacity - self.capacity))
            )
            self.capacity = capacity
        for state in self.states:
            if state.batch is self:
                state.capacity_bytes = self.capacity

    def refresh(self) -> None:
        for layer in self.layers:
            if isinstance(layer, DenseKV):
                layer.refresh()

    def view(self, states: tuple[LibraryState, ...]) -> list[Any]:
        indices = tuple(self.states.index(state) for state in states)
        return [
            KVView(layer, indices) if isinstance(layer, DenseKV) else layer.view(indices)
            for layer in self.layers
        ]

    def publish(self) -> None:
        for layer in self.layers:
            if isinstance(layer, DenseArrays):
                layer.publish()

    def remove(self, state: LibraryState) -> None:
        state.batch = None
        if not any(row.batch is self for row in self.states):
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self in self.store._batches:
            self.store._batches.remove(self)
        self.closed = True
        self._retire()

    def acquire(self) -> BatchLease:
        if self.closed:
            raise RuntimeError("cannot borrow detached native storage")
        self._users += 1
        return BatchLease(self)

    def _retire(self) -> None:
        if self.closed and not self._users:
            self.layers.clear()
            for charge in self.charges:
                charge.close()
            self.charges.clear()


class BatchLease:
    """Logical regrouping does not retire storage used by an earlier execution."""

    def __init__(self, batch: DenseBatch):
        self.batch, self.closed = batch, False

    def close(self) -> None:
        if not self.closed:
            self.batch._users -= 1
            self.batch._retire()
            self.closed = True


class DenseKV:
    def __init__(self, batch: DenseBatch, index: int, caches: list[Any]):
        self.batch, self.index = batch, index
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        populated = [cache for cache in caches if cache.keys is not None]
        if not populated:
            return
        sample = populated[0]
        width = max(cache.keys.shape[2] for cache in populated)
        self.keys = mx.zeros(
            (batch.width, sample.keys.shape[1], width, sample.keys.shape[3]),
            dtype=sample.keys.dtype,
        )
        self.values = mx.zeros(
            (batch.width, sample.values.shape[1], width, sample.values.shape[3]),
            dtype=sample.values.dtype,
        )
        for row, cache in enumerate(caches):
            if cache.keys is not None and cache.offset:
                self.keys[row : row + 1, :, : cache.offset] = cache.keys[:, :, : cache.offset]
                self.values[row : row + 1, :, : cache.offset] = cache.values[:, :, : cache.offset]

    def arrays(self) -> tuple[mx.array, ...]:
        if self.keys is None:
            return ()
        assert self.values is not None
        return self.keys, self.values

    def refresh(self) -> None:
        if self.keys is None:
            return
        assert self.values is not None
        for index, state in enumerate(self.batch.states):
            if state.batch is self.batch:
                cache = state.caches[self.index]
                cache.keys = self.keys[index : index + 1]
                cache.values = self.values[index : index + 1]


class DenseArrays:
    """Recurrent rows share storage but keep independent rollback images."""

    def __init__(self, batch: DenseBatch, index: int, caches: list[Any]):
        self.batch, self.index = batch, index
        self.cache_type = type(caches[0])
        self.values: list[mx.array | None] = []
        self.sources = [list(cache.state) for cache in caches]
        self.active: tuple[tuple[int, ...], Any] | None = None
        for column in zip(*self.sources, strict=True):
            sample = next((value for value in column if value is not None), None)
            self.values.append(
                None
                if sample is None
                else mx.concatenate(
                    [mx.zeros_like(sample) if value is None else value for value in column]
                )
            )

    def arrays(self) -> tuple[mx.array, ...]:
        return tuple(value for value in self.values if value is not None)

    def view(self, indices: tuple[int, ...]):
        if self.active is not None:
            raise RuntimeError("publish recurrent output before opening another view")
        for row in indices:
            current = self.batch.states[row].caches[self.index].state
            for field, value in enumerate(current):
                if value is not self.sources[row][field]:
                    backing = self.values[field]
                    if backing is not None:
                        backing[row : row + 1] = 0 if value is None else value
                    self.sources[row][field] = value
        cache = self.cache_type(len(self.values))
        for field, value in enumerate(self.values):
            if value is not None:
                cache[field] = (
                    value if indices == tuple(range(self.batch.width)) else value[mx.array(indices)]
                )
        self.active = indices, cache
        return cache

    def publish(self) -> None:
        if self.active is None:
            return
        indices, cache = self.active
        self.active = None
        for field, result in enumerate(cache.state):
            if result is None:
                continue
            if indices == tuple(range(self.batch.width)):
                backing = result
            else:
                backing = self.values[field]
                if backing is None:
                    backing = mx.zeros((self.batch.width, *result.shape[1:]), dtype=result.dtype)
                backing[mx.array(indices)] = result
            self.values[field] = backing
            for row in indices:
                value = backing[row : row + 1]
                self.batch.states[row].caches[self.index][field] = value
                self.sources[row][field] = value


class KVView:
    """A selected set of logical rows borrowing persistent dense storage."""

    def __init__(self, layer: DenseKV, indices: tuple[int, ...]):
        self.layer, self.indices = layer, indices
        self.positions = tuple(layer.batch.states[i].caches[layer.index].offset for i in indices)
        self.offset = mx.array(self.positions, dtype=mx.int32)

    def make_mask(self, N: int, *, window_size=None, return_array=False):
        if len(set(self.positions)) == 1 and window_size is None and not return_array:
            return None if N == 1 else "causal"
        queries = self.offset[:, None] + mx.arange(N)[None]
        keys = mx.arange(max(self.positions) + N)
        mask = keys[None, None, :] <= queries[:, :, None]
        if window_size is not None:
            mask &= keys[None, None, :] > queries[:, :, None] - window_size
        return mask[:, None]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if (
            keys.ndim != 4
            or values.ndim != 4
            or keys.shape[:3] != values.shape[:3]
            or keys.shape[0] != len(self.indices)
        ):
            raise ValueError("batched KV updates must align with selected rows")
        layer = self.layer
        end = max(self.positions) + keys.shape[2]
        capacity = ((end + 255) // 256) * 256
        if layer.keys is None:
            layer.keys = mx.zeros(
                (layer.batch.width, keys.shape[1], capacity, keys.shape[3]), dtype=keys.dtype
            )
            layer.values = mx.zeros(
                (layer.batch.width, values.shape[1], capacity, values.shape[3]), dtype=values.dtype
            )
        elif end > layer.keys.shape[2]:
            assert layer.values is not None
            extra = capacity - layer.keys.shape[2]
            layer.keys = mx.concatenate(
                [
                    layer.keys,
                    mx.zeros(
                        (layer.batch.width, keys.shape[1], extra, keys.shape[3]), dtype=keys.dtype
                    ),
                ],
                axis=2,
            )
            layer.values = mx.concatenate(
                [
                    layer.values,
                    mx.zeros(
                        (layer.batch.width, values.shape[1], extra, values.shape[3]),
                        dtype=values.dtype,
                    ),
                ],
                axis=2,
            )
        assert layer.values is not None
        contiguous = self.indices == tuple(
            range(self.indices[0], self.indices[0] + len(self.indices))
        )
        if contiguous and len(set(self.positions)) == 1:
            start = self.positions[0]
            layer.keys[self.indices[0] : self.indices[-1] + 1, :, start:end] = keys
            layer.values[self.indices[0] : self.indices[-1] + 1, :, start:end] = values
        else:
            rows = mx.array(self.indices)[:, None, None]
            positions = self.offset[:, None] + mx.arange(keys.shape[2])[None]
            heads = mx.arange(keys.shape[1])[None, :, None]
            layer.keys[rows, heads, positions[:, None], :] = keys
            layer.values[rows, heads, positions[:, None], :] = values
        for index, position in zip(self.indices, self.positions, strict=True):
            layer.batch.states[index].caches[layer.index].offset = position + keys.shape[2]
        self.offset += keys.shape[2]
        layer.refresh()
        if contiguous:
            return (
                layer.keys[self.indices[0] : self.indices[-1] + 1, :, :end],
                layer.values[self.indices[0] : self.indices[-1] + 1, :, :end],
            )
        selected = mx.array(self.indices)
        return layer.keys[selected, :, :end], layer.values[selected, :, :end]

    def size(self):
        return max(self.positions)
