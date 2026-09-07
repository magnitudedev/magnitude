"""State-owned append preparation around one-input pure tensor execution."""

from __future__ import annotations

import mlx.core as mx

from .pages import SequencePages


class PreparedDecodeAppend:
    """Borrow pinned buffers and writable addresses; install one staged boundary.

    The caller reserves capacity before preparation and keeps its execution pin
    through device completion. Tensor execution returns functional buffer versions;
    this object neither donates old storage nor grants writes to protected prefixes.
    Installation stages KV only. The owning transaction publishes or rejects it.
    """

    def __init__(self, states: tuple[SequencePages, ...]):
        if not states or len(set(states)) != len(states):
            raise ValueError("decode append requires distinct nonempty sequence states")
        store = states[0].store
        if any(state.store is not store for state in states):
            raise ValueError("decode append requires one physical state store")
        self._arena = store.arena
        if self._arena._closed or not self._arena._pins:
            raise RuntimeError("decode append requires a live execution pin")
        self._states = states
        self._starts = tuple(state.length for state in states)
        for state, start in zip(states, self._starts, strict=True):
            for layer in range(len(self._arena.layers)):
                state._validate_write(layer, start, start + 1)
        self.page_size = self._arena.page_size
        self.capacity = self._arena.allocator.capacity * self.page_size
        self.table = store.table(states)
        self.positions = mx.array(self._starts, mx.int32)
        self.destinations = mx.array(
            [
                state.addresses[start // self.page_size] * self.page_size + start % self.page_size
                for state, start in zip(states, self._starts, strict=True)
            ],
            mx.int32,
        )
        self.keys = self._arena.keys
        self.values = self._arena.values
        self._installed = False

    def install(self, keys: tuple[mx.array, ...], values: tuple[mx.array, ...]) -> None:
        """Validate the entire result before assigning buffers or staged horizons."""
        arena = self._arena
        if self._installed:
            raise RuntimeError("decode append is already installed")
        if arena._closed or not arena._pins:
            raise RuntimeError("decode append requires a live execution pin")
        if arena.keys is not self.keys or arena.values is not self.values:
            raise RuntimeError("decode append input buffers were superseded")
        if len(keys) != len(self.keys) or len(values) != len(self.values):
            raise ValueError("decode append output must contain every physical layer")
        for output, original in zip((*keys, *values), (*self.keys, *self.values), strict=True):
            if not isinstance(output, mx.array) or (
                output.shape != original.shape or output.dtype != original.dtype
            ):
                raise ValueError("decode append output differs from reserved physical geometry")
        for state, start in zip(self._states, self._starts, strict=True):
            if state.length != start:
                raise RuntimeError("decode append committed boundary changed")
            for layer in range(len(arena.layers)):
                state._validate_write(layer, start, start + 1)
        arena.keys, arena.values = tuple(keys), tuple(values)
        for state, start in zip(self._states, self._starts, strict=True):
            state._written = [start + 1] * len(arena.layers)
        writes = len(self._states) * len(arena.layers)
        arena.counters["kv_write_runs"] += writes
        arena.counters["kv_written_tokens"] += writes
        self._installed = True


def prepare_decode_append(states: tuple[SequencePages, ...]) -> PreparedDecodeAppend:
    return PreparedDecodeAppend(states)
