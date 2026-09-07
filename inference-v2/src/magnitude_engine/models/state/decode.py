"""Prepare bounded append outputs while the retained KV arena stays read-only."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx

from ..execution import ExecutionScope
from .pages import SequencePages
from .tail import TailImage, split_tail


@dataclass
class DecodeKV:
    """Tensor state for one resident step; allocation and commitment stay outside the trace."""

    page_size: int
    table_width: int
    positions: mx.array
    offsets: mx.array
    pages: mx.array
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    tails: list[mx.array]
    starts: mx.array

    def append(self, layer: int, keys: mx.array, values: mx.array) -> None:
        self.tails[layer] = update_tail(
            self.tails[layer], keys, values, self.offsets, self.page_size
        )

    def tail(self, layer: int) -> tuple[mx.array, mx.array, mx.array]:
        keys, values = split_tail(
            self.tails[layer],
            self.positions.size,
            self.keys[layer].shape[0],
            self.page_size,
            self.keys[layer].shape[-1],
            self.values[layer].shape[-1],
        )
        return keys, values, self.starts


@cache
def _append_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_append_tail",
        input_names=["previous", "keys", "values", "offsets"],
        output_names=["output"],
        source="""
        uint i = thread_position_in_grid.x;
        constexpr uint NK = B * H * C * K;
        if (i >= B * H * C * (K + V)) return;
        bool key = i < NK;
        uint j = key ? i : i - NK;
        uint width = key ? K : V;
        uint channel = j % width;
        uint position = (j / width) % C;
        uint head = (j / width / C) % H;
        uint row = j / width / C / H;
        int token = int(position) - offsets[row];
        if (token >= 0 && token < N) {
            uint source = ((row * H + head) * N + uint(token)) * width + channel;
            output[i] = key ? keys[source] : values[source];
        } else {
            output[i] = previous[i];
        }
    """,
    )


def update_tail(
    buffer: mx.array, new_keys: mx.array, new_values: mx.array, offsets: mx.array, capacity: int
) -> mx.array:
    """One bounded write over contiguous K and V regions of the same allocation."""
    batch, heads, count, key_width = new_keys.shape
    value_width = new_values.shape[-1]
    return _append_kernel()(
        inputs=[buffer, new_keys, new_values, offsets],
        template=[
            ("B", batch),
            ("H", heads),
            ("C", capacity),
            ("K", key_width),
            ("V", value_width),
            ("N", count),
        ],
        grid=(buffer.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[buffer.shape],
        output_dtypes=[buffer.dtype],
    )[0]


class PreparedDecodeAppend:
    def __init__(self, states: tuple[SequencePages, ...], scope: ExecutionScope):
        if not states or len(set(states)) != len(states):
            raise ValueError("decode append requires distinct nonempty sequence states")
        store = states[0].store
        if any(state.store is not store for state in states):
            raise ValueError("decode append requires one physical state store")
        arena = self._arena = store.arena
        if arena._closed or not arena._pins:
            raise RuntimeError("decode append requires a live execution pin")
        self._states = states
        self._starts = tuple(state.length for state in states)
        for state, start in zip(states, self._starts, strict=True):
            for layer in range(len(arena.layers)):
                state._validate_write(layer, start, start + 1)
        self.page_size = arena.page_size
        self.capacity = arena.allocator.capacity * self.page_size
        self.table = store.table(states)
        self.positions = mx.array(self._starts, mx.int32)
        self.keys, self.values = arena.keys, arena.values
        self._installed = False
        tails = []
        for state in states:
            if state.tail is None:
                image = TailImage(arena, 1)
                try:
                    image.zeros()
                    state.tail = image.acquire(0, state.length)
                except BaseException:
                    image.close()
                    raise
            tail = state.tail
            if state.length + 1 > tail.start + tail.image.capacity:
                raise RuntimeError("append capacity must be prepared before execution")
            scope.acquire(tail.acquire)
            tails.append(tail)
        self.tail_starts = mx.array([tail.start for tail in tails], mx.int32)
        self.offsets = mx.array(
            [s.length - t.start for s, t in zip(states, tails, strict=True)], mx.int32
        )
        image = tails[0].image
        if image.width == len(tails) and all(
            t.image is image and t.index == i for i, t in enumerate(tails)
        ):
            self.tails = image.buffers
        else:
            size = len(tails) * image.capacity * arena.page_bytes // arena.page_size
            scope.acquire(lambda: arena.budget.reserve("kv.append-batch", size))
            self.tails = tuple(
                mx.concatenate(
                    [
                        mx.stack([t.layer(i)[0] for t in tails]).reshape(-1),
                        mx.stack([t.layer(i)[1] for t in tails]).reshape(-1),
                    ]
                )
                for i in range(len(arena.layers))
            )
        self._tails = tuple(tails)
        self._output = TailImage(arena, len(states))
        # A construction lease retires partially built outputs on execution failure.
        scope.acquire(lambda: self._output.acquire(0, states[0].length))

    def padded_table(self) -> tuple[int, mx.array]:
        """Bucket compiled page-map geometry at the append allocation boundary."""
        # Launch regimes depend on visible history; allocation slabs and append
        # sizes must not change the mathematical attention horizon.
        width = 1
        while width < self.table.width:
            width *= 2
        return width, mx.pad(
            self.table.device, [(0, 0), (0, width - self.table.width)], constant_values=-1
        )

    def install(self, buffers: tuple[mx.array, ...]) -> None:
        arena = self._arena
        if self._installed:
            raise RuntimeError("decode append is already installed")
        if arena._closed or not arena._pins:
            raise RuntimeError("decode append requires a live execution pin")
        if arena.keys is not self.keys or arena.values is not self.values:
            raise RuntimeError("decode append input buffers were superseded")
        for state, start, tail in zip(self._states, self._starts, self._tails, strict=True):
            if state.length != start or state.tail is not tail:
                raise RuntimeError("decode append boundary changed")
            for layer in range(len(arena.layers)):
                state._validate_write(layer, start, start + 1)
        self._output.install(buffers)
        for row, (state, start, tail) in enumerate(
            zip(self._states, self._starts, self._tails, strict=True)
        ):
            state.tail = self._output.acquire(row, tail.start)
            tail.close()
            state._written = [start + 1] * len(arena.layers)
        arena.counters["tail_appended_tokens"] += len(self._states) * len(arena.layers)
        self._installed = True


def prepare_decode_append(
    states: tuple[SequencePages, ...], scope: ExecutionScope
) -> PreparedDecodeAppend:
    return PreparedDecodeAppend(states, scope)
