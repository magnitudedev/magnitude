"""Bounded functional KV outputs, independently owned from read-only paged history."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .arena import KVArena


class TailImage:
    """One immutable batch of append buffers; rows and execution leases share its charge."""

    def __init__(self, arena: KVArena, width: int):
        if width < 1:
            raise ValueError("append image requires a positive width")
        self.capacity = arena.page_size
        self.width = width
        self.geometry = arena.layers
        self.dtype = arena.dtype
        self.reservation = arena.budget.reserve(
            "kv.append-tail", width * self.capacity * arena.page_bytes // arena.page_size
        )
        self.buffers: tuple[mx.array, ...] = ()
        self.users = 0
        self.closed = False

    def install(self, buffers: tuple[mx.array, ...]) -> None:
        if self.closed or self.buffers:
            raise RuntimeError("append image is already written or closed")
        if len(buffers) != len(self.geometry):
            raise ValueError("append image must contain every KV producer")
        for geometry, buffer in zip(self.geometry, buffers, strict=True):
            size = (
                self.width
                * geometry.heads
                * self.capacity
                * (geometry.key_width + geometry.value_width)
            )
            if buffer.shape != (size,) or buffer.dtype != self.dtype:
                raise ValueError("append image differs from reserved geometry")
        self.buffers = buffers

    def zeros(self) -> None:
        self.install(
            tuple(
                mx.zeros(
                    self.width * g.heads * self.capacity * (g.key_width + g.value_width), self.dtype
                )
                for g in self.geometry
            )
        )

    def layer(self, index: int) -> tuple[mx.array, mx.array]:
        g = self.geometry[index]
        return split_tail(
            self.buffers[index], self.width, g.heads, self.capacity, g.key_width, g.value_width
        )

    def acquire(self, index: int, start: int) -> TailRow:
        if self.closed or not 0 <= index < self.width or start < 0:
            raise ValueError("append image row is unavailable")
        self.users += 1
        return TailRow(self, index, start)

    def close(self) -> None:
        if not self.users and not self.closed:
            self.buffers = ()
            self.reservation.close()
            self.closed = True


@dataclass
class TailRow:
    image: TailImage
    index: int
    start: int
    closed: bool = False

    def acquire(self) -> TailRow:
        if self.closed:
            raise RuntimeError("append row is closed")
        return self.image.acquire(self.index, self.start)

    def layer(self, index: int) -> tuple[mx.array, mx.array]:
        if self.closed:
            raise RuntimeError("append row is closed")
        k, v = self.image.layer(index)
        return k[self.index], v[self.index]

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.image.users -= 1
            self.image.close()


def split_tail(
    buffer: mx.array, batch: int, heads: int, capacity: int, key_width: int, value_width: int
) -> tuple[mx.array, mx.array]:
    """Contiguous K/V views of one backing allocation, including unequal widths."""
    boundary = batch * heads * capacity * key_width
    k, v = mx.split(buffer, [boundary])
    return (
        k.reshape(batch, heads, capacity, key_width),
        v.reshape(batch, heads, capacity, value_width),
    )
