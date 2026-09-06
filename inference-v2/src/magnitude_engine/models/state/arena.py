"""Heads-major MLX KV storage, independent of prefix identity and retention."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.resources.budget import MemoryBudget, Reservation

from .gather import gather_pages
from .placement import (
    PageAllocator,
    PlacementHints,
    RepackedLayout,
    runs,
)


@dataclass(frozen=True)
class LayerGeometry:
    heads: int
    key_width: int
    value_width: int

    def __post_init__(self) -> None:
        if min(self.heads, self.key_width, self.value_width) <= 0:
            raise ValueError("KV dimensions must be positive")


class KVArena:
    """Own arrays addressed by [head, physical_page * page_size + token, channel].

    Replacing buffers has a measured memory peak: the old allocation stays reserved
    until the replacement is materialized. Active execution pins the buffers against
    growth, shrink, page release and relocation; the caller retires the pin only after
    GPU consumers complete. Tensor writes remain lazy within the pinned execution.
    """

    def __init__(
        self,
        layers: tuple[LayerGeometry, ...],
        *,
        page_size: int,
        slab_pages: int,
        max_pages: int,
        budget: MemoryBudget,
        dtype: mx.Dtype = mx.bfloat16,
        owner: str = "target.kv",
    ):
        if not layers or page_size <= 0:
            raise ValueError("KV storage requires layers and a positive page size")
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("KV dtype must be float16, bfloat16 or float32")
        self.layers = layers
        self.page_size = page_size
        self.dtype = dtype
        self.owner = owner
        self.allocator = PageAllocator(slab_pages, max_pages)
        self.budget = budget
        self.keys: tuple[mx.array, ...] = ()
        self.values: tuple[mx.array, ...] = ()
        self.counters: Counter[str] = Counter()
        self._allocation: Reservation | None = None
        self._pins = 0
        self._closed = False

    @property
    def page_bytes(self) -> int:
        itemsize = 4 if self.dtype == mx.float32 else 2
        return (
            self.page_size
            * itemsize
            * sum(layer.heads * (layer.key_width + layer.value_width) for layer in self.layers)
        )

    def _idle(self) -> None:
        if self._closed:
            raise RuntimeError("KV arena is closed")
        if self._pins:
            raise RuntimeError("KV layout cannot change while execution holds a pin")

    @contextmanager
    def pin(self) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("KV arena is closed")
        self._pins += 1
        try:
            yield
        finally:
            self._pins -= 1

    def allocate(
        self,
        count: int,
        hints: PlacementHints | None = None,
        *,
        grow: bool = True,
        before: int | None = None,
    ) -> tuple[int, ...]:
        self._idle()
        plan = self.allocator.plan(count, hints, grow=grow, before=before)
        if plan.capacity != self.allocator.capacity:
            self._resize(plan.capacity)
        self.allocator.commit(plan)
        self.counters["pages_allocated"] += count
        self.counters["allocation_runs"] += len(runs(plan.pages))
        self.counters["adjacency_hits"] += plan.adjacency_hit
        self.counters["frontiers_consumed"] += plan.frontiers_consumed
        return plan.pages

    def _resize(self, capacity: int, moves: tuple[tuple[int, int], ...] = ()) -> None:
        old_capacity = self.allocator.capacity
        allocation = self.budget.reserve(self.owner, capacity * self.page_bytes)
        try:
            keys, values = [], []
            for index, layer in enumerate(self.layers):
                for width, current, output in (
                    (layer.key_width, self.keys, keys),
                    (layer.value_width, self.values, values),
                ):
                    if not capacity:
                        continue
                    if capacity < old_capacity:
                        # A view may retain the original allocation for one-head
                        # geometries. Explicit construction guarantees a new buffer.
                        result = mx.array(current[index][:, : capacity * self.page_size, :])
                    else:
                        tail = mx.zeros(
                            (layer.heads, (capacity - old_capacity) * self.page_size, width),
                            dtype=self.dtype,
                        )
                        result = mx.concatenate([current[index], tail], axis=1) if current else tail
                    for source, destination in moves:
                        data = current[index][
                            :, source * self.page_size : (source + 1) * self.page_size, :
                        ]
                        result = mx.slice_update(
                            result, data, mx.array([destination * self.page_size]), axes=[1]
                        )
                    output.append(result)
            mx.eval(keys, values)
        except BaseException:
            allocation.close()
            raise
        self.keys, self.values = tuple(keys), tuple(values)
        previous, self._allocation = self._allocation, allocation
        if previous:
            previous.close()
        self.counters["slabs_grown"] += max(0, capacity - old_capacity) // self.allocator.slab_pages

    def release(self, pages: tuple[int, ...]) -> None:
        self._idle()
        self.allocator.release(pages)
        self.counters["pages_freed"] += len(pages)

    def shrink(self) -> int:
        self._idle()
        capacity = self.allocator.releasable_capacity()
        released = self.allocator.capacity - capacity
        if released:
            self._resize(capacity)
            self.allocator.shrink(capacity)
            self.counters["slabs_released"] += released // self.allocator.slab_pages
        return released * self.page_bytes

    def repack(self, layout: RepackedLayout) -> int:
        """Materialize replacement storage before publishing any eviction or address change."""
        self._idle()
        if layout.revision != self.allocator.revision:
            raise RuntimeError("physical reclamation plan became stale")
        old_capacity = self.allocator.capacity
        if layout.capacity >= old_capacity:
            raise ValueError("reclamation must release at least one slab")
        self._resize(layout.capacity, layout.moves)
        self.allocator.commit_repack(layout)
        released = old_capacity - layout.capacity
        self.counters["slabs_released"] += released // self.allocator.slab_pages
        self.counters["reclamation_pages_moved"] += len(layout.moves)
        self.counters["reclamation_pages_discarded"] += len(layout.released)
        return released * self.page_bytes

    def write(self, layer: int, page: int, offset: int, keys: mx.array, values: mx.array) -> None:
        """Write one contiguous physical run, starting within an owned page."""
        if self._closed or not self.allocator.owns(page):
            raise ValueError("write requires an owned page")
        geometry = self.layers[layer]
        count = keys.shape[1]
        if (
            offset < 0
            or offset >= self.page_size
            or keys.shape != (geometry.heads, count, geometry.key_width)
            or values.shape != (geometry.heads, count, geometry.value_width)
            or keys.dtype != self.dtype
            or values.dtype != self.dtype
        ):
            raise ValueError("KV write does not match page geometry or dtype")
        if not count:
            return
        touched = (offset + count + self.page_size - 1) // self.page_size
        if not all(self.allocator.owns(address) for address in range(page, page + touched)):
            raise ValueError("write requires an owned physical run")
        start = mx.array([page * self.page_size + offset])
        k, v = list(self.keys), list(self.values)
        k[layer] = mx.slice_update(k[layer], keys, start, axes=[1])
        v[layer] = mx.slice_update(v[layer], values, start, axes=[1])
        self.keys, self.values = tuple(k), tuple(v)
        self.counters["kv_write_runs"] += 1
        self.counters["kv_written_tokens"] += count

    def gather(self, layer: int, pages: tuple[int, ...], tokens: int) -> tuple[mx.array, mx.array]:
        if tokens < 0 or tokens > len(pages) * self.page_size:
            raise ValueError("visible tokens exceed page capacity")
        if not all(self.allocator.owns(page) for page in pages):
            raise ValueError("read requires owned pages")
        return gather_pages(self.keys[layer], self.values[layer], self.page_size, pages, 0, tokens)

    def copy(self, pairs: tuple[tuple[int, int], ...], *, tokens: int | None = None) -> None:
        self._idle()
        if len({destination for _, destination in pairs}) != len(pairs):
            raise ValueError("copy destinations must be distinct")
        count = self.page_size if tokens is None else tokens
        if not 0 <= count <= self.page_size:
            raise ValueError("invalid page copy length")
        if not all(self.allocator.owns(page) for pair in pairs for page in pair):
            raise ValueError("copy requires owned source and destination pages")
        # Materialize every source before writing any destination. This preserves
        # overlap semantics and prevents slice views from keeping the arena live.
        size = len(pairs) * self.page_bytes * count // self.page_size
        with self.budget.reserve(f"{self.owner}.copy", size):
            copies = [
                (
                    mx.array(
                        self.keys[layer][
                            :, source * self.page_size : source * self.page_size + count
                        ]
                    ),
                    mx.array(
                        self.values[layer][
                            :, source * self.page_size : source * self.page_size + count
                        ]
                    ),
                )
                for layer in range(len(self.layers))
                for source, _ in pairs
            ]
            mx.eval(copies)
            for layer in range(len(self.layers)):
                for index, (_, destination) in enumerate(pairs):
                    k, v = copies[layer * len(pairs) + index]
                    self.write(layer, destination, 0, k, v)
            self.complete()
        self.counters["pages_copied"] += len(pairs)
        self.counters["bytes_copied"] += size

    def complete(self) -> None:
        mx.eval(self.keys, self.values)

    def close(self) -> None:
        if self._closed:
            return
        self._idle()
        if self.allocator.owned:
            raise RuntimeError("cannot close KV arena with owned pages")
        self.complete()
        self.keys, self.values = (), ()
        if self._allocation:
            self._allocation.close()
        self._closed = True
