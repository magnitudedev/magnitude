"""Read-only snapshots of staged layer KV, borrowed within an execution scope."""

from dataclasses import dataclass

import mlx.core as mx

from .gather import gather_pages
from .pages import SequencePages
from .table import PageTable


@dataclass(frozen=True)
class PagedKV:
    """Heads-major buffers and row page maps, with no append or allocator authority.

    The caller holds the arena execution pin through every consumer. Capture after
    all row writes so each row reads the same functional MLX buffer version. The
    visible horizon may include staged, not yet committed, causal state.
    """

    keys: mx.array
    values: mx.array
    page_size: int
    table: PageTable
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.keys.ndim != 3
            or self.values.ndim != 3
            or self.keys.shape[:2] != self.values.shape[:2]
            or min(self.keys.shape) < 1
            or self.values.shape[-1] < 1
            or self.keys.dtype != self.values.dtype
            or type(self.page_size) is not int
            or self.page_size < 1
            or self.keys.shape[1] % self.page_size
            or not self.lengths
            or len(self.lengths) != len(self.table.rows)
        ):
            raise ValueError("invalid read-only KV geometry")
        capacity = self.keys.shape[1] // self.page_size
        if self.table.maximum >= capacity:
            raise ValueError("page table exceeds the physical layer capacity")
        for length, row in zip(self.lengths, self.table.rows, strict=True):
            if type(length) is not int or not 1 <= length <= len(row.addresses) * self.page_size:
                raise ValueError("read-only KV horizon or page address is out of bounds")

    @property
    def pages(self) -> tuple[tuple[int, ...], ...]:
        return self.table.addresses

    def gather(self, row: int, start: int = 0) -> tuple[mx.array, mx.array]:
        return gather_pages(
            self.keys, self.values, self.page_size, self.pages[row], start, self.lengths[row]
        )


def read_layer(
    states: tuple[SequencePages, ...], layer: int, *, pending_tokens: int = 0
) -> PagedKV:
    if not states or len({id(state) for state in states}) != len(states):
        raise ValueError("KV read requires distinct nonempty sequence states")
    arena = states[0].store.arena
    if any(state.store.arena is not arena for state in states):
        raise ValueError("KV read batch must share one physical arena")
    if type(pending_tokens) is not int or pending_tokens < 0:
        raise ValueError("requested pending KV horizon must be nonnegative")
    lengths = tuple(state.length + pending_tokens for state in states)
    if any(
        length > state.visible_length(layer) for state, length in zip(states, lengths, strict=True)
    ):
        raise ValueError("requested KV horizon has not been staged by its producer")
    if min(lengths) < 1:
        raise ValueError("KV read requires a nonempty staged or committed horizon")
    return PagedKV(
        arena.keys[layer],
        arena.values[layer],
        arena.page_size,
        states[0].store.table(states),
        lengths,
    )
