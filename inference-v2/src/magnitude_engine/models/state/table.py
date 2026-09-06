"""Immutable physical mappings, shared by all layer reads until placement changes."""

from dataclasses import dataclass
from functools import cached_property

import mlx.core as mx


@dataclass(frozen=True, eq=False)
class PageMap:
    addresses: tuple[int, ...]
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity < 0 or any(
            type(page) is not int or not 0 <= page < self.capacity for page in self.addresses
        ):
            raise ValueError("page address is outside physical storage")

    @cached_property
    def maximum(self) -> int:
        return max(self.addresses, default=-1)


@dataclass(frozen=True)
class PageTable:
    rows: tuple[PageMap, ...]

    @cached_property
    def addresses(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row.addresses for row in self.rows)

    @cached_property
    def width(self) -> int:
        return max(map(len, self.addresses), default=0)

    @cached_property
    def maximum(self) -> int:
        return max((row.maximum for row in self.rows), default=-1)

    @cached_property
    def device(self) -> mx.array:
        return mx.array(
            [(*row, *([-1] * (self.width - len(row)))) for row in self.addresses],
            mx.int32,
        )
