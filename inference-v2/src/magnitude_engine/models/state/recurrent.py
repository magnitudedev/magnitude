"""Immutable recurrent images and independently owned logical rows.

An image is one physical batch across all recurrent layers. Readable rows own its
tensor handles; execution pins retain the allocation charge after those handles
are no longer needed. Neither may retire device storage before completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.resources.budget import MemoryBudget


@dataclass(frozen=True)
class StateTensor:
    shape: tuple[int, ...]
    dtype: mx.Dtype

    @property
    def nbytes(self) -> int:
        size = 4 if self.dtype == mx.float32 else 2
        for dimension in self.shape:
            size *= dimension
        return size


@dataclass(frozen=True)
class RecurrentLayout:
    tensors: tuple[StateTensor, ...]
    trace_bytes_per_token: int

    def __post_init__(self) -> None:
        if (
            not self.tensors
            or self.trace_bytes_per_token < 0
            or any(
                t.dtype not in (mx.float16, mx.bfloat16, mx.float32)
                or not t.shape
                or t.shape[0] != 1
                or any(d <= 0 for d in t.shape)
                for t in self.tensors
            )
        ):
            raise ValueError("invalid recurrent layout")

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)


@component("STATE:RECURRENT:MAG:CHECKPOINTED")
class RecurrentImage:
    """Write each layer once, then share immutable rows until their last lease ends."""

    def __init__(self, layouts: tuple[RecurrentLayout, ...], width: int, budget: MemoryBudget):
        if width < 1:
            raise ValueError("recurrent image requires a positive width")
        self.layouts, self.width = layouts, width
        self.reservation = budget.reserve("recurrent-state", width * sum(s.nbytes for s in layouts))
        self._layers: list[tuple[mx.array, ...] | None] = [None] * len(layouts)
        self._views: dict[tuple[int, int], tuple[mx.array, ...]] = {}
        self._users = 0
        self._pins = 0
        self._retired = False
        self.closed = False

    def acquire(self, index: int) -> RecurrentRow:
        if self._retired or self.closed or not 0 <= index < self.width:
            raise ValueError("recurrent row is unavailable")
        self._users += 1
        return RecurrentRow(self, index)

    def pin(self) -> RecurrentPin:
        """Retain allocation accounting without retaining obsolete tensor graphs."""
        if self.closed or self._retired:
            raise RuntimeError("recurrent image is unavailable")
        self._pins += 1
        return RecurrentPin(self)

    def write(self, layer: int, values: tuple[mx.array, ...]) -> None:
        if self._retired or self.closed or self._layers[layer] is not None:
            raise RuntimeError("recurrent image layer is already written or closed")
        layout = self.layouts[layer]
        if len(values) != len(layout.tensors) or any(
            a.shape != (self.width, *spec.shape[1:]) or a.dtype != spec.dtype
            for a, spec in zip(values, layout.tensors, strict=True)
        ):
            raise ValueError("recurrent output does not match its physical layout")
        self._layers[layer] = values

    def read(self, layer: int) -> tuple[mx.array, ...]:
        values = self._layers[layer]
        if self.closed or values is None:
            raise RuntimeError("recurrent image layer is unavailable")
        return values

    def row_values(self, layer: int, index: int) -> tuple[mx.array, ...]:
        values = self.read(layer)
        if self.width == 1:
            return values
        key = (layer, index)
        if key not in self._views:
            self._views[key] = tuple(a[index : index + 1] for a in values)
        return self._views[key]

    def _release(self) -> None:
        self._users -= 1
        if self._users == 0:
            self._retired = True
            self._views.clear()
            self._layers = [None] * len(self.layouts)
        self._retire()

    def _retire(self) -> None:
        if not self._users and not self._pins:
            self.reservation.close()
            self.closed = True


class RecurrentPin:
    """Submitted consumers own buffers; this lease keeps their allocation charged."""

    def __init__(self, image: RecurrentImage):
        self.image, self.closed = image, False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.image._pins -= 1
            self.image._retire()


class RecurrentRow:
    def __init__(self, image: RecurrentImage, index: int):
        self.image, self.index = image, index
        self.closed = False

    def acquire(self) -> RecurrentRow:
        if self.closed:
            raise RuntimeError("recurrent row lease is closed")
        return self.image.acquire(self.index)

    def values(self, layer: int) -> tuple[mx.array, ...]:
        if self.closed:
            raise RuntimeError("recurrent row lease is closed")
        return self.image.row_values(layer, self.index)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.image._release()


class RecurrentTransition(Protocol):
    @property
    def length(self) -> int: ...
    @property
    def values(self) -> tuple[mx.array, ...]: ...
    def prefix(self, count: int) -> tuple[mx.array, ...]: ...


@dataclass(frozen=True)
class RecurrentBoundaries:
    initial: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    length: int

    def prefix(self, count: int) -> tuple[mx.array, ...]:
        if count not in (0, self.length):
            raise ValueError("recurrent advance has no interior-prefix trace")
        return self.initial if count == 0 else self.values


class RecurrentSlot:
    def __init__(self, source: RecurrentRow, layer: int):
        self.source, self.layer = source, layer
        self.layout = source.image.layouts[layer]
        self.destination: RecurrentRow | None = None
        self.pending: RecurrentTransition | None = None

    @property
    def values(self) -> tuple[mx.array, ...]:
        return self.source.values(self.layer)

    def stage(self, transition: RecurrentTransition) -> None:
        if self.pending is not None or self.destination is None:
            raise RuntimeError("recurrent layer is not ready to stage an advance")
        if transition.values is not self.destination.values(self.layer):
            raise ValueError("recurrent transition must use its reserved destination")
        self.pending = transition


def read_batch(slots: tuple[RecurrentSlot, ...]) -> tuple[mx.array, ...]:
    first = slots[0]
    image = first.source.image
    if image.width == len(slots) and all(
        slot.source.image is image and slot.source.index == row and slot.layer == first.layer
        for row, slot in enumerate(slots)
    ):
        return image.read(first.layer)
    if len(slots) == 1:
        return first.values
    return tuple(
        mx.concatenate([slot.values[i] for slot in slots]) for i in range(len(first.layout.tensors))
    )


def write_batch(
    slots: tuple[RecurrentSlot, ...], values: tuple[mx.array, ...]
) -> tuple[tuple[mx.array, ...], ...]:
    destinations = tuple(slot.destination for slot in slots)
    if any(row is None for row in destinations):
        raise RuntimeError("recurrent batch has no reserved destination")
    rows = tuple(row for row in destinations if row is not None)
    image = rows[0].image
    layer = slots[0].layer
    if image.width == len(rows) and all(
        row.image is image and row.index == index and slot.layer == layer
        for index, (row, slot) in enumerate(zip(rows, slots, strict=True))
    ):
        image.write(layer, values)
    else:
        # Independently prepared transactions may still share tensor computation.
        # Their physical destinations remain distinct and individually charged.
        for index, (row, slot) in enumerate(zip(rows, slots, strict=True)):
            row.image.write(
                slot.layer,
                values if len(rows) == 1 else tuple(mx.array(a[index : index + 1]) for a in values),
            )
    return tuple(row.values(slot.layer) for row, slot in zip(rows, slots, strict=True))


def stage_boundaries(
    slots: tuple[RecurrentSlot, ...], values: tuple[mx.array, ...], length: int
) -> None:
    """Publish a reserved recurrent output whose accepted prefix is either endpoint."""
    outputs = write_batch(slots, values)
    for slot, output in zip(slots, outputs, strict=True):
        slot.stage(RecurrentBoundaries(slot.values, output, length))
