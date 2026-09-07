"""Page sharing and append ownership without token hashes or retention policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import mlx.core as mx

from magnitude_engine import components as c
from magnitude_engine.components import component
from magnitude_engine.resources.retention import RetainedStorage

from .arena import KVArena
from .placement import PlacementHints, runs
from .table import PageMap, PageTable


@dataclass(eq=False)
class _Page:
    address: int
    written: int = 0
    writer: int | None = None
    readers: dict[int, int] = field(default_factory=dict)
    checkpoints: dict[int, int] = field(default_factory=dict)
    retained_tails: dict[int, int] = field(default_factory=dict)
    continuations: int = 0

    @property
    def protected(self) -> int:
        return max((*self.readers.values(), *self.checkpoints.values()), default=0)


class KVCheckpoint:
    """An immutable visible prefix. Physical addresses may change between executions."""

    def __init__(self, store: PageStore, identity: int, pages: tuple[_Page, ...], length: int):
        self._store = store
        self._identity = identity
        self._pages = pages
        self.length = length
        self.closed = False

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        return tuple(RetainedStorage(page, self._store.arena.page_bytes) for page in self._pages)

    @property
    def reclaimable(self) -> bool:
        return not self._store.arena._pins

    def close(self) -> None:
        if self.closed:
            return
        self._store.arena._idle()
        for page in self._pages:
            del page.checkpoints[self._identity]
            page.retained_tails.pop(self._identity, None)
            self._store._collect(page)
        del self._store._checkpoints[self._identity]
        self.closed = True


class SequencePages:
    """A model sequence's logical KV position and writable suffix."""

    def __init__(self, store: PageStore, identity: int):
        self.store = store
        self._identity = identity
        self._pages: list[_Page] = []
        self._mapping: PageMap | None = None
        self.length = 0
        self.origin = 0
        self._written = [0] * len(store.arena.layers)
        self.closed = False

    @property
    def addresses(self) -> tuple[int, ...]:
        self._live()
        return self.mapping.addresses

    @property
    def mapping(self) -> PageMap:
        self._live()
        if self._mapping is None:
            self._mapping = PageMap(
                tuple(p.address for p in self._pages),
                self.store.arena.allocator.capacity,
            )
        return self._mapping

    def _live(self) -> None:
        if self.closed:
            raise RuntimeError("sequence state is closed")

    def reserve(self, end: int) -> None:
        self._live()
        if end < self.length:
            raise ValueError("capacity cannot precede committed length")
        size = self.store.arena.page_size
        need = (end + size - 1) // size - len(self._pages)
        if need > 0:
            addresses = self.store.arena.allocate(need, self.store._hints(self))
            self._mapping = None
            for address in addresses:
                page = _Page(address, writer=self._identity)
                self.store._pages[address] = page
                self._pages.append(page)

    def _validate_write(self, layer: int, start: int, end: int) -> None:
        """Check append authority before either direct or prepared tensor execution."""
        self._live()
        size = self.store.arena.page_size
        if (
            not 0 <= layer < len(self._written)
            or start != self._written[layer]
            or start < self.length
            or end < start
            or end > len(self._pages) * size
        ):
            raise ValueError("layer writes must append contiguously within reserved capacity")
        # Validate every destination before enqueuing any layer mutation. Immutable
        # prefixes remain protected even when physical adjacency permits one write.
        if end > start:
            for index in range(start // size, (end - 1) // size + 1):
                page = self._pages[index]
                offset = start % size if index == start // size else 0
                if page.writer != self._identity or offset < page.protected:
                    raise RuntimeError("write would alter an immutable page prefix")

    @component(c.KV_APPEND, source=c.Source.MAG, variant="CONTIGUOUS_RUNS")
    def write(self, layer: int, start: int, keys: mx.array, values: mx.array) -> None:
        """Append one layer's KV. Commit publishes length only after every layer wrote it."""
        end = start + keys.shape[1]
        if values.shape[1] != keys.shape[1]:
            raise ValueError("layer writes require matching key/value lengths")
        self._validate_write(layer, start, end)
        size = self.store.arena.page_size
        cursor = start
        while cursor < end:
            index, offset = divmod(cursor, size)
            page = self._pages[index]
            stop = min(end, (index + 1) * size)
            while stop < end:
                next_index = stop // size
                if self._pages[next_index].address != page.address + next_index - index:
                    break
                stop = min(end, (next_index + 1) * size)
            self.store.arena.write(
                layer,
                page.address,
                offset,
                keys[:, cursor - start : stop - start, :],
                values[:, cursor - start : stop - start, :],
            )
            cursor = stop
        self._written[layer] = end

    def visible_length(self, layer: int) -> int:
        self._live()
        if not 0 <= layer < len(self._written):
            raise ValueError("KV layer is outside the sequence state")
        return self._written[layer]

    def commit(self, end: int) -> None:
        self._live()
        if end < self.length or any(position < end for position in self._written):
            raise ValueError("commit requires KV written by every layer")
        size = self.store.arena.page_size
        # Only the appended interval can change a page's published extent.
        for index in range(self.length // size, (end + size - 1) // size):
            page = self._pages[index]
            if page.writer == self._identity:
                page.written = max(page.written, min(size, end - index * size))
        self.length = end

    def checkpoint(self) -> KVCheckpoint:
        self._live()
        if any(position != self.length for position in self._written):
            raise RuntimeError("checkpoint requires a reconciled model state")
        self.store.arena.complete()
        identity = next(self.store._identities)
        size = self.store.arena.page_size
        pages = tuple(self._pages[: (self.length + size - 1) // size])
        for index, page in enumerate(pages):
            page.checkpoints[identity] = min(size, self.length - index * size)
        if pages:
            pages[-1].retained_tails[identity] = (self.length - 1) % size + 1
            if self.origin:
                pages[-1].continuations += 1
        checkpoint = KVCheckpoint(self.store, identity, pages, self.length)
        self.store._checkpoints[identity] = checkpoint
        return checkpoint

    def trim(self, end: int) -> None:
        self._live()
        self.store.arena._idle()
        if not self.origin <= end <= self.length:
            raise ValueError("rollback must stay within the sequence's own continuation")
        size = self.store.arena.page_size
        for index, page in enumerate(self._pages):
            if (
                page.writer == self._identity
                and min(size, max(0, end - index * size)) < page.protected
            ):
                raise RuntimeError(
                    "rollback crosses a retained checkpoint; restore it as a new state"
                )
        keep = (end + size - 1) // size
        for page in self._pages[keep:]:
            self.store._detach(self, page)
        if keep < len(self._pages):
            self._mapping = None
        del self._pages[keep:]
        for index, page in enumerate(self._pages):
            if page.writer == self._identity:
                page.written = min(size, end - index * size)
        self.length = end
        self._written = [end] * len(self._written)

    def read(self, layer: int) -> tuple[mx.array, mx.array]:
        return self.store.arena.gather(layer, self.addresses, self.length)

    def compact(self) -> int:
        """Use existing holes only, and only when the complete run table improves."""
        self._live()
        self.store.arena._idle()
        movable = [
            p
            for p in self._pages
            if p.writer == self._identity and not p.readers and not p.checkpoints
        ]
        if len(movable) < 2:
            return 0
        try:
            destinations = self.store.arena.allocate(len(movable), grow=False)
        except MemoryError:
            return 0
        mapping = {p.address: address for p, address in zip(movable, destinations, strict=True)}
        proposed = tuple(mapping.get(address, address) for address in self.addresses)
        if len(runs(proposed)) >= len(runs(self.addresses)):
            self.store.arena.release(destinations)
            return 0
        sources = tuple(p.address for p in movable)
        try:
            self.store.arena.copy(tuple(zip(sources, destinations, strict=True)))
        except BaseException:
            self.store.arena.release(destinations)
            raise
        self.store._relocate(movable, destinations)
        self.store.arena.release(sources)
        return len(movable)

    def close(self) -> None:
        if self.closed:
            return
        self.store.arena._idle()
        self.store.arena.complete()
        for page in self._pages:
            self.store._detach(self, page)
        self._pages.clear()
        self._mapping = None
        del self.store._sequences[self._identity]
        self.closed = True


@component(c.KV_STORE, source=c.Source.MAG, variant="PAGED")
class PageStore:
    """Own physical sharing and state handles; retention holds checkpoint handles only."""

    def __init__(self, arena: KVArena):
        self.arena = arena
        self._identities = count()
        self._pages: dict[int, _Page] = {}
        self._sequences: dict[int, SequencePages] = {}
        self._checkpoints: dict[int, KVCheckpoint] = {}
        self._read_table: PageTable | None = None

    def table(self, states: tuple[SequencePages, ...]) -> PageTable:
        if any(state.store is not self for state in states):
            raise ValueError("page table rows belong to different stores")
        mappings = tuple(state.mapping for state in states)
        if self._read_table is None or self._read_table.rows != mappings:
            self._read_table = PageTable(mappings)
        return self._read_table

    @component(c.KV_BRANCH, source=c.Source.MAG, variant="COPY_ON_WRITE")
    def create(self, checkpoint: KVCheckpoint | None = None) -> SequencePages:
        self.arena._idle()
        if checkpoint and (checkpoint.closed or checkpoint._store is not self):
            raise ValueError("checkpoint is closed or belongs to a different state store")
        sequence = SequencePages(self, next(self._identities))
        self._sequences[sequence._identity] = sequence
        if checkpoint is None:
            return sequence
        size = self.arena.page_size
        try:
            for index, page in enumerate(checkpoint._pages):
                visible = min(size, checkpoint.length - index * size)
                if visible == size:
                    page.readers[sequence._identity] = size
                    sequence._pages.append(page)
                elif page.writer is None and page.written == visible:
                    page.writer = sequence._identity
                    sequence._pages.append(page)
                    self.arena.counters["partial_page_reuses"] += 1
                else:
                    address = self.arena.allocate(1, self._hints(sequence))[0]
                    try:
                        self.arena.copy(((page.address, address),), tokens=visible)
                    except BaseException:
                        self.arena.release((address,))
                        raise
                    branch = _Page(address, written=visible, writer=sequence._identity)
                    self._pages[address] = branch
                    sequence._pages.append(branch)
                    self.arena.counters["partial_page_copies"] += 1
            sequence.origin = sequence.length = checkpoint.length
            sequence._written = [checkpoint.length] * len(self.arena.layers)
        except BaseException:
            sequence.close()
            raise
        return sequence

    def _hints(self, sequence: SequencePages) -> PlacementHints:
        adjacent = sequence._pages[-1].address + 1 if sequence._pages else None
        frontiers = {
            s._pages[-1].address + 1
            for s in self._sequences.values()
            if s is not sequence and s._pages
        }
        frontiers.update(
            page.address + 1
            for page in self._pages.values()
            if page.continuations and page.written in page.retained_tails.values()
        )
        return PlacementHints(adjacent, frozenset(frontiers - {adjacent}))

    def _detach(self, sequence: SequencePages, page: _Page) -> None:
        if page.writer == sequence._identity:
            page.writer = None
        else:
            del page.readers[sequence._identity]
        self._collect(page)

    def _collect(self, page: _Page) -> None:
        if page.writer is None:
            page.written = page.protected
        if page.writer is None and not page.readers and not page.checkpoints:
            self.arena.release((page.address,))
            del self._pages[page.address]

    def _relocate(self, pages: list[_Page], destinations: tuple[int, ...]) -> None:
        for page, address in zip(pages, destinations, strict=True):
            del self._pages[page.address]
            page.address = address
            self._pages[address] = page
            if page.writer is not None:
                self._sequences[page.writer]._mapping = None
            for identity in page.readers:
                self._sequences[identity]._mapping = None

    def validate(self) -> None:
        self.arena.allocator.validate()
        assert set(self._pages) == self.arena.allocator.owned
        for address, page in self._pages.items():
            assert address == page.address
            assert page.writer is not None or page.readers or page.checkpoints
            assert page.protected <= page.written <= self.arena.page_size
            if page.writer is not None:
                assert page in self._sequences[page.writer]._pages
            for identity in page.readers:
                assert page in self._sequences[identity]._pages


def append_layer(
    states: tuple[SequencePages, ...], layer: int, keys: mx.array, values: mx.array
) -> None:
    """Stage one producer layer for each row, before acquiring any shared read view."""
    if not states or len({id(state) for state in states}) != len(states):
        raise ValueError("KV append requires distinct nonempty sequence states")
    arena = states[0].store.arena
    if any(state.store.arena is not arena for state in states) or not 0 <= layer < len(
        arena.layers
    ):
        raise ValueError("KV append batch must share a valid physical layer")
    geometry = arena.layers[layer]
    if (
        keys.ndim != 4
        or values.ndim != 4
        or keys.shape[0] != len(states)
        or values.shape[0] != len(states)
        or keys.shape[1] != geometry.heads
        or values.shape[1] != geometry.heads
        or keys.shape[2] != values.shape[2]
        or keys.shape[3] != geometry.key_width
        or values.shape[3] != geometry.value_width
        or keys.dtype != arena.dtype
        or values.dtype != arena.dtype
    ):
        raise ValueError("KV append does not match the physical layer geometry")
    for row, state in enumerate(states):
        state.write(layer, state.length, keys[row], values[row])
