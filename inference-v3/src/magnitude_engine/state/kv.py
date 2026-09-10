"""Dynamic KV slabs with run placement and completion-safe extent reuse."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise

from magnitude_engine.platform.execution import (
    CapacityError,
    DeviceContext,
    DType,
    ResourceLease,
    Tensor,
    TensorSpec,
    reclaimable_bytes,
)


@dataclass(frozen=True)
class KVLayout:
    layers: int
    heads: int
    width: int
    dtype: DType = DType.F32

    def __post_init__(self):
        if min(self.layers, self.heads, self.width) <= 0 or self.dtype not in (
            DType.F32,
            DType.F16,
            DType.BF16,
        ):
            raise ValueError("invalid KV layout")

    @property
    def token_bytes(self) -> int:
        return self.layers * 2 * self.heads * self.width * self.dtype.itemsize


@dataclass(frozen=True)
class KVSpan:
    run: KVRun
    start: int
    length: int

    def __post_init__(self):
        if self.start < 0 or not 0 < self.length <= self.run.capacity:
            raise ValueError("invalid visible KV span")


@dataclass(frozen=True)
class KVReadRun:
    """One contiguous read operand, retaining separately owned allocation extents."""

    spans: tuple[KVSpan, ...]

    def __post_init__(self):
        if not self.spans or any(not _consecutive(a, b) for a, b in pairwise(self.spans)):
            raise ValueError("KV read run must be contiguous in physical and logical space")

    @property
    def start(self) -> int:
        return self.spans[0].start

    @property
    def length(self) -> int:
        return sum(span.length for span in self.spans)

    @property
    def capacity(self) -> int:
        return sum(span.run.capacity for span in self.spans)

    def views(self, layer: int) -> tuple[Tensor, Tensor]:
        return _run_views(tuple(span.run for span in self.spans), layer)


def _consecutive(first: KVSpan, second: KVSpan) -> bool:
    return (
        first.start + first.length == second.start
        and first.length == first.run.capacity
        and first.run.adjacent(second.run)
    )


def physical_runs(history: tuple[KVSpan, ...]) -> tuple[KVReadRun, ...]:
    if any(b.start < a.start + a.length for a, b in pairwise(history)):
        raise ValueError("KV history must contain ordered disjoint visible spans")
    groups: list[list[KVSpan]] = []
    for span in history:
        if groups and _consecutive(groups[-1][-1], span):
            groups[-1].append(span)
        else:
            groups.append([span])
    return tuple(KVReadRun(tuple(group)) for group in groups)


@dataclass(frozen=True)
class KVReadGroup:
    """Segments sharing backing storage, with explicit visibility for every segment.

    The operand window may cover gaps. Only the listed segments may be read;
    their allocation claims, rather than the gaps, are retained by the views.
    """

    runs: tuple[KVReadRun, ...]

    def __post_init__(self):
        if not self.runs:
            raise ValueError("KV read group requires visible runs")
        slab = self.runs[0].spans[0].run._extent.slab
        for run in self.runs:
            for span in run.spans:
                span.run._check()
                if span.run._extent.slab is not slab:
                    raise ValueError("KV read group requires shared backing storage")
        if any(b.start < a.start + a.length for a, b in pairwise(self.runs)):
            raise ValueError("KV read group requires ordered disjoint visible runs")

    @property
    def _origin(self) -> int:
        return min(
            run.spans[0].run._extent.start * run.spans[0].run._extent.pool.page_tokens
            for run in self.runs
        )

    @property
    def capacity(self) -> int:
        # The physical operand stays stable as visible extents grow. Metadata,
        # never this window, grants permission to read an element.
        return self.runs[0].spans[0].run._extent.slab.capacity - self._origin

    @property
    def segment_capacity(self) -> int:
        return max(run.capacity for run in self.runs)

    @property
    def segments(self) -> tuple[tuple[int, int, int], ...]:
        """Physical offset, logical start, visible length, in logical order."""
        origin = self._origin
        return tuple(
            (
                run.spans[0].run._extent.start * run.spans[0].run._extent.pool.page_tokens - origin,
                run.start,
                run.length,
            )
            for run in self.runs
        )

    def acquire(self) -> KVReadLease:
        return KVReadLease(_ReadOwnership(self))

    def views(self, layer: int) -> tuple[Tensor, Tensor]:
        lease = self.acquire()
        try:
            return lease.views(layer)
        finally:
            lease.close()


class _ReadOwnership:
    def __init__(self, group: KVReadGroup):
        with ExitStack() as cleanup:
            runs = []
            for segment in group.runs:
                for span in segment.spans:
                    run = span.run.fork()
                    cleanup.callback(run.close)
                    runs.append(run)
            self.runs = tuple(runs)
            self.context = self.runs[0].context
            self.origin, self.capacity = group._origin, group.capacity
            self.claims = 0
            cleanup.pop_all()


class KVReadLease:
    """One retained extent set shared by all layer views and derived commands."""

    def __init__(self, ownership: _ReadOwnership):
        self._ownership, self._closed = ownership, False
        ownership.claims += 1

    @property
    def context(self) -> DeviceContext:
        return self._ownership.context

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("KV read lease is closed")
        self._ownership.runs[0]._check()

    def fork(self) -> KVReadLease:
        self._check()
        return KVReadLease(self._ownership)

    def views(self, layer: int) -> tuple[Tensor, Tensor]:
        self._check()
        owner = self._ownership
        return _storage_views(owner.runs, layer, owner.origin, owner.capacity, pins=(self,))

    def close(self) -> None:
        self.context.check_thread()
        if not self._closed:
            self._closed = True
            self._ownership.claims -= 1
            if self._ownership.claims == 0:
                for run in self._ownership.runs:
                    run.close()


def grouped_reads(history: tuple[KVReadRun, ...]) -> tuple[KVReadGroup, ...]:
    """Group by backing identity without changing any segment's logical visibility."""
    if any(b.start < a.start + a.length for a, b in pairwise(history)):
        raise ValueError("KV history must contain ordered disjoint visible runs")
    groups: dict[_Slab, list[KVReadRun]] = {}
    for run in history:
        groups.setdefault(run.spans[0].run._extent.slab, []).append(run)
    return tuple(KVReadGroup(tuple(runs)) for runs in groups.values())


@dataclass(frozen=True)
class KVWrite:
    run: KVRun
    destination: int
    source: int
    length: int

    def __post_init__(self):
        if (
            min(self.destination, self.source) < 0
            or self.length <= 0
            or self.destination + self.length > self.run.capacity
        ):
            raise ValueError("invalid KV write span")


class _Slab:
    def __init__(self, pool: KVPool, pages: int):
        self.pool, self.pages = pool, pages
        self.capacity = pages * pool.page_tokens
        layout = pool.layout
        self.storage = pool.context.allocate(
            TensorSpec((layout.layers, 2, self.capacity, layout.heads, layout.width), layout.dtype)
        )
        self.free = [(0, pages)]


@dataclass
class _Extent:
    pool: KVPool
    slab: _Slab
    start: int
    pages: int
    generation: int
    claims: int = 0


class KVRun:
    """Owned physical pages; logical visible positions belong to the sequence."""

    def __init__(self, extent: _Extent):
        self._extent = extent
        self._closed = False
        extent.claims += 1

    @property
    def context(self) -> DeviceContext:
        return self._extent.pool.context

    @property
    def capacity(self) -> int:
        return self._extent.pages * self._extent.pool.page_tokens

    @property
    def generation(self) -> int:
        return self._extent.generation

    @property
    def exclusive(self) -> bool:
        self._check()
        return self._extent.claims == 1

    def _check(self) -> None:
        self.context.check()
        if self._closed or self._extent.pool._closed:
            raise RuntimeError("KV run is unavailable")

    def fork(self) -> KVRun:
        self._check()
        return KVRun(self._extent)

    def adjacent(self, other: KVRun) -> bool:
        self._check()
        other._check()
        a, b = self._extent, other._extent
        return a.slab is b.slab and a.start + a.pages == b.start

    def views(self, layer: int) -> tuple[Tensor, Tensor]:
        return _run_views((self,), layer)

    def close(self) -> None:
        self.context.check_thread()
        if not self._closed:
            self._closed = True
            self._extent.claims -= 1
            if self._extent.claims == 0:
                self._extent.pool._release(self._extent)


def _run_views(runs: tuple[KVRun, ...], layer: int) -> tuple[Tensor, Tensor]:
    for run in runs:
        run._check()
    if not runs or any(not a.adjacent(b) for a, b in pairwise(runs)):
        raise ValueError("KV views require adjacent live extents")
    return _storage_views(
        runs,
        layer,
        runs[0]._extent.start * runs[0]._extent.pool.page_tokens,
        sum(run.capacity for run in runs),
    )


def _storage_views(
    runs: tuple[KVRun, ...],
    layer: int,
    start: int,
    capacity: int,
    *,
    pins: tuple[ResourceLease, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    if pins is None:
        for run in runs:
            run._check()
    extent, layout = runs[0]._extent, runs[0]._extent.pool.layout
    if not 0 <= layer < layout.layers:
        raise ValueError("KV layer is outside the physical layout")
    spec = TensorSpec((capacity, layout.heads, layout.width), layout.dtype)
    stride = layout.heads * layout.width * layout.dtype.itemsize
    result: list[Tensor] = []
    try:
        for kind in range(2):
            offset = ((layer * 2 + kind) * extent.slab.capacity + start) * stride
            view = extent.slab.storage.view(spec, offset)
            try:
                result.append(view.pin(*(runs if pins is None else pins)))
            finally:
                view.close()
        return result[0], result[1]
    except BaseException:
        for view in result:
            view.close()
        raise


class KVPool:
    def __init__(
        self, context: DeviceContext, layout: KVLayout, page_tokens: int = 128, slab_pages: int = 32
    ):
        if min(page_tokens, slab_pages) <= 0:
            raise ValueError("KV allocation units must be positive")
        self.context, self.layout = context, layout
        self.page_tokens, self.slab_pages = page_tokens, slab_pages
        self._slabs: list[_Slab] = []
        self._extents: dict[int, _Extent] = {}
        self._next_generation = 0
        self._closed = False

    @property
    def slab_count(self) -> int:
        return len(self._slabs)

    @property
    def allocated_tokens(self) -> int:
        return sum(slab.capacity for slab in self._slabs)

    def reserve(
        self, tokens: int, *, after: KVRun | None = None, preferred_tokens: int = 0
    ) -> tuple[KVRun, ...]:
        """Claim tokens; grow backing toward a known horizon when capacity permits.

        The preference claims no extra pages and does not make a feasible advance
        fail. Unclaimed backing remains available to every sequence in this pool.
        """
        if type(preferred_tokens) is not int or preferred_tokens < 0:
            raise ValueError("KV growth preference must be nonnegative")
        try:
            return (
                self._allocate_contiguous(tokens, after=after, preferred_tokens=preferred_tokens),
            )
        except CapacityError:
            remaining = (tokens + self.page_tokens - 1) // self.page_tokens
            if sum(count for slab in self._slabs for _, count in slab.free) < remaining:
                raise
            runs: list[KVRun] = []
            try:
                while remaining:
                    largest = max(count for slab in self._slabs for _, count in slab.free)
                    count = min(remaining, largest)
                    run = self._allocate_contiguous(
                        count * self.page_tokens, after=runs[-1] if runs else after
                    )
                    runs.append(run)
                    remaining -= count
                return tuple(runs)
            except BaseException:
                for run in reversed(runs):
                    run.close()
                raise

    def _allocate_contiguous(
        self, tokens: int, *, after: KVRun | None = None, preferred_tokens: int = 0
    ) -> KVRun:
        self.context.check()
        if self._closed or type(tokens) is not int or tokens <= 0:
            raise ValueError("invalid KV allocation request")
        if after is not None:
            after._check()
            if after._extent.pool is not self:
                raise ValueError("preferred predecessor belongs to another KV pool")
        pages = (tokens + self.page_tokens - 1) // self.page_tokens
        candidates = []
        for slab_index, slab in enumerate(self._slabs):
            frontiers = {e.start + e.pages for e in self._extents.values() if e.slab is slab}
            for index, (start, available) in enumerate(slab.free):
                if available < pages:
                    continue
                own_frontier = (
                    after is not None
                    and after._extent.slab is slab
                    and after._extent.start + after._extent.pages == start
                )
                priority = 0 if own_frontier else 2 if start in frontiers else 1
                candidates.append((priority, available, slab_index, index))
        if candidates:
            _, _, slab_index, index = min(candidates)
            slab = self._slabs[slab_index]
        else:
            target_pages = max(
                self.slab_pages,
                pages,
                (preferred_tokens + self.page_tokens - 1) // self.page_tokens,
            )
            try:
                slab = _Slab(self, target_pages)
            except CapacityError:
                if target_pages == pages:
                    raise
                slab = _Slab(self, pages)
            self._slabs.append(slab)
            index = 0
        start, available = slab.free[index]
        if available == pages:
            del slab.free[index]
        else:
            slab.free[index] = (start + pages, available - pages)
        generation = self._next_generation
        self._next_generation += 1
        extent = _Extent(self, slab, start, pages, generation)
        self._extents[generation] = extent
        return KVRun(extent)

    def reclaimable(self, runs: tuple[KVRun, ...]) -> int:
        """Whole backing allocations exclusively releasable by this victim set."""
        owned: dict[int, set[int]] = {}
        for run in runs:
            run._check()
            if run._extent.pool is not self:
                raise ValueError("reclamation query belongs to another KV pool")
            owned.setdefault(run.generation, set()).add(id(run))
        released = {
            generation
            for generation, claims in owned.items()
            if self._extents[generation].claims == len(claims)
        }
        return sum(
            reclaimable_bytes((slab.storage,))
            for slab in self._slabs
            if all(e.generation in released for e in self._extents.values() if e.slab is slab)
        )

    def _release(self, extent: _Extent) -> None:
        self.context.check_thread()
        del self._extents[extent.generation]
        free = sorted((*extent.slab.free, (extent.start, extent.pages)))
        merged: list[tuple[int, int]] = []
        for start, pages in free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous, count = merged[-1]
                merged[-1] = (previous, count + pages)
            else:
                merged.append((start, pages))
        extent.slab.free = merged
        if merged == [(0, extent.slab.pages)]:
            extent.slab.storage.close()
            self._slabs.remove(extent.slab)

    def close(self) -> None:
        self.context.check_thread()
        self._closed = True
        # Live extents (including prepared/submitted views) retain their slab.
        # Their final release frees the backing even after pool retirement.
