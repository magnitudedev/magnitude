"""Plan page placement without allocating tensors or mutating live ownership."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class PageRun:
    start: int
    count: int

    @property
    def end(self) -> int:
        return self.start + self.count


def runs(pages: tuple[int, ...]) -> tuple[PageRun, ...]:
    result: list[PageRun] = []
    for page in pages:
        if result and result[-1].end == page:
            last = result[-1]
            result[-1] = PageRun(last.start, last.count + 1)
        else:
            result.append(PageRun(page, 1))
    return tuple(result)


@dataclass(frozen=True)
class PlacementHints:
    adjacent: int | None = None
    peer_frontiers: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Placement:
    revision: int
    pages: tuple[int, ...]
    capacity: int
    remaining: tuple[PageRun, ...]
    adjacency_hit: bool
    frontiers_consumed: int


@dataclass(frozen=True)
class RepackedLayout:
    revision: int
    capacity: int
    released: tuple[int, ...]
    moves: tuple[tuple[int, int], ...]
    owned: frozenset[int]
    free: tuple[PageRun, ...]


class PageAllocator:
    """Slab-bounded extents with transactional placement and explicit revision checks.

    Frontiers are preferences, never reservations. Ordinary growth tries adjacency,
    then an unprotected contiguous range, then any contiguous range, then fragments.
    Physical storage publishes a plan only after its tensor allocation succeeds.
    """

    def __init__(self, slab_pages: int, max_pages: int):
        if slab_pages <= 0 or max_pages < slab_pages:
            raise ValueError("at least one positive slab must fit")
        self.slab_pages = slab_pages
        self.max_pages = max_pages
        self.capacity = 0
        self.revision = 0
        self._free: tuple[PageRun, ...] = ()
        self._owned: set[int] = set()

    @property
    def free(self) -> tuple[PageRun, ...]:
        return self._free

    @property
    def owned(self) -> frozenset[int]:
        return frozenset(self._owned)

    def owns(self, page: int) -> bool:
        return page in self._owned

    def plan(
        self,
        count: int,
        hints: PlacementHints | None = None,
        *,
        grow: bool = True,
        before: int | None = None,
    ) -> Placement:
        if count < 0:
            raise ValueError("page count must be nonnegative")
        hints = hints or PlacementHints()
        capacity = self.capacity
        available = list(self._free)
        if before is not None:
            if grow:
                raise ValueError("relocation cannot grow storage")
            available = [
                PageRun(r.start, min(r.end, before) - r.start)
                for r in available
                if r.start < before
            ]
        while sum(r.count for r in available) < count:
            if not grow or capacity + self.slab_pages > self.max_pages:
                raise MemoryError("insufficient physical pages")
            available.append(PageRun(capacity, self.slab_pages))
            capacity += self.slab_pages

        def fits(start: int) -> bool:
            return any(r.start <= start and start + count <= r.end for r in available)

        adjacent = count > 0 and hints.adjacent is not None and fits(hints.adjacent)
        start = hints.adjacent if adjacent else None
        if start is None and count:
            for extent in available:
                cursor = extent.start
                blockers = sorted(p for p in hints.peer_frontiers if extent.start <= p < extent.end)
                for stop in [*blockers, extent.end]:
                    if stop - cursor >= count:
                        start = cursor
                        break
                    cursor = stop + 1
                if start is not None:
                    break
        if start is None:
            start = next((r.start for r in available if r.count >= count), None)
        if start is not None:
            chosen = tuple(range(start, start + count))
        else:
            selected: list[int] = []
            for extent in available:
                take = min(count - len(selected), extent.count)
                selected.extend(range(extent.start, extent.start + take))
                if len(selected) == count:
                    break
            chosen = tuple(selected)

        # Subtract from the complete free map, not the clipped relocation candidates.
        complete = [*self._free]
        complete.extend(
            PageRun(p, self.slab_pages) for p in range(self.capacity, capacity, self.slab_pages)
        )
        for selection in runs(chosen):
            updated: list[PageRun] = []
            for extent in complete:
                lo, hi = max(extent.start, selection.start), min(extent.end, selection.end)
                if lo >= hi:
                    updated.append(extent)
                else:
                    if extent.start < lo:
                        updated.append(PageRun(extent.start, lo - extent.start))
                    if hi < extent.end:
                        updated.append(PageRun(hi, extent.end - hi))
            complete = updated
        return Placement(
            self.revision,
            chosen,
            capacity,
            tuple(complete),
            adjacent,
            len(set(chosen) & hints.peer_frontiers),
        )

    def commit(self, plan: Placement) -> None:
        if plan.revision != self.revision:
            raise RuntimeError("page placement became stale")
        self._free = plan.remaining
        self.capacity = plan.capacity
        self._owned.update(plan.pages)
        self.revision += 1

    def release(self, pages: tuple[int, ...]) -> None:
        if len(set(pages)) != len(pages) or not set(pages) <= self._owned:
            raise ValueError("release requires distinct owned pages")
        free = sorted([*self._free, *(PageRun(p, 1) for p in pages)])
        merged: list[PageRun] = []
        for extent in free:
            if (
                merged
                and merged[-1].end == extent.start
                and merged[-1].start // self.slab_pages == extent.start // self.slab_pages
            ):
                left = merged[-1]
                merged[-1] = PageRun(left.start, left.count + extent.count)
            else:
                merged.append(extent)
        self._free = tuple(merged)
        self._owned.difference_update(pages)
        self.revision += 1

    def releasable_capacity(self) -> int:
        capacity = self.capacity
        for extent in reversed(self._free):
            if extent == PageRun(capacity - self.slab_pages, self.slab_pages):
                capacity -= self.slab_pages
            else:
                break
        return capacity

    def shrink(self, capacity: int) -> None:
        if capacity != self.releasable_capacity():
            raise RuntimeError("slab release must match current free trailing capacity")
        self._free = tuple(r for r in self._free if r.start < capacity)
        self.capacity = capacity
        self.revision += 1

    def plan_repack(
        self, released: tuple[int, ...], moves: tuple[tuple[int, int], ...]
    ) -> RepackedLayout:
        sources = {source for source, _ in moves}
        destinations = {destination for _, destination in moves}
        if (
            len(sources) != len(moves)
            or len(destinations) != len(moves)
            or len(set(released)) != len(released)
            or sources & set(released)
            or not sources | set(released) <= self._owned
            or destinations & self._owned
            or any(not 0 <= p < self.capacity for p in destinations)
        ):
            raise ValueError("invalid cold-page relocation")
        owned = frozenset((self._owned - sources - set(released)) | destinations)
        capacity = ((max(owned, default=-1) + self.slab_pages) // self.slab_pages) * self.slab_pages
        extents = []
        for boundary in range(0, capacity, self.slab_pages):
            start = boundary
            occupied = sorted(p for p in owned if boundary <= p < boundary + self.slab_pages)
            for stop in [*occupied, boundary + self.slab_pages]:
                if start < stop:
                    extents.append(PageRun(start, stop - start))
                start = stop + 1
        return RepackedLayout(self.revision, capacity, released, moves, owned, tuple(extents))

    def commit_repack(self, layout: RepackedLayout) -> None:
        if layout.revision != self.revision:
            raise RuntimeError("physical reclamation plan became stale")
        self.capacity, self._free = layout.capacity, layout.free
        self._owned = set(layout.owned)
        self.revision += 1

    def validate(self) -> None:
        free: set[int] = set()
        prior: PageRun | None = None
        for extent in self._free:
            assert extent.count > 0 and 0 <= extent.start < extent.end <= self.capacity
            assert extent.start // self.slab_pages == (extent.end - 1) // self.slab_pages
            if prior:
                assert prior.end <= extent.start
                assert (
                    prior.end != extent.start
                    or prior.start // self.slab_pages != extent.start // self.slab_pages
                )
            free.update(range(extent.start, extent.end))
            prior = extent
        assert not free & self._owned
        assert free | self._owned == set(range(self.capacity))
