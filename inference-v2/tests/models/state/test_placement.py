import pytest
from hypothesis import given
from hypothesis import strategies as st

from magnitude_engine.models.state.placement import PageAllocator, PlacementHints


def test_frontiers_are_preferences_and_adjacency_wins():
    allocator = PageAllocator(8, 16)
    first = allocator.plan(2)
    allocator.commit(first)
    protected = allocator.plan(2, PlacementHints(peer_frontiers=frozenset({2})))
    assert protected.pages == (3, 4)
    allocator.commit(protected)
    adjacent = allocator.plan(1, PlacementHints(2, frozenset({2})))
    assert adjacent.pages == (2,)
    assert adjacent.adjacency_hit and adjacent.frontiers_consumed == 1
    allocator.commit(adjacent)
    allocator.validate()


def test_failed_and_stale_plans_cannot_change_ownership():
    allocator = PageAllocator(4, 4)
    first = allocator.plan(3)
    assert allocator.capacity == 0 and not allocator.owned
    allocator.commit(first)
    before = (allocator.free, allocator.owned, allocator.revision)
    with pytest.raises(MemoryError):
        allocator.plan(2)
    with pytest.raises(RuntimeError, match="stale"):
        allocator.commit(first)
    with pytest.raises(ValueError):
        allocator.release((0, 0))
    assert (allocator.free, allocator.owned, allocator.revision) == before


@given(st.lists(st.tuples(st.booleans(), st.integers(0, 12)), max_size=150))
def test_random_allocation_release_traces_preserve_partition(trace):
    allocator = PageAllocator(8, 64)
    for allocate, size in trace:
        if allocate:
            try:
                plan = allocator.plan(size)
            except MemoryError:
                continue
            assert len(plan.pages) == size
            assert not set(plan.pages) & allocator.owned
            allocator.commit(plan)
        else:
            allocator.release(tuple(sorted(allocator.owned)[:size]))
            allocator.shrink(allocator.releasable_capacity())
        allocator.validate()


def test_relocation_only_uses_existing_lower_holes():
    allocator = PageAllocator(4, 12)
    allocator.commit(allocator.plan(10))
    allocator.release((1, 3, 6, 9))
    before = allocator.owned
    with pytest.raises(MemoryError):
        allocator.plan(4, grow=False, before=8)
    assert allocator.owned == before
    plan = allocator.plan(3, grow=False, before=8)
    assert plan.pages == (1, 3, 6)
    allocator.commit(plan)
    allocator.validate()
