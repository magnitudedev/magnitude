import pytest

from magnitude_engine.platform.execution import CapacityError, DeviceContext, Executable, Prepared
from magnitude_engine.state.kv import KVLayout, KVPool


class Buffer:
    def __init__(self, size):
        self.allocated_bytes = size
        self.closed = False

    def close(self):
        assert not self.closed
        self.closed = True


class Completion:
    def __init__(self):
        self.completed = False

    def ready(self):
        return self.completed

    def wait(self):
        self.completed = True


class Driver:
    def allocate(self, size):
        return Buffer(size)

    def dispatch(self, commands):
        for command in commands:
            assert all(not buffer.closed for buffer, _, _ in command.arguments)

    def record(self):
        return Completion()


class Command:
    def __init__(self, arguments):
        self.arguments = arguments

    def close(self):
        self.arguments = ()


class Kernel:
    def bind(self, arguments):
        return Command(arguments)


def test_prepared_and_submitted_views_pin_extent_against_reuse():
    context = DeviceContext(Driver(), 512)
    pool = KVPool(context, KVLayout(2, 1, 4), page_tokens=2, slab_pages=4)
    (first,) = pool.reserve(3)
    (peer,) = pool.reserve(3)
    generation = first.generation
    keys, values = first.views(0)
    descendant = keys.view(keys.spec)
    keys.close()
    prepared = Prepared(
        context,
        Executable((descendant.spec, values.spec), Kernel(), context.driver),
        [descendant, values],
    )
    descendant.close()
    values.close()
    first.close()
    with pytest.raises(CapacityError):
        pool.reserve(1)
    ticket = context.submit([prepared])
    with pytest.raises(CapacityError):
        pool.reserve(1)
    ticket.wait()
    (replacement,) = pool.reserve(1)
    assert replacement.generation != generation
    assert pool.slab_count == 1
    assert context.allocated_bytes == 512
    peer.close()
    replacement.close()
    assert pool.slab_count == 0
    assert context.allocated_bytes == 0
    pool.close()
    context.close()


def test_fragmented_capacity_is_used_when_new_slab_cannot_fit():
    context = DeviceContext(Driver(), 64)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=8)
    runs = [pool.reserve(2)[0] for _ in range(4)]
    runs[0].close()
    runs[2].close()
    fragments = pool.reserve(4)
    assert len(fragments) == 2
    assert sum(run.capacity for run in fragments) == 4
    assert context.allocated_bytes == 64
    for run in [*runs, *fragments]:
        run.close()
    assert context.allocated_bytes == 0
    pool.close()


def test_own_adjacent_growth_and_new_requests_preserve_peer_frontiers():
    context = DeviceContext(Driver(), 64)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=8)
    first, middle, last = (pool.reserve(2)[0] for _ in range(3))
    first.close()
    (unrelated,) = pool.reserve(1)
    (extension,) = pool.reserve(1, after=last)
    assert last.adjacent(extension)
    assert not last.adjacent(unrelated)
    for run in (middle, last, unrelated, extension):
        run.close()
    assert context.allocated_bytes == 0
    pool.close()


def test_pool_retirement_defers_backing_release_through_completion():
    context = DeviceContext(Driver(), 32)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=4)
    (run,) = pool.reserve(1)
    keys, values = run.views(0)
    prepared = Prepared(context, Executable((keys.spec,), Kernel(), context.driver), [keys])
    keys.close()
    values.close()
    run.close()
    ticket = context.submit([prepared])
    pool.close()
    assert context.allocated_bytes == 32
    ticket.wait()
    assert context.allocated_bytes == 0
    context.close()


def test_minimum_slab_fallback_and_independent_forks():
    context = DeviceContext(Driver(), 8)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=32)
    (run,) = pool.reserve(1)
    assert run.exclusive
    fork = run.fork()
    assert not run.exclusive
    run.close()
    assert context.allocated_bytes == 8
    assert fork.exclusive
    fork.close()
    assert context.allocated_bytes == 0
    pool.close()


def test_contiguous_read_view_retains_every_extent_until_its_consumer_finishes():
    from magnitude_engine.state.kv import KVSpan, physical_runs

    context = DeviceContext(Driver(), 256)
    pool = KVPool(context, KVLayout(2, 1, 2), page_tokens=2, slab_pages=4)
    (first,) = pool.reserve(2)
    (second,) = pool.reserve(3, after=first)
    (peer,) = pool.reserve(2, after=second)
    runs = physical_runs((KVSpan(first, 10, 2), KVSpan(second, 12, 3)))
    assert len(runs) == 1
    assert (runs[0].start, runs[0].length, runs[0].capacity) == (10, 5, 6)
    keys, values = runs[0].views(1)
    assert keys.offset == 128 and values.offset == 192
    descendant = keys.view(keys.spec)
    keys.close()
    prepared = Prepared(
        context,
        Executable((descendant.spec, values.spec), Kernel(), context.driver),
        (descendant, values),
    )
    descendant.close()
    values.close()
    first.close()
    second.close()
    ticket = context.submit((prepared,))
    with pytest.raises(CapacityError):
        pool.reserve(1)
    ticket.wait()
    (replacement,) = pool.reserve(6)
    assert replacement.capacity == 6 and pool.slab_count == 1
    replacement.close()
    peer.close()
    pool.close()
    assert context.allocated_bytes == 0
    context.close()


def test_read_runs_preserve_unused_slots_logical_gaps_and_foreign_allocations():
    from magnitude_engine.state.kv import KVSpan, physical_runs

    context = DeviceContext(Driver(), 128)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=2, slab_pages=4)
    (first,) = pool.reserve(2)
    (second,) = pool.reserve(2, after=first)
    assert first.adjacent(second)
    assert len(physical_runs((KVSpan(first, 0, 1), KVSpan(second, 1, 2)))) == 2
    assert len(physical_runs((KVSpan(first, 0, 2), KVSpan(second, 3, 2)))) == 2
    with pytest.raises(ValueError, match="ordered disjoint"):
        physical_runs((KVSpan(first, 0, 2), KVSpan(second, 1, 2)))
    (foreign,) = pool.reserve(8)
    assert len(physical_runs((KVSpan(second, 0, 2), KVSpan(foreign, 2, 8)))) == 2
    first.close()
    second.close()
    foreign.close()
    pool.close()
    assert context.allocated_bytes == 0
    context.close()


def test_segmented_views_pin_selected_extents_but_allow_gap_reuse():
    from magnitude_engine.state.kv import KVReadGroup, KVSpan, grouped_reads, physical_runs

    context = DeviceContext(Driver(), 64)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=8)
    first, gap, last = (pool.reserve(n)[0] for n in (2, 3, 3))
    history = physical_runs((KVSpan(last, 7, 2), KVSpan(first, 12, 2)))
    (group,) = grouped_reads(history)
    assert group.segments == ((5, 7, 2), (0, 12, 2))
    assert group.capacity == 8 and group.segment_capacity == 3
    with pytest.raises(ValueError, match="ordered disjoint"):
        KVReadGroup(tuple(reversed(history)))
    keys, values = group.views(0)
    prepared = Prepared(
        context, Executable((keys.spec, values.spec), Kernel(), context.driver), (keys, values)
    )
    keys.close()
    values.close()
    first.close()
    last.close()
    gap_generation = gap.generation
    gap.close()
    (replacement,) = pool.reserve(3)
    assert replacement.generation != gap_generation
    with pytest.raises(CapacityError):
        pool.reserve(1)
    ticket = context.submit((prepared,))
    with pytest.raises(CapacityError):
        pool.reserve(1)
    ticket.wait()
    replacement.close()
    pool.close()
    assert context.allocated_bytes == 0
    context.close()


def test_growth_horizon_shares_backing_without_claiming_future_pages():
    context = DeviceContext(Driver(), 1024)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=2)
    first = pool.reserve(2, preferred_tokens=32)[0]
    following = pool.reserve(2, after=first)[0]
    assert first.capacity == following.capacity == 2
    assert pool.slab_count == 1
    assert pool.allocated_tokens == 32
    first.close()
    following.close()
    assert context.allocated_bytes == 0
    pool.close()
    context.close()


def test_optional_growth_cannot_prevent_a_feasible_claim():
    context = DeviceContext(Driver(), 32)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=2)
    run = pool.reserve(2, preferred_tokens=1024)[0]
    assert run.capacity == 2
    assert pool.allocated_tokens == 2
    run.close()
    pool.close()
    context.close()


def test_read_window_stays_stable_as_visibility_grows():
    from magnitude_engine.state.kv import KVSpan, grouped_reads, physical_runs

    context = DeviceContext(Driver(), 256)
    pool = KVPool(context, KVLayout(1, 1, 1), page_tokens=1, slab_pages=32)
    first = pool.reserve(2)[0]
    (before,) = grouped_reads(physical_runs((KVSpan(first, 0, 2),)))
    following = pool.reserve(2, after=first)[0]
    (after,) = grouped_reads(physical_runs((KVSpan(first, 0, 2), KVSpan(following, 2, 2))))
    assert before.capacity == after.capacity == 32
    assert before.segments == ((0, 0, 2),)
    assert after.segments == ((0, 0, 4),)
    first.close()
    following.close()
    pool.close()
    context.close()
