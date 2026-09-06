import mlx.core as mx
import pytest

from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget


def store(*, slab_pages=4, budget=None):
    return PageStore(
        KVArena(
            (LayerGeometry(2, 3, 2), LayerGeometry(1, 2, 4)),
            page_size=4,
            slab_pages=slab_pages,
            max_pages=64,
            dtype=mx.float32,
            budget=budget or MemoryBudget(1 << 24),
        )
    )


def append(state, count, value):
    start, end = state.length, state.length + count
    state.reserve(end)
    for layer, geometry in enumerate(state.store.arena.layers):
        state.write(
            layer,
            start,
            mx.full((geometry.heads, count, geometry.key_width), value, mx.float32),
            mx.full((geometry.heads, count, geometry.value_width), -value, mx.float32),
        )
    state.commit(end)


def values(state):
    return state.read(0)[0][0, :, 0].tolist()


def test_adjacent_append_is_one_write_per_layer_and_keeps_shared_prefix_immutable():
    storage = store()
    original = storage.create()
    append(original, 6, 1)
    checkpoint = original.checkpoint()
    branch = storage.create(checkpoint)
    before = storage.arena.counters["kv_write_runs"]
    append(branch, 12, 2)
    assert storage.arena.counters["kv_write_runs"] - before == len(storage.arena.layers)
    assert values(original) == [1] * 6
    assert values(branch) == [1] * 6 + [2] * 12
    restored = storage.create(checkpoint)
    assert values(restored) == [1] * 6
    storage.validate()
    restored.close()
    branch.close()
    checkpoint.close()
    original.close()
    storage.arena.close()


def test_fragmented_append_never_writes_through_another_owners_page():
    storage = store()
    state = storage.create()
    guards = []
    for index in range(4):
        state.reserve((index + 1) * 4)
        guard = storage.arena.allocate(1)[0]
        guards.append(guard)
        storage.arena.write(0, guard, 0, mx.full((2, 4, 3), 99.0), mx.full((2, 4, 2), 99.0))
    before = storage.arena.counters["kv_write_runs"]
    append(state, 16, 7)
    assert storage.arena.counters["kv_write_runs"] - before == 4 * len(storage.arena.layers)
    assert values(state) == [7] * 16
    protected, _ = storage.arena.gather(0, tuple(guards), 16)
    assert mx.all(protected == 99).item()
    state.close()
    storage.arena.release(tuple(guards))
    storage.arena.close()


def test_physical_run_write_rejects_an_unowned_hole_before_mutating_storage():
    storage = store()
    first, hole = storage.arena.allocate(2)
    storage.arena.release((hole,))
    previous = storage.arena.keys[0]
    with pytest.raises(ValueError, match="owned physical run"):
        storage.arena.write(0, first, 0, mx.ones((2, 8, 3)), mx.ones((2, 8, 2)))
    assert storage.arena.keys[0] is previous
    assert storage.arena.counters["kv_write_runs"] == 0
    storage.arena.release((first,))
    storage.arena.close()


def test_linear_partial_continuation_shares_then_competing_writer_copies_one_page():
    storage = store()
    initial = storage.create()
    append(initial, 6, 1)
    prefix = initial.checkpoint()
    addresses = initial.addresses
    initial.close()
    first = storage.create(prefix)
    assert first.addresses == addresses
    second = storage.create(prefix)
    assert second.addresses[0] == addresses[0]
    assert second.addresses[1] != addresses[1]
    assert storage.arena.counters["partial_page_reuses"] == 1
    assert storage.arena.counters["partial_page_copies"] == 1
    assert storage.arena.counters["bytes_copied"] == storage.arena.page_bytes // 2
    append(first, 3, 2)
    append(second, 2, 3)
    assert values(first) == [1] * 6 + [2] * 3
    assert values(second) == [1] * 6 + [3] * 2
    prefix.close()
    storage.validate()
    first.close()
    second.close()
    storage.validate()
    storage.arena.shrink()
    assert storage.arena.budget.snapshot().reserved == 0


def test_older_retained_boundary_branches_without_altering_later_checkpoint():
    storage = store()
    sequence = storage.create()
    append(sequence, 2, 1)
    older = sequence.checkpoint()
    append(sequence, 1, 2)
    later = sequence.checkpoint()
    sequence.close()
    branch = storage.create(older)
    continuation = storage.create(later)
    assert branch.addresses != continuation.addresses
    append(branch, 2, 9)
    append(continuation, 1, 3)
    assert values(branch) == [1, 1, 9, 9]
    assert values(continuation) == [1, 1, 2, 3]
    older.close()
    later.close()
    branch.close()
    continuation.close()
    storage.validate()


def test_cancel_discards_append_suffix_but_keeps_retained_prefix():
    storage = store()
    sequence = storage.create()
    append(sequence, 2, 1)
    checkpoint = sequence.checkpoint()
    sequence.close()
    aborted = storage.create(checkpoint)
    original = aborted.addresses
    append(aborted, 8, 7)
    aborted.close()
    restored = storage.create(checkpoint)
    assert restored.addresses == original
    append(restored, 1, 5)
    assert values(restored) == [1, 1, 5]
    checkpoint.close()
    restored.close()
    storage.validate()


def test_speculative_trim_releases_pages_and_rewrites_rejected_suffix():
    storage = store()
    sequence = storage.create()
    append(sequence, 3, 1)
    prefix = sequence.checkpoint()
    append(sequence, 8, 2)
    sequence.trim(5)
    assert len(sequence.addresses) == 2
    append(sequence, 2, 9)
    assert values(sequence) == [1] * 3 + [2] * 2 + [9] * 2
    with pytest.raises(RuntimeError, match="retained checkpoint"):
        sequence.trim(2)
    prefix.close()
    sequence.close()
    storage.validate()


def test_reservation_failure_does_not_publish_capacity_or_leave_charges():
    budget = MemoryBudget(1024)
    storage = store(slab_pages=1, budget=budget)
    sequence = storage.create()
    append(sequence, 4, 1)
    before = storage.arena.allocator.capacity, budget.snapshot().reserved, sequence.addresses
    # Each page is 256 bytes. Replacing 256 with 1024 requires a 1280-byte peak.
    with pytest.raises(MemoryError):
        sequence.reserve(16)
    assert (
        storage.arena.allocator.capacity,
        budget.snapshot().reserved,
        sequence.addresses,
    ) == before
    assert values(sequence) == [1] * 4
    sequence.close()
    storage.arena.close()
    assert budget.snapshot().reserved == 0


def test_execution_pin_blocks_layout_changes_but_allows_reserved_writes():
    storage = store()
    sequence = storage.create()
    sequence.reserve(4)
    with storage.arena.pin():
        with pytest.raises(RuntimeError, match="pin"):
            sequence.reserve(8)
        with pytest.raises(RuntimeError, match="pin"):
            sequence.close()
        append(sequence, 2, 1)
        storage.arena.complete()
    sequence.close()
    storage.validate()


def test_commit_requires_every_layer_and_checkpoint_rejects_uncommitted_writes():
    storage = store()
    sequence = storage.create()
    sequence.reserve(4)
    sequence.write(0, 0, mx.ones((2, 2, 3)), mx.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="every layer"):
        sequence.commit(2)
    with pytest.raises(RuntimeError, match="reconciled"):
        sequence.checkpoint()
    sequence.close()
    storage.validate()


def test_foreign_or_released_checkpoint_cannot_be_restored():
    storage = store()
    sequence = storage.create()
    checkpoint = sequence.checkpoint()
    with pytest.raises(ValueError):
        store().create(checkpoint)
    checkpoint.close()
    with pytest.raises(ValueError):
        storage.create(checkpoint)
    sequence.close()


def test_cached_continuation_frontier_is_preserved_without_reserving_capacity():
    storage = store()
    initial = storage.create()
    append(initial, 4, 1)
    first = initial.checkpoint()
    initial.close()
    continuation = storage.create(first)
    append(continuation, 1, 2)
    second = continuation.checkpoint()
    assert continuation.addresses == (0, 1)
    continuation.close()
    unrelated = storage.create()
    append(unrelated, 1, 3)
    assert unrelated.addresses == (3,)
    resumed = storage.create(second)
    append(resumed, 4, 4)
    assert resumed.addresses == (0, 1, 2)
    for state in (unrelated, resumed):
        state.close()
    first.close()
    second.close()
    storage.validate()
    storage.arena.close()
