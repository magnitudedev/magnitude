import mlx.core as mx
import pytest

from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.models.state.reclamation import plan_reclamation, reclaim
from magnitude_engine.resources.budget import MemoryBudget


def fixture():
    store = PageStore(
        KVArena(
            (LayerGeometry(1, 2, 2),),
            page_size=4,
            slab_pages=2,
            max_pages=16,
            budget=MemoryBudget(1 << 20),
            dtype=mx.float32,
        )
    )
    checkpoints = []
    for value in (1.0, 2.0, 3.0):
        sequence = store.create()
        sequence.reserve(4)
        sequence.write(0, 0, mx.full((1, 4, 2), value), mx.full((1, 4, 2), -value))
        sequence.commit(4)
        checkpoints.append(sequence.checkpoint())
        sequence.close()
    return store, checkpoints


def read(store, checkpoint):
    sequence = store.create(checkpoint)
    values = sequence.read(0)[0].tolist()
    address = sequence.addresses[0]
    sequence.close()
    return address, values


def test_cold_relocation_updates_every_checkpoint_view_and_physically_releases_a_slab():
    store, checkpoints = fixture()
    before = read(store, checkpoints[2])
    checkpoints[0].close()
    plan = plan_reclamation(store, frozenset(), max_copy_bytes=store.arena.page_bytes)
    assert plan.layout is not None and plan.layout.moves == ((2, 0),)
    result = reclaim(store, plan, frozenset())
    after = read(store, checkpoints[2])
    assert before[0] == 2 and after[0] == 0
    assert before[1] == after[1]
    assert result.pages_moved == 1 and not result.retired
    assert result.bytes_released == store.arena.page_bytes * 2
    assert store.arena.allocator.capacity == 2
    assert store.arena.budget.snapshot().reserved == store.arena.page_bytes * 2
    store.validate()
    for checkpoint in checkpoints:
        checkpoint.close()
    store.arena.close()


def test_retention_eligibility_changes_invalidate_a_plan_without_any_eviction():
    store, checkpoints = fixture()
    eligible = frozenset({checkpoints[2]})
    plan = plan_reclamation(store, eligible, max_copy_bytes=0)
    with pytest.raises(RuntimeError, match="eligibility changed"):
        reclaim(store, plan, frozenset())
    assert not any(checkpoint.closed for checkpoint in checkpoints)
    assert store.arena.allocator.capacity == 4
    result = reclaim(store, plan, eligible)
    assert result.retired == (checkpoints[2],)
    assert result.pages_discarded == 1 and result.pages_moved == 0
    store.validate()
    for checkpoint in checkpoints:
        checkpoint.close()
    store.arena.close()


def test_active_reader_prevents_reclamation_and_new_reader_invalidates_an_existing_plan():
    store, checkpoints = fixture()
    plan = plan_reclamation(store, frozenset({checkpoints[2]}), max_copy_bytes=0)
    active = store.create(checkpoints[2])
    blocked = plan_reclamation(store, frozenset({checkpoints[2]}), max_copy_bytes=0)
    assert blocked.layout is None and "active" in blocked.reason
    with pytest.raises(RuntimeError, match="ownership"):
        reclaim(store, plan, frozenset({checkpoints[2]}))
    active.close()
    for checkpoint in checkpoints:
        checkpoint.close()
    store.arena.close()


def test_replacement_allocation_failure_preserves_retained_state_and_addresses():
    store, checkpoints = fixture()
    checkpoints[0].close()
    before = read(store, checkpoints[2])
    plan = plan_reclamation(store, frozenset(), max_copy_bytes=store.arena.page_bytes)
    store.arena.budget.limit = store.arena.budget.snapshot().reserved
    with pytest.raises(MemoryError):
        reclaim(store, plan, frozenset())
    assert store.arena.allocator.capacity == 4
    assert read(store, checkpoints[2]) == before
    store.validate()
    for checkpoint in checkpoints:
        checkpoint.close()
    store.arena.close()
