import pytest

import ops
from engine.state.sequence import StateStore
from tests.ops.test_compiler import Runtime


class Done:
    def __init__(self, device, check=lambda: None):
        self.device, self.check = device, check
        self.waits = 0

    def wait(self):
        self.check()
        self.waits += 1


@pytest.fixture
def store():
    device = ops.DeviceRuntime(Runtime(), budget_bytes=1 << 20)
    value = ops.TensorSpec((4,), ops.DType.F32)
    state = StateStore(
        device,
        context_capacity=16,
        history_capacity=32,
        history_specs=(ops.TensorSpec((32, 4), ops.DType.F32),),
        initial_values=lambda: (device.allocate(value),),
    )
    yield state
    state.close()
    device.close()


def accept(state, count):
    advance = state.begin(count)
    following = tuple(state.store.device.allocate(v.spec) for v in state.values)
    advance.submitted(Done(state.store.device), following)
    advance.commit()
    return advance


def test_prefix_shared_and_private_tail_isolated(store):
    parent = store.create()
    accept(parent, 8)
    checkpoint = parent.checkpoint()
    reserved = store.device.allocated_bytes
    branches = [checkpoint.fork() for _ in range(6)]
    assert store.device.allocated_bytes == reserved
    assert store.occupied_rows == 8
    assert all(b.history_ranges == ((0, 8),) for b in branches)
    accept(parent, 2)
    accept(branches[0], 3)
    assert checkpoint.position == 8
    assert branches[0].history_ranges == ((0, 8), (10, 3))
    assert branches[1].history_ranges == ((0, 8),)
    assert store.occupied_rows == 13
    parent.close()
    checkpoint.close()
    for b in branches:
        b.close()
    assert store.occupied_rows == 0


def test_abort_waits_before_recycling_destinations(store):
    state = store.create()
    advance = state.begin(5)
    done = Done(
        store.device,
        lambda: (_ for _ in ()).throw(AssertionError()) if store.occupied_rows != 5 else None,
    )
    advance.submitted(done, (store.device.allocate(state.values[0].spec),))
    advance.abort()
    assert done.waits == 1
    assert state.position == 0 and store.occupied_rows == 0
    assert state.begin(5).destinations == (0, 1, 2, 3, 4)


def test_pending_state_cannot_checkpoint_or_advance(store):
    state = store.create()
    state.begin(2)
    with pytest.raises(RuntimeError):
        state.checkpoint()
    with pytest.raises(RuntimeError):
        state.begin(2)
    state.pending.abort()
    accept(state, 2)
    assert state.position == 2


def test_reclamation_counts_shared_values_once(store):
    parent = store.create()
    child = parent.checkpoint()
    fork = child.fork()
    assert store.reclaimable((parent, fork)) == 0
    child.close()
    assert store.reclaimable((parent,)) == 0
    assert store.reclaimable((parent, fork, parent)) == parent.values[0].allocated_bytes


def test_fragmented_history_reservation_and_reuse(store):
    a, b, c = (store.create() for _ in range(3))
    accept(a, 8)
    accept(b, 8)
    accept(c, 8)
    b.close()
    advance = a.begin(8)
    assert advance.destinations == tuple(range(8, 16))
    advance.abort()
    assert a.history_ranges == ((0, 8),)
    assert store.occupied_rows == 16


def test_idle_history_can_be_reallocated(store):
    state = store.create()
    old = store.history[0]
    assert store.release_idle() == 0
    state.close()
    assert store.release_idle() > 0
    fresh = store.history[0]
    assert fresh is not old


def test_trim_keeps_checkpoint_logical_history_and_position(store):
    parent = store.create()
    accept(parent, 8)
    checkpoint = parent.checkpoint()
    parent.trim_history(5)
    assert parent.position == 8
    assert parent.history_ranges == ((5, 3),)
    descendant = parent.checkpoint()
    fork = descendant.fork()
    assert fork.history_ranges == ((5, 3),)
    original = checkpoint.fork()
    assert original.history_ranges == ((0, 8),)
    accept(parent, 2)
    parent.trim_history(8)
    assert parent.history_ranges == ((8, 2),)
    assert store.occupied_rows == 10
    original.close()
    checkpoint.close()
    fork.close()
    descendant.close()
    assert store.occupied_rows == 2
    parent.trim_history(10)
    assert store.occupied_rows == 0 and parent.position == 10


def test_successor_schema_failure_preserves_accepted_collection(store):
    state = store.create()
    old = state.values
    advance = state.begin(3)
    with pytest.raises(ValueError):
        advance.submitted(Done(store.device), old)
    with pytest.raises(ValueError):
        advance.submitted(Done(store.device), ())
    assert state.values == old and state.position == 0
    advance.abort()
    assert store.occupied_rows == 0
    retained = old[0].fork()
    advance = state.begin(1)
    advance.submitted(Done(store.device), (retained,))
    advance.commit()
    assert state.values == (retained,)
    assert retained.allocated_bytes > 0


def test_forks_of_forks_survive_parent_first_close(store):
    a = store.create()
    accept(a, 4)
    checkpoint = a.checkpoint()
    b = checkpoint.fork()
    a.close()
    checkpoint.close()
    accept(b, 2)
    checkpoint = b.checkpoint()
    c = checkpoint.fork()
    b.close()
    checkpoint.close()
    assert c.position == 6 and c.history_ranges == ((0, 6),)
    accept(c, 1)
    assert store.occupied_rows == 7


def test_capacity_failure_is_atomic(store):
    a, b = store.create(), store.create()
    accept(a, 16)
    accept(b, 16)
    c = store.create()
    with pytest.raises(ops.CapacityError):
        c.begin(1)
    assert c.pending is None and c.position == 0
    assert store.occupied_rows == 32


@pytest.mark.parametrize("history_only", [True, False])
def test_history_and_values_do_not_require_each_other(history_only):
    device = ops.DeviceRuntime(Runtime(), budget_bytes=1 << 20)
    value_spec = ops.TensorSpec((4,), ops.DType.F32)
    store = StateStore(
        device,
        context_capacity=16,
        history_capacity=32,
        history_specs=(ops.TensorSpec((32, 4), ops.DType.F32),) if history_only else (),
        initial_values=lambda: () if history_only else (device.allocate(value_spec),),
    )
    try:
        parent = store.create()
        accept(parent, 4)
        checkpoint = parent.checkpoint()
        branch = checkpoint.fork()
        accept(branch, 2)
        assert parent.position == 4 and branch.position == 6
        assert store.occupied_rows == (6 if history_only else 0)
    finally:
        store.close()
        device.close()
