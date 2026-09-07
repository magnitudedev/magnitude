"""Prepared writes preserve the same page authority as incremental appends."""

import mlx.core as mx
import pytest

from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.state.decode import prepare_decode_append
from magnitude_engine.models.state.views import read_layer
from tests.models.state.test_pages import append, store, values


@pytest.fixture(autouse=True)
def cpu_execution():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def scope():
    owner = ExecutionOwner()
    with owner.scope() as scope:
        yield scope
    owner.close()


@pytest.fixture
def storage():
    storage = store()
    yield storage
    for checkpoint in tuple(storage._checkpoints.values()):
        checkpoint.close()
    for state in tuple(storage._sequences.values()):
        state.close()
    storage.arena.close()
    assert storage.arena.budget.snapshot().reserved == 0


def tensor_append(prepared):
    """Pure reference transition; no page or arena mutation inside this function."""
    keys, values = [], []
    from magnitude_engine.models.state.tail import split_tail

    for buffer, geometry in zip(prepared.tails, prepared._arena.layers, strict=True):
        k, v = split_tail(
            buffer,
            prepared.positions.size,
            geometry.heads,
            prepared.page_size,
            geometry.key_width,
            geometry.value_width,
        )
        for row in range(prepared.positions.size):
            destination = mx.concatenate(
                [mx.array([row], mx.int32), prepared.offsets[row : row + 1]]
            )
            k = mx.slice_update(
                k, mx.full((1, k.shape[1], 1, k.shape[3]), 10 + row), destination, axes=[0, 2]
            )
            v = mx.slice_update(
                v, mx.full((1, v.shape[1], 1, v.shape[3]), -10 - row), destination, axes=[0, 2]
            )
        keys.append(k)
        values.append(v)
    return tuple(
        mx.concatenate([k.reshape(-1), v.reshape(-1)]) for k, v in zip(keys, values, strict=True)
    )


@pytest.mark.parametrize("invalid", ["rows", "start", "horizon", "heads", "dtype"])
def test_append_view_rejects_mismatched_storage_before_kernel_launch(storage, invalid):
    from dataclasses import replace

    from magnitude_engine.models.state.views import AppendView

    row = storage.create()
    append(row, 3, 1)
    view = read_layer((row,), 0)
    k = mx.zeros((view.keys.shape[0], 4, view.keys.shape[-1]), view.keys.dtype)
    v = mx.zeros((view.values.shape[0], 4, view.values.shape[-1]), view.values.dtype)
    tail = AppendView(k, v, 1)
    if invalid == "start":
        tail = replace(tail, start=-1)
    elif invalid == "horizon":
        tail = replace(tail, keys=k[:, :1], values=v[:, :1])
    elif invalid == "heads":
        tail = replace(tail, keys=mx.concatenate([k, k]), values=mx.concatenate([v, v]))
    elif invalid == "dtype":
        tail = replace(tail, keys=k.astype(mx.float16))
    with pytest.raises(ValueError):
        replace(view, tails=(tail, tail) if invalid == "rows" else (tail,))


@pytest.mark.parametrize("lengths", [(0,), (3,), (4,), (1, 4, 7)])
def test_prepared_append_stages_one_token_per_row_without_committing(storage, scope, lengths):
    states = tuple(storage.create() for _ in lengths)
    for state, length in zip(states, lengths, strict=True):
        append(state, length, 1)
        state.reserve(length + 1)
    before = storage.arena.budget.snapshot().reserved
    with storage.arena.pin():
        prepared = prepare_decode_append(states, scope)
        assert prepared.positions.tolist() == list(lengths)
        assert prepared.capacity == storage.arena.allocator.capacity * prepared.page_size
        assert prepared.table.addresses == tuple(state.addresses for state in states)
        output = tensor_append(prepared)
        mx.eval(output)
        assert tuple(state.length for state in states) == lengths
        assert all(
            state.visible_length(0) == length for state, length in zip(states, lengths, strict=True)
        )
        prepared.install(output)
        assert tuple(state.length for state in states) == lengths
        for layer in range(len(storage.arena.layers)):
            staged = read_layer(states, layer, pending_tokens=1)
            assert staged.lengths == tuple(length + 1 for length in lengths)
        with pytest.raises(RuntimeError, match="already installed"):
            prepared.install(output)
        for state, length in zip(states, lengths, strict=True):
            state.commit(length + 1)
    scope.seal().complete()
    expected_tail_bytes = len(states) * storage.arena.page_bytes
    assert storage.arena.budget.snapshot().reserved == before + expected_tail_bytes
    for row, (state, length) in enumerate(zip(states, lengths, strict=True)):
        assert values(state) == [1] * length + [10 + row]
    storage.validate()


def test_prepared_append_preserves_checkpoint_and_old_buffer_versions(storage, scope):
    original = storage.create()
    append(original, 3, 1)
    checkpoint = original.checkpoint()
    branch = storage.create(checkpoint)
    assert branch.addresses != original.addresses  # Existing tail COW owns this decision.
    branch.reserve(4)
    with storage.arena.pin():
        prepared = prepare_decode_append((branch,), scope)
        old = tuple(mx.array(a) for a in (*prepared.keys, *prepared.values))
        mx.eval(old)
        output = tensor_append(prepared)
        prepared.install(output)
        mx.eval(output)
        assert all(
            mx.array_equal(a, b).item()
            for a, b in zip(old, (*prepared.keys, *prepared.values), strict=True)
        )
        branch.commit(4)
    restored = storage.create(checkpoint)
    assert values(original) == values(restored) == [1, 1, 1]
    assert values(branch) == [1, 1, 1, 10]


def test_preparation_requires_a_pin_and_reserved_capacity(storage, scope):
    state = storage.create()
    with pytest.raises(RuntimeError, match="execution pin"):
        prepare_decode_append((state,), scope)
    with storage.arena.pin(), pytest.raises(ValueError, match="reserved capacity"):
        prepare_decode_append((state,), scope)


def test_invalid_later_row_does_not_stage_an_earlier_row(storage, scope):
    first, second = storage.create(), storage.create()
    first.reserve(1)
    old = storage.arena.keys
    with storage.arena.pin(), pytest.raises(ValueError, match="reserved capacity"):
        prepare_decode_append((first, second), scope)
    assert storage.arena.keys is old
    assert first.visible_length(0) == second.visible_length(0) == 0


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype"])
def test_invalid_result_is_rejected_before_any_publication(storage, scope, invalid):
    state = storage.create()
    state.reserve(1)
    with storage.arena.pin():
        prepared = prepare_decode_append((state,), scope)
        buffers = tensor_append(prepared)
        if invalid == "missing":
            buffers = buffers[:-1]
        elif invalid == "shape":
            buffers = (*buffers[:-1], buffers[-1][:1])
        else:
            buffers = (*buffers[:-1], buffers[-1].astype(mx.float16))
        with pytest.raises(ValueError, match="KV producer|reserved geometry"):
            prepared.install(buffers)
        assert storage.arena.keys is prepared.keys
        assert storage.arena.values is prepared.values
        assert state.visible_length(0) == state.visible_length(1) == 0


def test_superseded_buffer_version_cannot_overwrite_another_append(storage, scope):
    first, second = storage.create(), storage.create()
    first.reserve(1)
    second.reserve(1)
    with storage.arena.pin():
        prepared = prepare_decode_append((first,), scope)
        output = tensor_append(prepared)
        geometry = storage.arena.layers[0]
        second.write(
            0,
            0,
            mx.ones((geometry.heads, 1, geometry.key_width)),
            mx.ones((geometry.heads, 1, geometry.value_width)),
        )
        current = storage.arena.keys
        with pytest.raises(RuntimeError, match="superseded"):
            prepared.install(output)
        assert storage.arena.keys is current
        assert first.visible_length(0) == 0
        assert second.visible_length(0) == 1


def test_installation_requires_the_execution_pin_to_remain_live(storage, scope):
    state = storage.create()
    state.reserve(1)
    with storage.arena.pin():
        prepared = prepare_decode_append((state,), scope)
        output = tensor_append(prepared)
    with pytest.raises(RuntimeError, match="execution pin"):
        prepared.install(output)
    assert state.visible_length(0) == 0


def test_prepared_append_reuses_immutable_prefix_validation(storage, scope):
    original = storage.create()
    append(original, 3, 1)
    checkpoint = original.checkpoint()
    # Ask the shared validator to authorize an overwrite of retained input. Both
    # direct writes and prepared transitions must use this same authority check.
    original._written = [2] * len(storage.arena.layers)
    original.length = 2
    with storage.arena.pin(), pytest.raises(RuntimeError, match="immutable page prefix"):
        prepare_decode_append((original,), scope)
    original.length = 3
    original._written = [3] * len(storage.arena.layers)
    checkpoint.close()


def append_token(states):
    owner = ExecutionOwner()
    for state in states:
        state.reserve(state.length + 1)
    with owner.scope() as scope:
        scope.enter(states[0].store.arena.pin())
        prepared = prepare_decode_append(states, scope)
        output = tensor_append(prepared)
        prepared.install(output)
        for state in states:
            state.commit(state.length + 1)
        scope.seal(*output[0], *output[1]).complete()
    owner.close()


def test_tail_crosses_capacity_and_batch_changes_without_copying_history_each_step(storage):
    states = (storage.create(), storage.create())
    append(states[0], 3, 1)
    append(states[1], 7, 2)
    capacity = storage.arena.page_size
    for _ in range(capacity + 3):
        append_token(states)
    assert values(states[0]) == [1] * 3 + [10] * (capacity + 3)
    assert values(states[1]) == [2] * 7 + [11] * (capacity + 3)
    # Taking one row out of a shared image must retain the other's physical charge.
    append_token((states[1],))
    assert values(states[1]) == [2] * 7 + [11] * (capacity + 3) + [10]
    assert storage.arena.counters["tail_seals"] == 2
    storage.validate()


def test_tail_snapshot_survives_replacement_and_rejection(storage):
    state = storage.create()
    append(state, 3, 1)
    append_token((state,))
    snapshot = read_layer((state,), 0)
    append_token((state,))
    state.trim(4)
    assert snapshot.gather(0)[0][0, :, 0].tolist() == [1, 1, 1, 10]
    assert values(state) == [1, 1, 1, 10]
    append_token((state,))
    checkpoint = state.checkpoint()
    assert state.tail is None
    restored = storage.create(checkpoint)
    assert values(restored) == values(state) == [1, 1, 1, 10, 10]
    storage.validate()


def test_compiled_append_does_not_replace_or_return_full_history(storage, scope):
    state = storage.create()
    append(state, 31, 1)
    state.reserve(32)
    original = storage.arena.keys, storage.arena.values
    with storage.arena.pin():
        prepared = prepare_decode_append((state,), scope)
        result = tensor_append(prepared)
        prepared.install(result)
        state.commit(32)
        assert storage.arena.keys is original[0] and storage.arena.values is original[1]
        assert sum(a.nbytes for a in result) == storage.arena.page_bytes
    assert values(state) == [1] * 31 + [10]


@pytest.mark.parametrize("widths", [(32, 32), (7, 5)])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_combined_tail_write_preserves_independent_rows_and_unequal_widths(widths, count, dtype):
    from magnitude_engine.models.state.decode import update_tail
    from magnitude_engine.models.state.tail import split_tail

    batch, heads, capacity = 3, 2, 16
    dk, dv = widths
    k = mx.random.normal((batch, heads, capacity, dk)).astype(dtype)
    v = mx.random.normal((batch, heads, capacity, dv)).astype(dtype)
    new_k = mx.random.normal((batch, heads, count, dk)).astype(dtype)
    new_v = mx.random.normal((batch, heads, count, dv)).astype(dtype)
    offsets = mx.array([0, 5, capacity - count], mx.int32)
    buffer = mx.concatenate([k.reshape(-1), v.reshape(-1)])
    result = update_tail(buffer, new_k, new_v, offsets, capacity)
    actual = split_tail(result, batch, heads, capacity, dk, dv)
    expected = []
    for old, new in ((k, new_k), (v, new_v)):
        for row, offset in enumerate([0, 5, capacity - count]):
            old = mx.slice_update(old, new[row : row + 1], mx.array([row, offset]), axes=[0, 2])
        expected.append(old)
    mx.eval(*actual, *expected)
    assert all(mx.array_equal(x, y).item() for x, y in zip(actual, expected, strict=True))
