"""Prepared writes preserve the same page authority as incremental appends."""

import mlx.core as mx
import pytest

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
    for k, v in zip(prepared.keys, prepared.values, strict=True):
        for row in range(prepared.positions.size):
            destination = prepared.destinations[row : row + 1]
            k = mx.slice_update(
                k, mx.full((k.shape[0], 1, k.shape[2]), 10 + row), destination, axes=[1]
            )
            v = mx.slice_update(
                v, mx.full((v.shape[0], 1, v.shape[2]), -10 - row), destination, axes=[1]
            )
        keys.append(k)
        values.append(v)
    return tuple(keys), tuple(values)


@pytest.mark.parametrize("lengths", [(0,), (3,), (4,), (1, 4, 7)])
def test_prepared_append_stages_one_token_per_row_without_committing(storage, lengths):
    states = tuple(storage.create() for _ in lengths)
    for state, length in zip(states, lengths, strict=True):
        append(state, length, 1)
        state.reserve(length + 1)
    before = storage.arena.budget.snapshot().reserved
    with storage.arena.pin():
        prepared = prepare_decode_append(states)
        assert prepared.positions.tolist() == list(lengths)
        assert prepared.capacity == storage.arena.allocator.capacity * prepared.page_size
        assert prepared.table.addresses == tuple(state.addresses for state in states)
        output = tensor_append(prepared)
        mx.eval(output)
        assert tuple(state.length for state in states) == lengths
        assert all(
            state.visible_length(0) == length for state, length in zip(states, lengths, strict=True)
        )
        prepared.install(*output)
        assert tuple(state.length for state in states) == lengths
        for layer in range(len(storage.arena.layers)):
            staged = read_layer(states, layer, pending_tokens=1)
            assert staged.lengths == tuple(length + 1 for length in lengths)
        with pytest.raises(RuntimeError, match="already installed"):
            prepared.install(*output)
        for state, length in zip(states, lengths, strict=True):
            state.commit(length + 1)
    assert storage.arena.budget.snapshot().reserved == before
    for row, (state, length) in enumerate(zip(states, lengths, strict=True)):
        assert values(state) == [1] * length + [10 + row]
    storage.validate()


def test_prepared_append_preserves_checkpoint_and_old_buffer_versions(storage):
    original = storage.create()
    append(original, 3, 1)
    checkpoint = original.checkpoint()
    branch = storage.create(checkpoint)
    assert branch.addresses != original.addresses  # Existing tail COW owns this decision.
    branch.reserve(4)
    with storage.arena.pin():
        prepared = prepare_decode_append((branch,))
        old = tuple(mx.array(a) for a in (*prepared.keys, *prepared.values))
        mx.eval(old)
        output = tensor_append(prepared)
        prepared.install(*output)
        mx.eval(output)
        assert all(
            mx.array_equal(a, b).item()
            for a, b in zip(old, (*prepared.keys, *prepared.values), strict=True)
        )
        branch.commit(4)
    restored = storage.create(checkpoint)
    assert values(original) == values(restored) == [1, 1, 1]
    assert values(branch) == [1, 1, 1, 10]


def test_preparation_requires_a_pin_and_reserved_capacity(storage):
    state = storage.create()
    with pytest.raises(RuntimeError, match="execution pin"):
        prepare_decode_append((state,))
    with storage.arena.pin(), pytest.raises(ValueError, match="reserved capacity"):
        prepare_decode_append((state,))


def test_invalid_later_row_does_not_stage_an_earlier_row(storage):
    first, second = storage.create(), storage.create()
    first.reserve(1)
    old = storage.arena.keys
    with storage.arena.pin(), pytest.raises(ValueError, match="reserved capacity"):
        prepare_decode_append((first, second))
    assert storage.arena.keys is old
    assert first.visible_length(0) == second.visible_length(0) == 0


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype"])
def test_invalid_result_is_rejected_before_any_publication(storage, invalid):
    state = storage.create()
    state.reserve(1)
    with storage.arena.pin():
        prepared = prepare_decode_append((state,))
        keys, vals = tensor_append(prepared)
        if invalid == "missing":
            vals = vals[:-1]
        elif invalid == "shape":
            vals = (*vals[:-1], vals[-1][:, :, :1])
        else:
            vals = (*vals[:-1], vals[-1].astype(mx.float16))
        with pytest.raises(ValueError, match="physical"):
            prepared.install(keys, vals)
        assert storage.arena.keys is prepared.keys
        assert storage.arena.values is prepared.values
        assert state.visible_length(0) == state.visible_length(1) == 0


def test_superseded_buffer_version_cannot_overwrite_another_append(storage):
    first, second = storage.create(), storage.create()
    first.reserve(1)
    second.reserve(1)
    with storage.arena.pin():
        prepared = prepare_decode_append((first,))
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
            prepared.install(*output)
        assert storage.arena.keys is current
        assert first.visible_length(0) == 0
        assert second.visible_length(0) == 1


def test_installation_requires_the_execution_pin_to_remain_live(storage):
    state = storage.create()
    state.reserve(1)
    with storage.arena.pin():
        prepared = prepare_decode_append((state,))
        output = tensor_append(prepared)
    with pytest.raises(RuntimeError, match="execution pin"):
        prepared.install(*output)
    assert state.visible_length(0) == 0


def test_prepared_append_reuses_immutable_prefix_validation(storage):
    original = storage.create()
    append(original, 3, 1)
    checkpoint = original.checkpoint()
    # Ask the shared validator to authorize an overwrite of retained input. Both
    # direct writes and prepared transitions must use this same authority check.
    original._written = [2] * len(storage.arena.layers)
    original.length = 2
    with storage.arena.pin(), pytest.raises(RuntimeError, match="immutable page prefix"):
        prepare_decode_append((original,))
    original.length = 3
    original._written = [3] * len(storage.arena.layers)
    checkpoint.close()
