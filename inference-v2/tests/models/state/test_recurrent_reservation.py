"""Admission and batch-contract checks; no tensors or device work are created."""

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from magnitude_engine.models.architectures.qwen35.recurrence.operation import RecurrentMixer
from magnitude_engine.models.state.hybrid import HybridState, HybridStateStore, HybridTransaction
from magnitude_engine.models.state.recurrent import RecurrentImage, RecurrentLayout, StateTensor
from magnitude_engine.resources.budget import MemoryBudget


@pytest.fixture
def admission():
    budget = MemoryBudget(4096)
    layouts = (RecurrentLayout((StateTensor((1, 64), mx.float32),), 48),)
    store = HybridStateStore(Mock(), layouts, budget)
    initial = RecurrentImage(layouts, 1, budget).acquire(0)
    destination = RecurrentImage(layouts, 1, budget).acquire(0)
    pages = Mock(length=7)
    state = HybridState(store, pages, initial)
    try:
        yield state, destination, budget
    finally:
        if state.active is not None:
            state.active.close()
        initial.close()
        destination.close()
        assert budget.snapshot().reserved == 0


@pytest.mark.parametrize(("count", "committed"), [(1, 0), (1, 1), (4, 4), (1024, 1024)])
def test_boundary_only_admission_needs_no_repair_capacity(admission, count, committed):
    state, destination, budget = admission
    state_bytes = budget.snapshot().reserved
    budget.limit = state_bytes
    transaction = HybridTransaction(state, count, destination, committed_inputs=committed)
    assert budget.snapshot().reserved == state_bytes
    assert budget.snapshot().owners == {"recurrent-state": state_bytes}
    state.pages.reserve.assert_called_once_with(7 + count)
    transaction.close()
    # Transaction closure retires its destination, but leaves the committed image.
    assert budget.snapshot().reserved == state.recurrent.image.reservation.size


@pytest.mark.parametrize("committed", [0, 1, 3])
def test_tentative_admission_reserves_the_full_retained_trace(admission, committed):
    state, destination, budget = admission
    state_bytes = budget.snapshot().reserved
    budget.limit = state_bytes + 4 * 48
    transaction = HybridTransaction(state, 4, destination, committed_inputs=committed)
    assert budget.snapshot().owners == {
        "recurrent-state": state_bytes,
        "recurrent-advance": 4 * 48,
    }
    transaction.close()
    assert "recurrent-advance" not in budget.snapshot().owners


def test_tentative_admission_fails_before_page_reservation(admission):
    state, destination, budget = admission
    state_bytes = budget.snapshot().reserved
    budget.limit = state_bytes
    with pytest.raises(MemoryError, match="recurrent-advance"):
        HybridTransaction(state, 4, destination)
    assert budget.snapshot().reserved == state_bytes
    assert state.active is None
    state.pages.reserve.assert_not_called()


def test_page_admission_failure_releases_repair_capacity(admission):
    state, destination, budget = admission
    state_bytes = budget.snapshot().reserved
    state.pages.reserve.side_effect = MemoryError("page admission")
    with pytest.raises(MemoryError, match="page admission"):
        HybridTransaction(state, 4, destination)
    assert budget.snapshot().reserved == state_bytes
    assert state.active is None


@pytest.mark.parametrize("commitments", [(0, 4), (4, 0), (1, 3)])
def test_recurrent_mixer_rejects_mixed_commitments_before_execution(commitments):
    operation = Mock()
    mixer = RecurrentMixer(0, operation)
    states = tuple(SimpleNamespace(active=SimpleNamespace(committed_inputs=n)) for n in commitments)
    with pytest.raises(ValueError, match="uniform committed-input prefix"):
        mixer.compute_batch(Mock(), states, Mock())
    operation.compute_batch.assert_not_called()


@pytest.mark.parametrize("committed", [0, 1, 4])
def test_recurrent_mixer_preserves_uniform_commitment(committed):
    operation = Mock()
    mixer = RecurrentMixer(0, operation)
    slots = (Mock(), Mock())
    states = tuple(
        SimpleNamespace(active=SimpleNamespace(committed_inputs=committed), slots=(slot,))
        for slot in slots
    )
    hidden, scope = Mock(), Mock()
    result = mixer.compute_batch(hidden, states, scope)
    assert result is operation.compute_batch.return_value
    operation.compute_batch.assert_called_once_with(
        hidden, slots, scope, committed_inputs=committed
    )
