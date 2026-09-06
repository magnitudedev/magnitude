from types import SimpleNamespace

import mlx.core as mx
import pytest

from magnitude_engine.models.residency import ModelDescriptor, ModelResources, PagedRequirements
from magnitude_engine.models.state.arena import LayerGeometry
from magnitude_engine.models.state.binding import PagedHybridFactory
from magnitude_engine.resources.budget import MemoryBudget


@pytest.mark.parametrize("context,rows", [(1, 1), (17, 3), (513, 1), (1088, 2)])
def test_declared_context_is_reachable_for_all_rows_with_whole_slab_growth(context, rows):
    budget = MemoryBudget(16 << 20)
    resources = ModelResources(budget=budget, context_tokens=context, max_active=rows)
    bound = SimpleNamespace(
        descriptor=ModelDescriptor("fixture", context, 64, "fixture", "fixture"),
        state=PagedRequirements((LayerGeometry(1, 2, 2),), mx.float32),
    )
    states = PagedHybridFactory(page_size=16, slab_pages=32).create(bound, resources)
    sequences = []
    try:
        for _ in range(rows):
            state = states.create()
            sequences.append(state)
            states.reserve(state, context)
        assert all(len(state.addresses) * 16 >= context for state in sequences)
        assert states.pages.arena.allocator.capacity % 32 == 0
    finally:
        for state in sequences:
            states.release(state)
        resources.close()
    assert budget.snapshot().reserved == 0
