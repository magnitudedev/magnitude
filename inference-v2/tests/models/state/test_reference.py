"""Opt-in differential checks against an external, read-only PoC checkout.

Run with MLX_POC_REFERENCE pointing at the reviewed snapshot's engine directory.
The oracle is imported for tests only; it is not distributed with this engine.
"""

import importlib
import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from magnitude_engine.models.state.placement import PageAllocator, PlacementHints


@pytest.fixture(scope="module")
def reference():
    path = os.environ.get("MLX_POC_REFERENCE")
    if not path:
        pytest.skip("set MLX_POC_REFERENCE to run the external differential oracle")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("mlxengine.kv.pool")
    finally:
        sys.path.remove(path)


class OracleBudget:
    def can_allocate(self, size):
        return True

    def charge(self, size, owner):
        pass

    def release(self, size, owner):
        pass


@settings(max_examples=80, deadline=None)
@given(
    st.lists(
        st.tuples(st.booleans(), st.integers(1, 8), st.integers(0, 31)), min_size=1, max_size=80
    )
)
def test_placement_matches_reviewed_mechanism(reference, trace):
    old = reference.PagePool([], page_size=4, slab_pages=8, max_pages=64, accountant=OracleBudget())
    new = PageAllocator(8, 64)
    for allocate, count, adjacent in trace:
        if allocate:
            frontiers = frozenset({(adjacent + 3) % 64, (adjacent + 7) % 64})
            try:
                expected = old.alloc(
                    count, context=reference.AllocationContext(adjacent, frontiers)
                )
            except MemoryError:
                with pytest.raises(MemoryError):
                    new.plan(count, PlacementHints(adjacent, frontiers))
                continue
            plan = new.plan(count, PlacementHints(adjacent, frontiers))
            assert plan.pages == tuple(expected)
            new.commit(plan)
        else:
            pages = tuple(sorted(new.owned)[:count])
            old.free(list(pages))
            new.release(pages)
        assert tuple((r.start, r.count) for r in new.free) == tuple(
            (r.start_page, r.n_pages) for r in old._free_extents
        )
        old.check_invariants()
        new.validate()
