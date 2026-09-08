import mlx.core as mx
import pytest

from magnitude_engine.models.features import Feature, FeatureCache
from magnitude_engine.resources.budget import MemoryBudget


def ready(budget, value):
    lease = Feature(16, budget).acquire()
    lease.feature.value = mx.full((1, 1, 4), value)
    mx.eval(lease.value)
    return lease


def test_feature_eviction_respects_borrowers_and_bound_encoder_identity():
    budget = MemoryBudget(64)
    cache = FeatureCache(32)
    first, second = object(), object()
    value = ready(budget, 3)
    cache.put(first, b"pixels", value)
    assert cache.get(second, b"pixels") is None
    borrower = cache.get(first, b"pixels")
    assert borrower is not None
    value.close()
    assert not cache.reclaim()  # Pressure cannot reclaim a borrower's allocation.
    cache.close()
    assert budget.snapshot().reserved == 16
    assert borrower.value.tolist() == [[[3, 3, 3, 3]]]
    borrower.close()
    assert budget.snapshot().reserved == 0


def test_feature_cache_lru_releases_only_completed_owned_entries():
    budget = MemoryBudget(64)
    cache = FeatureCache(32)
    encoder = object()
    for key in (b"a", b"b"):
        value = ready(budget, 1)
        cache.put(encoder, key, value)
        value.close()
    lease = cache.get(encoder, b"a")
    assert lease is not None
    lease.close()
    value = ready(budget, 2)
    cache.put(encoder, b"c", value)
    value.close()
    assert cache.get(encoder, b"b") is None
    assert cache.nbytes == budget.snapshot().reserved == 32
    assert cache.reclaim() and cache.reclaim() and not cache.reclaim()
    empty = Feature(16, budget).acquire()
    with pytest.raises(RuntimeError, match="ready"):
        cache.put(encoder, b"unfinished", empty)
    empty.close()
    cache.close()
    assert budget.snapshot().reserved == 0


def test_failed_or_cancelled_preparation_never_publishes_an_unready_lease():
    from magnitude_engine.models.features import FeatureSet

    budget = MemoryBudget(64)
    cache = FeatureCache()
    features = FeatureSet(object(), budget, cache)
    task = features.prepare(b"image", 16, lambda lease: object())
    next(task)
    assert budget.snapshot().reserved == 16 and features.leases == {}
    with pytest.raises(MemoryError):
        task.throw(MemoryError("pre-submit capacity"))
    assert budget.snapshot().reserved == 0 and features.leases == {} and not cache.entries
    task = features.prepare(b"image", 16, lambda lease: object())
    next(task)
    task.close()
    assert budget.snapshot().reserved == 0 and features.leases == {}
