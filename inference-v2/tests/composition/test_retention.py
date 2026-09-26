from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from magnitude_engine import blueprints as bp
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.engine.prefixes.contracts import RetentionPolicy
from magnitude_engine.engine.prefixes.index import PrefixIdentity
from magnitude_engine.resources.retention import RetainedStorage


@dataclass
class Checkpoint:
    blocks: tuple[RetainedStorage, ...]
    length: int = 1
    reclaimable: bool = True
    closed: bool = False

    def retained_storage(self):
        return self.blocks

    def close(self):
        assert self.reclaimable
        assert not self.closed
        self.closed = True


def identity(token):
    return PrefixIdentity(b"test", ((token, b""),))


def test_retention_counts_shared_storage_once_and_evicts_least_recently_used():
    graph = bp.engine.prefixes.Radix(
        retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=4, max_bytes=10)
    )
    with bp.build(graph) as store:
        shared = RetainedStorage(object(), 8)
        first, second = Checkpoint((shared,)), Checkpoint((shared,))
        store.retain(identity(1), first)
        store.retain(identity(2), second)
        assert store.retained_bytes == 8
        newest = Checkpoint((RetainedStorage(object(), 9),))
        store.retain(identity(3), newest)
        assert first.closed and second.closed and not newest.closed
        assert store.retained_bytes == 9


def test_retention_revisits_pins_and_restore_leases_at_safe_boundaries():
    with bp.build(
        bp.engine.prefixes.Radix(
            retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=1),
        )
    ) as store:
        first = Checkpoint((), reclaimable=False)
        store.retain(identity(1), first)
        lease = store.match(identity(1), exclude_last=0)
        second = Checkpoint((), reclaimable=False)
        store.retain(identity(2), second)
        assert len(store) == 2
        first.reclaimable = True
        store.maintain()
        assert not first.closed  # The restore lease still owns it.
        lease.close()
        store.maintain()
        assert first.closed and len(store) == 1
        second.reclaimable = True


def test_io_pressure_cannot_mutate_owner_thread_prefix_state():
    with bp.build(bp.engine.memory.Budgeted(limit_bytes=10)) as budget:
        held = budget.reserve("weights", 10)
        calls = []

        def reclaim():
            calls.append(True)
            held.close()
            return True

        budget.bind_reclaimer(reclaim)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(MemoryError):
                pool.submit(budget.reserve, "io", 1).result()
        assert not calls and budget.snapshot().reserved == 10
        acquired = budget.reserve("owner", 1)
        assert calls == [True]
        acquired.close()
        assert budget.snapshot().reserved == 0


def test_retention_implementation_is_injected_through_its_contract():
    class KeepOldest(RetentionPolicy):
        def __init__(self, *, max_entries: int, max_bytes: int | None):
            self.max_entries, self.max_bytes = max_entries, max_bytes

        def select(self, eligible):
            return eligible[-1:]

    @blueprint
    class KeepOldestBP(Blueprint[RetentionPolicy]):
        max_entries: int = 1
        max_bytes: int | None = None

        @staticmethod
        def implementation():
            return KeepOldest

    with bp.build(bp.engine.prefixes.Radix(retention=KeepOldestBP())) as store:
        first, second = Checkpoint(()), Checkpoint(())
        store.retain(identity(1), first)
        store.retain(identity(2), second)
        assert not first.closed and second.closed
