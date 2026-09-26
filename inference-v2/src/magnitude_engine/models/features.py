"""Shared, budgeted model features retained by continuations and checkpoints."""

from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.retention import RetainedStorage

if TYPE_CHECKING:
    from .computation import Computation
    from .operations import Task


class Feature:
    def __init__(self, nbytes: int, budget: MemoryBudget):
        self.reservation = budget.reserve("model-input-features", nbytes)
        self.value: mx.array | None = None
        self.users = 0
        self.closed = False

    def acquire(self) -> "FeatureLease":
        if self.closed:
            raise RuntimeError("model feature is closed")
        self.users += 1
        return FeatureLease(self)


class FeatureLease:
    def __init__(self, feature: Feature):
        self.feature = feature
        self.closed = False

    @property
    def value(self) -> mx.array:
        if self.closed or self.feature.value is None:
            raise RuntimeError("model feature is not ready")
        return self.feature.value

    def fork(self) -> "FeatureLease":
        if self.closed:
            raise RuntimeError("model feature lease is closed")
        return self.feature.acquire()

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        return (RetainedStorage(self.feature, self.feature.reservation.size),)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.feature.users -= 1
        if not self.feature.users:
            self.feature.value = None
            self.feature.reservation.close()
            self.feature.closed = True


@component("MODEL:INPUT_FEATURES:MAG:LRU")
class FeatureCache:
    """Bounded completed-feature retention, independent of decoder-prefix retention.

    Keys include the live bound encoder and its prepared content identity. Borrowers
    and executions retain their own leases; eviction releases only cache ownership.
    No capacity is allocated until a successful computation is published.
    """

    def __init__(self, max_bytes: int = 64 << 20):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("feature retention needs a nonnegative byte limit")
        self.max_bytes = max_bytes
        self.entries: OrderedDict[tuple[object, bytes], FeatureLease] = OrderedDict()
        self.nbytes = 0

    def get(self, encoder: object, identity: bytes) -> FeatureLease | None:
        key = (encoder, identity)
        lease = self.entries.get(key)
        if lease is None:
            return None
        self.entries.move_to_end(key)
        return lease.fork()

    def put(self, encoder: object, identity: bytes, feature: FeatureLease) -> None:
        _ = feature.value
        size = feature.feature.reservation.size
        key = (encoder, identity)
        if size > self.max_bytes or key in self.entries:
            return
        while self.nbytes + size > self.max_bytes:
            self._discard(next(iter(self.entries)))
        self.entries[key] = feature.fork()
        self.nbytes += size

    def _discard(self, key) -> bool:
        lease = self.entries.pop(key)
        self.nbytes -= lease.feature.reservation.size
        released = lease.feature.users == 1
        lease.close()
        return released

    def reclaim(self) -> bool:
        for key, lease in self.entries.items():
            if lease.feature.users == 1:
                return self._discard(key)
        return False

    def close(self):
        for lease in self.entries.values():
            lease.close()
        self.entries.clear()
        self.nbytes = 0


class FeatureSet:
    """One input continuation's feature leases, independent of input geometry.

    The adapter selects identities and computation; this owner handles completion,
    cache publication, execution pins, and release of obsolete features.
    """

    def __init__(
        self,
        encoder: object,
        budget: MemoryBudget,
        cache: FeatureCache,
        leases: dict[bytes, FeatureLease] | None = None,
    ):
        self.encoder, self.budget, self.cache = encoder, budget, cache
        self.leases = {} if leases is None else leases
        self.cache_hits = 0

    def prepare(
        self, identity: bytes, size: int, computation: Callable[[FeatureLease], "Computation"]
    ) -> "Task[None]":
        from .operations import Complete, compute

        if identity in self.leases:
            return
        cached = self.cache.get(self.encoder, identity)
        if cached is not None:
            self.leases[identity] = cached
            self.cache_hits += 1
            return
        lease = Feature(size, self.budget).acquire()
        try:
            result = yield from compute(computation(lease))
            yield Complete(result.execution)
            self.cache.put(self.encoder, identity, lease)
            self.leases[identity] = lease
        except BaseException:
            lease.close()
            raise

    def keep(self, identities: set[bytes]) -> None:
        for key in tuple(self.leases):
            if key not in identities:
                self.leases.pop(key).close()

    def value(self, identity: bytes) -> mx.array:
        return self.leases[identity].value

    def pin(self, identities: Iterable[bytes]) -> ExitStack:
        pins = ExitStack()
        try:
            for identity in identities:
                pins.callback(self.leases[identity].fork().close)
        except BaseException:
            pins.close()
            raise
        return pins

    def checkpoint(self, identities: set[bytes]) -> dict[bytes, FeatureLease]:
        return {key: lease.fork() for key, lease in self.leases.items() if key in identities}

    def close(self) -> None:
        for lease in self.leases.values():
            lease.close()
        self.leases.clear()
