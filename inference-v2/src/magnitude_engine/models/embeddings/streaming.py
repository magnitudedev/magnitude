"""Stream embedding rows through bounded IO, decode and row-cache lifetimes."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock

import mlx.core as mx
import numpy as np

from magnitude_engine.artifacts.tensors import DTYPE_BYTES
from magnitude_engine.components import component
from magnitude_engine.resources.budget import MemoryBudget, Reservation
from magnitude_engine.resources.io.reader import PositionalReader, Read

from ..execution import ExecutionScope
from .table import AffineRowTable


@dataclass(frozen=True)
class _Rows:
    encoded: np.ndarray
    inverse: np.ndarray
    shape: tuple[int, ...]


class RowLease:
    def __init__(self, owner: StreamedEmbedding, reservation: Reservation, future: Future[_Rows]):
        self.owner, self.reservation = owner, reservation
        self.future: Future[_Rows] | None = future
        self.closed = False

    def decode(self) -> mx.array:
        if self.closed or self.future is None:
            raise RuntimeError("row lease is closed")
        rows = self.future.result()
        components, offset = [], 0
        for part in self.owner.table.shards[0]:
            width = part.shape[1] * DTYPE_BYTES[part.dtype]
            raw = np.ascontiguousarray(rows.encoded[:, offset : offset + width])
            dtype = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[
                part.dtype
            ]
            value = mx.array(raw.view(dtype))
            if part.dtype == "BF16":
                value = value.view(mx.bfloat16)
            components.append(value)
            offset += width
        encoding = self.owner.table.encoding
        decoded = mx.dequantize(
            *components, bits=encoding.bits, group_size=encoding.group_size, mode="affine"
        )
        return decoded[mx.array(rows.inverse, dtype=mx.int32)].reshape(
            *rows.shape, self.owner.table.width
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.future is not None and not self.future.cancel():
                self.future.result()
        finally:
            self.reservation.close()
            self.closed = True
            self.future = None
            with self.owner._lock:
                self.owner._pending.remove(self)


@component("MODEL:EMBEDDING:MAG:STREAMED")
class StreamedEmbedding:
    """Bounded row staging and encoded LRU cache, with serialized lookahead planning.

    Each request owns its staging allocation. Eviction cannot invalidate a GPU
    consumer. One planner deduplicates overlapping lookahead against the previous
    read, while the injected positional reader performs bounded parallel IO.
    """

    def __init__(
        self,
        table: AffineRowTable,
        reader: PositionalReader,
        budget: MemoryBudget,
        *,
        cache_bytes: int,
        max_pending: int = 2,
        owner: str = "target.embedding",
    ):
        if cache_bytes < 0 or not 1 <= max_pending <= 8:
            raise ValueError("invalid embedding cache or request limit")
        self.table, self.reader, self.budget = table, reader, budget
        self.cache_bytes, self.max_pending, self.owner = cache_bytes, max_pending, owner
        self._cache: OrderedDict[int, tuple[bytes, Reservation]] = OrderedDict()
        self._cache_size = 0
        self._pending: set[RowLease] = set()
        self._lock = RLock()
        self._planner = ThreadPoolExecutor(1, thread_name_prefix="embedding-plan")
        self._closed = False
        self.metrics = {
            "rows": 0,
            "unique_rows": 0,
            "cache_hits": 0,
            "bytes_read": 0,
            "read_calls": 0,
        }
        self._boundaries = [0]
        for shard in table.shards:
            self._boundaries.append(self._boundaries[-1] + shard[0].shape[0])

    def _evict(self) -> bool:
        if not self._cache:
            return False
        _, (encoded, reservation) = self._cache.popitem(last=False)
        self._cache_size -= len(encoded) + 128
        reservation.close()
        return True

    def prepare(self, ids: mx.array | np.ndarray) -> RowLease:
        shape = tuple(ids.shape)
        count = int(np.prod(shape))
        if not shape or count <= 0:
            raise ValueError("embedding row IDs must have a nonempty shape")
        # Reserve before host readback, unique/inverse arrays, staging or dequantization.
        charge = count * (3 * self.table.row_bytes + 4 * self.table.width + 256) + 65536
        with self._lock:
            if self._closed:
                raise RuntimeError("embedding lookup is closed")
            if len(self._pending) >= self.max_pending:
                raise MemoryError("embedding lookahead queue is full")
            while True:
                try:
                    reservation = self.budget.reserve(f"{self.owner}.staging", charge)
                    break
                except MemoryError:
                    if not self._evict():
                        raise
            try:
                ids = np.array(ids, copy=True)
                if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.table.rows):
                    raise ValueError("embedding row IDs must be in-range integers")
                future = self._planner.submit(self._fetch, ids)
                lease = RowLease(self, reservation, future)
                self._pending.add(lease)
                return lease
            except BaseException:
                reservation.close()
                raise

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        lease = scope.acquire(lambda: self.prepare(rows))
        output = lease.decode()
        scope.depend(output)
        return output

    def _fetch(self, ids: np.ndarray) -> _Rows:
        unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        encoded = np.empty((len(unique), self.table.row_bytes), dtype=np.uint8)
        missing = []
        with self._lock:
            for index, row in enumerate(unique):
                cached = self._cache.get(int(row))
                if cached is None:
                    missing.append(index)
                else:
                    encoded[index] = np.frombuffer(cached[0], dtype=np.uint8)
                    self._cache.move_to_end(int(row))
            self.metrics["rows"] += ids.size
            self.metrics["unique_rows"] += len(unique)
            self.metrics["cache_hits"] += len(unique) - len(missing)
        reads = []
        for index in missing:
            row = int(unique[index])
            shard = bisect_right(self._boundaries, row) - 1
            local = row - self._boundaries[shard]
            offset = 0
            for part in self.table.shards[shard]:
                source = part.row(local)
                reads.append(
                    Read(source, memoryview(encoded[index, offset : offset + source.size]))
                )
                offset += source.size
        batch = self.reader.submit(tuple(reads))
        try:
            measurements = batch.result()
        finally:
            batch.close()
        with self._lock:
            self.metrics["bytes_read"] += measurements.bytes
            self.metrics["read_calls"] += measurements.calls
            for index in missing:
                size = self.table.row_bytes + 128
                while self._cache and self._cache_size + size > self.cache_bytes:
                    self._evict()
                if size > self.cache_bytes:
                    continue
                try:
                    reservation = self.budget.reserve(f"{self.owner}.cache", size)
                except MemoryError:
                    continue
                self._cache[int(unique[index])] = (encoded[index].tobytes(), reservation)
                self._cache_size += size
        return _Rows(encoded, inverse, tuple(ids.shape))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._pending:
                raise RuntimeError("retire embedding execution leases before closing the operation")
            self._closed = True
        self._planner.shutdown(wait=True)
        while self._evict():
            pass
