"""Bounded positional reads into caller-owned storage; no model-specific policy."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock

from magnitude_engine.artifacts.tensors import FileSlice


@dataclass(frozen=True)
class Read:
    source: FileSlice
    destination: memoryview

    def __post_init__(self) -> None:
        if (
            self.destination.readonly
            or not self.destination.c_contiguous
            or self.destination.nbytes != self.source.size
        ):
            raise ValueError("read destination must be writable, contiguous and exactly sized")


@dataclass(frozen=True)
class ReadMetrics:
    bytes: int = 0
    calls: int = 0


@dataclass(frozen=True)
class _ReadGroup:
    path: str
    offset: int
    targets: tuple[memoryview, ...]
    size: int


class ReadBatch:
    def __init__(self, reader: PositionalReader, futures: tuple[Future[ReadMetrics], ...]):
        self._reader = reader
        self._futures = futures
        self._closed = False

    def result(self) -> ReadMetrics:
        if self._closed:
            raise RuntimeError("read batch is closed")
        failures, total, calls = [], 0, 0
        # Join every writer even after one fails. A failed batch never publishes
        # partially written data or returns a still-writable buffer to its owner.
        for future in self._futures:
            try:
                result = future.result()
                total += result.bytes
                calls += result.calls
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("positional read batch failed", failures)
        return ReadMetrics(total, calls)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.result()
        finally:
            self._closed = True
            self._futures = ()
            with self._reader._lock:
                self._reader._pending.remove(self)


class PositionalReader:
    """At most workers × max_pending queued jobs, with descriptors owned until drain."""

    def __init__(
        self,
        *,
        workers: int = 4,
        max_pending: int = 2,
        uncached: bool = False,
        batch_bytes: int = 64 << 20,
    ):
        if not 1 <= workers <= 64 or not 1 <= max_pending <= 8:
            raise ValueError("read worker or pending-batch limit is invalid")
        if batch_bytes <= 0:
            raise ValueError("read batch byte limit must be positive")
        self.workers, self.max_pending, self.uncached = workers, max_pending, uncached
        self.batch_bytes = batch_bytes
        self._executor = ThreadPoolExecutor(workers, thread_name_prefix="model-read")
        self._files: dict[str, int] = {}
        self._pending: set[ReadBatch] = set()
        self._lock = RLock()
        self._closed = False

    def submit(self, reads: tuple[Read, ...]) -> ReadBatch:
        with self._lock:
            if self._closed:
                raise RuntimeError("positional reader is closed")
            if len(self._pending) >= self.max_pending:
                raise MemoryError("positional read queue is full")
            for read in reads:
                path = str(read.source.path)
                if path not in self._files:
                    descriptor = os.open(path, os.O_RDONLY)
                    try:
                        if self.uncached:
                            import fcntl

                            # Apple descriptor-local F_NOCACHE, never a global cache flush.
                            fcntl.fcntl(descriptor, 48, 1)
                    except BaseException:
                        os.close(descriptor)
                        raise
                    self._files[path] = descriptor
            # Coalesce adjacent source ranges before assigning jobs. Destination
            # buffers need not be adjacent: preadv writes directly into each one.
            groups: list[_ReadGroup] = []
            iov_limit = min(512, os.sysconf("SC_IOV_MAX"))
            for read in sorted(reads, key=lambda r: (str(r.source.path), r.source.offset)):
                target = read.destination.cast("B")
                for position in range(0, len(target), self.batch_bytes):
                    part = target[position : position + self.batch_bytes]
                    path, offset = str(read.source.path), read.source.offset + position
                    if (
                        groups
                        and groups[-1].path == path
                        and groups[-1].offset + groups[-1].size == offset
                        and groups[-1].size + len(part) <= self.batch_bytes
                        and len(groups[-1].targets) < iov_limit
                    ):
                        last = groups[-1]
                        groups[-1] = _ReadGroup(
                            path, last.offset, (*last.targets, part), last.size + len(part)
                        )
                    else:
                        groups.append(_ReadGroup(path, offset, (part,), len(part)))
            jobs: list[list[_ReadGroup]] = [[] for _ in range(min(self.workers, len(groups)))]
            sizes = [0] * len(jobs)
            for group in groups:
                worker = min(range(len(jobs)), key=lambda index: sizes[index])
                jobs[worker].append(group)
                sizes[worker] += group.size
            futures = []
            try:
                for job in jobs:
                    futures.append(self._executor.submit(self._read, tuple(job)))
            except BaseException:
                for future in futures:
                    try:
                        future.result()
                    except BaseException:
                        pass
                raise
            batch = ReadBatch(self, tuple(futures))
            self._pending.add(batch)
            return batch

    def _read(self, reads: tuple[_ReadGroup, ...]) -> ReadMetrics:
        total, calls = 0, 0
        for read in reads:
            descriptor = self._files[read.path]
            remaining = list(read.targets)
            position = read.offset
            while remaining:
                try:
                    received = os.preadv(descriptor, remaining, position)
                except InterruptedError:
                    continue
                calls += 1
                if not received:
                    raise EOFError(f"short tensor read at offset {position}")
                position += received
                total += received
                while remaining and received >= remaining[0].nbytes:
                    received -= remaining.pop(0).nbytes
                if remaining and received:
                    remaining[0] = remaining[0][received:]
        return ReadMetrics(total, calls)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)
        failures = []
        for batch in tuple(self._pending):
            try:
                batch.close()
            except BaseException as error:
                failures.append(error)
        for descriptor in self._files.values():
            os.close(descriptor)
        self._files.clear()
        if failures:
            raise BaseExceptionGroup("read service shutdown failed", failures)
