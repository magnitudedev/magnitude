import os
import threading

import pytest

from magnitude_engine.artifacts.tensors import FileSlice
from magnitude_engine.resources.io.reader import PositionalReader, Read


def test_coalesced_reads_retry_interruptions_and_short_reads_into_disjoint_destinations(
    tmp_path, monkeypatch
):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefghijkl")
    destinations = bytearray(4), bytearray(3), bytearray(5)
    reads, offset = [], 0
    for buffer in destinations:
        reads.append(Read(FileSlice(path, offset, len(buffer)), memoryview(buffer)))
        offset += len(buffer)
    original = os.preadv
    calls = 0

    def partial(fd, targets, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError()
        return original(fd, [targets[0][:2]], offset)

    monkeypatch.setattr(os, "preadv", partial)
    reader = PositionalReader(workers=1)
    batch = reader.submit(tuple(reads))
    metrics = batch.result()
    assert metrics.bytes == 12
    assert metrics.calls == calls - 1
    assert b"".join(destinations) == path.read_bytes()
    batch.close()
    reader.close()


def test_backpressure_counts_unretired_batches_and_close_joins_every_writer(tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdef")
    started, release = threading.Event(), threading.Event()
    original = os.preadv

    def blocked(fd, targets, offset):
        started.set()
        assert release.wait(3)
        return original(fd, targets, offset)

    monkeypatch.setattr(os, "preadv", blocked)
    reader = PositionalReader(workers=1, max_pending=1)
    output = bytearray(6)
    batch = reader.submit((Read(FileSlice(path, 0, 6), memoryview(output)),))
    assert started.wait(3)
    with pytest.raises(MemoryError, match="queue"):
        reader.submit(())
    release.set()
    reader.close()
    assert output == b"abcdef"
    batch.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.submit(())
