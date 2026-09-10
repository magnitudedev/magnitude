"""Read-only artifact access, independent of container interpretation."""

from __future__ import annotations

import hashlib
import mmap
from pathlib import Path
from typing import Protocol


class ByteSource(Protocol):
    @property
    def size(self) -> int: ...

    def read(self, offset: int, length: int) -> bytes: ...


class ZeroSource:
    def __init__(self, size: int):
        self.size = size

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("zero source read outside range")
        return bytes(length)


class FileSource:
    """An immutable open-file snapshot. Returned bytes do not borrow the mapping.

    Artifact acquisition must publish by rename, never mutate an open artifact.
    Mapping makes metadata scans cheap; it does not promise physical residency.
    """

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self._file.close()
            raise
        self._size = len(self._map)

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self._size:
            raise ValueError(f"read outside artifact: [{offset}, {offset + length})")
        return self._map[offset : offset + length]

    def close(self) -> None:
        self._map.close()
        self._file.close()

    def digest(self) -> str:
        digest = hashlib.sha256()
        for offset in range(0, self.size, 8 * 1024 * 1024):
            digest.update(self.read(offset, min(8 * 1024 * 1024, self.size - offset)))
        return digest.hexdigest()

    def __enter__(self) -> FileSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ConcatenatedSource:
    """A bounded read across immutable byte ranges; no full concatenation copy."""

    def __init__(self, ranges: tuple[tuple[ByteSource, int, int], ...]):
        if not ranges or any(
            start < 0 or size < 0 or start + size > source.size for source, start, size in ranges
        ):
            raise ValueError("invalid concatenated byte ranges")
        self.ranges = ranges
        self.size = sum(size for _, _, size in ranges)

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("read outside concatenated byte source")
        pieces = []
        cursor = 0
        end = offset + length
        for source, start, size in self.ranges:
            lo, hi = max(offset, cursor), min(end, cursor + size)
            if hi > lo:
                pieces.append(source.read(start + lo - cursor, hi - lo))
            cursor += size
            if cursor >= end:
                break
        return b"".join(pieces)
