"""Read-only artifact access, independent of container interpretation."""

from __future__ import annotations

import hashlib
import mmap
import os
from pathlib import Path

from ops.binding import SourceInfo, SourceKind


class FileSource:
    """An immutable open-file snapshot. Returned bytes do not borrow the mapping.

    Artifact acquisition must publish by rename, never mutate an open artifact.
    Mapping makes metadata scans cheap; it does not promise physical residency.
    """

    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        self._file = self.path.open("rb")
        try:
            stat = os.fstat(self._file.fileno())
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self._file.close()
            raise
        self._size = len(self._map)
        self.info = SourceInfo(
            f"file:{self.path}", f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}",
            SourceKind.FILE, location=str(self.path),
        )

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self._size:
            raise ValueError(f"read outside artifact: [{offset}, {offset + length})")
        return self._map[offset : offset + length]

    def read_into(self, offset: int, destination: memoryview) -> int:
        length = destination.nbytes
        if offset < 0 or offset + length > self._size:
            raise ValueError("read outside artifact")
        with memoryview(self._map) as source:
            destination[:] = source[offset:offset + length]
        return length

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
