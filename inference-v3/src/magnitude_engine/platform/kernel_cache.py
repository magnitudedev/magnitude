"""Typed, content-addressed generated artifacts; never live execution ownership."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from magnitude_engine.data import Record
from magnitude_engine.platform.publication import publish


def encoded(record: Record) -> bytes:
    return json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class ArtifactKey(Record):
    format: Literal[1] = 1
    program: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler: str = Field(pattern=r"^[0-9a-f]{64}$")
    lowering: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def digest(self) -> str:
        return fingerprint(encoded(self))


class Envelope[T: Record](Record):
    key: ArtifactKey
    artifact: T
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class CacheStatistics(Record):
    hits: int
    misses: int
    invalid: int
    read_errors: int
    write_errors: int
    last_error: str | None


@dataclass
class _Counters:
    hits: int = 0
    misses: int = 0
    invalid: int = 0
    read_errors: int = 0
    write_errors: int = 0
    last_error: str | None = None

    def snapshot(self) -> CacheStatistics:
        return CacheStatistics(**vars(self))


def cache_directory() -> Path:
    configured = os.environ.get("MAGNITUDE_KERNEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "magnitude" / "kernel-artifacts"


class ArtifactCache[T: Record]:
    """Atomic cache publication; absent/corrupt/unavailable storage is a cache miss.

    A loaded artifact contains code and its launch contract only. The execution
    driver must still validate and realize it on the current device. No pickle,
    callable, buffer, model state or device pipeline is stored in this layer.
    """

    def __init__(self, artifact_type: type[T], root: Path | None = None):
        self.root = cache_directory() if root is None else root
        self._envelope = Envelope[artifact_type]
        self._counters = _Counters()

    @property
    def statistics(self) -> CacheStatistics:
        return self._counters.snapshot()

    def read(self, key: ArtifactKey) -> T | None:
        try:
            content = (self.root / (key.digest + ".json")).read_bytes()
        except FileNotFoundError:
            self._counters.misses += 1
            return None
        except OSError as error:
            self._counters.misses += 1
            self._counters.read_errors += 1
            self._counters.last_error = str(error)
            return None
        try:
            envelope = self._envelope.model_validate_json(content)
            if envelope.key != key or fingerprint(encoded(envelope.artifact)) != envelope.checksum:
                raise ValueError("compiled artifact identity or checksum mismatch")
        except (ValidationError, ValueError) as error:
            self._counters.invalid += 1
            self._counters.misses += 1
            self._counters.last_error = str(error)
            return None
        self._counters.hits += 1
        return envelope.artifact

    def write(self, key: ArtifactKey, artifact: T) -> None:
        envelope = self._envelope(
            key=key, artifact=artifact, checksum=fingerprint(encoded(artifact))
        )
        try:
            publish(self.root / (key.digest + ".json"), encoded(envelope))
        except OSError as error:
            self._counters.write_errors += 1
            self._counters.last_error = str(error)
