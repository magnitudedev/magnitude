"""Content-verified upstream sources in a machine-local cache; never repository data."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from pydantic import Field

from .canonical import Record

DATA = Path(__file__).parent / "data"


class LockedFile(Record):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def cache_root() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "magnitude" / "evaluation" / "sources"


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".fetch-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def fetch(url: str, sha256: str, path: Path) -> bytes:
    """Return the cached bytes at ``path``, downloading ``url`` when absent or stale."""
    content = path.read_bytes() if path.is_file() else b""
    if hashlib.sha256(content).hexdigest() == sha256:
        return content
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"fixture source checksum mismatch: {url}")
    write_atomic(path, content)
    return content
