"""Content-verified downloads and atomic machine-local caches."""

import hashlib
import os
import tempfile
from pathlib import Path

import httpx


def cache_root() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "magnitude/benchmarks"
    )


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".prepare-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


async def fetch(client: httpx.AsyncClient, url: str, sha256: str, path: Path) -> bytes:
    content = path.read_bytes() if path.is_file() else b""
    if hashlib.sha256(content).hexdigest() == sha256:
        return content
    response = await client.get(url)
    response.raise_for_status()
    content = response.content
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"fixture source checksum mismatch: {url}")
    write_atomic(path, content)
    return content
