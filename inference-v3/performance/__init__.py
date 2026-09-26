"""Inference service performance evidence and provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def source_identity(root: Path) -> str:
    """Hash all engine, tensor-system, and measurement source used by a run."""
    digest = hashlib.sha256()
    paths = (
        *root.joinpath("src/ops").rglob("*.py"),
        *root.joinpath("src/engine").rglob("*.py"),
        *root.joinpath("performance").rglob("*.py"),
        root / "pyproject.toml",
        root / "uv.lock",
    )
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = ["source_identity"]
