"""Digests as the evaluation contracts define them (lowercase hex SHA-256)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    """SHA-256 of the canonical JSON encoding: sorted keys, no insignificant whitespace, UTF-8."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The numerical implementation: everything that decides a reference's rows and sanity values.
# Performance options (`performance.py`), the CLI, records and validation tools are outside it.
NUMERICAL = ("families", "model.py", "gguf.py", "contract.py", "sanity.py", "run.py")


def implementation() -> str:
    """The reference implementation's identity: the canonical digest of {relative path: SHA-256} over
    the Python sources of `NUMERICAL`."""
    files = sorted(path for name in NUMERICAL
                   for path in ((PACKAGE / name).rglob("*.py") if (PACKAGE / name).is_dir() else [PACKAGE / name]))
    return canonical({path.relative_to(PACKAGE).as_posix(): sha256_file(path) for path in files})
