"""Content identity of the analytical rules loaded by this process."""

from hashlib import sha256
from pathlib import Path


def _loaded_revision():
    root = Path(__file__).parent
    value = sha256()
    for name in (
        "records.py",
        "derive.py",
        "evidence.py",
        "propagation.py",
        "behavior.py",
        "composition.py",
    ):
        value.update(name.encode())
        value.update((root / name).read_bytes())
    return value.hexdigest()


_REVISION = _loaded_revision()


def revision():
    return _REVISION
