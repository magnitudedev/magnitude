"""Identity of the compiler binaries actually loaded by the platform runtime."""

import hashlib
from functools import cache
from importlib.metadata import version
from pathlib import Path

from magnitude_engine.data import Record


class CompilerBinary(Record):
    name: str
    sha256: str


class CompilerBuild(Record):
    distribution_version: str
    binaries: tuple[CompilerBinary, ...]


@cache
def compiler_build() -> CompilerBuild:
    import tilelang
    from tilelang import tvm

    base = tvm.base

    paths = (
        Path(tilelang._LIB_PATH),
        Path(base._LIB._name),
        Path(base._LIB_RUNTIME._name),
    )
    binaries = []
    for path in dict.fromkeys(paths):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        binaries.append(CompilerBinary(name=path.name, sha256=digest))
    return CompilerBuild(distribution_version=version("tilelang"), binaries=tuple(binaries))
