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


class CompilerImplementation(Record):
    binaries: CompilerBuild
    python_sha256: str
    ffi_version: str


@cache
def compiler_implementation() -> CompilerImplementation:
    """Actual lowering implementation, including editable Python compiler code."""
    import tilelang
    import tvm_ffi
    from tilelang import tvm

    digest = hashlib.sha256()
    for label, package in (("tilelang", tilelang), ("tvm", tvm), ("tvm_ffi", tvm_ffi)):
        location = package.__file__
        if location is None:
            raise RuntimeError(f"compiler package {label} has no source location")
        root = Path(location).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(f"{label}/{path.relative_to(root)}".encode())
            digest.update(b"\x00")
            digest.update(path.read_bytes())
    return CompilerImplementation(
        binaries=compiler_build(),
        python_sha256=digest.hexdigest(),
        ffi_version=version("apache-tvm-ffi"),
    )
