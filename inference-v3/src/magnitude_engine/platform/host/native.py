"""Native dependency lookup belongs to the platform, including packaged bundles."""

import ctypes.util
import sys
from pathlib import Path


def library_path(name: str) -> str:
    discovered = ctypes.util.find_library(name)
    if discovered is not None:
        return discovered
    suffix = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
    directories = [Path(__file__).parent / "_native", Path(sys.prefix) / "lib"]
    if sys.platform == "darwin":
        directories += [Path("/opt/homebrew/lib"), Path("/usr/local/lib")]
    for directory in directories:
        path = directory / f"lib{name}{suffix}"
        if path.is_file():
            return str(path)
    raise RuntimeError(f"required native library {name!r} was not found")
