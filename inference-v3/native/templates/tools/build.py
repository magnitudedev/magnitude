"""Build and bundle the standalone native library for the engine's Python module."""

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path


class Buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_uint64), ("owner", ctypes.c_uint64)]


def main():
    native = Path(__file__).resolve().parents[1]
    engine = native.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=engine / "src/templates/_native")
    parser.add_argument("--build-dir", type=Path, default=engine / "build/templates")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--without-tests", action="store_true")
    args = parser.parse_args()
    ninja = shutil.which("ninja")
    if ninja is None:
        raise RuntimeError("Native templates build requires Ninja")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(args.build_dir),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DBUILD_TESTING={'OFF' if args.without_tests else 'ON'}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(args.build_dir), "--parallel", str(args.jobs)], check=True
    )
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    binary = args.build_dir / f"libtemplates{suffix}"
    library = ctypes.CDLL(str(binary))
    library.templates_build_info.argtypes = [ctypes.POINTER(Buffer), ctypes.POINTER(Buffer)]
    library.templates_build_info.restype = ctypes.c_int32
    library.templates_buffer_release.argtypes = [ctypes.c_uint64]
    library.templates_buffer_release.restype = ctypes.c_int32
    output, error = Buffer(), Buffer()
    status = library.templates_build_info(ctypes.byref(output), ctypes.byref(error))
    try:
        if status:
            raise RuntimeError(ctypes.string_at(error.data, error.size).decode())
        identity = json.loads(ctypes.string_at(output.data, output.size))
    finally:
        library.templates_buffer_release(output.owner)
        library.templates_buffer_release(error.owner)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, args.output / binary.name)
    (args.output / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
    shutil.copy2(native / "manifest.json", args.output / "manifest.json")
    shutil.copy2(native / "upstream/LICENSE", args.output / "LICENSE.llama.cpp")
    # nlohmann's license is embedded in the vendored header.
    header = (native / "upstream/vendor/nlohmann/json.hpp").read_text()
    (args.output / "NOTICE.nlohmann").write_text(header[: header.index("#ifndef")])
    shutil.copytree(native / "patches", args.output / "patches", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
