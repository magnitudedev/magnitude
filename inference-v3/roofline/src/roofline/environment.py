"""Build accepted source using only worker-owned environments and caches."""

import json
import os
import shutil
import subprocess
import sys

from .contracts import digest, encoded
from .sources import materialize
from .store import atomic_write


def worker_environment(root):
    env = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "TVM_LIBRARY_PATH",
        "TVM_IMPORT_PYTHON_PATH",
        "TILELANG_CACHE_DIR",
    ):
        env.pop(name, None)
    env.update(
        UV_CACHE_DIR=str(root / "cache/uv"),
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        UV_MANAGED_PYTHON="1",
        XDG_CACHE_HOME=str(root / "cache"),
        TILELANG_CACHE_DIR=str(root / "cache/tilelang"),
        CUDA_CACHE_PATH=str(root / "cache/cuda"),
        TORCH_HOME=str(root / "cache/torch"),
        USE_METAL="ON" if sys.platform == "darwin" else "OFF",
        USE_CUDA="OFF" if sys.platform == "darwin" else "ON",
        USE_ROCM="OFF",
        # Workers have the CUDA toolkit and driver. Use TileLang's supported
        # direct-link build instead of its optional lazy-loading CUDA shims.
        CMAKE_ARGS="-DTILELANG_USE_CUDA_STUBS=OFF" if sys.platform != "darwin" else "",
    )
    native = sorted((root / "native/usr/lib").glob("*-linux-gnu"))
    if sys.platform == "linux":
        # TileLang's direct CUDA linkage needs the driver in the global loader
        # scope as well as cudart. Load the host driver before importing runtime.
        env["LD_PRELOAD"] = "libcuda.so.1"
    if native:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, native))
    return env


def dependency_identity(source):
    return digest(
        {
            file.path: file.blob
            for file in source.files
            if (
                file.path in {
                    "pyproject.toml", "uv.lock", "formula-performance/pyproject.toml",
                    "hatch_build.py",
                }
                or file.path.startswith(("tilelang/", "native/"))
            )
            and not {".claude", ".agents", ".codex"}.intersection(file.path.split("/"))
        }
    )


def build_profile(env):
    return {
        **{name: env[name] for name in ("USE_METAL", "USE_CUDA", "USE_ROCM", "CMAKE_ARGS")},
        "python": sys.version.split()[0],
        "dependency_groups": ["performance"],
    }


def available_environment(root, dependency, store):
    """Read existing build provenance without materializing or building anything."""
    profile = build_profile(worker_environment(root))
    for marker in sorted((root / "executors").glob("*/.ready")):
        candidate = marker.parent
        if not (candidate / ".venv/bin/python").is_file():
            continue
        if (
            json.loads(marker.read_bytes()).get("build_profile") == profile
            and dependency_identity(store.source(candidate.name)) == dependency
        ):
            return candidate
    return None


def execution_paths(destination, prepared, env):
    python = prepared / ".venv/bin/python"
    paths = [
        destination / "roofline/src",
        destination / "src",
        destination / "formula-performance/src",
        prepared / "tilelang",
        prepared / "tilelang/3rdparty/tvm/python",
        *sorted((python.parent.parent / "lib").glob("python*/site-packages")),
    ]
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE="1",
        TVM_IMPORT_PYTHON_PATH=str(prepared / "tilelang/3rdparty/tvm/python"),
        TVM_LIBRARY_PATH=str(prepared / "tilelang/build/lib"),
    )
    return destination, python, env


def prepare_environment(root, source_id, store):
    from .admission import preparation_reservation

    with preparation_reservation():
        return _prepare_environment(root, source_id, store)


def _prepare_environment(root, source_id, store):
    destination = root / "executors" / source_id
    source = store.source(source_id)
    env = worker_environment(root)
    if not (destination / ".ready").exists():
        materialize(source, store, destination)
    if not (destination / "uv.lock").is_file():
        raise ValueError("accepted source is missing its dependency lock")
    # Only reuse environments built by this worker from identical dependency and
    # compiler inputs. Python engine code always comes from the requested snapshot.
    key = dependency_identity(source)
    prepared = available_environment(root, key, store)
    if prepared is None:
        (destination / ".ready").unlink(missing_ok=True)
        with (root / "environment.log").open("ab") as log:
            subprocess.run(
                [
                    str(root / "bin/uv"),
                    "sync",
                    "--frozen",
                    "--no-dev",
                    "--group",
                    "performance",
                    "--python",
                    "3.12",
                ],
                cwd=destination,
                env=env,
                stdout=log,
                stderr=log,
                check=True,
                timeout=1800,
            )
        prepared = destination
    python = prepared / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError("worker-owned execution environment is missing")
    if prepared != destination:
        for relative in ("tilelang/build", "src/templates/_native"):
            built = prepared / relative
            if not built.exists():
                continue
            build = destination / relative
            if not build.is_symlink():
                if build.exists():
                    shutil.rmtree(build)
                build.parent.mkdir(parents=True, exist_ok=True)
                build.symlink_to(built, target_is_directory=True)
    atomic_write(
        destination / ".ready",
        encoded({"source_id": source_id, "build_profile": build_profile(env)}),
    )
    return execution_paths(destination, prepared, env)


def main():
    import argparse
    from pathlib import Path

    from .store import Store
    from .transport import send

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    try:
        with Store(args.root) as store:
            destination, python, env = prepare_environment(args.root, args.source, store)
        send(
            sys.stdout.buffer, {"destination": str(destination), "python": str(python), "env": env}
        )
    except Exception as exc:
        send(sys.stdout.buffer, {"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
