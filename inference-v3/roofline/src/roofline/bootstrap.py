"""Standard-library-only installation on a host with Python 3 and uv."""

import ctypes.util
import fcntl
import gzip
import hashlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time


def main():
    root = pathlib.Path(sys.argv[1]).expanduser().resolve()
    archive = sys.stdin.buffer.read()
    release_id = hashlib.sha256(gzip.decompress(archive)).hexdigest()
    marker = root / ".roofline-worker"
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise RuntimeError(f"refusing to install into a non-worker directory: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.touch()
    uv = shutil.which("uv") or str(pathlib.Path.home() / ".local/bin/uv")
    if not pathlib.Path(uv).is_file():
        raise RuntimeError("worker setup requires uv on the host")
    (root / "bin").mkdir(exist_ok=True)
    owned_uv = root / "bin/uv"
    if pathlib.Path(uv).resolve() != owned_uv.resolve():
        temporary_uv = root / "bin/.uv-next"
        shutil.copy2(uv, temporary_uv)
        temporary_uv.replace(owned_uv)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(
        UV_CACHE_DIR=str(root / "cache/uv"),
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        UV_MANAGED_PYTHON="1",
    )
    if sys.platform == "linux" and ctypes.util.find_library("hwloc") is None:
        native = root / "native"
        if not list(native.glob("usr/lib/*/libhwloc.so")):
            if not shutil.which("apt-get") or not shutil.which("dpkg-deb"):
                raise RuntimeError(
                    "worker execution requires libhwloc; install it using the host package manager"
                )
            native.mkdir(exist_ok=True)
            subprocess.run(
                ["apt-get", "download", "libhwloc15"], cwd=native, check=True, timeout=120
            )
            for package in native.glob("libhwloc15_*.deb"):
                subprocess.run(["dpkg-deb", "-x", str(package), str(native)], check=True)
            for library in native.glob("usr/lib/*/libhwloc.so.15"):
                library.with_name("libhwloc.so").symlink_to(library.name)
    release = root / "installations" / release_id
    release.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for entry in tar:
            path = pathlib.PurePosixPath(entry.name)
            if not entry.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"invalid worker package entry: {entry.name}")
            destination = release / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(entry) as source:
                destination.write_bytes(source.read())
    install_env = {**env, "UV_PROJECT_ENVIRONMENT": str(release / ".venv")}
    subprocess.run(
        [str(owned_uv), "sync", "--project", "roofline", "--frozen", "--python", "3.12"],
        cwd=release,
        env=install_env,
        check=True,
    )
    current = root / "current"
    if current.exists():
        # Ask the existing worker to exit only when idle. Never interrupt a measurement.
        stop = (
            "from pathlib import Path; from roofline.transport import local_call; "
            "import sys; "
            'local_call(Path(sys.argv[1]), {"op":"shutdown"})'
        )
        result = subprocess.run(
            [str(release / ".venv/bin/python"), "-I", "-c", stop, str(root)],
            env=env,
            capture_output=True,
            text=True,
        )
        if (
            result.returncode
            and "ConnectionRefusedError" not in result.stderr
            and "FileNotFoundError" not in result.stderr
        ):
            raise RuntimeError(result.stderr)
    # The old process must release its lock before the new control environment starts.
    with (root / "service.lock").open("a") as lock:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("worker did not stop after shutdown") from None
                time.sleep(0.05)
    link = root / ".current-next"
    link.unlink(missing_ok=True)
    link.symlink_to(release)
    link.replace(current)
    subprocess.run(
        [
            str(current / ".venv/bin/python"),
            "-I",
            "-c",
            "from pathlib import Path; import sys; "
            "from roofline.transport import ensure_service, local_call; "
            'root=Path(sys.argv[1]); ensure_service(root, None, "worker"); '
            'print(local_call(root, {"op":"ping"}))',
            str(root),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
