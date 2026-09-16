"""Install workers in their own directories, independently of engine checkouts."""

import io
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from .transport import DEFAULT_WORKER_ROOT, WorkerClient


def setup_worker(config, name):
    target = config.targets[name]
    archive = io.BytesIO()
    base = config.root
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        paths = [
            base / "roofline/pyproject.toml",
            base / "roofline/uv.lock",
            base / "formula-performance/pyproject.toml",
        ]
        paths += sorted(
            p
            for directory in ("roofline/src", "formula-performance/src")
            for p in (base / directory).rglob("*")
            if p.is_file() and p.suffix in (".py", ".cpp")
        )
        for path in paths:
            # Stable package identity, independent of file timestamps and ownership.
            content = path.read_bytes()
            entry = tarfile.TarInfo(path.relative_to(base).as_posix())
            entry.size = len(content)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(content))
    bootstrap = (Path(__file__).parent / "bootstrap.py").read_text()
    command = ["python3", "-c", bootstrap, target.worker_root or DEFAULT_WORKER_ROOT]
    if target.connection.kind == "ssh":
        command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            target.connection.host,
            shlex.join(command),
        ]
    else:
        command[0] = sys.executable
    subprocess.run(command, input=archive.getvalue(), check=True, timeout=600, stdout=sys.stderr)
    with WorkerClient(target, config.root) as client:
        return client.call({"op": "ping"})
