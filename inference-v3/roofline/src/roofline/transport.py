"""Length-framed JSON control over local sockets or an SSH stdio bridge."""

import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from .contracts import encoded

DEFAULT_WORKER_ROOT = "~/.local/share/roofline"

MAX_FRAME = 16 << 20


class WorkerError(RuntimeError):
    """A worker rejected an operation; retrying requires an explicit transient reason."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def failure(exc):
    return {
        "error": str(exc),
        "code": exc.code if isinstance(exc, WorkerError) else type(exc).__name__,
    }


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("control connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def receive(stream):
    (size,) = struct.unpack("!I", read_exact(stream, 4))
    if size > MAX_FRAME:
        raise ValueError("oversized control frame")
    return json.loads(read_exact(stream, size))


def send(stream, value):
    content = encoded(value)
    if len(content) > MAX_FRAME:
        raise ValueError("oversized control frame")
    stream.write(struct.pack("!I", len(content)) + content)
    stream.flush()


def socket_path(root):
    import hashlib

    return Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"roofline-{os.getuid()}-{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20]}.sock"
    )


def local_call(root, message, *, timeout=60):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path(root)))
        with client.makefile("rwb") as stream:
            send(stream, message)
            response = receive(stream)
    if "error" in response:
        raise WorkerError(response["error"], response["code"])
    return response["result"]


def ensure_service(root, project, kind):
    try:
        local_call(root, {"op": "ping"})
        return
    except (OSError, EOFError):
        pass
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    command = [
        sys.executable,
        "-m",
        "roofline.service",
        "--root",
        str(root),
        "--kind",
        kind,
    ]
    if kind == "worker":
        command.insert(1, "-I")
    if project is not None:
        command.extend(["--project", str(project)])
    with (root / f"{kind}.log").open("ab") as log:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            local_call(root, {"op": "ping"})
            return
        except (OSError, EOFError):
            time.sleep(0.05)
    raise RuntimeError(f"{kind} did not start; see {root / (kind + '.log')}")


class WorkerClient:
    def __init__(self, target, project=None):
        import shlex
        import tempfile

        self.target = target
        self.process = None
        self.errors = None
        root = target.worker_root or DEFAULT_WORKER_ROOT
        if target.connection.kind == "local":
            self.root = Path(root).expanduser().resolve()
            python = self.root / "current/.venv/bin/python"
            if not python.is_file():
                if project is None:
                    raise ValueError(
                        "local worker is not installed; run roofline workers setup local"
                    )
                from types import SimpleNamespace

                from .setup import setup_worker

                setup_worker(SimpleNamespace(root=project, targets={"local": target}), "local")
            subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "from pathlib import Path; import sys; "
                    "from roofline.transport import ensure_service; "
                    "ensure_service(Path(sys.argv[1]), None, 'worker')",
                    str(self.root),
                ],
                check=True,
                timeout=30,
            )
        else:
            # System Python only resolves the user-owned installation and execs it.
            launcher = (
                "import os, pathlib, sys; root=pathlib.Path(sys.argv[1]).expanduser(); "
                "python=str(root/'current/.venv/bin/python'); "
                "os.execv(python, [python, '-I', '-m', 'roofline.service', "
                "'--root', str(root), '--bridge'])"
            )
            remote = shlex.join(["python3", "-c", launcher, root])
            self.errors = tempfile.TemporaryFile()
            self.process = subprocess.Popen(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ServerAliveInterval=10",
                    "-o",
                    "ServerAliveCountMax=3",
                    target.connection.host,
                    remote,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.errors,
            )

    def call(self, message):
        if self.process is None:
            return local_call(
                self.root, message, timeout=3600 if message["op"] == "discover" else 60
            )
        send(self.process.stdin, message)
        try:
            response = receive(self.process.stdout)
        except (EOFError, OSError) as exc:
            self.errors.seek(0)
            detail = self.errors.read().decode(errors="replace")[-4000:]
            raise ConnectionError(f"worker connection failed: {detail or exc}") from exc
        if "error" in response:
            raise WorkerError(response["error"], response["code"])
        return response["result"]

    def close(self):
        if self.process:
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)

        if self.errors is not None:
            self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
