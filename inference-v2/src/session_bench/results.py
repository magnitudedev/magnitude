"""Incremental evidence storage and alias-independent public reproduction commands."""

import json
import os
import platform
import queue
import shlex
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psutil

from .models import Target, file_hash
from .sessions import encoded


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def public_command(
    targets: list[Target],
    sections: tuple[str, ...],
    contexts: tuple[int, ...],
    categories: tuple[str, ...],
    repeat: int,
    case: str | None = None,
    prose: bool = False,
) -> str:
    args = ["uv", "run", "--frozen", "session-bench", "run"]
    for target in targets:
        args += ["--target", f"{target.engine}={target.reference}"]
    args += [
        "--suite",
        ",".join(sections),
        "--context",
        ",".join(map(str, contexts)),
        "--repeat",
        str(repeat),
    ]
    args += ["--prose"] if prose else ["--category", ",".join(categories)]
    if case:
        args += ["--case", case]
    return shlex.join(args)


class RunStore:
    def __init__(self, root: Path, command: str, selection: dict):
        from magnitude_engine.host_info import capture_hardware

        identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
        self.path = root / "runs" / "session-bench" / identifier
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "logs").mkdir()
        self.command = command
        self.root = root
        self.started = now()
        self.hardware = capture_hardware().model_dump(mode="json")
        self._streams = queue.SimpleQueue()
        self._writer = None
        self._write_error = None
        (self.path / "command.txt").write_text(f"cd {shlex.quote(str(root))}\n{command}\n")
        atomic_json(
            self.path / "run.json",
            {
                "format": 1,
                "id": identifier,
                "started_at": self.started,
                "pid": os.getpid(),
                "process_started_at": psutil.Process().create_time(),
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
                "command": command,
                "selection": selection,
                "host": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": sys.version,
                    "hardware": self.hardware,
                },
            },
        )
        self.event("created")

    def append(self, filename: str, value: dict) -> None:
        with (self.path / filename).open("a") as stream:
            stream.write(encoded(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record_stream(self, filename: str, value: dict) -> None:
        """Keep disk writes out of timed response parsing and the asyncio event loop."""
        if self._write_error:
            raise OSError("stream evidence writer failed") from self._write_error
        if self._writer is None:
            self._writer = threading.Thread(target=self._write_streams, daemon=True)
            self._writer.start()
        self._streams.put((filename, value))

    def _write_streams(self):
        try:
            while (item := self._streams.get()) is not None:
                filename, value = item
                with (self.path / filename).open("a") as stream:
                    stream.write(encoded(value) + "\n")
        except Exception as exc:
            self._write_error = exc

    def close_streams(self):
        if self._writer:
            self._streams.put(None)
            self._writer.join()
            self._writer = None
        if self._write_error:
            raise OSError("stream evidence writer failed") from self._write_error

    def event(self, event: str, **fields):
        self.append("events.jsonl", {"at": now(), "event": event, **fields})

    def snapshot(self, label: str, root: Path) -> dict:
        files = {}
        source = self.path / "source" / label
        source.mkdir(parents=True, exist_ok=True)
        candidates = sorted(
            p for p in (root / "src").rglob("*") if p.is_file() and p.suffix in (".py", ".json")
        )
        candidates += [p for p in (root / "pyproject.toml", root / "uv.lock") if p.is_file()]
        for path in candidates:
            relative = path.relative_to(root)
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            files[str(relative)] = file_hash(destination)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        record = {"root": str(root), "revision": revision.stdout.strip(), "files": files}
        atomic_json(source / "identity.json", record)
        return record

    def complete(self, summary: dict, markdown: str):
        self.close_streams()
        atomic_json(self.path / "summary.json", summary)
        temporary = self.path / "report.tmp"
        temporary.write_text(markdown)
        temporary.replace(self.path / "report.md")
        self.event("finished", status=summary["status"])
        from performance.session import ingest
        from performance.store import Store

        ingest(self.path, Store(self.root / "runs" / "performance"))


def inspect_run(path: Path) -> dict:
    if (path / "summary.json").is_file():
        value = json.loads((path / "summary.json").read_text())
        if value.get("format") != 1:
            return {"id": path.name, "status": "unsupported-format", "path": str(path)}
        return value

    run = json.loads((path / "run.json").read_text())
    if run.get("format") != 1:
        return {"id": path.name, "status": "unsupported-format", "path": str(path)}
    active = False
    try:
        process = psutil.Process(run["pid"])
        active = process.create_time() == run.get("process_started_at")
    except psutil.Error:
        pass
    return {
        "id": run["id"],
        "status": "running" if active else "interrupted",
        "command": run["command"],
        "path": str(path),
    }
