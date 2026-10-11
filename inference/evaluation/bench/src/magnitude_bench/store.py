"""Incremental evidence in the run directory: every row is persisted as it is observed, so a
failed or cancelled run keeps what it measured."""

import json
import os
import queue
import threading
from datetime import UTC, datetime
from pathlib import Path

from magnitude_fixtures.canonical import encoded


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "logs").mkdir(exist_ok=True)
        self._streams = queue.SimpleQueue()
        self._writer = None
        self._write_error = None
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

    def complete(self, summary: dict, markdown: str):
        self.close_streams()
        atomic_json(self.path / "summary.json", summary)
        temporary = self.path / "report.tmp"
        temporary.write_text(markdown)
        temporary.replace(self.path / "report.md")
        self.event("finished", status=summary["status"])
