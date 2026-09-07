"""Immutable runs, idempotent import and a single atomic assessed graph."""

from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from performance.assessment import rebuild
from performance.records import Assembly, Observation, Profile, digest, encoded

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "runs" / "performance"


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded(value) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def validate_run(run: dict) -> None:
    if run.get("schema_version") != 1 or run.get("status") not in (
        "running",
        "complete",
        "failed",
        "interrupted",
    ):
        raise ValueError("invalid run schema/status")
    identity = run.get("id", "")
    if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("invalid run identity")
    graph = Assembly.read(run["assembly"])
    if run["node"] not in graph.nodes:
        raise ValueError("measurement node absent from composition")
    Profile(**run["profile"])
    if run.get("checksum") != digest({k: v for k, v in run.items() if k != "checksum"}):
        raise ValueError("run checksum mismatch")
    if (
        not run.get("boundary")
        or not run.get("benchmark")
        or not isinstance(run.get("workload"), dict)
    ):
        raise ValueError("missing observation contract")
    samples = run.get("samples", [])
    for sample in samples:
        if sample.get("phase") not in ("warmup", "measured") or not isinstance(
            sample.get("index"), int
        ):
            raise ValueError("invalid sample identity")
        elapsed = sample.get("elapsed_ns")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid completed duration")
        if "observation" in sample:
            Observation(**sample["observation"])
        elif not sample.get("error"):
            raise ValueError("sample lacks observation or failure")
    if run["status"] == "complete":
        for phase, field in (("warmup", "warmup"), ("measured", "repetitions")):
            selected = [s for s in samples if s["phase"] == phase]
            if [s["index"] for s in selected] != list(range(run[field])) or any(
                s.get("error") for s in selected
            ):
                raise ValueError("completed run has incomplete or failed samples")
    encoded(run)


class Store:
    def __init__(self, root: Path = DEFAULT_STORE):
        self.root = Path(root)

    def runs(self) -> list[dict]:
        result = []
        for path in sorted((self.root / "runs").glob("*/run.json")):
            record = json.loads(path.read_text())
            if record.get("status") != "running":
                validate_run(record)
                result.append(record)
        return result

    def _publish(self) -> dict:
        state = rebuild(self.runs())
        atomic(self.root / "state.json", state)
        return state

    def refresh(self) -> dict:
        with locked(self.root):
            return self._publish()

    def state(self) -> dict:
        path = self.root / "state.json"
        return json.loads(path.read_text()) if path.exists() else self.refresh()

    def ingest(self, run: dict) -> bool:
        return bool(self.ingest_many([run])["imported"])

    def ingest_many(self, runs: list[dict]) -> dict:
        for run in runs:
            validate_run(run)
            if run["status"] == "running":
                raise ValueError("unfinished run requires journal recovery, not assessment")
        with locked(self.root):
            pending, duplicate = {}, 0
            for run in runs:
                path = self.root / "runs" / run["id"] / "run.json"
                existing = pending.get(run["id"])
                if existing is None and path.exists():
                    existing = json.loads(path.read_text())
                if existing is not None:
                    if encoded(existing) != encoded(run):
                        raise ValueError("conflicting content for existing run ID")
                    duplicate += 1
                else:
                    pending[run["id"]] = run
            # Validate the complete transaction before writing any new evidence.
            for identity, run in pending.items():
                atomic(self.root / "runs" / identity / "run.json", run)
            if pending:
                self._publish()
            return {"imported": len(pending), "duplicate": duplicate}

    def finalize(self, run: dict) -> None:
        validate_run(run)
        with locked(self.root):
            path = self.root / "runs" / run["id"] / "run.json"
            existing = json.loads(path.read_text())
            if existing.get("status") != "running":
                raise ValueError("finalized evidence is immutable")
            atomic(path, run)
            self._publish()

    def incomplete(self) -> list[dict]:
        return [
            r
            for path in sorted((self.root / "runs").glob("*/run.json"))
            if (r := json.loads(path.read_text())).get("status") == "running"
        ]

    def recover(self, identity: str) -> dict:
        import psutil

        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid run identity")
        path = self.root / "runs" / identity / "run.json"
        record = json.loads(path.read_text())
        if record["status"] != "running":
            raise ValueError("only unfinished journals can be recovered")
        process = record.get("process", {})
        if process.get("hostname") != platform.node():
            raise ValueError("recover an unfinished run on its original host")
        try:
            active = psutil.Process(process["pid"]).create_time() == process["started_at"]
        except psutil.NoSuchProcess:
            active = False
        if active:
            raise ValueError("measurement process is still active")
        samples = []
        journal = path.parent / "samples.jsonl"
        if journal.exists():
            for line in journal.read_text().splitlines():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        record.update(
            status="interrupted",
            completed_at=datetime.now(UTC).isoformat(),
            samples=samples,
            error="recovered unfinished journal; process no longer exists",
        )
        record["checksum"] = digest(record)
        self.finalize(record)
        return record

    def import_directory(self, source: Path) -> dict:
        paths = [source] if source.is_file() else sorted(source.rglob("run.json"))
        runs = [json.loads(path.read_text()) for path in paths]
        return self.ingest_many([r for r in runs if r.get("status") != "running"])

    def pull(self, host: str, remote_directory: str) -> dict:
        # rsync transfers data only. No remote Python, shell pipeline or source sync.
        if not host or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
            for c in host
        ):
            raise ValueError("use a configured SSH host alias")
        if not remote_directory or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/~+-"
            for c in remote_directory
        ):
            raise ValueError("invalid remote directory")
        with tempfile.TemporaryDirectory(prefix="performance-import-") as temporary:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--",
                    f"{host}:{remote_directory.rstrip('/')}/",
                    temporary + "/",
                ],
                check=True,
            )
            return self.import_directory(Path(temporary))
