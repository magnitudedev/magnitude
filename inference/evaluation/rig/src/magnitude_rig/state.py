"""This machine's view of its runs: one state file per run in flight.

`~/.magnitude/rig/runs/<run id>.json` is written by the run's own process until the run is
answered (`mrig runs`, `mrig stop`, `mrig logs`, `mrig wait` read it), then removed; the run's
record and logs stay in its results directory. A state file whose process is gone is
removed by whoever lists it next.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from pathlib import Path

HOME = Path.home() / ".magnitude" / "rig"
RUNS = HOME / "runs"
# The environment variable each coding agent gives its commands to identify its session.
AGENT_SESSIONS = (("claude", "CLAUDE_CODE_SESSION_ID"), ("codex", "CODEX_THREAD_ID"))


def agent_session() -> str | None:
    for _, name in AGENT_SESSIONS:
        if os.environ.get(name):
            return os.environ[name]
    return None


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def alive(pid: int) -> bool:
    """Whether process `pid` has not ended. A child of this process that has exited still answers
    signals until it is reaped, so it is reaped here: a run's caller follows its own child."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:  # not this process's child: its own parent reaps it
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


class RunState:
    """The live state of one run, kept current by its own process."""

    def __init__(self, run_id: str, fields: dict):
        self.path = self._path(run_id)
        self.value = {"run": run_id, "pid": os.getpid(), **fields}
        self.write()

    def update(self, **fields) -> None:
        self.value.update(fields)
        self.write()

    def write(self) -> None:
        _atomic(self.path, self.value)

    @staticmethod
    def _path(run_id: str) -> Path:
        return RUNS / f"{run_id}.json"

    @classmethod
    def remove(cls, run_id: str) -> None:
        """The run is answered: no longer in flight."""
        cls._path(run_id).unlink(missing_ok=True)


def account(run: dict, now: float) -> tuple[float, float]:
    """A live run's seconds so far, as (queued, held), by the host's one definition
    (`host.agent.account`) over the waits its runner announced (`waited`, and `waiting_since`
    while it waits): its estimate is judged against `held` alone."""
    from .host.agent import account as accounted

    return accounted(run["started"], {"waited": run.get("waited", 0.0), "since": run.get("waiting_since")}, now)


def live_runs() -> list[dict]:
    runs = []
    for path in sorted(RUNS.glob("*.json")) if RUNS.is_dir() else ():
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if alive(value["pid"]):
            runs.append(value)
        else:
            path.unlink(missing_ok=True)
    return runs


class Launched:
    """A run this machine started on a host and has not fetched whole yet:
    `~/.magnitude/rig/launched/<run id>.json`, written once its runner owns the run and removed
    once its directory is fetched, so a run whose caller is gone is still answered and fetched
    (`engine.collect`)."""

    DIRECTORY = HOME / "launched"

    def __init__(self, run_id: str, host: str, agent: list[str], results: Path, command: tuple[str, ...]):
        # `command`: the mrig command line that asked for the run, which answers it once fetched.
        self.value = {"run": run_id, "host": host, "agent": agent, "results": str(results),
                      "command": list(command)}

    def write(self) -> None:
        _atomic(self.DIRECTORY / f"{self.value['run']}.json", self.value)

    @classmethod
    def read(cls, run_id: str) -> dict:
        return json.loads((cls.DIRECTORY / f"{run_id}.json").read_text())

    @classmethod
    def remove(cls, run_id: str) -> None:
        (cls.DIRECTORY / f"{run_id}.json").unlink(missing_ok=True)

    @classmethod
    @contextlib.contextmanager
    def claim(cls, run_id: str) -> Iterator[bool]:
        """Whether this process alone holds the run's record for the block: an exclusive `flock`
        on the record file, taken without waiting. The kernel grants it to one open file at a
        time and releases it when its holder ends, so a collector that dies mid-fetch leaves the
        run to the next. False when another holds it, or has removed the record (a lock taken
        on a file already unlinked holds nothing)."""
        path = cls.DIRECTORY / f"{run_id}.json"
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            yield False
            return
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                current = os.stat(path).st_ino == os.fstat(descriptor).st_ino
            except FileNotFoundError:
                current = False
            yield current
        finally:
            os.close(descriptor)

    @classmethod
    def gone(cls) -> list[dict]:
        """The launched runs no process here follows any more."""
        found = []
        for path in sorted(cls.DIRECTORY.glob("*.json")) if cls.DIRECTORY.is_dir() else ():
            try:
                value = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # collected meanwhile by another command
            if run_state(value["run"]) is None:
                found.append(value)
        return found


def run_state(run_id: str) -> dict | None:
    for value in live_runs():
        if value["run"] == run_id or value["run"].endswith(run_id):
            return value
    return None
