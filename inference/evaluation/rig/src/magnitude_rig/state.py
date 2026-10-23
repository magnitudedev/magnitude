"""This machine's view of its runs: one state file per run in flight.

`~/.magnitude/rig/runs/<run id>.json` is written by the run's own process while it lives
(`mrig runs`, `mrig stop`, `mrig logs`, `mrig wait` read it) and removed when it ends; the
run's record and logs stay in its results directory. A state file whose process is gone is
removed by whoever lists it next.
"""

from __future__ import annotations

import json
import os
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
        self.path = RUNS / f"{run_id}.json"
        self.value = {"run": run_id, "pid": os.getpid(), **fields}
        self.write()

    def update(self, **fields) -> None:
        self.value.update(fields)
        self.write()

    def write(self) -> None:
        _atomic(self.path, self.value)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


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


def run_state(run_id: str) -> dict | None:
    for value in live_runs():
        if value["run"] == run_id or value["run"].endswith(run_id):
            return value
    return None
