"""Hosts for the engine's end-to-end tests: directories on this machine reached through an `ssh`
stand-in on PATH that runs the remote command line here, so the real agent, rsync and host
package serve every run unchanged. While the file `link_down(tmp_path)` exists, the stand-in
fails as ssh does when its host does not answer.

Every test owns the processes it starts, through the stand-in or not: each inherits the test's
own `RIG_TEST_SCOPE`, which no detaching (the agent's runners start sessions of their own)
sheds, and each still running when the test ends, passed, failed or interrupted, is killed
(`_owned_processes`). A run stopped from outside (`timeout`, `kill`) interrupts as Ctrl-C does,
so its teardown runs too."""

import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from magnitude_rig import state, store
from magnitude_rig.build import BuildIdentity
from magnitude_rig.hosts import HostConfig, Hosts
from magnitude_rig.sources import Source

PACKAGE = Path(__file__).parents[1] / "src" / "magnitude_rig"
SSH = """#!/bin/sh
if [ -e "$RIG_LINK_DOWN" ]; then echo "ssh: connect to host: Operation timed out" >&2; exit 255; fi
while [ $# -gt 0 ]; do
  case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac
done
shift
exec /bin/sh -c "$*"
"""


SCOPE = "RIG_TEST_SCOPE"


def pytest_configure(config):
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stopped from outside: tear down


def _scoped(token: str) -> list[int]:
    """The live processes started with `SCOPE` set to `token`, this one aside."""
    marker = f"{SCOPE}={token}"
    if os.path.isdir("/proc/self"):
        found = []
        for name in os.listdir("/proc"):
            try:
                with open(f"/proc/{name}/environ", "rb") as stream:
                    if marker.encode() in stream.read().split(b"\0"):
                        found.append(int(name))
            except (OSError, ValueError):
                continue
    else:  # BSD ps: -E shows each process's environment after its command
        listing = subprocess.run(["ps", "-Eww", "-axo", "pid=,command="], capture_output=True, text=True,
                                 check=True).stdout  # run unscoped, or it would list itself
        found = [int(line.split(None, 1)[0]) for line in listing.splitlines() if marker in line.split()]
    return [pid for pid in found if pid != os.getpid()]


@pytest.fixture(autouse=True)
def _owned_processes():
    """Set up before, so torn down after, every fixture of the test, its monkeypatches undone:
    the kill sees the real `subprocess` and `os`."""
    token = uuid.uuid4().hex
    os.environ[SCOPE] = token
    try:
        yield
    finally:
        del os.environ[SCOPE]
    deadline = time.monotonic() + 10
    while pids := _scoped(token):  # a runner may start a step between listing and its kill
        assert time.monotonic() < deadline, f"processes {pids} outlived every kill"
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.05)


def link_down(tmp_path: Path) -> Path:
    return tmp_path / "link-down"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    if shutil.which("rsync") is None:
        pytest.skip("rsync is needed to mirror sources")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(SSH)
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("RIG_LINK_DOWN", str(link_down(tmp_path)))
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(store, "results", lambda: tmp_path / "results")
    monkeypatch.setattr(state, "RUNS", tmp_path / "state" / "runs")
    monkeypatch.setattr(state.Launched, "DIRECTORY", tmp_path / "state" / "launched")
    hosts = {}
    for alias in ("alpha", "beta", "local"):
        root = tmp_path / alias
        (root / "rig").mkdir(parents=True)
        (root / "rig" / "ready.json").write_text("ready")
        hosts[alias] = HostConfig(alias, str(root), str(tmp_path / alias / "models"), local=alias == "local")
    source_dir = tmp_path / "source"
    shutil.copytree(PACKAGE, source_dir / "evaluation" / "rig" / "src" / "magnitude_rig",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (source_dir / "data.txt").write_text("the source\n")
    paths = tuple(sorted(p.relative_to(source_dir).as_posix() for p in source_dir.rglob("*") if p.is_file()))
    # Fixed for the test, as a ref is for its run: never read again (`Source.current`).
    source = Source("fixture", "test-checkout", source_dir, paths, BuildIdentity("c" * 40, None), "test-branch",
                    tmp_path / "test-worktree")
    return Hosts(HostConfig("", str(tmp_path / "default"), str(tmp_path / "models")),
                 {"pool": ("alpha", "beta")}, hosts), source, tmp_path
