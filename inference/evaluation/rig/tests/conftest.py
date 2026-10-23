"""Hosts for the engine's end-to-end tests: directories on this machine reached through an `ssh`
stand-in on PATH that runs the remote command line here, so the real agent, rsync and host
package serve every run unchanged."""

import shutil
from pathlib import Path

import pytest

from magnitude_rig import engine, state
from magnitude_rig.build import BuildIdentity
from magnitude_rig.hosts import HostConfig, Hosts
from magnitude_rig.sources import Source

PACKAGE = Path(__file__).parents[1] / "src" / "magnitude_rig"
SSH = """#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac
done
shift
exec /bin/sh -c "$*"
"""


@pytest.fixture
def rig(tmp_path, monkeypatch):
    if shutil.which("rsync") is None:
        pytest.skip("rsync is needed to mirror sources")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(SSH)
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    monkeypatch.setattr(engine, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(state, "RUNS", tmp_path / "state" / "runs")
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
    source = Source("worktree", "test-checkout", source_dir, paths, BuildIdentity("c" * 40, None))
    return Hosts(HostConfig("", str(tmp_path / "default"), str(tmp_path / "models")),
                 {"pool": ("alpha", "beta")}, hosts), source, tmp_path
