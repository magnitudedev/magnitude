"""The two builds' sources and identities, on the controlling machine.

A side is the working tree (synchronized as the job's source tree) or a git ref. A ref is
materialized with `git archive` of its `inference/` tree, keeping file times (the commit
time) so an unchanged ref stays built, and becomes one of the job's extra trees. Both sides
measure with the same tool: the working tree's harness replaces the ref's before it builds,
so a comparison never mixes two versions of the measurement itself. The ref's workspace must
already list the harness.
"""

import io
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

from magnitude_remote.build import BuildIdentity
from magnitude_remote.build import identify as identify_worktree
from magnitude_remote.flow import Tree
from magnitude_remote.sync import INFERENCE, manifest, mirrored

from .plan import WORKTREE, Build

REPOSITORY = INFERENCE.parent
HARNESS = "evaluation/harness"


class BuildUnavailable(Exception):
    """A side cannot be built with the evaluation harness."""


def _git(*arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(REPOSITORY), *arguments], capture_output=True)
    if result.returncode != 0:
        raise BuildUnavailable(f"git {' '.join(arguments)}: {result.stderr.decode().strip()}")
    return result.stdout


def identify(source: str) -> Build:
    """The working tree's identity (remote's, per the contract), or a ref's commit with no
    uncommitted digest."""
    if source == WORKTREE:
        return Build(WORKTREE, identify_worktree())
    commit = (
        _git("rev-parse", "--verify", "--end-of-options", f"{source}^{{commit}}").decode().strip()
    )
    return Build(source, BuildIdentity(commit, None))


def check_workspace(build: Build, cargo_toml: str) -> None:
    members = tomllib.loads(cargo_toml)["workspace"]["members"]
    if HARNESS not in members:
        raise BuildUnavailable(
            f"{build.source} ({build.commit[:12]}) predates the evaluation harness: its workspace does "
            f"not list {HARNESS}, so it cannot be measured; compare against a later commit"
        )


def check_ref(build: Build) -> None:
    """Whether a ref can be built with the harness, read from git without materializing it."""
    check_workspace(build, _git("show", f"{build.commit}:inference/Cargo.toml").decode())


def tree_name(side: str) -> str:
    return f"compare-{side}"


def materialize(side: str, build: Build, root: Path) -> Tree:
    """The ref's `inference/` tree at `<root>/trees/compare-<side>`, with the working tree's
    harness in place of its own."""
    directory = root / "trees" / tree_name(side)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    archive = _git("archive", "--format=tar", f"{build.commit}:inference")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = [member for member in tar.getmembers() if mirrored(member.name)]
        tar.extractall(directory, members, filter="data")
    check_workspace(build, (directory / "Cargo.toml").read_text())
    harness = directory / HARNESS
    if harness.exists():
        shutil.rmtree(harness)
    for path in manifest():
        if path.startswith(HARNESS + "/"):
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(INFERENCE / path, directory / path)
    paths = sorted(
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if not path.is_dir()
    )
    return Tree(tree_name(side), directory, tuple(paths))
