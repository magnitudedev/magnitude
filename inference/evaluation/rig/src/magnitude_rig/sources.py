"""What a run builds: this checkout's working tree, or a git ref, as a keyed source.

Each source has a key, and every host keeps one source tree and one build target per key
(`<root>/checkouts/<key>/`), so runs from different checkouts never rebuild each other's
targets. The working tree's key names the checkout's directory and its path. A ref is
materialized for the run in a temporary directory with `git archive` of its `inference/` tree,
keeping file times (the commit time), and removed when the run ends: the host keeps the built
checkout under the ref's key, so the next run of the same ref transfers and rebuilds nothing.
The working tree's `evaluation/` tree (the harness, this rig's host package, the references,
the benchmark) replaces the ref's, so two sources are always measured by the same tools, and a
ref from before any of them still runs them. The ref's key names its commit and the harness it
carries, the only part of those tools cargo builds.
"""

import hashlib
import io
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .build import BuildIdentity, identify
from .sync import INFERENCE, manifest, mirrored

REPOSITORY = INFERENCE.parent
TOOLS = "evaluation"
HARNESS = "evaluation/harness"
WORKTREE = "worktree"


class SourceUnavailable(Exception):
    """A ref cannot be resolved or built with the evaluation harness."""


@dataclass(frozen=True)
class Source:
    name: str  # `worktree` or the ref as given
    key: str
    directory: Path
    paths: tuple[str, ...]
    identity: BuildIdentity


def _git(*arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(REPOSITORY), *arguments], capture_output=True)
    if result.returncode != 0:
        raise SourceUnavailable(f"git {' '.join(arguments)}: {result.stderr.decode().strip()}")
    return result.stdout


def worktree() -> Source:
    path = str(INFERENCE.resolve())
    key = f"{REPOSITORY.name}-{hashlib.sha256(path.encode()).hexdigest()[:10]}"
    return Source(WORKTREE, key, INFERENCE, tuple(manifest()), identify())


def _harness_digest(tools: list[str]) -> str:
    material = hashlib.sha256()
    for path in tools:
        if path.startswith(HARNESS + "/"):
            material.update(path.encode() + b"\0" + (INFERENCE / path).read_bytes() + b"\0")
    return material.hexdigest()[:10]


def commit(ref: str) -> str:
    return _git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()


@contextmanager
def ref(name: str) -> Iterator[Source]:
    """The ref's `inference/` tree with this worktree's tools, for as long as it is open."""
    resolved = commit(name)
    members = tomllib.loads(_git("show", f"{resolved}:inference/Cargo.toml").decode())
    if HARNESS not in members["workspace"]["members"]:
        raise SourceUnavailable(
            f"{name} ({resolved[:12]}) predates the evaluation harness: its workspace does not "
            f"list {HARNESS}"
        )
    tools = [path for path in manifest() if path.startswith(TOOLS + "/")]
    key = f"ref-{resolved[:12]}-{_harness_digest(tools)}"
    with tempfile.TemporaryDirectory(prefix=f"mrig-{key}-") as temporary:
        directory = Path(temporary)
        archive = _git("archive", "--format=tar", f"{resolved}:inference")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            kept = [member for member in tar.getmembers() if mirrored(member.name)]
            tar.extractall(directory, kept, filter="data")
        if (directory / TOOLS).exists():
            shutil.rmtree(directory / TOOLS)
        for path in tools:
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(INFERENCE / path, directory / path)
        paths = sorted(
            path.relative_to(directory).as_posix() for path in directory.rglob("*") if not path.is_dir()
        )
        yield Source(name, key, directory, tuple(paths), BuildIdentity(resolved, None))


@contextmanager
def opened(name: str) -> Iterator[Source]:
    """The named source (`worktree` or a git ref), for as long as a run needs it."""
    if name == WORKTREE:
        yield worktree()
    else:
        with ref(name) as source:
            yield source
