"""What a run builds: the working tree of the git worktree that contains the current directory
(`checkout`), or a git ref, as a keyed source. Never the tree mrig itself was launched from (its
`--project`): an agent in a worktree that runs another checkout's mrig builds its own worktree,
and a command run outside any git worktree is refused, for `worktree` and a ref alike.

Every source states what it is as `branch@commit (dirty|clean)` (`Source.described`): each run's
start line shows it and its record keeps it, with the tree it came from.

Each source has a key, which a host's checkouts record as the source last installed into them
(`host/agent.py`): a run claims a free checkout last built from its key when there is one, so
the next run of the same source transfers and rebuilds nothing, and otherwise one last built
from another source, rebuilding only what differs. The working tree's key names the checkout's
directory and its path. A ref is materialized for the run in a temporary directory with
`git archive` of its `inference/` tree, keeping file times (the commit time), and removed when
the run ends.
The working tree's `evaluation/` tools (this rig's host package, the references, the
benchmark, the fixtures) replace the ref's, so two sources are always measured by the same
tools, and a ref from before any of them still runs them. The harness is the exception: it is
versioned with the executor it drives, so a ref keeps its own (`evaluation/harness`, with the
lock its workspace pins), and each instrument reads what either harness records. The ref's key
is its commit and a digest of the tools overlaid on it (the host package and the venv synced
from them are the checkout's), so runs of one ref with the same tools find the checkout they
last built.
"""

import functools
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
from .sync import manifest, mirrored

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
    branch: str  # the worktree's branch (`HEAD` when detached), or the ref as given
    tree: Path  # the worktree it was staged from: its own tree, or a ref's tools

    def described(self) -> str:
        """`branch@commit (dirty|clean)`."""
        state = "clean" if self.identity.dirty_digest is None else "dirty"
        return f"{self.branch}@{self.identity.commit[:9]} ({state})"


@functools.cache
def checkout() -> Path:
    """The `inference/` tree of the git worktree that contains the current directory: what
    `worktree` builds and whose tools a ref is measured by. SourceUnavailable outside a git
    worktree, or in one with no `inference/` workspace."""
    found = subprocess.run(["git", "-C", str(Path.cwd()), "rev-parse", "--show-toplevel"], capture_output=True)
    if found.returncode != 0:
        raise SourceUnavailable(f"{Path.cwd()} is not inside a git worktree: run mrig from the worktree "
                                "whose tree it builds")
    tree = Path(found.stdout.decode().strip()) / "inference"
    if not (tree / "Cargo.toml").is_file():
        raise SourceUnavailable(f"{tree.parent} has no inference/ workspace: run mrig from a magnitude worktree")
    return tree


def _git(*arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(checkout().parent), *arguments], capture_output=True)
    if result.returncode != 0:
        raise SourceUnavailable(f"git {' '.join(arguments)}: {result.stderr.decode().strip()}")
    return result.stdout


def worktree() -> Source:
    tree = checkout()
    key = f"{tree.parent.name}-{hashlib.sha256(str(tree.resolve()).encode()).hexdigest()[:10]}"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    return Source(WORKTREE, key, tree, tuple(manifest(tree)), identify(tree), branch, tree.parent)


def _overlaid(path: str) -> bool:
    """Whether the working tree's copy of `path` replaces a ref's: every evaluation tool but
    the harness."""
    return path.startswith(TOOLS + "/") and not path.startswith(HARNESS + "/")


@functools.cache
def descends(commit: str, ancestor: str) -> bool:
    """Whether `commit` is `ancestor` or descends from it; False when either is unknown here."""
    return subprocess.run(["git", "-C", str(checkout().parent), "merge-base", "--is-ancestor", ancestor, commit],
                          capture_output=True).returncode == 0


def commit(ref: str) -> str:
    return _git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()


@contextmanager
def ref(name: str) -> Iterator[Source]:
    """The ref's `inference/` tree, its harness its own and its other evaluation tools this
    worktree's, for as long as it is open."""
    resolved = commit(name)
    members = tomllib.loads(_git("show", f"{resolved}:inference/Cargo.toml").decode())
    if HARNESS not in members["workspace"]["members"]:
        raise SourceUnavailable(
            f"{name} ({resolved[:12]}) predates the evaluation harness: its workspace does not "
            f"list {HARNESS}"
        )
    tree = checkout()
    tools = [path for path in manifest(tree) if _overlaid(path)]
    overlay = hashlib.sha256()
    for path in tools:
        overlay.update(path.encode() + b"\0" + (tree / path).read_bytes() + b"\0")
    key = f"ref-{resolved[:12]}-{overlay.hexdigest()[:10]}"
    with tempfile.TemporaryDirectory(prefix=f"mrig-{key}-") as temporary:
        directory = Path(temporary)
        archive = _git("archive", "--format=tar", f"{resolved}:inference")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            kept = [
                member for member in tar.getmembers()
                if mirrored(member.name) and not _overlaid(member.name)
            ]
            tar.extractall(directory, kept, filter="data")
        for path in tools:
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tree / path, directory / path)
        paths = sorted(
            path.relative_to(directory).as_posix() for path in directory.rglob("*") if not path.is_dir()
        )
        yield Source(name, key, directory, tuple(paths), BuildIdentity(resolved, None), name, tree.parent)


@contextmanager
def opened(name: str) -> Iterator[Source]:
    """The named source (`worktree` or a git ref), for as long as a run needs it."""
    if name == WORKTREE:
        yield worktree()
    else:
        with ref(name) as source:
            yield source
