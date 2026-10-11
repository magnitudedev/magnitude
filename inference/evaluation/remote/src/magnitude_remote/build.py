"""Build identity of the local working tree: the contract's `build` object.

`dirty_digest` is the SHA-256 of `git diff HEAD --binary` followed by each untracked,
non-ignored file of the repository in path order as `path NUL bytes NUL`, or None when
both are empty. An untracked symlink contributes its link target text, never the file it
points to, so broken or out-of-tree links are hashed like any other change.

The controller computes it once from the tree it synchronizes; hosts never need git.
"""

import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .sync import INFERENCE


@dataclass(frozen=True)
class BuildIdentity:
    commit: str
    dirty_digest: str | None

    def to_json(self) -> dict:
        return asdict(self)


def _git(tree: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(tree), *arguments], capture_output=True, check=True
    ).stdout


def _content(path: Path) -> bytes:
    if path.is_symlink():
        return os.fsencode(os.readlink(path))
    return path.read_bytes()


def identify(tree: Path = INFERENCE) -> BuildIdentity:
    commit = _git(tree, "rev-parse", "HEAD").decode().strip()
    root = Path(_git(tree, "rev-parse", "--show-toplevel").decode().strip())
    material = hashlib.sha256()
    diff = _git(tree, "diff", "HEAD", "--binary")
    material.update(diff)
    listed = _git(tree, "ls-files", "--others", "--exclude-standard", "-z", "--full-name", ":/")
    untracked = sorted(path for path in listed.split(b"\0") if path)
    for path in untracked:
        material.update(path + b"\0" + _content(root / os.fsdecode(path)) + b"\0")
    return BuildIdentity(commit, material.hexdigest() if diff or untracked else None)
