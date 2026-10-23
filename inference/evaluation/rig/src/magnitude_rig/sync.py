"""Mirroring local trees to a host, and fetching run directories back.

The source mirror is git's view of the working tree (tracked plus untracked-not-ignored
files). Build outputs and the top-level `native/` submodule are excluded;
`engine/templates/native` is ordinary source. A ref's materialized tree (`sources.ref`) is
mirrored the same way from its own path list. Remote files a manifest no longer lists are
pruned by the agent.

A mirrored file's time on the host is when its content last changed there, never the
controller's file time: a file is transferred only when its bytes differ from the host's
(compared by checksum) and is then written with the host's current time, and a file with
the same bytes keeps the time it has. Tools that decide by file time (cargo) then rebuild
exactly what changed, in either direction: a file whose content went back to older bytes still
reads as changed.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from .connection import SSH_OPTIONS, Remote

INFERENCE = Path(__file__).resolve().parents[4]


def mirrored(path: str) -> bool:
    """Whether a path of an inference tree belongs in a mirror (not a build output or `native/`)."""
    parts = path.split("/")
    return parts[0] != "native" and "target" not in parts[:-1]


def manifest(tree: Path = INFERENCE) -> list[str]:
    listed = subprocess.run(
        ["git", "-C", str(tree), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = []
    for path in dict.fromkeys(listed.split("\0")):
        if not path or not mirrored(path):
            continue
        if os.path.lexists(tree / path):  # unstaged deletions are still listed by git
            paths.append(path)
    return paths


CONTENT = ("--links", "--perms", "--checksum")  # no --times: see the module documentation
ARCHIVE = ("-a",)


def _rsync(mode: tuple[str, ...], arguments: list[str]) -> None:
    result = subprocess.run(
        # rsync starts the remote shell only for `host:path` operands, never for this machine's
        ["rsync", *mode, "-e", " ".join(("ssh", *SSH_OPTIONS)), *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OSError(f"rsync {' '.join(arguments[-2:])} exited {result.returncode}: {result.stderr}")


def mirror(paths: list[str], tree: Path, destination: str) -> None:
    """Make `paths` (relative to the local `tree`) under `destination` (an rsync directory,
    local or `host:path`) equal in content, changing the time of only the files that differ."""
    with tempfile.NamedTemporaryFile("w", suffix=".files") as listing:
        listing.write("\n".join(paths) + "\n")
        listing.flush()
        _rsync(CONTENT, [f"--files-from={listing.name}", f"{tree}/", f"{destination}/"])


def push(remote: Remote, paths: list[str], tree: Path, destination: str) -> None:
    """Mirror `paths` (relative to the local `tree`) into `destination` on the host."""
    mirror(paths, tree, remote.place(destination))


def fetch(remote: Remote, run_dir: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    _rsync(ARCHIVE, [f"{remote.place(run_dir)}/", f"{destination}/"])
