"""Staging local trees on a host, and fetching run directories back (the stages and raw
logits stay on the host, `HOST_ONLY`; the request's files are the caller's own, `RUN_INPUTS`).

A source is git's view of the working tree (tracked plus untracked-not-ignored files). Build
outputs and the top-level `native/` submodule are excluded; `engine/templates/native` is
ordinary source. A ref's materialized tree (`sources.ref`) is staged the same way from its own
path list. The caller stages each side's source into its run's own `stage-<label>/src`, sending
only the files whose bytes (compared by checksum) the side's checkout does not already have;
the runner then installs the stage into the checkout (the agent's `install`, which keeps the
time of every file whose bytes did not change).

An ended run's directory comes back in two fetches: `fetch_answer`, everything its answer may
read, as soon as it ends; then `fetch`, the rest (the runner's own records), which no answer
waits for. Both compress the stream (results and logs are text, sent over links as slow as tens
of KB/s), and both are rsync, so either resumes, file by file, what an earlier attempt copied.
"""

import os
import resource
import subprocess
import tempfile
from pathlib import Path

from .connection import SSH_OPTIONS, CommandFailed, HostUnavailable, Remote, reachable
from .host.agent import EVENTS, RUN_LOCKS, RUNNER_LOG

INFERENCE = Path(__file__).resolve().parents[4]


def mirrored(path: str) -> bool:
    """Whether a path of an inference tree belongs in a mirror (not a build output or `native/`)."""
    parts = path.split("/")
    return parts[0] != "native" and "target" not in parts[:-1]


def manifest(tree: Path) -> list[str]:
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


# By checksum, so a file the checkout has with other times still matches it (`--copy-dest`).
CONTENT = ("--links", "--perms", "--checksum")
ARCHIVE = ("-a",)
# openrsync (macOS's rsync) keeps a descriptor open, until its process ends, for each file whose
# `--copy-dest` copy has other bytes: a receiver may need one per file it is sent, beyond these
# of its own, however far the checkout it copies from is from the source.
RSYNC_DESCRIPTORS = 64
STAGE = "stage-"  # a run directory's `stage-<label>`, one per side
RUN_KERNELS = "kernels"  # a run directory's own kernel cache, fresh per run
RUN_CLONES = "clones"  # a run directory's clones of its model's files, per fresh job (`host/fresh.py`)
RUN_INPUTS = "inputs"  # a run directory's copy of the request's files, staged from its results


def _descriptors(count: int) -> None:
    """Raise this process's descriptor limit (which its children inherit) to `count`, as far as
    its hard limit allows; never lower it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < count:
        resource.setrlimit(resource.RLIMIT_NOFILE, (count if hard == resource.RLIM_INFINITY else min(count, hard), hard))


def _rsync(mode: tuple[str, ...], arguments: list[str], remote: Remote | None,
           descriptors: int = RSYNC_DESCRIPTORS, ssh: tuple[str, ...] = ()) -> None:
    """rsync with `remote` (None: between this machine's directories), with `descriptors` open
    files allowed here (where a local receiver runs) and `ssh`'s options beyond the rig's. A
    failure is `HostUnavailable` when the host does not answer once rsync has failed: rsync's
    code cannot say (openrsync, macOS's rsync, exits 1 for a dropped link as for anything)."""
    _descriptors(descriptors)
    result = subprocess.run(
        # rsync starts the remote shell only for `host:path` operands, never for this machine's
        ["rsync", *mode, "-e", " ".join(("ssh", *SSH_OPTIONS, *ssh)), *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        failure = CommandFailed if remote is None or reachable(remote) else HostUnavailable
        raise failure(f"rsync {' '.join(arguments[-2:])} exited {result.returncode}: {result.stderr}")


def copy(paths: list[str], tree: Path, destination: str, *, alike: str | None = None,
         remote: Remote | None = None) -> None:
    """Copy `paths` (relative to the local `tree`) into `destination` (a directory on `remote`,
    or here for None); a file whose bytes `alike` (a directory beside `destination`) has is
    copied from there rather than sent. A missing `alike` sends every file. Copied, not
    hard-linked (`--link-dest`): openrsync, macOS's rsync, links a file whose mode alone changed
    and keeps the old mode, which no later step could restore. The receiver may hold a
    descriptor per file (`RSYNC_DESCRIPTORS`): here, and in a POSIX host's shell, it is allowed
    that many."""
    descriptors = len(paths) + RSYNC_DESCRIPTORS
    options = (f"--copy-dest={alike}",) if alike is not None else ()
    if remote is not None and not (remote.local or remote.windows):
        options += (f"--rsync-path=ulimit -Sn {descriptors} 2>/dev/null; exec rsync",)
    target = destination if remote is None else remote.place(destination)
    with tempfile.NamedTemporaryFile("w", suffix=".files") as listing:
        listing.write("\n".join(paths) + "\n")
        listing.flush()
        _rsync(CONTENT + options, [f"--files-from={listing.name}", f"{tree}/", f"{target}/"], remote, descriptors)


def push(remote: Remote, paths: list[str], tree: Path, destination: str, *, alike: str | None = None) -> None:
    """Copy `paths` (relative to the local `tree`) into `destination` on the host (`copy`)."""
    copy(paths, tree, destination, alike=alike, remote=remote)


# The stages are the caller's copies of what the run installs, already in its checkouts. The
# run's kernel cache is compiled for the host's device, and what of it a run's answer shows is
# in its jobs' logs. Clones are the model's files, which a killed job leaves behind: kept with
# the run's directory, so its removal or eviction takes them. Raw logits stay on the host: they are gigabytes at a long context, and
# everything that reads them (the references' `kl`, the logits digests) runs there. The run's
# locks are its host processes' own (`host/agent.py`).
HOST_ONLY = ("--exclude", f"/{STAGE}*", "--exclude", f"/{RUN_KERNELS}/", "--exclude", f"/{RUN_CLONES}/",
             "--exclude", "logits*.f32",
             *(argument for lock in RUN_LOCKS for argument in ("--exclude", f"/{lock}")))
# Never fetched: what stays on the host, and the request's files, which the caller staged from
# its own results (`engine.Run.stage`).
UNFETCHED = (*HOST_ONLY, "--exclude", f"/{RUN_INPUTS}/")
# The runner's own records, which no answer reads: its events (every line of output again, which
# its phases' logs hold) and its own output. Fetched after the answer.
RECORDS = (EVENTS, RUNNER_LOG)
# A fetch is megabytes of text: compressing it saves minutes on a slow link and costs a fast one
# milliseconds.
COMPRESSED = ("-o", "Compression=yes")


def _fetch(remote: Remote, run_dir: str, destination: Path, excluded: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    _rsync(ARCHIVE + excluded, [f"{remote.place(run_dir)}/", f"{destination}/"], remote, ssh=COMPRESSED)


def fetch_answer(remote: Remote, run_dir: str, destination: Path) -> None:
    """Fetch everything of the run's directory an answer may read: its record of its end, its
    jobs' directories and its phases' logs (the caller already mirrors most of them, so rsync
    sends what they lack)."""
    _fetch(remote, run_dir, destination,
           (*UNFETCHED, *(argument for record in RECORDS for argument in ("--exclude", f"/{record}"))))


def fetch(remote: Remote, run_dir: str, destination: Path) -> None:
    """Fetch the run's whole directory: after `fetch_answer`, the runner's records alone."""
    _fetch(remote, run_dir, destination, UNFETCHED)
