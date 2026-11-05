"""A job that starts with none of its model's pages cached, so it reads them from storage and
every page of them it finds cached is one its own load cached.

The engine reads streamed weights uncached (`magnitude_artifacts` reader). On Linux that is
`O_DIRECT`, which reads from storage whatever the page cache holds but evicts none of it: the
job drops the target's cached pages first (`POSIX_FADV_DONTNEED`, which needs no privilege;
on a GB10 it took a model file from 1,433,118 cached pages to none), so pages an earlier
load cached (a resident load reads through the cache) are not read as the job's. On macOS it
is `F_NOCACHE`, which still serves pages already cached, and macOS keeps cached pages per
file: there the job runs on APFS clones of the target's files (`cp -c`), which share the
files' blocks, so take no space, but none of their cached pages. The clones live under
`clones`, in the run's directory, for the job alone; they are removed when it ends, and one a
killed job leaves goes with the run's directory. Elsewhere nothing makes a job fresh, and it
is refused.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from . import trace

# A split GGUF's shard name; the engine opens the first and finds the rest beside it.
SHARD = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")


def target_files(target: Path) -> list[Path]:
    """The files the engine reads for `target`: every shard of a split GGUF, or the file."""
    match = SHARD.match(target.name)
    if match is None:
        return [target]
    count = int(match["count"])
    width = len(match["count"])
    return [
        target.with_name(f"{match['stem']}-{index:0{width}d}-of-{match['count']}.gguf")
        for index in range(1, count + 1)
    ]


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


def drop_cached(path: Path) -> None:
    """Drop every page of `path` the page cache holds (Linux)."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def run(artifacts: Path, output: Path, clones: Path, argv: list[str]) -> int:
    """Run `argv` with `output` as its acquisition object: `artifacts`', its target on
    clones under `clones` on macOS, its target's cached pages dropped on Linux."""
    if sys.platform not in ("darwin", "linux"):
        sys.exit(f"fresh: no way to start {sys.platform}'s job with none of its model's pages cached")
    acquisition = json.loads(artifacts.read_text())
    cloned = sys.platform == "darwin"
    signal.signal(signal.SIGTERM, _terminate)
    try:
        target = Path(acquisition["target"])
        if cloned:
            clones.mkdir(parents=True)
            for path in target_files(target):
                subprocess.run(["cp", "-c", str(path), str(clones / path.name)], check=True)
            acquisition["target"] = str(clones / target.name)
        else:  # Linux
            for path in target_files(target):
                drop_cached(path)
        trace.atomic(output, acquisition)
        child = subprocess.Popen(argv)
        try:
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    finally:
        if cloned and clones.exists():
            shutil.rmtree(clones)
