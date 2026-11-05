"""A step that waits until a test creates a path, so the run holding it holds what it holds until
then. Whatever starts it (a runner, the host's `record`, a shim), it lives no longer than the
process that asked for it, named by pid when the step is made (the test's own, by default), and
never past a hard deadline: neither depends on when it starts, so one started after its owner
is gone exits at once, and a test killed before it creates the path leaves no wait behind."""

import os
import sys
import time
from pathlib import Path

POLL_SECONDS = 0.2
DEADLINE_SECONDS = 600  # past any test's own time
UNTIL_EXISTS = f"""
import os, sys, time
path, owner, deadline = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
while not os.path.exists(path):
    if time.time() > deadline:
        sys.exit("until-exists: past its deadline")
    try:
        os.kill(owner, 0)
    except ProcessLookupError:
        sys.exit("until-exists: its owner is gone")
    except PermissionError:
        pass  # alive, another user's
    time.sleep({POLL_SECONDS})
"""


def until_exists(path: Path, *, before: str = "", owner: int | None = None,
                 seconds: float = DEADLINE_SECONDS) -> list[str]:
    """The step's command: it waits for `path` while process `owner` (this one by default)
    lives, `seconds` at most from now; `before`, Python it runs first."""
    owner = os.getpid() if owner is None else owner
    return [sys.executable, "-c", before + UNTIL_EXISTS, str(path), str(owner), str(time.time() + seconds)]
