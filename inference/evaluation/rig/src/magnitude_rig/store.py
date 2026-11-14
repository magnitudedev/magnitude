"""The repository's one store of run results: the main checkout's
`inference/evaluation/results/`, whichever worktree runs. A worktree is removed with everything
under it, so runs recorded there would go with it; the main checkout is the working tree of the
repository's common git directory (`git rev-parse --git-common-dir`), which outlives every
worktree. Runs are keyed by run ID, which no two runs share.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

from . import sources

RELATIVE = Path("evaluation") / "results"  # under the checkout's `inference/`


class StoreUnavailable(Exception):
    """The checkout's repository has no main working tree to hold the store."""


@functools.cache
def results() -> Path:
    """The store of the invoking checkout's repository (`store_for`; `sources.checkout`: the one owner of which)."""
    return store_for(sources.checkout())


def store_for(inference: Path) -> Path:
    """The store of the repository whose checkout's `inference/` is `inference`, created."""
    store = main_checkout(inference) / "inference" / RELATIVE
    store.mkdir(parents=True, exist_ok=True)
    return store


def main_checkout(inside: Path) -> Path:
    """The working tree of the repository `inside` belongs to: the common git directory's."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=inside, capture_output=True, text=True,
    )
    if common.returncode != 0:
        raise StoreUnavailable(f"{inside}: no git repository: {common.stderr.strip()}")
    directory = Path(common.stdout.strip())
    if directory.name != ".git":
        raise StoreUnavailable(f"{directory}: a bare repository has no main checkout for results")
    return directory.parent
