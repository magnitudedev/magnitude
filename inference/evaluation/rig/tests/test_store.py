"""The results store: one per repository, in its main checkout, whichever worktree runs."""

import subprocess

import pytest

from magnitude_rig import store


def git(*argv, cwd):
    subprocess.run(["git", *argv], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path):
    """A main checkout and two linked worktrees, each with an `inference/`."""
    main = tmp_path / "main"
    (main / "inference").mkdir(parents=True)
    (main / "inference" / "README").write_text("inference\n")
    git("init", "-q", cwd=main)
    git("add", ".", cwd=main)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c", cwd=main)
    worktrees = [tmp_path / name for name in ("first", "second")]
    for worktree in worktrees:
        git("worktree", "add", "-q", str(worktree), cwd=main)
    return main, worktrees


def test_two_worktrees_write_to_the_main_checkouts_store(repository):
    main, (first, second) = repository
    stores = [store.store_for(worktree / "inference") for worktree in (first, second)]
    assert stores[0] == stores[1] == (main / "inference" / "evaluation" / "results").resolve()
    (stores[0] / "20261024T000000Z-test-aaaaaa").mkdir()
    (stores[1] / "20261024T000001Z-test-bbbbbb").mkdir()
    assert sorted(p.name for p in stores[0].iterdir()) == [
        "20261024T000000Z-test-aaaaaa", "20261024T000001Z-test-bbbbbb"]
    # The main checkout's own store is the same one.
    assert store.store_for(main / "inference") == stores[0]
