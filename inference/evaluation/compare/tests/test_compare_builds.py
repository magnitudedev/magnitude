"""Build sides: identities, ref materialization with the working tree's harness, and the
workspace check."""

import io
import tarfile

import pytest
from magnitude_remote.build import BuildIdentity
from magnitude_remote.sync import INFERENCE, manifest

from magnitude_compare import builds
from magnitude_compare.plan import WORKTREE, Build

COMMIT_TIME = 1_700_000_000
WORKSPACE = '[workspace]\nmembers = ["engine", "evaluation/harness"]\n'


def archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), COMMIT_TIME
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def git_archive(monkeypatch):
    files = {"Cargo.toml": WORKSPACE}

    def git(*arguments: str) -> bytes:
        assert (
            arguments[:2] == ("archive", "--format=tar") and arguments[2] == f"{'c' * 40}:inference"
        )
        return archive(files)

    monkeypatch.setattr(builds, "_git", git)
    return files


def test_materialize_overlays_the_working_tree_harness(tmp_path, git_archive):
    git_archive.update(
        {
            "engine/src/lib.rs": "engine",
            "native/llama/x.c": "submodule",
            "engine/target/release/out": "build output",
            "evaluation/harness/src/stale.rs": "the ref's own harness",
        }
    )
    tree = builds.materialize("base", Build("v1", BuildIdentity("c" * 40, None)), tmp_path)
    directory = tmp_path / "trees" / "compare-base"
    assert tree.name == "compare-base" and tree.local == directory
    assert (directory / "engine/src/lib.rs").read_text() == "engine"
    assert (directory / "engine/src/lib.rs").stat().st_mtime == COMMIT_TIME
    assert not (directory / "native").exists() and not (directory / "engine/target").exists()
    harness = [path for path in manifest() if path.startswith("evaluation/harness/")]
    assert "evaluation/harness/Cargo.toml" in harness
    assert not (directory / "evaluation/harness/src/stale.rs").exists()
    for path in harness:
        assert (directory / path).read_bytes() == (INFERENCE / path).read_bytes()
    assert set(tree.paths) == {"Cargo.toml", "engine/src/lib.rs", *harness}
    # Materializing again replaces the tree.
    git_archive.pop("engine/src/lib.rs")
    again = builds.materialize("base", Build("v1", BuildIdentity("c" * 40, None)), tmp_path)
    assert "engine/src/lib.rs" not in again.paths and not (directory / "engine/src/lib.rs").exists()


def test_ref_without_the_harness_is_unavailable(tmp_path, git_archive):
    git_archive["Cargo.toml"] = '[workspace]\nmembers = ["engine"]\n'
    with pytest.raises(builds.BuildUnavailable, match="predates the evaluation harness"):
        builds.materialize("base", Build("old", BuildIdentity("c" * 40, None)), tmp_path)


def test_identities(monkeypatch):
    worktree_identity = BuildIdentity("d" * 40, "e" * 64)
    monkeypatch.setattr(builds, "identify_worktree", lambda: worktree_identity)
    assert builds.identify(WORKTREE) == Build(WORKTREE, worktree_identity)
    head = builds.identify("HEAD")
    assert head.identity.dirty_digest is None and len(head.commit) == 40 and not head.worktree
    with pytest.raises(builds.BuildUnavailable):
        builds.identify("no-such-ref-for-compare")
