"""`worktree` is the git worktree that contains the current directory, never the tree mrig was
launched from, and nothing outside one; a ref is measured by that worktree's tools and driven by
its own harness: its `evaluation/` tree is the worktree's but for the harness, which is
versioned with its engine, and its key is its commit and the tools overlaid on it."""

import subprocess

import pytest

from magnitude_rig import sources


def test_a_refs_key_names_its_commit_and_the_tools_overlaid_on_it(monkeypatch):
    with sources.ref("HEAD~1") as first, sources.ref("HEAD~1") as again:
        assert first.key == again.key  # the same tools: one checkout
    listed = sources.manifest(sources.checkout())
    tool = next(path for path in listed if sources._overlaid(path))
    monkeypatch.setattr(sources, "manifest", lambda tree: [path for path in listed if path != tool])
    with sources.ref("HEAD~1") as other:
        assert other.key != first.key and other.key.startswith(first.key.rsplit("-", 1)[0])


def test_a_ref_carries_this_worktrees_tools_its_own_harness_and_its_own_engine():
    base = sources.commit("HEAD~1")
    with sources.ref("HEAD~1") as ref:
        assert ref.key.startswith(f"ref-{base[:12]}-")
        assert ref.identity.commit == base and ref.identity.dirty_digest is None
        assert (ref.directory / "evaluation/rig/src/magnitude_rig/host/agent.py").read_bytes() == (
            sources.checkout() / "evaluation/rig/src/magnitude_rig/host/agent.py"
        ).read_bytes()
        harness = "evaluation/harness/src/time.rs"
        assert (ref.directory / harness).read_bytes() == sources._git("show", f"{base}:inference/{harness}")
        assert "evaluation/rig/src/magnitude_rig/engine.py" in ref.paths
        assert "engine/Cargo.toml" in ref.paths and harness in ref.paths
        assert all(
            path.startswith(sources.HARNESS + "/") or (sources.checkout() / path).exists()
            for path in ref.paths if path.startswith(sources.TOOLS + "/")
        ), "no evaluation tool but the harness comes from the ref"
        directory = ref.directory
    assert not directory.exists()  # materialized for the run only


def _git(tree, *arguments):
    return subprocess.run(["git", "-C", str(tree), *arguments], capture_output=True, check=True, text=True).stdout


@pytest.fixture
def elsewhere(monkeypatch):
    """Commands run from another directory: the worktree is found again from it, and the
    directory the rig's own tests run in afterwards."""
    sources.checkout.cache_clear()
    yield monkeypatch
    sources.checkout.cache_clear()


def test_worktree_is_the_git_worktree_of_the_current_directory_and_nothing_outside_one(tmp_path, elsewhere):
    main = tmp_path / "main"
    (main / "inference" / "evaluation").mkdir(parents=True)
    (main / "inference" / "Cargo.toml").write_text("[workspace]\n")
    (main / "inference" / "evaluation" / "tool.txt").write_text("a tool\n")
    _git(tmp_path, "init", "-q", "-b", "trunk", str(main))
    _git(main, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "base")
    _git(main, "add", ".")
    _git(main, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "tree")
    branch = tmp_path / "feature"
    _git(main, "worktree", "add", "-q", "-b", "feature", str(branch))
    (branch / "inference" / "evaluation" / "new.txt").write_text("the branch's change\n")
    head = _git(branch, "rev-parse", "HEAD").strip()

    # Run from inside the branch's worktree, whatever checkout's mrig runs: its tree is staged.
    elsewhere.chdir(branch / "inference" / "evaluation")
    source = sources.worktree()
    assert source.directory.resolve() == (branch / "inference").resolve()
    assert "evaluation/new.txt" in source.paths and source.tree.name == "feature"
    assert source.described() == f"feature@{head[:9]} (dirty)" and source.key.startswith("feature-")
    sources.checkout.cache_clear()
    elsewhere.chdir(main)
    trunk = sources.worktree()
    assert trunk.described() == f"trunk@{head[:9]} (clean)" and "evaluation/new.txt" not in trunk.paths
    assert trunk.key != source.key

    # Outside any git worktree, refused, for `worktree` and a ref alike.
    sources.checkout.cache_clear()
    outside = tmp_path / "outside"
    outside.mkdir()
    elsewhere.chdir(outside)
    with pytest.raises(sources.SourceUnavailable, match="not inside a git worktree"):
        sources.worktree()
    with pytest.raises(sources.SourceUnavailable, match="not inside a git worktree"):
        sources.commit("HEAD")
