"""Build identity: the contract's {commit, dirty_digest} of a git working tree."""

import hashlib
import subprocess

from magnitude_remote.build import BuildIdentity, identify


def git(tree, *arguments):
    return subprocess.run(
        ["git", "-C", str(tree), *arguments], capture_output=True, check=True, text=True
    ).stdout


def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("ignored\n")
    (tmp_path / "tracked.txt").write_text("one\n")
    (tmp_path / "sub").mkdir()
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "initial")
    return git(tmp_path, "rev-parse", "HEAD").strip()


def test_clean_tree_has_no_dirty_digest(tmp_path):
    commit = repository(tmp_path)
    (tmp_path / "ignored").write_text("never hashed")
    assert identify(tmp_path) == BuildIdentity(commit, None)


def test_dirty_digest_follows_the_contract_and_hashes_link_text(tmp_path):
    commit = repository(tmp_path)
    (tmp_path / "tracked.txt").write_text("two\n")
    (tmp_path / "b.txt").write_bytes(b"untracked")
    (tmp_path / "a-broken-link").symlink_to("/nonexistent/target")
    identity = identify(tmp_path / "sub")  # any directory of the repository
    diff = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "HEAD", "--binary"], capture_output=True, check=True
    ).stdout
    material = diff + b"a-broken-link\0/nonexistent/target\0" + b"b.txt\0untracked\0"
    assert identity == BuildIdentity(commit, hashlib.sha256(material).hexdigest())
    assert identity.to_json() == {"commit": commit, "dirty_digest": identity.dirty_digest}
