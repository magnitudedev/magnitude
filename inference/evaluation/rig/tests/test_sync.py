"""Mirror semantics, against a local directory standing in for a host: a mirrored file's time
is when its content last changed there, whatever the controller's file times say."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from magnitude_rig import sync

AGENT = Path(__file__).parents[1] / "src" / "magnitude_rig" / "host" / "agent.py"
FROZEN = 1_600_000_000  # every controller file time: long before anything on the "host"
BUILT = 1_700_000_000  # when the host last built: after FROZEN, before the next mirror


def write_tree(directory: Path, files: dict[str, str]) -> Path:
    """A tree whose files all carry the same old time, like a frozen copy."""
    for path, text in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_text(text)
        os.utime(directory / path, (FROZEN, FROZEN))
    return directory


def mirror(tree: Path, host: Path) -> None:
    """The flow's mirror step: content push, then the agent's prune of unlisted files."""
    paths = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
    host.mkdir(exist_ok=True)
    sync.mirror(paths, tree, str(host))
    subprocess.run(
        [sys.executable, str(AGENT), "prune", "--dir", str(host)],
        input="\n".join(paths) + "\n", text=True, check=True, capture_output=True,
    )


def settle(host: Path) -> None:
    """Stand in for a build finishing after the mirror: every host file is older than BUILT."""
    for path in host.rglob("*"):
        if path.is_file():
            os.utime(path, (BUILT - 1, BUILT - 1))


def changed_since_build(host: Path) -> set[str]:
    return {
        p.relative_to(host).as_posix()
        for p in host.rglob("*")
        if p.is_file() and p.stat().st_mtime > BUILT
    }


def test_changed_content_reads_as_changed_even_when_its_file_time_is_older(tmp_path):
    x = write_tree(tmp_path / "x", {"src/a.rs": "tree X", "src/b.rs": "shared"})
    y = write_tree(tmp_path / "y", {"src/a.rs": "tree Y", "src/b.rs": "shared"})  # same sizes
    host = tmp_path / "host"
    mirror(x, host)
    assert changed_since_build(host) == {"src/a.rs", "src/b.rs"}  # never the controller's times
    settle(host)
    mirror(x, host)
    assert changed_since_build(host) == set()  # an unchanged tree changes nothing
    mirror(y, host)
    assert (host / "src/a.rs").read_text() == "tree Y"
    assert changed_since_build(host) == {"src/a.rs"}


def test_going_back_to_older_content_reads_as_changed_again(tmp_path):
    x = write_tree(tmp_path / "x", {"a.rs": "older", "b.rs": "shared"})
    y = write_tree(tmp_path / "y", {"a.rs": "newer", "b.rs": "shared"})
    os.utime(y / "a.rs", (FROZEN + 500, FROZEN + 500))
    host = tmp_path / "host"
    mirror(x, host)
    for tree, text in ((y, "newer"), (x, "older"), (y, "newer")):
        settle(host)
        mirror(tree, host)
        assert (host / "a.rs").read_text() == text
        assert changed_since_build(host) == {"a.rs"}


def test_deleted_and_renamed_files(tmp_path):
    x = write_tree(tmp_path / "x", {"lib.rs": "mod old;", "old.rs": "body", "keep.rs": "kept"})
    y = write_tree(tmp_path / "y", {"lib.rs": "mod new;", "new.rs": "body", "keep.rs": "kept"})
    host = tmp_path / "host"
    mirror(x, host)
    settle(host)
    mirror(y, host)
    assert sorted(p.name for p in host.iterdir()) == ["keep.rs", "lib.rs", "new.rs"]
    assert changed_since_build(host) == {"lib.rs", "new.rs"}  # the renamed file is new here
    settle(host)
    mirror(x, host)
    assert sorted(p.name for p in host.iterdir()) == ["keep.rs", "lib.rs", "old.rs"]
    assert changed_since_build(host) == {"lib.rs", "old.rs"}


def test_permissions_follow_without_reading_as_a_content_change(tmp_path):
    x = write_tree(tmp_path / "x", {"tool.sh": "#!/bin/sh\n"})
    host = tmp_path / "host"
    mirror(x, host)
    settle(host)
    (x / "tool.sh").chmod(0o755)
    mirror(x, host)
    assert os.access(host / "tool.sh", os.X_OK)
    assert changed_since_build(host) == set()


CRATE = {
    "Cargo.toml": '[package]\nname = "probe"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n',
    "src/main.rs": 'mod who;\nfn main() { println!("{}", who::WHO); }\n',
}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")
def test_cargo_rebuilds_alternating_trees_in_one_target_directory(tmp_path):
    """The observed failure: with file times preserved, the second build below finished without
    compiling and ran the first tree's binary."""
    x = write_tree(tmp_path / "x", {**CRATE, "src/who.rs": 'pub const WHO: &str = "X";\n'})
    y = write_tree(tmp_path / "y", {**CRATE, "src/who.rs": 'pub const WHO: &str = "Y";\n'})
    host, target = tmp_path / "host", tmp_path / "target"

    def build_and_run(tree: Path) -> tuple[str, bool]:
        mirror(tree, host)
        build = subprocess.run(
            ["cargo", "build", "--release", "--offline", "--target-dir", str(target)],
            cwd=host, capture_output=True, text=True, check=True,
        )
        ran = subprocess.run([str(target / "release" / "probe")], capture_output=True, text=True)
        return ran.stdout.strip(), "Compiling probe" in build.stderr

    assert build_and_run(x) == ("X", True)
    assert build_and_run(y) == ("Y", True)
    assert build_and_run(x) == ("X", True)
    assert build_and_run(x) == ("X", False)  # incremental: unchanged content is not rebuilt
