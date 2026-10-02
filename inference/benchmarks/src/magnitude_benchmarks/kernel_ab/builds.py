"""Engine measurement tools built from a git revision or the working tree.

A revision builds in a persistent worktree with its own Cargo target directory, so moving it to
another revision rebuilds incrementally. The working tree builds in place. Each build's binaries
are copied to a directory named by its identity, which is what a run records.

Both sides measure with the same tools: the harness checkout's measurement examples replace the
revision's before it builds, so a comparison never mixes two versions of the measurement itself.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = ("forward_bench", "token_logits")
FEATURES = "pinned-tuning"
WORKING_TREE = "working-tree"


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def executable(name: str) -> str:
    return name + (".exe" if sys.platform == "win32" else "")


def tool_sources(repository: Path) -> list[Path]:
    return [repository / "inference" / "engine" / "examples" / f"{tool}.rs" for tool in TOOLS]


def tools_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for path in tool_sources(repository):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def identity(repository: Path, spec: str) -> dict:
    """The commit a spec names, for the working tree a digest of its uncommitted changes, and the
    digest of the measurement tools it is built with."""
    tools = tools_digest(repository)
    if spec != WORKING_TREE:
        return {
            "spec": spec,
            "commit": git(repository, "rev-parse", f"{spec}^{{commit}}"),
            "dirty": None,
            "tools": tools,
        }
    diff = subprocess.run(
        ["git", "-C", str(repository), "diff", "HEAD", "--binary", "--", "inference"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "spec": spec,
        "commit": git(repository, "rev-parse", "HEAD"),
        "dirty": hashlib.sha256(diff).hexdigest()[:12] if diff else None,
        "tools": tools,
    }


def build_name(build: dict) -> str:
    dirty = f"-{build['dirty']}" if build["dirty"] else ""
    return f"{build['commit'][:12]}{dirty}-tools-{build['tools']}"


def build(repository: Path, spec: str, root: Path, log) -> tuple[dict, Path]:
    """Build the tools for `spec` and return the build identity and its binary directory.

    A spec naming a directory that holds a `build.json` is a build made elsewhere (for example
    on a faster machine of the same platform) and is used as it is.
    """
    prebuilt = Path(spec).expanduser()
    if (prebuilt / "build.json").is_file():
        return json.loads((prebuilt / "build.json").read_text()), prebuilt.resolve()
    built = identity(repository, spec)
    output = root / "builds" / build_name(built)
    if (output / "build.json").is_file():
        return built, output
    if spec == WORKING_TREE:
        source, target = repository, repository / "inference" / "target"
    else:
        source = root / "worktrees" / "build"
        target = root / "worktrees" / "target"
        if not source.exists():
            git(repository, "worktree", "add", "--detach", str(source), built["commit"])
        else:
            git(source, "checkout", "--detach", "--force", built["commit"])
        for path in tool_sources(repository):
            shutil.copy2(path, source / path.relative_to(repository))
    log(f"building {spec} ({build_name(built)})")
    command = ["cargo", "build", "--release", "-p", "magnitude-engine", "--features", FEATURES]
    for tool in TOOLS:
        command += ["--example", tool]
    subprocess.run(
        command,
        cwd=source / "inference",
        check=True,
        env={**os.environ, "CARGO_TARGET_DIR": str(target)},
    )
    output.mkdir(parents=True, exist_ok=True)
    for tool in TOOLS:
        shutil.copy2(target / "release" / "examples" / executable(tool), output / executable(tool))
    (output / "build.json").write_text(json.dumps(built, indent=2))
    return built, output
