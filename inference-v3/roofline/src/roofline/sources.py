"""Freeze first-party and compiler sources; never execute a mutable checkout."""

import json
import os
import stat
from pathlib import Path

from .contracts import Source, SourceFile

# Generated output, VCS metadata and user caches are not executable source inputs.
EXCLUDED = {
    ".git",
    ".claude",
    ".agents",
    ".codex",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".cache",
    "runs",
    "results",
    ".egg-info",
    "_native",
}
ROOTS = ("formula-performance/src", "src", "native", "performance", "roofline/src", "tilelang")
FILES = (
    "hatch_build.py",
    "formula-performance/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
    "roofline/pyproject.toml",
    "roofline/uv.lock",
    "roofline/models.json",
    "roofline/targets.json",
)


def source_paths(root):
    for name in FILES:
        path = root / name
        if path.is_file():
            yield path
    for name in ROOTS:
        base = root / name
        if not base.exists():
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.endswith(".egg-info"))
            for name in tuple(dirs):
                if (Path(directory) / name).is_symlink():
                    raise ValueError(
                        f"source directory links are unsupported: {Path(directory) / name}"
                    )
            for filename in sorted(files):
                if filename.endswith((".pyc", ".so", ".dylib", ".a", ".o")):
                    continue
                path = Path(directory) / filename
                if path.name == ".git":
                    continue
                yield path


def capture(root, store):
    from .admission import preparation_reservation

    with preparation_reservation():
        return _capture(root, store)


def _capture(root, store):
    paths = list(source_paths(root))
    before = {p: p.stat() for p in paths}
    records = []
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source link leaves inference-v3: {path}")
        content = path.read_bytes()
        mode = bool(before[path].st_mode & stat.S_IXUSR)
        records.append(
            SourceFile(
                path=path.relative_to(root).as_posix(),
                blob=store.put_blob(content),
                executable=mode,
            )
        )
    if paths != list(source_paths(root)) or any(
        (p.stat().st_mtime_ns, p.stat().st_size, p.stat().st_ino, p.stat().st_mode)
        != (old.st_mtime_ns, old.st_size, old.st_ino, old.st_mode)
        for p, old in before.items()
    ):
        raise ValueError("source changed while being captured; retry the measurement")
    source = Source(files=tuple(sorted(records, key=lambda f: f.path)))
    store.put("source", source.source_id, source)
    return source


def materialize(source, store, destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / ".source-files.json"
    expected = {f.path for f in source.files}
    previous = json.loads(manifest.read_text()) if manifest.exists() else []
    for name in set(previous) - expected:
        from .contracts import safe_path

        (destination / safe_path(name)).unlink(missing_ok=True)
    for entry in source.files:
        path = destination / entry.path
        content = store.blob(entry.blob)
        if not path.exists() or path.read_bytes() != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            # This is a reconstructible cache; durable source bytes live in the
            # store. Fsyncing every compiler header makes cold preparation costly.
            path.write_bytes(content)
        path.chmod(0o755 if entry.executable else 0o644)
    manifest.write_text(json.dumps(sorted(expected)))
    return destination


def fixture_inputs(source, store):
    import hashlib
    import urllib.request

    lock_path = "src/benchmark_fixtures/data/moby-dick.lock.json"
    entry = next(f for f in source.files if f.path == lock_path)
    lock = json.loads(store.blob(entry.blob))
    checksum = lock["sha256"]
    if store.blob_path(checksum).exists():
        store.blob(checksum)
        return {"prose.moby-dick": checksum}
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cached = cache / "magnitude/benchmarks/sources" / checksum / "moby-dick.txt"
    content = cached.read_bytes() if cached.exists() else b""
    if hashlib.sha256(content).hexdigest() != checksum:
        with urllib.request.urlopen(lock["url"], timeout=120) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError(
            "pinned prose source is unavailable: downloaded bytes differ from its lock"
        )
    store.put_blob(content)
    return {"prose.moby-dick": checksum}
