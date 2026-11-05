"""Build provenance: proof that an executable was compiled from the sources now on the host.

Cargo decides what to recompile from file times and its own fingerprints. This module never
consults either. The rig sets `RUSTC_WORKSPACE_WRAPPER` to a shim that runs `rustc` through
`wrap`, so every compilation of a workspace crate leaves a record beside its outputs:

    <out-dir>/<crate><extra-filename>.provenance.json
    {"schema": "build-provenance-unit/1", "crate": ..., "sources": {path: sha256},
     "outputs": {path: sha256}, "externs": {path: sha256}}

`sources` are the files rustc itself reports reading (its dep-info: modules, `include!`d and
generated files) and `externs` the dependency artifacts it was given, all hashed when the
compilation finished. `verify` then starts from an executable, finds the record whose
output has the executable's bytes, and walks `externs`: the executable is `verified` when
every unit it was linked from still has the outputs its record names, was compiled against
the dependency bytes present now, and was compiled from sources with the same bytes as the
files on the host now. It is `contradicted` when a source or dependency differs (a stale
unit was reused), and `unverified` when a unit compiling synchronized sources has no record
(it was not built through the rig). Units with no record whose sources all lie outside the
synchronized roots are registry dependencies, pinned by `Cargo.lock`.

No job runs an executable of the checkout's target that the proof does not tie to the sources:
`verify` refuses the run when one its texts name is not verified, and withdraws every other
executable of the target's profile directories that is not (its file, and the outputs of the
unit that produced it, which cargo then rebuilds when asked), so a job that reaches one only at
run time (a script, an executable another starts) finds none rather than a stale one.

Outside the proof: what a build script produced without rustc reading it (native objects
it compiled and linked); whether cargo re-ran a build script still rests on file times.

This file is also run directly as the wrapper, so it imports nothing from the package
(standard library, Python 3.9+):

    provenance.py rustc RUSTC ARGS...
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

UNIT_SCHEMA = "build-provenance-unit/1"
SCHEMA = "build-provenance/1"
RECORD_SUFFIX = ".provenance.json"
_UNESCAPED_SPACE = re.compile(r"(?<!\\) ")


def digest(path: str):
    """SHA-256 of a regular file's bytes; None when there is no such file."""
    try:
        with open(path, "rb") as stream:
            material = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                material.update(block)
    except OSError:
        return None
    return material.hexdigest()


def _write(path: str, value: dict) -> None:
    temporary = "%s.%d.tmp" % (path, os.getpid())
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2)
    os.replace(temporary, path)


def dep_info(path: str):
    """A rustc dep-info file as (outputs, sources): the rule targets that have prerequisites,
    and the union of those prerequisites, in file order."""
    outputs, sources = [], {}
    with open(path) as stream:
        for line in stream:
            line = line.rstrip("\n")
            target, separator, listed = line.partition(": ")
            if not separator or line.startswith("#"):
                continue
            outputs.append(target.replace("\\ ", " "))
            for source in _UNESCAPED_SPACE.split(listed):
                if source:
                    sources[source.replace("\\ ", " ")] = None
    return outputs, list(sources)


class Unit:
    """One rustc invocation that compiles a crate into an output directory."""

    def __init__(self, crate: str, out_dir: str, extra: str, externs: list):
        self.crate, self.externs = crate, externs
        stem = os.path.join(out_dir, crate + extra)
        self.dep_info, self.record = stem + ".d", stem + RECORD_SUFFIX

    @classmethod
    def parse(cls, arguments: list):
        """The unit `arguments` compile, or None for rustc's probes and queries."""
        crate = out_dir = None
        extra, externs, emits = "", [], False
        following = iter(arguments[1:] + [""])
        for argument, value in zip(arguments, following):
            if argument == "--crate-name":
                crate = value
            elif argument == "--out-dir":
                out_dir = value
            elif argument == "--extern" and "=" in value:
                externs.append(os.path.abspath(value.split("=", 1)[1]))
            elif argument == "-C" and value.startswith("extra-filename="):
                extra = value.split("=", 1)[1]
            elif argument.startswith("-Cextra-filename="):
                extra = argument.split("=", 1)[1]
            elif argument.startswith("--emit"):
                emits = True
        if crate is None or out_dir is None or not emits:
            return None
        return cls(crate, os.path.abspath(out_dir), extra, externs)

    def write_record(self) -> None:
        outputs, sources = dep_info(self.dep_info)
        own = os.path.abspath(self.dep_info)
        _write(
            self.record,
            {
                "schema": UNIT_SCHEMA,
                "crate": self.crate,
                "sources": {p: digest(p) for p in map(os.path.abspath, sources)},
                "outputs": {p: digest(p) for p in map(os.path.abspath, outputs) if p != own},
                "externs": {path: digest(path) for path in self.externs},
            },
        )


def wrap(argv: list) -> int:
    """Run rustc; a successful compilation leaves its record, and none survives a failed one."""
    unit = Unit.parse(argv[1:])
    if unit is not None:
        try:
            os.remove(unit.record)
        except FileNotFoundError:
            pass
    code = subprocess.call(argv)
    if unit is not None and code == 0 and os.path.exists(unit.dep_info):
        unit.write_record()
    return code


def install(root: str) -> str:
    """Write the wrapper shim at a fixed path under `root` (a checkout's directory on the host)
    and return that path. The path is constant per checkout because cargo keys workspace
    artifacts by it."""
    directory = os.path.join(root, "provenance")
    os.makedirs(directory, exist_ok=True)
    script = os.path.abspath(__file__)
    if os.name == "nt":
        path = os.path.join(directory, "rustc.cmd")
        text = '@"%s" "%s" rustc %%*\r\n' % (sys.executable, script)
    else:
        path = os.path.join(directory, "rustc")
        text = '#!/bin/sh\nexec %s %s rustc "$@"\n' % (shlex.quote(sys.executable), shlex.quote(script))
    temporary = "%s.%d.tmp" % (path, os.getpid())
    with open(temporary, "w") as stream:
        stream.write(text)
    os.chmod(temporary, 0o755)
    os.replace(temporary, path)
    return path


def _within(path: str, roots: list) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def _records(directory: str) -> list:
    found = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(RECORD_SUFFIX):
            path = os.path.join(directory, name)
            with open(path) as stream:
                found.append((path, json.load(stream)))
    return found


def _extern_record(extern: str):
    """Where the record of the unit that produced `extern` is, if that unit left one."""
    directory, name = os.path.split(extern)
    stem = os.path.splitext(name)[0]
    stems = [stem[3:], stem] if stem.startswith("lib") else [stem]
    for candidate in stems:
        record = os.path.join(directory, candidate + RECORD_SUFFIX)
        if os.path.exists(record):
            return record, None
    for candidate in stems:
        dependency = os.path.join(directory, candidate + ".d")
        if os.path.exists(dependency):
            return None, dependency
    return None, None


def _words(text: str) -> list:
    try:
        return shlex.split(text)
    except ValueError:  # not a shell command line
        return []


def _built(path: str) -> bool:
    """Whether `path` is a cargo-built executable: an executable file of a profile directory
    (one with `deps/` beside it)."""
    return (os.path.isfile(path) and os.access(path, os.X_OK)
            and os.path.isdir(os.path.join(os.path.dirname(path), "deps")))


def built_executables(texts: list) -> list:
    """The cargo-built executables `texts` name, given as a whole text, as a word of a shell
    command line, or after the last `=` of either."""
    found = {}
    for text in texts:
        for candidate in {part.rsplit("=", 1)[-1] for part in (text, *_words(text))}:
            if os.path.isabs(candidate) and _built(os.path.realpath(candidate)):
                found[os.path.realpath(candidate)] = None
    return list(found)


def target_executables(target: str) -> list:
    """Every cargo-built executable of `target`'s profile directories: `<target>/<profile>/`
    and, for a named target triple, `<target>/<triple>/<profile>/`."""
    found = []
    for top in sorted(_subdirectories(target)):
        for directory in (top, *sorted(_subdirectories(top))):
            if os.path.isdir(os.path.join(directory, "deps")):
                found += [path for path in (os.path.realpath(os.path.join(directory, name))
                                            for name in sorted(os.listdir(directory))) if _built(path)]
    return found


def _subdirectories(directory: str) -> list:
    names = os.listdir(directory) if os.path.isdir(directory) else []
    return [os.path.join(directory, name) for name in names if os.path.isdir(os.path.join(directory, name))]


def verify_executable(path: str, roots: list, digest=digest, records=_records) -> dict:
    sha256 = digest(path)
    deps = os.path.join(os.path.dirname(path), "deps")
    root_unit = next(
        (record for record, unit in records(deps) if sha256 in unit["outputs"].values()), None
    )
    result = {"path": path, "sha256": sha256, "unit": root_unit, "units": 0, "sources": 0, "external_units": 0}
    if root_unit is None:
        reason = "no compilation record has this executable's bytes: it was not built by the rig"
        return {**result, "status": "unverified", "reasons": [reason]}
    contradicted, unverified = [], []
    compiled = {}
    pending, seen = [root_unit], {root_unit}
    while pending:
        record = pending.pop()
        with open(record) as stream:
            unit = json.load(stream)
        result["units"] += 1
        for output, recorded in unit["outputs"].items():
            if digest(output) != recorded:
                unverified.append(f"{unit['crate']}: {output} is not the output its record names")
        for source, recorded in unit["sources"].items():
            compiled[source] = recorded
            if digest(source) != recorded:
                contradicted.append(f"{unit['crate']}: compiled from other content of {source}")
        for extern, recorded in unit["externs"].items():
            if digest(extern) != recorded:
                contradicted.append(f"{unit['crate']}: compiled against other bytes of {extern}")
            dependency, info = _extern_record(extern)
            if dependency is not None:
                if dependency not in seen:
                    seen.add(dependency)
                    pending.append(dependency)
                continue
            sources = dep_info(info)[1] if info is not None else None
            if sources is None:
                unverified.append(f"{unit['crate']}: dependency {extern} has no record or dep-info")
            elif all(os.path.isabs(s) and not _within(os.path.realpath(s), roots) for s in sources):
                if extern not in seen:
                    seen.add(extern)
                    result["external_units"] += 1
            else:
                unverified.append(
                    f"{unit['crate']}: dependency {extern} compiles synchronized sources "
                    "but has no compilation record"
                )
    material = hashlib.sha256()
    for source in sorted(compiled):
        material.update(f"{source}\0{compiled[source]}\n".encode())
    result.update(sources=len(compiled), sources_digest=material.hexdigest())
    status = "contradicted" if contradicted else "unverified" if unverified else "verified"
    return {**result, "status": status, "reasons": sorted(set(contradicted + unverified))}


def verify(texts: list, roots: list, target: str) -> dict:
    """The provenance of every cargo-built executable `texts` name, and of every other one of
    `target` (`target_executables`), each of those not verified withdrawn. `status` is
    `verified`, `no-built-executable` (the texts name none), or `refused`."""
    roots = [os.path.realpath(root) for root in roots]
    known, listed = {}, {}

    def once(path: str):  # units share dependencies; nothing changes until the withdrawal
        if path not in known:
            known[path] = digest(path)
        return known[path]

    def records(deps: str) -> list:
        if deps not in listed:
            listed[deps] = _records(deps)
        return listed[deps]

    named = built_executables(texts)
    executables = [verify_executable(path, roots, once, records) for path in named]
    others = [verify_executable(path, roots, once, records) for path in target_executables(target) if path not in named]
    withdrawn = [other for other in others if other["status"] != "verified"]
    for executable in withdrawn:
        _withdraw(executable)
    if not executables:
        status = "no-built-executable"
    elif all(e["status"] == "verified" for e in executables):
        status = "verified"
    else:
        status = "refused"
    return {"schema": SCHEMA, "status": status, "roots": roots, "executables": executables,
            "withdrawn": withdrawn}


def _withdraw(executable: dict) -> None:
    """Remove an executable no job may run: its file, and the outputs and record of the unit
    that produced it, so cargo builds that unit again when asked."""
    paths = [executable["path"]]
    if executable["unit"] is not None:
        with open(executable["unit"]) as stream:
            paths += [*json.load(stream)["outputs"], executable["unit"]]
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def main(argv: list) -> int:
    if argv[:1] != ["rustc"] or len(argv) < 2:
        sys.stderr.write("usage: provenance.py rustc RUSTC ARGS...\n")
        return 2
    return wrap(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
