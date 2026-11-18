"""The ledger: what every cargo test run executed, and from it what is stale. The one owner of
test results across runs, and of how a test's execution is judged.

A test run's job records what each test executable ran (`host tests`, its `ran.json`): each
test's outcome as libtest printed it and, from the test's record
(`seismic_lang::registry::TestDevice`), the backends it asked for, those whose devices it
opened, those it did work on (submitted work or allocated memory) and those it selected none
of. Each test is judged on the run's backend (`judge`): it ran there (it worked there); it is
outside the backend (it declared it selected none of it); it was skipped there (it chose its
devices and did neither: never a pass); or it is a host test, which records nothing because it
chooses no device. Device use the record cannot tie to a test of the executable (on a thread
no test runs on) makes its silent tests unknown, never host tests, and a run with an unknown
test is not run. Documentation tests, run by cargo, are host tests.

When the run is answered, the ledger gains one row per test executable (`rows`, appended once
per run): its key across builds (`host.agent.binary_key`), the run, its host, the key of what
it was built in (`configured`: the configuration cargo's arguments name,
`instruments.cargo.configuration`, and the environment its build read), the digest of the
sources it was built from (its build provenance's `sources_digest`), the tests it has and
ignores, and on the run's backend the hardware class it ran on and its tests by judgement
there; host tests under `HOST`. A package's documentation tests are a row of their own, keyed
by its library's sources.

A test is stale on a backend and hardware class in a configuration (`Ledger.stale`) unless a
row at the sources it would be built from now records it passed there or outside it, or, for
a host test, passed anywhere. A tree's sources are digested from the paths a run's provenance
recorded for the executable (`unit_sources`), read from the tree, so nothing is built: an
executable whose sources changed since every row of it is stale whole, one no run built is
stale whole, and otherwise each test of it no row credits is. That stale set is what a change
reaches; `mrig test --stale` runs it.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import store
from .engine import PROVENANCE, TESTS
from .host.agent import binary_key, target_kind
from .host.provenance import path_digest, tree_digest

LEDGER = "ledger.jsonl"  # in the results store
MARK = "ledgered"  # in a run's results: its rows are in the ledger
RAN = "ran.json"  # in a test job's results: what each executable ran (`host tests`)
CPU, HOST = "cpu", "host"  # a host test runs on the CPU, and is judged under `HOST`
# A test's judgement on a backend (`judge`); the ledger keeps all but `unknown`.
RAN_THERE, SKIPPED, OUTSIDE, HOST_TEST, UNKNOWN = "ran", "skipped", "outside", "host", "unknown"
CREDITED = ("passed", "outside")  # what makes a test current on a backend
DOC = "doc"  # the kind of a package's documentation tests' row (`Binary.kind`)


def path() -> Path:
    return store.results() / LEDGER


def hardware(facts: dict, backend: str) -> str:
    """The hardware class a backend's tests ran on: the CPU's chip for the CPU and for host
    tests, the GPUs' names for a GPU backend, each with the OS."""
    if backend in (CPU, HOST):
        return f"{facts['os']} {facts['architecture']} {facts.get('chip') or 'unknown CPU'}"
    names = sorted({gpu["name"] for gpu in facts.get("gpus") or []})
    return f"{facts['os']} {', '.join(names) or 'unknown GPU'}"


def unattributed(record: dict) -> list[str]:
    """The record's device use no test of its executable made: the threads it was on."""
    return sorted(name for name in record["executions"] if name not in record["outcomes"])


def judge(record: dict, name: str, backend: str) -> str:
    """How test `name` of executable record `record` executed on `backend` (see the module)."""
    execution = record["executions"].get(name)
    if execution is None:
        return UNKNOWN if unattributed(record) else HOST_TEST
    if backend in execution["ran"]:
        return RAN_THERE
    if backend in execution["none"] and backend not in execution["opened"]:
        return OUTSIDE
    return SKIPPED  # it chose its devices, and did no work on `backend` nor declared it outside


def results(record: dict, backend: str) -> dict:
    """The record's tests by place and kind (`passed`, `failed`, `outside`, `skipped`), as a
    run on `backend` executed them: under `backend`, a host test under `HOST`; an ignored test,
    and one whose execution is unknown, under none."""
    found: dict = {}
    for name, outcome in record["outcomes"].items():
        judged = judge(record, name, backend)
        if outcome == "ignored" or judged == UNKNOWN:
            continue
        place = HOST if judged == HOST_TEST else backend
        # A failure is one whatever else it is; a pass is credited only where it ran.
        kind = "failed" if outcome == "FAILED" else "passed" if judged in (RAN_THERE, HOST_TEST) else judged
        found.setdefault(place, {}).setdefault(kind, []).append(name)
    return found


def configured(configuration: str, environment: dict) -> str:
    """The key of what an executable was built in: the configuration cargo's arguments name
    (`instruments.cargo.configuration`) and each variable of the environment its build read
    (provenance's `environment`), with its value. A rig build sets none of them, so the key
    of a build that read one set (on a host whose own environment sets it) credits no other."""
    return " ".join([configuration, *(f"{name}={environment[name]}" for name in sorted(environment))])


def rows(results_dir: Path, job: Path, backend: str, configuration: str, doc: dict | None = None) -> list[dict]:
    """The ledger rows of a test run (its results directory, its job's) on `backend`, built in
    `configuration`; `doc`, its documentation tests' outcomes by crate, each the row of its
    package's library's sources, where the run built them."""
    ran = json.loads((job / RAN).read_text())
    facts = json.loads((job / "host.json").read_text())
    proof = json.loads((results_dir / PROVENANCE.format(side="source")).read_text())
    proven = {os.path.basename(executable["path"]): executable for executable in proof["executables"]}
    found = []

    def row(binary: str, executable: str, listed: list, ignored: list, by_place: dict) -> dict:
        return {"run": results_dir.name, "binary": binary, "executable": executable, "host": facts["alias"],
                "configuration": configured(configuration, proven[executable]["environment"]),
                "digest": proven[executable]["sources_digest"], "listed": listed, "ignored": ignored,
                "hardware": {place: hardware(facts, place) for place in by_place}, "results": by_place}

    for record in ran:
        found.append(row(record["binary"], record["executable"], record["listed"], record["ignored"],
                         results(record, backend)))
    libraries = {}  # each library's unit tests the run built, by crate: package directory, executable
    for key, executable in (built(results_dir, proof) if doc else {}).items():
        package, kind, crate = key.rsplit(":", 2)
        if kind == "lib":
            libraries[crate] = (package, executable)
    for crate, outcomes in (doc or {}).items():
        if crate not in libraries:
            continue  # a library whose unit tests the run did not build: no sources to key them by
        package, executable = libraries[crate]
        by_kind: dict = {}
        for name, outcome in outcomes.items():
            if outcome != "ignored":
                by_kind.setdefault("failed" if outcome == "FAILED" else "passed", []).append(name)
        found.append(row(binary_key(package, DOC, crate), executable, list(outcomes),
                         [name for name, outcome in outcomes.items() if outcome == "ignored"],
                         {HOST: by_kind} if by_kind else {}))
    return found


def built(results_dir: Path, proof: dict) -> dict[str, str]:
    """The test executables a run's build produced (`TESTS`, as the agent wrote them), by key:
    their file names."""
    tests = json.loads((results_dir / TESTS.format(side="source")).read_text())["tests"]
    found = {}
    for path, manifest, name, kind in tests:
        directory = os.path.realpath(os.path.dirname(manifest))
        root = next(root for root in proof["roots"] if directory == root or directory.startswith(root + os.sep))
        found[binary_key(os.path.relpath(directory, root), kind, name)] = os.path.basename(path)
    return found


def record(results_dir: Path, found: list[dict]) -> None:
    """Append a run's rows, once: a run answered again (`mrig fetch`) adds none."""
    if (results_dir / MARK).exists():
        return
    with open(path(), "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)  # the runs of every checkout append to one ledger
        stream.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in found))
    (results_dir / MARK).write_text("")


def read() -> list[dict]:
    """Every row, oldest first."""
    if not path().exists():
        return []
    with open(path()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


@dataclass(frozen=True)
class Binary:
    """A test executable of a tree, as cargo builds it."""

    key: str
    package: str  # the cargo package name
    kind: str  # `lib`, one of `host.agent.NAMED_KINDS`, or `DOC`: the documentation tests of a library
    name: str  # the target name

    def selection(self) -> tuple[str, ...]:
        """Cargo test's target selection of what it is built from: a library's for its
        documentation tests, whose row its unit tests' executable keys (`rows`)."""
        return ("--lib",) if self.kind in ("lib", DOC) else (f"--{self.kind}", self.name)


def binaries(tree: Path) -> dict[str, Binary]:
    """The test executables of the workspace at `tree` (its `inference/`), by key: every target
    `cargo test` builds tests of, and the documentation tests of each library that has them."""
    listed = subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline",
                             "--manifest-path", str(tree / "Cargo.toml")], capture_output=True, text=True)
    if listed.returncode != 0:
        raise SystemExit(f"mrig: cargo metadata of {tree}: {listed.stderr.strip()}")
    found = {}
    for package in json.loads(listed.stdout)["packages"]:
        directory = os.path.relpath(os.path.dirname(package["manifest_path"]), tree)
        for target in package["targets"]:
            kind = target_kind(target["kind"])
            kinds = ([kind] if target.get("test") else []) + ([DOC] if kind == "lib" and target.get("doctest") else [])
            for each in kinds:
                key = binary_key(directory, each, target["name"])
                found[key] = Binary(key, package["name"], each, target["name"])
    return found


@dataclass(frozen=True)
class Stale:
    binary: Binary
    tests: tuple[str, ...] | None  # None: the whole executable
    why: str


class Ledger:
    """The rows, read once, judged against a tree."""

    def __init__(self, found: list[dict] | None = None):
        self.rows = read() if found is None else found
        self._proofs: dict[str, dict | None] = {}
        self._paths: dict[tuple[Path, str], str | None] = {}

    def hardware_of(self, host: str, backend: str) -> str | None:
        """The hardware class `host` last ran `backend` tests on, or None when it has not."""
        return next((row["hardware"][backend] for row in reversed(self.rows)
                     if row["host"] == host and backend in row["hardware"]), None)

    def classes(self, backend: str) -> dict[str, list[str]]:
        """Every hardware class `backend` tests ran on, with the hosts of each."""
        found: dict[str, list[str]] = {}
        for row in self.rows:
            if backend in row["hardware"]:
                hosts = found.setdefault(row["hardware"][backend], [])
                if row["host"] not in hosts:
                    hosts.append(row["host"])
        return found

    def _proof(self, run: str) -> dict | None:
        if run not in self._proofs:
            proof = store.results() / run / PROVENANCE.format(side="source")
            self._proofs[run] = json.loads(proof.read_text()) if proof.exists() else None
        return self._proofs[run]

    def _path(self, tree: Path, relative: str) -> str | None:
        if (tree, relative) not in self._paths:
            self._paths[tree, relative] = path_digest(str(tree / relative))
        return self._paths[tree, relative]

    def _current(self, tree: Path, row: dict) -> bool:
        """Whether `tree` holds the sources `row`'s executable was built from."""
        proof = self._proof(row["run"])
        if proof is None:
            return False
        executable = next(e for e in proof["executables"] if os.path.basename(e["path"]) == row["executable"])
        paths = {relative for unit in executable["linked"] for relative in proof["unit_sources"][unit]}
        return tree_digest({relative: self._path(tree, relative) for relative in paths}) == row["digest"]

    def stale(self, tree: Path, backend: str, hardware: str | None, configuration: str,
              universe: dict[str, Binary]) -> list[Stale]:
        """The stale tests of `universe` at `tree` on `backend` and `hardware` (None: a class no
        run recorded, where nothing has passed), built in `configuration` (a `configured` key:
        cargo's arguments' configuration, its build reading no environment, as the rig's)."""
        by_binary: dict[str, list[dict]] = {}
        for row in self.rows:
            if row["configuration"] == configuration:
                by_binary.setdefault(row["binary"], []).append(row)
        found = []
        for key, binary in sorted(universe.items()):
            history = by_binary.get(key)
            if not history:
                found.append(Stale(binary, None, "never run"))
                continue
            judged: dict[str, bool] = {}  # the rows of one digest are of one tree
            for row in history:
                if row["digest"] not in judged:
                    judged[row["digest"]] = self._current(tree, row)
            current = [row for row in history if judged[row["digest"]]]
            if not current:
                found.append(Stale(binary, None, f"changed since its last run, {history[-1]['run']}"))
                continue
            latest = current[-1]
            tests = [name for name in latest["listed"] if name not in set(latest["ignored"])]
            host = {name for row in current for name in row["results"].get(HOST, {}).get("passed", [])}
            there = {name for row in current if hardware is not None and row["hardware"].get(backend) == hardware
                     for kind in CREDITED for name in row["results"][backend].get(kind, [])}
            missing = tuple(name for name in tests if name not in host | there)
            if missing:
                found.append(Stale(binary, None if len(missing) == len(tests) else missing,
                                   f"{len(missing)} of {len(tests)} never passed at these sources"))
        return found


def summary(stale: list[Stale]) -> str:
    whole = sum(1 for item in stale if item.tests is None)
    named = sum(len(item.tests) for item in stale if item.tests is not None)
    return f"{len(stale)} executables stale ({whole} whole, {named} tests of the others)"
