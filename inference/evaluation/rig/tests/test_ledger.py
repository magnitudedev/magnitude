"""A cargo test run answers from what each test executed and enters it in the ledger; the
ledger finds what is stale at a tree without building it."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from magnitude_rig import cli, engine, ledger, store
from magnitude_rig.host.provenance import tree_digest
from magnitude_rig.instruments import REGISTRY, JobResult

FACTS = {"alias": "m4", "os": "Darwin", "architecture": "arm64", "chip": "Apple M4 Pro",
         "gpus": [{"name": "Apple M4 Pro", "cores": 20, "memory_bytes": 1}]}
GPU, CPU = "Darwin Apple M4 Pro", "Darwin arm64 Apple M4 Pro"
RAN = {"asked": ["metal"], "opened": ["metal"], "ran": ["metal"], "none": []}
OUTSIDE = {"asked": ["metal"], "opened": [], "ran": [], "none": ["metal"]}
SKIPPED = {"asked": ["metal"], "opened": ["metal"], "ran": [], "none": []}
ELSEWHERE = {"asked": ["metal"], "opened": ["cpu"], "ran": ["cpu"], "none": []}  # declared nothing of metal
ASKED = {"asked": ["metal"], "opened": [], "ran": [], "none": []}


def parse(*argv: str) -> argparse.Namespace:
    return cli.parser().parse_args(list(argv))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "results", lambda: tmp_path / "results")
    return tmp_path / "results"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run(results: Path, name: str, ran: list, tree: dict, doc: str | None = None,
        environment: dict | None = None) -> dict[str, JobResult]:
    """A fetched test run `name` of the sources `tree` ({path: text}) whose job ran `ran`, each
    executable built reading `environment`, and whose documentation tests printed `doc`."""
    job = results / name / "test"
    job.mkdir(parents=True)
    files = {path: _sha(text) for path, text in tree.items()}
    executables = [{"path": f"/host/target/release/deps/{record['executable']}", "sources_digest": tree_digest(files),
                    "linked": ["unit"], "environment": environment or {}} for record in ran]
    (results / name / engine.PROVENANCE.format(side="source")).write_text(
        json.dumps({"roots": ["/host/src"], "executables": executables, "unit_sources": {"unit": files}}))
    built = [[f"/host/target/release/deps/{record['executable']}", f"/host/src/{record['binary'].split(':')[0]}/Cargo.toml",
              record["binary"].rsplit(":", 1)[1], record["binary"].split(":")[1]] for record in ran]
    (results / name / engine.TESTS.format(side="source")).write_text(json.dumps({"tests": built, "build_scripts": {}}))
    (job / ledger.RAN).write_text(json.dumps(ran))
    (job / "host.json").write_text(json.dumps(FACTS))
    log = job / "job.log"
    log.write_text("")
    found = {"test": JobResult(job, log, 0)}
    if doc is not None:
        (results / name / "doc").mkdir()
        (results / name / "doc.log").write_text(doc)
        found["doc"] = JobResult(results / name / "doc", results / name / "doc.log", 0)
    return found


def executable(binary: str, outcomes: dict, executions: dict, listed=None, ignored=()) -> dict:
    counts = [sum(1 for o in outcomes.values() if o == kind) for kind in ("ok", "FAILED", "ignored")]
    return {"binary": binary, "executable": binary.rsplit(":", 1)[1] + "-0123", "code": 0, "outcomes": outcomes,
            "counts": counts, "listed": list(outcomes) if listed is None else listed, "ignored": list(ignored),
            "executions": executions}


def answer(found: dict[str, JobResult], *given: str, backend: str = "metal"):
    arguments = parse("test", "--backend", backend, "--purpose", "p", "--estimate", "1m", "--", *given)
    return REGISTRY["test"].answer(arguments, found)


def test_a_test_that_does_no_work_on_the_backend_and_declares_it_not_outside_is_not_run(results):
    """A test that opened the device and did no work there, one that worked only on another
    backend, and one that asked for the backend and did nothing, each skipped it: one makes the
    run NOT RUN however many others ran. A test outside the backend (it declared it selected
    none of it) and host tests are reported and fail nothing; the ledger records each where it
    executed, and a run answered again adds no rows."""
    kernels = executable("k:test:routed", {"a": "ok", "b": "ok", "c": "ok", "e": "ok", "q": "ok", "h": "ok"},
                         {"a": RAN, "b": OUTSIDE, "c": SKIPPED, "e": ELSEWHERE, "q": ASKED})
    found = run(results, "r1", [kernels], {"k/src/lib.rs": "one"})
    judged = answer(found, "-p", "k", "--test", "routed")
    assert not judged.ok and judged.lines[0] == "NOT RUN 1 ran, 1 host, 1 outside, 3 skipped (metal)"
    assert ("NOT RUN on metal: chose its devices, did no work there and declared it outside none: "
            "k:test:routed c, k:test:routed e, k:test:routed q") in judged.lines
    (row,) = ledger.read()
    assert row["results"] == {"metal": {"passed": ["a"], "outside": ["b"], "skipped": ["c", "e", "q"]},
                              "host": {"passed": ["h"]}}
    assert row["hardware"] == {"metal": GPU, "host": CPU} and row["configuration"] == "test"
    answer(found, "-p", "k", "--test", "routed")
    assert len(ledger.read()) == 1


def test_a_build_that_read_environment_keys_its_rows_apart(results, tmp_path):
    """An executable whose build read RUSTFLAGS (or a variable its build script reruns for) is
    keyed with it, so its passes credit no rig build, which reads none."""
    record = executable("k:test:routed", {"a": "ok"}, {"a": RAN})
    answer(run(results, "r1", [record], {"k/src/lib.rs": "one"}, environment={"RUSTFLAGS": "-Ctarget-cpu=native"}),
           "-p", "k", "--release")
    (row,) = ledger.read()
    assert row["configuration"] == "release RUSTFLAGS=-Ctarget-cpu=native"
    tree = tmp_path / "tree"
    (tree / "k" / "src").mkdir(parents=True)
    (tree / "k" / "src" / "lib.rs").write_text("one")
    universe = {"k:test:routed": ledger.Binary("k:test:routed", "k", "test", "routed")}
    assert ledger.Ledger().stale(tree, "metal", GPU, "release", universe)[0].why == "never run"


def test_a_run_passes_on_what_ran_there_whatever_its_filter_missed_elsewhere(results):
    """An executable the filter matched nothing in, or whose tests are all ignored, fails
    nothing; a run on a GPU backend of host tests alone ran nothing there; on the CPU, host
    tests and documentation tests are what runs."""
    ran = executable("k:test:routed", {"a": "ok"}, {"a": RAN})
    missed = executable("k:test:gguf", {}, {}, listed=["gguf::iq1m::a"])
    ignored = executable("k:test:timed", {"t": "ignored"}, {}, ignored=["t"])
    assert answer(run(results, "r1", [ran, missed, ignored], {"k/src/lib.rs": "one"}), "-p", "k").ok
    host = executable("k:lib:k", {"h": "ok"}, {})
    assert answer(run(results, "r2", [host], {"k/src/lib.rs": "one"}), "-p", "k").lines[0] == \
        "NOT RUN 1 host (metal)"
    doc = "   Doc-tests k\nrunning 1 test\ntest src/lib.rs - f (line 3) ... ok\n" \
          "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n"
    on_cpu = answer(run(results, "r3", [host], {"k/src/lib.rs": "one"}, doc), "-p", "k", backend="cpu")
    assert on_cpu.ok and on_cpu.lines[0] == "PASS 2 host (cpu)"
    doc_row = ledger.read()[-1]
    assert doc_row["binary"] == "k:doc:k" and doc_row["results"] == {"host": {"passed": ["src/lib.rs - f (line 3)"]}}
    assert doc_row["executable"] == "k-0123" and doc_row["digest"] == ledger.read()[-2]["digest"]


def test_device_use_no_test_made_leaves_the_silent_tests_unknown_and_the_run_not_run(results):
    record = executable("k:test:engine", {"a": "ok", "q": "ok"}, {"a": RAN, "": SKIPPED})
    judged = answer(run(results, "r1", [record], {"k/src/lib.rs": "one"}), "-p", "k")
    assert not judged.ok and judged.lines[0] == "NOT RUN 1 ran, 1 unknown (metal)"
    assert "NOT RUN on metal: unknown whether it used a device: k:test:engine q" in judged.lines
    assert ledger.read()[0]["results"] == {"metal": {"passed": ["a"]}}


def test_outcomes_libtest_did_not_count_are_unreadable(results):
    record = executable("k:test:routed", {"a": "ok"}, {"a": RAN})
    record["counts"] = [2, 0, 0]  # a test's line broken by output it printed
    judged = answer(run(results, "r1", [record], {"k/src/lib.rs": "one"}), "-p", "k")
    assert not judged.ok and any(line.startswith("UNREADABLE: k:test:routed") for line in judged.lines)


def test_a_hosts_paths_are_read_as_written_never_resolved_here(tmp_path):
    """A host's test manifests and source roots are the host's, resolved there: where this machine
    resolves the same path elsewhere (macOS's `/home` link), they still key the executable, and a
    manifest outside every root is an invariant violation naming both."""
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "home").symlink_to(tmp_path / "elsewhere")
    root = str(tmp_path / "home" / "src")
    directory = tmp_path / "r1"
    directory.mkdir()

    def built(manifest: str) -> dict:
        (directory / engine.TESTS.format(side="source")).write_text(json.dumps(
            {"tests": [["/host/target/debug/deps/k-0123", manifest, "k", "lib"]], "build_scripts": {}}))
        return ledger.built(directory, {"roots": [root]})

    assert built(f"{root}/service/k/Cargo.toml") == {ledger.binary_key("service/k", "lib", "k"): "k-0123"}
    with pytest.raises(AssertionError, match=r"/elsewhere/k/Cargo\.toml .*outside every source root"):
        built(str(tmp_path / "elsewhere" / "k" / "Cargo.toml"))


def test_stale_is_what_no_row_at_the_trees_sources_credits(results, tmp_path):
    """At the sources of a row, a test no row there credits (passed or outside the backend, on its
    hardware, in the configuration; a host test passed anywhere) is stale, ignored ones aside; one
    that skipped stays stale; an executable whose sources changed since every row, or that no run
    built, is stale whole; on hardware no row records, every device test is. A library's
    documentation tests are stale by its sources."""
    tree = tmp_path / "tree"
    (tree / "k" / "src").mkdir(parents=True)
    (tree / "k" / "src" / "lib.rs").write_text("one")
    routed = executable("k:test:routed", {"a": "ok", "b": "ok", "c": "ok", "h": "ok", "i": "ignored"},
                        {"a": RAN, "b": OUTSIDE, "c": SKIPPED}, listed=["a", "b", "c", "d", "h", "i"], ignored=["i"])
    library = executable("k:lib:k", {"u": "ok"}, {})
    job = run(results, "r1", [routed, library], {"k/src/lib.rs": "one"})["test"].directory
    rows = ledger.rows(results / "r1", job, "metal", "release", {"k": {"src/lib.rs - f (line 3)": "ok"}})
    universe = {key: ledger.Binary(key, "k", kind, name)
                for key, kind, name in (("k:test:routed", "test", "routed"), ("k:lib:k", "lib", "k"),
                                        ("k:doc:k", "doc", "k"), ("j:doc:j", "doc", "j"))}
    found = ledger.Ledger(rows)
    stale = found.stale(tree, "metal", GPU, "release", universe)
    assert [(item.binary.key, item.tests, item.why) for item in stale] == [
        ("j:doc:j", None, "never run"), ("k:test:routed", ("c", "d"), "2 of 5 never passed at these sources")]
    assert universe["k:doc:k"].selection() == ("--lib",)  # what keys its row is built with it
    assert found.stale(tree, "cuda", None, "release", universe)[1].tests == ("a", "b", "c", "d")  # h: a host test
    assert found.stale(tree, "metal", GPU, "test", universe)[1].why == "never run"  # another configuration
    assert found.hardware_of("m4", "metal") == GPU and found.hardware_of("m4", "cuda") is None
    (tree / "k" / "src" / "lib.rs").write_text("two")
    assert {item.binary.key: (item.tests, item.why) for item in ledger.Ledger(rows).stale(
        tree, "metal", GPU, "release", universe)}["k:doc:k"] == (None, "changed since its last run, r1")
