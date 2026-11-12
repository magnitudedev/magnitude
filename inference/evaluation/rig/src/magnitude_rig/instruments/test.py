"""`test`: is it correct on this backend? A test suite, built under the build hold and run on
the device: cargo tests (`--suite cargo`, the default), or the evaluation tools' pytest suites
(`--suite python`, run in the side's environment with every package and extra). A cargo suite
that names its targets, none of them `--doc`, runs its test executables from the side's build
output (`host/agent.py`), each from its package's directory there with `CARGO_MANIFEST_DIR`
naming it, as cargo runs them; one that runs documentation tests, which compile from the source
when they run, runs as `cargo test` in the side's checkout, which it keeps until its job ends
(`Step.checkout`). With
`--model`, the catalog model is acquired first and its acquisition object is
`MAGNITUDE_ARTIFACTS` for the tests that check against a real model's files; otherwise a cargo
suite has the catalog's header export as `{catalog}`, its headers `MAGNITUDE_CATALOG_HEADERS`
(as the family and assessment tests read the release catalog), so a test of the suite never
lacks its inputs; a host keeps the export of each catalog (`engine.catalog_key`), so only a
catalog's first run there builds the catalog tool and exports it. The suite's arguments take
the run's tokens (`{catalog}`, `{models}`, `{root}`, and `{checkout}` for a suite run in it).
A cuda, metal or vulkan run names its backend to the tests (`SEISMIC_TEST_BACKEND`): they run on
it alone, and fail where the host has no such device, never pass by skipping it. `--timed` runs
tests that time the device (an ignored measurement test, a cost figure) under the timed hold, as
every timing instrument runs."""

import argparse
import re

from .. import selectors
from ..engine import Headers, Step, Weights
from . import Answer, Instrument, JobResult, Plan, cargo, environment

CARGO_RESULT = re.compile(r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored")
CARGO_FAILED = re.compile(r"^---- (\S+) stdout ----$")
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?)")
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)")
PYTEST = "{venv}/bin/pytest"
# The test executables a cargo suite's build produced, in the side's output (`host/agent.py`).
TESTS = "{output}/tests.json"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=("cargo", "python"), default="cargo",
                        help="cargo tests (default) or the evaluation tools' pytest suites")
    parser.add_argument("--timed", action="store_true",
                        help="the selected tests time the device: run them as a measurement, after the "
                             "thermal gate, with nothing else on the host")
    parser.add_argument("--model", type=selectors.model,
                        help="catalog model to acquire first, as MAGNITUDE_ARTIFACTS for the tests (else a "
                             "cargo suite has the catalog's headers)")
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="the suite's arguments after --: cargo test's, e.g. -- -p magnitude-kernels "
                             "--release --test routed -- general_routed; or pytest's, e.g. -- "
                             "evaluation/references/tests/test_released.py -k qwen")


def _given(arguments) -> list[str]:
    given = arguments.arguments[1:] if arguments.arguments[:1] == ["--"] else arguments.arguments
    if not given:
        raise SystemExit("test needs the suite's arguments after --, e.g. -- -p CRATE --test NAME -- FILTER")
    return given


def plan(arguments, source, host) -> Plan:
    # A device run names its backend, so a test fails where that backend is missing, never passes by skipping it.
    env = (("SEISMIC_TEST_BACKEND", arguments.backend),) if arguments.backend != "cpu" else ()
    if arguments.model is not None:
        env += (("MAGNITUDE_ARTIFACTS", "{artifacts}"),)
    # A cargo suite reads the release catalog's headers unless it reads one model's files.
    catalog = arguments.suite == "cargo" and arguments.model is None
    if catalog:
        env += (("MAGNITUDE_CATALOG_HEADERS", "{catalog}/headers"),)
    if arguments.suite == "cargo":
        own, binary = cargo.split(_given(arguments))
        builds = (Step("test build", ("cargo", "test", "--message-format=short", "--no-run", *own)),)
        if not cargo.runs_doc_tests(own):
            # Each executable gets what cargo would pass it: its test name filter, then what follows `--`.
            job = Step("test", ("{python}", "-m", "magnitude_rig.host", "tests", "--list", TESTS, "--",
                                *cargo.test_name(own), *binary), env)
        else:  # documentation tests compile from the source when they run: in the checkout, kept for them
            job = Step("test", ("cargo", "test", *own, "--", *binary), env, checkout=True)
    else:
        builds = (environment("environment", "--all-packages", "--all-extras"),)
        # Quiet comes from the project's pytest settings; a second -q would drop the summary line.
        job = Step("test", (PYTEST, "-rfE", *_given(arguments)), env)
    acquires = Weights(arguments.model) if arguments.model else Headers(None) if catalog else None
    device = "timed" if arguments.timed else None if arguments.backend == "cpu" else "device"
    return Plan(builds=builds, jobs=(("test", job),), device=device, acquires=acquires)


def _cargo(lines: list[str]) -> tuple[dict[str, int] | None, list[str]]:
    """The libtest counts, or None when no test binary printed them (a test without the libtest
    harness reports by its exit status alone)."""
    results = [match for line in lines if (match := CARGO_RESULT.match(line.strip()))]
    counts = {
        kind: sum(int(match[group]) for match in results)
        for kind, group in (("passed", 2), ("failed", 3), ("ignored", 4))
    } if results else None
    return counts, [match[1] for line in lines if (match := CARGO_FAILED.match(line.strip()))]


def _pytest(lines: list[str]) -> tuple[dict[str, int], list[str]]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    summary = next((line for line in reversed(lines) if " in " in line and PYTEST_COUNT.search(line)), "")
    for number, kind in PYTEST_COUNT.findall(summary):
        counts["failed" if kind.startswith("error") else kind] += int(number)
    return counts, [match[1] for line in lines if (match := PYTEST_FAILED.match(line.strip()))]


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["test"]
    lines = result.output()
    counts, names = (_cargo if arguments.suite == "cargo" else _pytest)(lines)
    panics = [line.strip() for line in lines if "panicked at" in line][:5]
    failed = counts["failed"] if counts else 0
    ok = result.code == 0 and failed == 0
    tally = ", ".join(f"{n} {kind}" for kind, n in counts.items()) if counts else f"exit {result.code}, no test counts"
    shown = [f"{'PASS' if ok else 'FAIL'} {tally} ({arguments.backend})"]
    if names:
        shown.append("failed: " + ", ".join(names[:20]))
    shown += [f"  {panic}" for panic in panics]
    if counts is None or (result.code != 0 and not failed):
        shown += [f"  {line}" for line in lines[-15:] if not line.startswith("[stderr]")]
    return Answer(tuple(shown), ok)


INSTRUMENT = Instrument(
    name="test",
    kind="correctness",
    question="Is it correct on this backend (a cargo or pytest selection)?",
    observable="test outcome",
    selectors=("backend", "host", "build", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
