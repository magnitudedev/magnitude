"""`test`: is it correct on this backend? A test suite, built under the build hold and run on
the device: cargo tests (`--suite cargo`, the default), or the evaluation tools' pytest suites
(`--suite python`, run in the checkout's environment with every package and extra). With
`--model`, the catalog model is acquired first and its acquisition object is
`MAGNITUDE_ARTIFACTS` for the tests that check against a real model's files; with `--catalog`,
the catalog's header export is `{catalog}` for the tests that read the release catalog. The
suite's arguments take the run's tokens (`{catalog}`, `{models}`, `{checkout}`, `{root}`).
`--backend vulkan` selects Vulkan where the kernel tests otherwise take CUDA or Metal
(`SEISMIC_TEST_BACKEND`, as the kernel tests read it)."""

import argparse
import re

from .. import selectors
from ..engine import Headers, Step, Weights
from . import Answer, Instrument, JobResult, Plan, cargo

CARGO_RESULT = re.compile(r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored")
CARGO_FAILED = re.compile(r"^---- (\S+) stdout ----$")
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?)")
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)")
PYTEST = "{venv}/bin/pytest"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=("cargo", "python"), default="cargo",
                        help="cargo tests (default) or the evaluation tools' pytest suites")
    acquired = parser.add_mutually_exclusive_group()
    acquired.add_argument("--model", type=selectors.model,
                          help="catalog model to acquire first, as MAGNITUDE_ARTIFACTS for the tests")
    acquired.add_argument("--catalog", action="store_true",
                          help="export the catalog's headers first, as {catalog} in the suite's arguments")
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="the suite's arguments after --: cargo test's, e.g. -- -p magnitude-kernels "
                             "--release --test routed -- general_routed; or pytest's, e.g. -- "
                             "evaluation/references/tests/test_released.py -k qwen")


def _given(arguments) -> list[str]:
    given = arguments.arguments[1:] if arguments.arguments[:1] == ["--"] else arguments.arguments
    if not given:
        raise SystemExit("test needs the suite's arguments after --, e.g. -- -p CRATE --test NAME -- FILTER")
    return given


def plan(arguments, source) -> Plan:
    env = (("SEISMIC_TEST_BACKEND", "vulkan"),) if arguments.backend == "vulkan" else ()
    if arguments.model is not None:
        env += (("MAGNITUDE_ARTIFACTS", "{artifacts}"),)
    if arguments.suite == "cargo":
        own, binary = cargo.split(_given(arguments))
        build = Step("test build", ("cargo", "test", "--message-format=short", "--no-run", *own))
        job = Step("test", ("cargo", "test", *own, "--", *binary), env)
    else:
        build = Step("environment", ("uv", "sync", "--frozen", "--project", "{src}/evaluation",
                                     "--all-packages", "--all-extras"))
        # Quiet comes from the project's pytest settings; a second -q would drop the summary line.
        job = Step("test", (PYTEST, "-rfE", *_given(arguments)), env)
    acquires = Weights(arguments.model) if arguments.model else Headers(None) if arguments.catalog else None
    return Plan(builds=(build,), jobs=(("test", job),), device=None if arguments.backend == "cpu" else "device",
                acquires=acquires)


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
