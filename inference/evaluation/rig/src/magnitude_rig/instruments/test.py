"""`test`: is it correct on this backend? A test suite, built under the build hold and run on
the device: cargo tests (`--suite cargo`, the default), or the evaluation tools' pytest suites
(`--suite python`, run in the side's environment with every package and extra). A cargo suite
is built without running (`cargo test --no-run`) and its test executables run from the side's
build output (`host/agent.py`), each from its package's directory there with
`CARGO_MANIFEST_DIR` naming it, as cargo runs them; its documentation tests, which compile
from the source when they run, run as `cargo test --doc` in the side's checkout, which it
keeps until its job ends (`Step.checkout`). With
`--model`, the catalog model is acquired first and its acquisition object is
`MAGNITUDE_ARTIFACTS` for the tests that check against a real model's files (every suite has the
host's model store as `MAGNITUDE_MODELS`, for the files a conversion leaves there); otherwise a cargo
suite whose tests declare they read catalog models (`reads_catalog`: as the family and
assessment tests read the release catalog) has the catalog's header export as `{catalog}`, its
headers `MAGNITUDE_CATALOG_HEADERS`, so a test of the suite never lacks its inputs, and no other
run resolves the catalog; a host keeps the export of each catalog (`engine.catalog_key`), so only a
catalog's first run there builds the catalog tool and exports it. The suite's arguments take
the run's tokens (`{catalog}`, `{models}`, `{root}`, and `{checkout}` for a suite run in it).
A run names its backend to the tests (`SEISMIC_TEST_BACKEND`), the CPU's too: they run on it
alone, and fail where the host has no such device, never pass by skipping it. `--timed` runs
tests that time the device (an ignored measurement test, a cost figure) under the timed hold, as
every timing instrument runs.

A cargo suite answers from what each of its tests executed (`ledger.judge`): a test that chose
its devices and neither worked on the backend nor declared it outside is NOT RUN, as is one
whose execution is unknown, and a run in which no test ran on the backend (host tests,
documentation tests among them, count on the CPU), never passed;
tests outside the backend's domain, and executables none of whose tests the filter matched, are
reported and fail nothing. Its results enter the ledger. `--stale` runs what the ledger finds
stale on the backend at the hardware of the host it is placed on, in the configuration the
arguments build: the executables, tests and documentation tests the sources reach, chosen for
you."""

import argparse
import dataclasses
import functools
import hashlib
import json
import re

from .. import ledger, selectors
from ..engine import Headers, Refused, Step, Weights
from ..host import libtest
from ..state import HOME
from . import Answer, Instrument, JobResult, Plan, cargo, environment

PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?)")
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)")
PYTEST = "{venv}/bin/pytest"
# The test executables a cargo suite's build produced, in the side's output (`host/agent.py`).
TESTS = "{output}/tests.json"
RAN = "{run_dir}/" + ledger.RAN
SELECTIONS = HOME / "selections"  # each `--stale` selection, by its digest, sent as an input


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=("cargo", "python"), default="cargo",
                        help="cargo tests (default) or the evaluation tools' pytest suites")
    parser.add_argument("--timed", action="store_true",
                        help="the selected tests time the device: run them as a measurement, after the "
                             "thermal gate, with nothing else on the host")
    parser.add_argument("--model", type=selectors.model,
                        help="catalog model to acquire first, as MAGNITUDE_ARTIFACTS for the tests (else a "
                             "cargo suite has the catalog's headers)")
    parser.add_argument("--stale", action="store_true",
                        help="run what is stale on the backend at the placed host's hardware (mrig stale): "
                             "cargo arguments after -- may narrow it to packages (-p) and set the profile "
                             "or features, but not choose targets or tests")
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="the suite's arguments after --: cargo test's, e.g. -- -p magnitude-kernels "
                             "--release --test routed -- general_routed; or pytest's, e.g. -- "
                             "evaluation/references/tests/test_released.py -k qwen")


def _given(arguments) -> list[str]:
    given = arguments.arguments[1:] if arguments.arguments[:1] == ["--"] else arguments.arguments
    if not given and not arguments.stale:
        raise SystemExit("test needs the suite's arguments after --, e.g. -- -p CRATE --test NAME -- FILTER")
    return given


@functools.cache
def _ledger() -> ledger.Ledger:
    """The ledger as this command found it, read once for every host it plans for."""
    return ledger.Ledger()


def stale_selection(found: ledger.Ledger, own: list[str], source, backend: str,
                    hardware: str | None) -> list[ledger.Stale]:
    """What `--stale` runs at `source` on `backend` and `hardware`: `found`'s stale tests of
    the packages cargo test's own arguments `own` select (all where they select none), in the
    configuration they build. Arguments that choose targets or tests are refused: the ledger
    chooses them."""
    chosen = {argument.split("=", 1)[0] for argument in own} & (cargo.TARGETS | {"--doc"})
    if chosen or cargo.test_name(own):
        raise SystemExit(f"--stale chooses the targets and tests; drop {sorted(chosen) or cargo.test_name(own)}")
    packages = cargo.packages(own)
    universe = {key: binary for key, binary in ledger.binaries(source.directory).items()
                if not packages or binary.package in packages}
    if packages - {binary.package for binary in universe.values()}:
        raise SystemExit(f"--stale: no test targets in {sorted(packages - {b.package for b in universe.values()})}")
    return found.stale(source.directory, backend, hardware, cargo.configuration(own), universe)


def _stale(arguments, source, host: str, runner: tuple) -> tuple[tuple[Step, ...], tuple, tuple, set[str]]:
    """A `--stale` run's builds on `host`, one per package, its jobs (`runner` on the selection,
    and the documentation tests of the packages whose are stale), the selection's file and its
    packages: what the ledger finds stale on the backend at `host`'s hardware."""
    if arguments.suite != "cargo":
        raise SystemExit("--stale selects cargo tests")
    own, binary = cargo.split(_given(arguments))
    hardware = _ledger().hardware_of(host, arguments.backend)
    stale = stale_selection(_ledger(), own, source, arguments.backend, hardware)
    if not stale:
        raise Refused(f"nothing is stale on {arguments.backend} at its hardware ({hardware})")
    kept = [argument for option, value in cargo.options(own) if option not in cargo.PACKAGES
            for argument in ([option] if value is None else [option, value])]
    by_package: dict[str, dict[tuple[str, ...], None]] = {}
    for item in stale:
        by_package.setdefault(item.binary.package, {})[item.binary.selection()] = None
    builds = tuple(Step("test build", ("cargo", "test", "--message-format=short", "--no-run", *kept, "-p", package,
                                       *(argument for targets in selections for argument in targets)))
                   for package, selections in sorted(by_package.items()))
    content = json.dumps({item.binary.key: item.tests and list(item.tests) for item in stale}, sort_keys=True)
    path = SELECTIONS / f"selection-{hashlib.sha256(content.encode()).hexdigest()[:16]}.json"
    if not path.is_file():
        SELECTIONS.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".tmp")
        staged.write_text(content)
        staged.replace(path)
    jobs = (("test", Step("test", (*runner, "--selection", "{inputs}/" + path.name, "--", *binary))),)
    documented = sorted({item.binary.package for item in stale if item.binary.kind == ledger.DOC})
    if documented:  # each package's whole: they compile from the source when they run, in the checkout
        doc = ("cargo", "test", "--doc", *kept, *(argument for package in documented for argument in ("-p", package)),
               "--", *binary)
        jobs += (("doc", Step("doc", doc, checkout=True)),)
    return builds, jobs, (path,), set(by_package)


def _documentation(own: list[str]) -> bool:
    """Whether cargo test's own arguments run documentation tests alone (`--doc` and no other
    target): nothing to build without running."""
    selected = {argument.split("=", 1)[0] for argument in own}
    return "--doc" in selected and not selected & cargo.TARGETS


# The crate whose `catalog-headers` feature reads the catalog's header export
# (`MAGNITUDE_CATALOG_HEADERS`): a test declares it reads catalog models by depending on it so.
CATALOG_READER, CATALOG_FEATURE = "magnitude-family-common", "catalog-headers"


def reads_catalog(tree, packages: set[str]) -> bool:
    """Whether a test of `packages` (every workspace package where it names none) of the
    workspace at `tree` can read catalog models: one of them, by its dependencies or its
    dev-dependencies, or a workspace package its dependencies reach, declares the header
    reader (`CATALOG_READER` with `CATALOG_FEATURE`)."""
    workspace = {package["name"]: package for package in ledger.metadata(tree)["packages"]}
    unknown = packages - set(workspace)
    if unknown:
        raise SystemExit(f"test: no workspace package {sorted(unknown)}")

    @functools.cache
    def reaches(name: str, tests: bool) -> bool:  # `tests`: its dev-dependencies too, as its tests have
        for dependency in workspace[name]["dependencies"]:
            if dependency["kind"] == "build" or (dependency["kind"] == "dev" and not tests):
                continue
            if dependency["name"] == CATALOG_READER and CATALOG_FEATURE in dependency["features"]:
                return True
            if dependency["name"] in workspace and dependency["name"] != name and reaches(dependency["name"], False):
                return True
        return False

    return any(reaches(name, True) for name in packages or set(workspace))


def plan(arguments, source, host) -> Plan:
    # A run names its backend, the CPU's too, so a test runs on it alone, fails where it is
    # missing, and never passes by skipping it.
    env = (("SEISMIC_TEST_BACKEND", arguments.backend),)
    # A test of real weights a conversion left in the host's model store (a layer slice) finds them there.
    env += (("MAGNITUDE_MODELS", "{models}"),)
    if arguments.model is not None:
        env += (("MAGNITUDE_ARTIFACTS", "{artifacts}"),)
    files = ()
    runner = ("{python}", "-m", "magnitude_rig.host", "tests", "--list", TESTS, "--ran", RAN)
    if arguments.suite == "cargo" and arguments.stale:
        builds, jobs, files, packages = _stale(arguments, source, host, runner)
    elif arguments.suite == "cargo":
        own, binary = cargo.split(_given(arguments))
        packages = cargo.packages(own)
        builds, jobs = (), ()
        if not _documentation(own):
            # Each executable gets what cargo would pass it: its test name filter, then what follows `--`.
            built = [argument for argument in own if argument != "--doc"]
            builds = (Step("test build", ("cargo", "test", "--message-format=short", "--no-run", *built)),)
            jobs += (("test", Step("test", (*runner, "--", *cargo.test_name(own), *binary))),)
        if cargo.runs_doc_tests(own):  # they compile from the source when they run: in the checkout, kept for them
            doc = ("cargo", "test", "--doc", *cargo.without_targets(own), "--", *cargo.test_name(own), *binary)
            jobs += (("doc", Step("doc", doc, checkout=True)),)
    else:
        builds, packages = (environment("environment", "--all-packages", "--all-extras"),), set()
        # Quiet comes from the project's pytest settings; a second -q would drop the summary line.
        jobs = (("test", Step("test", (PYTEST, "-rfE", *_given(arguments)))),)
    # A cargo suite has the release catalog's headers where a test of it declares it reads them
    # (`reads_catalog`), unless it reads one model's files; no other run resolves the catalog.
    catalog = arguments.suite == "cargo" and arguments.model is None and reads_catalog(source.directory, packages)
    if catalog:
        env += (("MAGNITUDE_CATALOG_HEADERS", "{catalog}/headers"),)
    jobs = tuple((name, dataclasses.replace(step, env=env)) for name, step in jobs)
    acquires = Weights(arguments.model) if arguments.model else Headers(None) if catalog else None
    device = "timed" if arguments.timed else None if arguments.backend == "cpu" else "device"
    return Plan(builds=builds, jobs=jobs, device=device, acquires=acquires, files=files)


def _pytest(lines: list[str]) -> tuple[dict[str, int], list[str]]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    summary = next((line for line in reversed(lines) if " in " in line and PYTEST_COUNT.search(line)), "")
    for number, kind in PYTEST_COUNT.findall(summary):
        counts["failed" if kind.startswith("error") else kind] += int(number)
    return counts, [match[1] for line in lines if (match := PYTEST_FAILED.match(line.strip()))]


def _cargo(arguments, results: dict[str, JobResult]) -> tuple[list[str], bool]:
    """A cargo suite's answer, from what each test executed (`ledger.judge`) and libtest's
    outcomes, its rows entered in the ledger: its lines, and whether it passed."""
    backend, shown, ok = arguments.backend, [], True
    judged: dict[str, list[str]] = {}
    failed: list[str] = []
    executables = results.get("test")
    ran = []
    if executables is not None:
        path = executables.directory / ledger.RAN
        if not path.exists():
            return ["NOT RUN: the job recorded no test executable it ran", *_tail(executables)], False
        ran = json.loads(path.read_text())
        ok = executables.code == 0
    for record in ran:
        if record["counts"] is not None and libtest.tallied(record["outcomes"]) != record["counts"]:
            shown.append(f"UNREADABLE: {record['binary']} printed outcomes {libtest.tallied(record['outcomes'])} "
                         f"for libtest's counts {record['counts']} (passed, failed, ignored)")
            ok = False
            continue
        if record["code"] != 0 and not any(outcome == "FAILED" for outcome in record["outcomes"].values()):
            shown.append(f"FAILED: {record['binary']} exited {record['code']}")
            ok = False
        if threads := ledger.unattributed(record):
            shown.append(f"  {record['binary']}: device use on threads no test runs on ({', '.join(threads)}): "
                         "its tests that recorded nothing are unknown, not host tests")
        for name, outcome in record["outcomes"].items():
            if outcome == "ignored":
                continue
            kind = ledger.judge(record, name, backend)
            judged.setdefault(kind, []).append(f"{record['binary']} {name}")
            failed += [f"{record['binary']} {name}"] if outcome == "FAILED" else []
    doc = {}
    if "doc" in results:
        lines = results["doc"].output()
        doc = {crate: libtest.outcomes(section) for crate, section in libtest.doc_sections(lines).items()}
        for crate, outcomes in doc.items():
            for name, outcome in outcomes.items():
                if outcome != "ignored":
                    judged.setdefault(ledger.HOST_TEST, []).append(f"doc {crate} {name}")
                    failed += [f"doc {crate} {name}"] if outcome == "FAILED" else []
        if results["doc"].code != 0 and not failed:
            shown += [f"FAILED: the documentation tests exited {results['doc'].code}", *_tail(results["doc"])]
        ok = ok and results["doc"].code == 0
    ran_on = len(judged.get(ledger.RAN_THERE, []))
    host_tests = len(judged.get(ledger.HOST_TEST, []))
    executed = ran_on + (host_tests if backend == ledger.CPU else 0)
    skipped, unknown = judged.get(ledger.SKIPPED, []), judged.get(ledger.UNKNOWN, [])
    counts = {kind: len(judged.get(kind, [])) for kind in (ledger.RAN_THERE, ledger.HOST_TEST, ledger.OUTSIDE,
                                                           ledger.SKIPPED, ledger.UNKNOWN)}
    verdict = "FAIL" if failed or not ok else "NOT RUN" if skipped or unknown or executed == 0 else "PASS"
    tally = ", ".join(f"{count} {kind}" for kind, count in counts.items() if count)
    shown.insert(0, f"{verdict} {tally or 'no tests'} ({backend})")
    if skipped:
        shown.append(f"NOT RUN on {backend}: chose its devices, did no work there and declared it outside "
                     "none: " + ", ".join(skipped[:20]))
    if unknown:
        shown.append(f"NOT RUN on {backend}: unknown whether it used a device: " + ", ".join(unknown[:20]))
    if failed:
        shown.append("failed: " + ", ".join(failed[:20]))
    if executables is not None:
        configuration = cargo.configuration(cargo.split(_given(arguments))[0])
        runs = executables.directory.parent
        ledger.record(runs, ledger.rows(runs, executables.directory, backend, configuration, doc))
    return shown, verdict == "PASS"


def _tail(result: JobResult) -> list[str]:
    return [f"  {line}" for line in result.output()[-15:] if not line.startswith(libtest.STDERR)]


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    if arguments.suite == "cargo":
        lines, passed = _cargo(arguments, results)
        logs = [line for result in results.values() for line in result.output()]
        panics = [line.strip() for line in logs if "panicked at" in line][:5]
        return Answer((*lines, *(f"  {panic}" for panic in panics)), passed)
    result = results["test"]
    lines = result.output()
    counts, names = _pytest(lines)
    ok = result.code == 0 and counts["failed"] == 0
    shown = [f"{'PASS' if ok else 'FAIL'} " + ", ".join(f"{n} {kind}" for kind, n in counts.items())
             + f" ({arguments.backend})"]
    if names:
        shown.append("failed: " + ", ".join(names[:20]))
    if result.code != 0 and not counts["failed"]:
        shown += _tail(result)
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
