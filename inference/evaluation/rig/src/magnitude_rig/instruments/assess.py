"""`assess`: what do the planner, assessment and estimator decide for this model on this
device? `magnitude-assess` on the model's acquired files, with the checkout's kernel cache
holding the machine's cost measurements (only points without a record are timed). Prints the
assessment's own JSON per model: whether it fits, how it would run, the estimated decode speed.
With `--breakdown`, the planned step at each depth by entry, from the plan: each entry's
launches per step and the estimator's time for it (the larger of its bytes and compute bound,
plus its fixed cost), with the step's own totals.

`--catalog` assesses every catalog model through the service's own path (its environment, the
automatic pool and the per-target assessor, the release catalog's header bundle), after the
machine's measurement as above: one line per model (fits, its context and memory, estimated
speeds; or what does not fit or is unsupported) and the pool's timings.
"""

import argparse
import json
import re
from collections import defaultdict

from ..engine import Headers, Step, Weights
from . import Answer, Instrument, JobResult, Plan
from .binaries import ASSESS, ASSESS_BUILD

RUNNER = (
    "import json, os, sys; a = json.load(open(sys.argv[1])); "
    "paths = a['target'] + (',,' + a['drafter']['path'] if a.get('drafter') else ''); "
    "os.execv(sys.argv[2], [sys.argv[2], '--device', sys.argv[3], '--kernels', sys.argv[4], *sys.argv[5:], paths])"
)
CATALOG = "{target}/release/examples/catalog_assessment"
BUILDS = (ASSESS_BUILD,)
CATALOG_BUILDS = BUILDS + (
    Step("catalog assessment", ("cargo", "build", "--release", "-p", "magnitude-service-server",
                                "--example", "catalog_assessment")),
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", action="store_true", help="every catalog model, through the service's assessment")
    parser.add_argument("--depths", help="context depths to estimate, e.g. 25000,50000 (assess's default otherwise)")
    parser.add_argument("--breakdown", action="store_true",
                        help="the planned step at each depth by entry: launches and estimated time")


def plan(arguments, source) -> Plan:
    if (arguments.model is None) == (not arguments.catalog):
        raise SystemExit("assess: give --model or --catalog")
    if arguments.catalog:
        if arguments.depths or arguments.breakdown:
            raise SystemExit("assess: --depths and --breakdown are per model, not with --catalog")
        # The checkout's kernel cache is the service cache root's `kernels`: the measurement
        # the first job stores there is the one the service's assessment reads.
        jobs = (
            ("measure", Step("measure", (ASSESS, "--device", arguments.backend, "--kernels", "{checkout}/kernels", "--topology"))),
            ("catalog", Step("catalog", (
                CATALOG, "--bundle", "{catalog}/model-planner-inputs.bundle", "--cache-root", "{checkout}",
                "--model-store", "{models}", "--snapshot", "{run_dir}/snapshot.json",
            ))),
        )  # fmt: skip
        return Plan(builds=CATALOG_BUILDS, jobs=jobs, device="timed", acquires=Headers(None))
    extra = (*(("--depths", arguments.depths) if arguments.depths else ()),
             *(("--breakdown",) if arguments.breakdown else ()))
    job = Step("assess", (
        "{python}", "-c", RUNNER, "{artifacts}", ASSESS, arguments.backend, "{checkout}/kernels", *extra,
    ))  # fmt: skip
    return Plan(
        builds=BUILDS,
        jobs=(("assess", job),),
        device="timed",  # it times the cost points the kernel cache has no record of
        acquires=Weights(arguments.model),
    )


DEPTH = re.compile(r"^magnitude-assess:\s+at (\d+): (.*)$")
CALL = re.compile(
    r"^magnitude-assess:\s+(\S+)\s+([\d.]+) µs bytes\s+([\d.]+) µs compute\s+([\d.]+) µs fixed (\d+) launches$"
)
SHOWN = 16


def breakdown(lines: list[str]) -> list[str]:
    """The planned step at each depth, by entry: launches and estimated µs per step."""
    shown, entries = [], None

    def flush():
        if entries is None:
            return
        shown.append(f"  {'entry':<34}{'launches':>9}{'est µs':>10}")
        ranked = sorted(entries.items(), key=lambda item: -item[1][1])
        for name, (launches, micros) in ranked[:SHOWN]:
            shown.append(f"  {name[:33]:<34}{launches:>9}{micros:>10.1f}")
        if len(ranked) > SHOWN:
            rest = ranked[SHOWN:]
            shown.append(f"  {f'{len(rest)} more entries':<34}{sum(launches for _, (launches, _) in rest):>9}"
                         f"{sum(micros for _, (_, micros) in rest):>10.1f}")

    for line in lines:
        text = line.removeprefix("[stderr] ").strip()
        if (match := DEPTH.match(text)):
            flush()
            entries = defaultdict(lambda: [0, 0.0])
            shown.append(f"step at depth {match[1]}: {match[2]}")
        elif (match := CALL.match(text)) and entries is not None:
            bytes_, compute, fixed, launches = (float(match[2]), float(match[3]), float(match[4]), int(match[5]))
            entry = entries[match[1]]
            entry[0] += launches
            entry[1] += max(bytes_, compute) + fixed
    flush()
    return shown


def _catalog(results: dict[str, JobResult]) -> Answer:
    for name, result in results.items():
        if result.code != 0:
            return Answer((f"{name} exited {result.code}", *result.output()[-15:]), False)
    lines = [line for line in results["catalog"].output() if not line.startswith("[stderr]")]
    return Answer(tuple(lines), True)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    if arguments.catalog:
        return _catalog(results)
    result = results["assess"]
    if result.code != 0:
        return Answer((f"magnitude-assess exited {result.code}", *result.output()[-15:]), False)
    shown = []
    for line in result.output():
        if line.startswith("[stderr]"):
            continue
        try:
            shown.append(json.dumps(json.loads(line), indent=1))
        except ValueError:
            shown.append(line)
    if arguments.breakdown:
        shown += breakdown(result.output())
    return Answer(tuple(shown), True)


INSTRUMENT = Instrument(
    name="assess",
    kind="decision",
    question="What do the planner, assessment and estimator decide for this model on this device "
             "(and, with --breakdown, its step's launches and time by entry)?",
    observable="assessment and estimate",
    selectors=("backend", "host", "build", "model-or-catalog", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
