"""`bench`: how does an engine serve this model over HTTP, against other engines? The serving
benchmark (`magnitude-bench job`), with its measurement options as this instrument's own, and
our engine and the harness built from the checkout as the measured engines need them.
Engineering questions about one engine's speed go to `step` and `kernel`; this is serving as
users see it.

Our engine serves from a kernel cache fresh per run (`{runs}/kernels`, which stays on the host
and goes with the run's directory), or the one the run names (`--magnitude-cache-dir`). Before
it serves, the product's measurer prices the machine into that cache: `magnitude-assess`, which
runs the function the service's measurement job runs (`worker::measure::measure_machine`),
times every executor cost point the cache has no record of. The bench's report shows the
engine's elastic expert bank over each pass (`magnitude_bench.report.expert_banks`).

Our engine and the harness are built by separate cargo invocations (`binaries`): the harness
enables the engine's development tuning tools, which one invocation would unify into the served
binary.
"""

import argparse

from .. import sync
from ..engine import Step
from ..sources import checkout
from . import Answer, Instrument, JobResult, Plan, environment, harness, held
from .binaries import ASSESS, ASSESS_BUILD, ENGINE, ENGINE_BUILD

BENCH = "{venv}/bin/magnitude-bench"
KERNELS = f"{{runs}}/{sync.RUN_KERNELS}"  # the run's own kernel cache
REPORT_LINES = 60


def configure(parser: argparse.ArgumentParser) -> None:
    from magnitude_bench import cli as bench

    held.option(parser)
    bench.measurement_options(parser)


def plan(arguments, source, host) -> Plan:
    from magnitude_bench import cli as bench
    held.subject("bench", arguments)
    if arguments.magnitude_device == "auto":
        arguments.magnitude_device = arguments.backend
    try:
        bench.check(arguments, checkout())
    except ValueError as error:
        raise SystemExit(f"bench: {error}") from None
    engines = arguments.engine or ["magnitude"]
    builds = [environment("bench", "--package", "magnitude-bench")]
    jobs, ours = [], []
    if "magnitude" in engines:
        if arguments.magnitude_binary is None:
            builds.append(ENGINE_BUILD)
            ours += ["--magnitude-binary", ENGINE]
        if arguments.magnitude_cache_dir is None:  # a named cache is passed with the bench's options
            cache = KERNELS
            ours += ["--magnitude-cache-dir", cache]
        else:
            cache = str(arguments.magnitude_cache_dir)
        builds.append(ASSESS_BUILD)
        jobs.append(("measure", Step("measure", (
            ASSESS, "--device", arguments.magnitude_device, "--kernels", cache, "--topology"))))
    if any(engine in bench.HOST_RENDERED for engine in engines) and arguments.harness is None:
        builds.append(harness.BUILD)
        ours += ["--harness", harness.EXECUTABLE]
    jobs.append(("bench", Step("bench", (
        BENCH, "job", "--inference", "{src}",
        *(("--model", arguments.model) if arguments.model is not None else ()),
        *bench.arguments(arguments), *ours,
        "--artifacts", "{artifacts}", "--output", "{run_dir}", "--build-identity", "{build_file}",
    ))))  # fmt: skip
    return Plan(builds=tuple(builds), jobs=(*held.jobs(arguments), *jobs), device="timed",
                acquires=held.acquires(arguments))


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    priced = results.get("measure")  # only when our engine serves
    if priced is not None and priced.code != 0:
        return Answer((f"the machine's measurement exited {priced.code}: our engine served unpriced",
                       *priced.output()[-15:]), False)
    result = results["bench"]
    report = result.directory / "report.md"
    if result.code != 0 or not report.exists():
        return Answer((f"magnitude-bench exited {result.code}", *result.output()[-15:]), False)
    lines = report.read_text().splitlines()
    shown = lines[:REPORT_LINES]
    if len(lines) > REPORT_LINES:
        shown.append(f"… {len(lines) - REPORT_LINES} more lines in {report}")
    return Answer(tuple(shown), True)


INSTRUMENT = Instrument(
    name="bench",
    kind="speed",
    question="How does an engine serve this model over HTTP, against other engines?",
    observable="serving throughput and latency",
    selectors=("backend", "host", "build", "model-or-checkpoint", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
