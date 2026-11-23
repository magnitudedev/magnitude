"""`bench`: how does an engine serve this model over HTTP, against other engines? The serving
benchmark (`magnitude-bench job`), with its measurement options as this instrument's own, on
`--template` (repeatable: one checkpoint each, every template of one fixture and response kind),
each built with the bench's traffic lanes and its qualification lane, and our engine and the
harness built from the checkout as the measured engines need them.
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
import json

from .. import sync, templates
from ..engine import Step
from ..sources import checkout
from . import Answer, Instrument, JobResult, Plan, environment, harness, held
from .binaries import ASSESS, ENGINE, SERVED_BUILD

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
    from magnitude_bench.policy import LANES

    # The traffic lanes and the qualification lane, the bench's last.
    files = tuple(templates.file(template, LANES + 1) for template in arguments.template)
    engines = arguments.engine or ["magnitude"]
    builds = [environment("bench", "--package", "magnitude-bench")]
    jobs, ours = [], []
    if "magnitude" in engines:
        builds.append(SERVED_BUILD)
        if arguments.magnitude_binary is None:
            ours += ["--magnitude-binary", ENGINE]
        if arguments.magnitude_cache_dir is None:  # a named cache is passed with the bench's options
            cache = KERNELS
            ours += ["--magnitude-cache-dir", cache]
        else:
            cache = str(arguments.magnitude_cache_dir)
        jobs.append(("measure", Step("measure", (
            ASSESS, "--device", arguments.magnitude_device, "--kernels", cache, "--topology"))))
    if any(engine in bench.HOST_RENDERED for engine in engines) and arguments.harness is None:
        builds.append(harness.BUILD)
        ours += ["--harness", harness.EXECUTABLE]
    jobs.append(("bench", Step("bench", (
        BENCH, "job", "--inference", "{src}",
        *(("--model", arguments.model) if arguments.model is not None else ()),
        *bench.arguments(arguments), *ours,
        *(token for file in files for token in ("--template", templates.argument(file))),
        "--artifacts", "{artifacts}", "--output", "{run_dir}", "--build-identity", "{build_file}",
    ))))  # fmt: skip
    return Plan(builds=tuple(builds), jobs=(*held.jobs(arguments), *jobs), device="timed",
                acquires=held.acquires(arguments), files=files)


def _tuning_alert(directory) -> list[str]:
    from magnitude_bench import record, report

    health = None
    summary_path = directory / "summary.json"
    if summary_path.exists():
        health = json.loads(summary_path.read_text()).get("tuning_health")
    if health is None:
        trace_path = directory / record.TUNING_TRACE
        if trace_path.exists():
            health = report.tuning_health(trace_path)
    return report.tuning_health_alert(health) if health else []


def _attribution_summary(directory) -> list[str]:
    from magnitude_bench import attribution

    measure_log = directory.parent / "logs" / "job-measure.log"
    log_path = measure_log if measure_log.is_file() else None
    data = attribution.extract(directory, measure_log=log_path)
    if not data["requests"] and not data["internal_requests"]:
        return []
    attribution.write(directory, measure_log=log_path)
    return attribution.format_summary(data)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    priced = results.get("measure")  # only when our engine serves
    if priced is not None and priced.code != 0:
        return Answer((f"the machine's measurement exited {priced.code}: our engine served unpriced",
                       *priced.output()[-15:]), False)
    result = results["bench"]
    report = result.directory / "report.md"
    if result.code != 0 or not report.exists():
        return Answer((f"magnitude-bench exited {result.code}", *result.output()[-15:]), False)
    text = report.read_text()
    lines = text.splitlines()
    shown = lines[:REPORT_LINES]
    if len(lines) > REPORT_LINES:
        shown.append(f"… {len(lines) - REPORT_LINES} more lines in {report}")
    if len(lines) > REPORT_LINES or "## Tuning health" not in text:
        shown.extend(_tuning_alert(result.directory))
    shown.extend(_attribution_summary(result.directory))
    return Answer(tuple(shown), True)


INSTRUMENT = Instrument(
    name="bench",
    kind="speed",
    question="How does an engine serve this model over HTTP, against other engines?",
    observable="serving throughput and latency",
    selectors=("backend", "host", "build", "model-or-checkpoint", "templates", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
