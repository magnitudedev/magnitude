"""`bench`: how does an engine serve this model over HTTP, against other engines? The serving
benchmark (`magnitude-bench job`), with its measurement options as this instrument's own, our
engine and the harness built from the checkout as the measured engines need them, and our
engine's persistent kernel caches unless the run names a cache. Engineering questions about one
engine's speed go to `step` and `kernel`; this is serving as users see it.

Our engine and the harness are built by separate cargo invocations (`binaries`): the harness
enables the engine's development tuning tools, which one invocation would unify into the served
binary.
"""

import argparse

from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Plan, harness
from .binaries import ENGINE, ENGINE_BUILD

BENCH = "{venv}/bin/magnitude-bench"
# A user's installation keeps compiled kernels and tuning across launches.
KERNEL_CACHE_ROOT = "{root}/kernel-cache"
REPORT_LINES = 60


def configure(parser: argparse.ArgumentParser) -> None:
    from magnitude_bench import cli as bench

    bench.measurement_options(parser)


def plan(arguments, source) -> Plan:
    from magnitude_bench import cli as bench

    if arguments.magnitude_device == "auto":
        arguments.magnitude_device = arguments.backend
    try:
        bench.check(arguments)
    except ValueError as error:
        raise SystemExit(f"bench: {error}") from None
    engines = arguments.engine or ["magnitude"]
    builds = [Step("bench", ("uv", "sync", "--frozen", "--project", "{src}/evaluation", "--package", "magnitude-bench"))]
    ours = []
    if "magnitude" in engines and arguments.magnitude_binary is None:
        builds.append(ENGINE_BUILD)
        ours += ["--magnitude-binary", ENGINE]
    if any(engine in bench.HOST_RENDERED for engine in engines) and arguments.harness is None:
        builds.append(harness.BUILD)
        ours += ["--harness", harness.EXECUTABLE]
    if "magnitude" in engines and arguments.magnitude_cache_dir is None and arguments.magnitude_kernel_cache_root is None:
        ours += ["--magnitude-kernel-cache-root", KERNEL_CACHE_ROOT]
    job = Step("bench", (
        BENCH, "job", "--model", arguments.model, *bench.arguments(arguments), *ours,
        "--artifacts", "{artifacts}", "--output", "{run_dir}", "--build-identity", "{build_file}",
    ))  # fmt: skip
    return Plan(builds=tuple(builds), jobs=(("bench", job),), device="timed", acquires=Weights(arguments.model))


def answer(arguments, results: dict[str, JobResult]) -> Answer:
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
    selectors=("backend", "host", "build", "model", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
