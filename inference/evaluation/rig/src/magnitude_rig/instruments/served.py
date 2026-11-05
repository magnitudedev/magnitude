"""`served`: does a load work through the real service? It runs the service executable
(`magnitude-inference serve`) on the host from an installation of the checkout's binary and the
catalog, drives it over its HTTP API as a client does, and reports each stage from what the API
answers and what the service logged (`host/served.py`).

The service runs its own measurement job beside the load, under its own device lock, so this is
the one instrument on which nothing has measured before a load. `--weight-cap` and `--context` are
the service's evaluation bounds: without them a load is the product's, at the model's declared
context, if assessment admits it.

Scenarios, each from an empty kernel cache of the run's own:

- `lifecycle`: a load arrives during the job and preempts it, serves under the no-cost policy,
  the job runs beside the idle load, a request preempts it, `Reprice` reaches the live load, a
  new service's load starts priced; and where each answer parts from a resident load's.
- `stop`: a load stopped while the job runs beside it, then a load in the same service.
- `stop-prepare`: a cold load stopped during its Prepare with a second load queued behind the
  stop: the queued load loads and answers, and no two inference workers ever run at once, by the
  service's own records of each worker's start and of its exit once shown reaped.
- `repeat`: two prompts answered in turn on one load, where each parts from its first answer.
- `failure`: a stretch of the model file read as EIO under two answers: those that named it
  fail, the load serves on, and where its answer parts from the one before.

Whether an answer is admitted is the precision system's, the `kl` instrument's at the same
weight cap: a served answer is never failed for its text, which paths computing the same
features in other arithmetic change within it. Each is reported with where it parts.
"""

import argparse
import json
import re

from ..engine import Step, Weights
from ..host import served
from . import Answer, Instrument, JobResult, Plan

SERVICE = "{target}/release/magnitude-inference"
# The service selects its device itself: what makes `--backend` hold where it would prefer another.
HIDDEN = {"vulkan": (("CUDA_VISIBLE_DEVICES", ""),)}
BUILD = Step("service", ("cargo", "build", "--release", "-p", "magnitude-service-server", "--bin", "magnitude-inference"))


def strip(line: str) -> str:
    """`line` without its terminal colours."""
    return re.sub(r"\x1b\[[0-9;]*m", "", line)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", choices=served.SCENARIOS, default="lifecycle", help="what is driven (default lifecycle)")
    parser.add_argument("--context", type=int, metavar="TOKENS", help="serve at this context")
    parser.add_argument("--weight-cap", type=int, metavar="BYTES", help="stream every capped load under this cap")
    parser.add_argument("--plain", action="store_true", help="decode plainly, the model's drafter left out")


def plan(arguments, source, host) -> Plan:
    bounds = (*(("--context", str(arguments.context)) if arguments.context else ()),
              *(("--weight-cap", str(arguments.weight_cap)) if arguments.weight_cap else ()),
              *(("--plain",) if arguments.plain else ()))
    job = Step("served", (
        "{python}", "-m", "magnitude_rig.host.served", "--service", SERVICE, "--bundle", "{catalog}/model-planner-inputs.bundle",
        "--models", "{models}", "--artifacts", "{artifacts}", "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--scratch", "{root}/served",
        "--run-dir", "{run_dir}", "--model", arguments.model, "--scenario", arguments.scenario, *bounds,
    ), HIDDEN.get(arguments.backend, ()), fresh=True)
    return Plan(builds=(BUILD,), jobs=(("served", job),), device="device",
                acquires=Weights(arguments.model, headers=True), files=(arguments.fixture,))


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["served"]
    written = result.directory / served.RESULTS
    if not written.exists():
        return Answer((f"served exited {result.code}", *result.output()[-25:]), False)
    selected = [line for log in sorted(result.directory.glob("service-*.log"))
                for line in log.read_text(errors="replace").splitlines() if "serving device selected" in line]
    wrong = [line for line in selected if f'backend="{arguments.backend}"' not in strip(line)]
    report = tuple(json.loads(written.read_text())["report"])
    if wrong:
        return Answer((f"the service did not select {arguments.backend}: {strip(wrong[0])[-120:]}", *report), False)
    return Answer(report, result.code == 0)


INSTRUMENT = Instrument(
    name="served",
    kind="correctness",
    question="Does a load work through the real service, its measurement job beside it?",
    observable="the service's API answers and logs",
    selectors=("backend", "host", "build", "model", "fixture", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
