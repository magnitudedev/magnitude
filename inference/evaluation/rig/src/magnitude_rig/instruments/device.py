"""`device`: what is this host's hardware, as the machine, the engine and the driver each state
it? Three owners, each of its own facts:

- the host record every run keeps (`host.facts`): CPU, memory, GPUs, OS;
- the engine's topology (`magnitude-assess --topology`): the devices and links the engine
  discovers, its discovery diagnostics, and the executor's costs, each with its samples and
  interval (points the checkout's kernel cache has no record of are timed, under the timed
  hold);
- on Vulkan, the driver's own facts and behaviour probes (`seismic/hardware/vulkan-facts`):
  identity, limits, features, memory heaps, and the FMA and specialization-constant probes.

Each job's whole output is its log; the answer shows each one's head and names the log.
"""

import argparse
import json

from ..engine import HOST_PACKAGE, Step
from . import Answer, Instrument, JobResult, Plan
from .binaries import ASSESS, ASSESS_BUILD

VULKAN_FACTS = "{target}/release/vulkan-facts"
SHOWN = 30


def configure(parser: argparse.ArgumentParser) -> None:
    pass


def plan(arguments, source) -> Plan:
    builds = [ASSESS_BUILD]
    jobs = [
        ("host", Step("host", ("{python}", "-m", "magnitude_rig.host", "facts"),
                      (("PYTHONPATH", f"{{src}}/{HOST_PACKAGE}"),))),
        ("topology", Step("topology", (ASSESS, "--device", arguments.backend, "--kernels", "{checkout}/kernels", "--topology"))),
    ]
    if arguments.backend == "vulkan":
        builds.append(Step("vulkan facts", ("cargo", "build", "--release", "--manifest-path",
                                            "{src}/seismic/hardware/vulkan-facts/Cargo.toml")))
        jobs.append(("vulkan", Step("vulkan", (VULKAN_FACTS,))))
    return Plan(builds=tuple(builds), jobs=tuple(jobs), device="timed")


def _readable(line: str) -> list[str]:
    """A line of output, a JSON document (the topology prints one per line) indented."""
    try:
        return json.dumps(json.loads(line), indent=1).splitlines()
    except ValueError:
        return [line]


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    lines, ok = [], True
    for name, result in results.items():
        output = [shown for line in result.output() if not line.startswith("[stderr]")
                  for shown in _readable(line)]
        ok &= result.code == 0
        lines.append(f"{name}: {result.log}" + ("" if result.code == 0 else f" · exited {result.code}"))
        lines += [f"  {line}" for line in output[:SHOWN]]
        if len(output) > SHOWN:
            lines.append(f"  … {len(output) - SHOWN} more lines")
    return Answer(tuple(lines), ok)


INSTRUMENT = Instrument(
    name="device",
    kind="hardware",
    question="What is this host's hardware, as the machine, the engine and the driver state it?",
    observable="hardware facts and the executor's measured costs",
    selectors=("backend", "host", "build", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
