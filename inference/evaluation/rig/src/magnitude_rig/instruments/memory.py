"""`memory`: what does a load of this model hold of a dedicated GPU, split into the terms
assessment prices? The host package's `memory` measurement (`host/memory.py`, which also runs
alone on a machine outside the rig) on the checkout's engine and assessment: a first load
prepares an isolated kernel cache, then a measured load samples the device's memory while it
opens, prepares its programs and loads, and splits what the device lost into the context, the
programs (per native kernel of the load's inventory) and the residual beyond what the engine
holds (`DeviceProcessCosts`). Device-wide memory is sampled, so the run holds the device alone.
"""

import argparse

from ..engine import HOST_PACKAGE, Step, Weights
from . import Answer, Instrument, JobResult, Plan
from .binaries import ASSESS_BUILD, ENGINE_BUILD

DEDICATED = ("cuda", "vulkan")  # a unified-memory device has no device-wide figure to split


def configure(parser: argparse.ArgumentParser) -> None:
    pass


def plan(arguments, source) -> Plan:
    if arguments.backend not in DEDICATED:
        raise SystemExit(f"memory: measures a dedicated GPU ({', '.join(DEDICATED)}), not {arguments.backend}")
    job = Step("memory", (
        "{python}", "-m", "magnitude_rig.host.memory", "--bin", "{target}/release",
        "--backend", arguments.backend, "--scratch", "{run_dir}", "--artifacts", "{artifacts}",
    ), (("PYTHONPATH", f"{{src}}/{HOST_PACKAGE}"),))  # fmt: skip
    return Plan(builds=(ENGINE_BUILD, ASSESS_BUILD), jobs=(("memory", job),), device="timed",
                acquires=Weights(arguments.model))


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["memory"]
    output = [line for line in result.output() if not line.startswith("[stderr]")]
    if result.code != 0 or not (result.directory / "device-costs.json").exists():
        return Answer((f"memory exited {result.code}", *result.output()[-15:]), False)
    return Answer(tuple(output), True)


INSTRUMENT = Instrument(
    name="memory",
    kind="memory",
    question="What does a load of this model hold of a dedicated GPU, by the terms assessment prices?",
    observable="device memory a load holds, by term",
    selectors=("backend", "host", "build", "model", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
