"""`memory`: what does a load of this model hold, by the terms assessment prices? The host
package's `memory` measurement (`host/memory.py`, which also runs alone on a machine outside the
rig, and fits the costs across loads) on the checkout's binaries.

Which measurement a load takes is whose memory its device is, as the engine's topology in the
placed host's latest `mrig device` run states it (`device.memory`; a host with none recorded is
refused), never its backend: a unified CUDA device (GB10's) measures as Metal does.

On a dedicated GPU (its own memory), with the engine and assessment: a first load prepares an
isolated kernel cache, then a measured load samples the device's memory while it opens, prepares
its programs and loads, and splits what the device lost into the context, the programs (per
native kernel of the load's inventory) and the residual beyond what the engine holds
(`DeviceProcessCosts`). Device-wide memory is sampled, so the run holds the device alone.

In unified memory (the host's), with `magnitude-harness memory`: the target loads plain in a worker
process, and its own memory (its footprint at load end less what its plan holds then) is set
against its charge's runtime, vocabulary and program terms (`HostProcessCosts`). Under
`--weight-cap BYTES` the load is planned under the cap and reads its routed experts from storage
(a `fresh` job), and the answer is that streamed load's own memory against everything its plan
charges, term by term: what it charges must equal what it is observed to hold. `--context TOKENS`
loads at that context; `--residency TOKENS` (under both) also reads the pages of the expert
stacks the file cache holds once loaded and once the load has decoded TOKENS tokens of the
`--fixture` set's first fixture (it must hold none). `--pressure BYTES` (under `--residency`) is the
forced-pressure gate: a sibling process presses the load BYTES below its planning line (it takes
everything above the line, read as the engine reads it, and BYTES more) from a quarter of that
decode to three quarters of it, and the answer is the owner's releases in order (each rung, the bank's slots
after each step, each step's stall, the charge and footprint on each side), then the same decode
on a resident load: the streamed tokens must equal its tokens.

`--recommit GB[,GB...]` (unified memory) loads nothing, and `--model` is not read: it times an
in-place recommit of a reserved tensor, an elastic expert bank's resize, at each commitment (one
2 MiB row shrunk and regrown, five times each way).

Re-measuring `HostProcessCosts`: one `mrig memory --model M` on a unified-memory device per load the
constants' comment names, then the fit over those runs' loads:

    uv run --project inference/evaluation python -m magnitude_rig.host.memory \\
        --fit inference/evaluation/results/<run>/memory/host-costs.json ...
"""

import argparse
import json

from ..engine import HOST_PACKAGE, Step, Weights
from ..host import memory
from ..host.agent import PLAN
from ..host.memory import RECOMMIT, RESULTS
from . import Answer, Instrument, JobResult, Plan, device, harness
from .binaries import ASSESS, ASSESS_BUILD, ENGINE_BUILD
from .assess import RUNNER


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weight-cap", type=int, metavar="BYTES",
                        help="unified memory: the load streamed under this weight cap, its experts read from storage")
    parser.add_argument("--context", type=int, metavar="TOKENS",
                        help="unified memory: load at this context (default: the model's declared one)")
    parser.add_argument("--residency", type=int, metavar="TOKENS",
                        help="under --weight-cap and --context: also read the pages of the expert stacks the file "
                             "cache holds once loaded and once the load has decoded TOKENS tokens of --fixture's "
                             "first fixture (it must hold none)")
    parser.add_argument("--pressure", type=int, metavar="BYTES",
                        help="under --residency: another process presses the load BYTES below its planning line "
                             "from a quarter of the decode to three quarters of it; report the owner's releases in order (the bank's "
                             "slots, each step's stall, charge and footprint) and the tokens against a resident "
                             "decode's (the forced-pressure gate)")
    parser.add_argument("--pressure-store", choices=("host", "device"), help="physical store to press (required with --pressure)")
    parser.add_argument("--pressure-prompt", action="store_true",
                        help="under --pressure: the pressure starts once the request is admitted, so its prompt's "
                             "groups run under it (the gate with a prompt group in flight)")
    parser.add_argument("--recommit", metavar="GB[,GB...]",
                        help="unified memory, loading nothing: what an in-place recommit of a reserved tensor (an "
                             "elastic expert bank's resize) costs at each commitment")


def _measurement(options: tuple[str, ...], fresh: bool) -> Step:
    """The host's measurement with `options`, judged as the host judges them (`refusal`)."""
    refused = memory.refusal(memory.parser().parse_args(options))
    if refused is not None:
        raise SystemExit(f"memory: {refused}")
    return Step("memory", ("{python}", "-m", "magnitude_rig.host.memory", *options),
                (("PYTHONPATH", f"{{src}}/{HOST_PACKAGE}"),), fresh=fresh)


def plan(arguments, source, host) -> Plan:
    kind = device.memory(host, arguments.backend)
    cap =("--weight-cap", str(arguments.weight_cap)) if arguments.weight_cap else ()
    # Every option as given: the host's judge (`refusal`) refuses a pair it cannot measure.
    pressure = (*(("--pressure", str(arguments.pressure)) if arguments.pressure else ()),
                *(("--pressure-store", arguments.pressure_store) if arguments.pressure_store else ()),
                *(("--pressure-prompt",) if arguments.pressure_prompt else ()))
    context = ("--context", str(arguments.context)) if arguments.context else ()
    residency = (("--residency", harness.EXECUTABLE, f"{{inputs}}/{arguments.fixture.name}",
                  str(arguments.residency)) if arguments.residency else ())
    given = ("--bin", "{target}/release", "--backend", arguments.backend, "--memory", kind, "--scratch", "{run_dir}",
             *cap, *context, *residency, *pressure)
    if arguments.recommit:
        job = _measurement((*given, "--recommit", arguments.recommit), fresh=False)
        return Plan(builds=(harness.BUILD,), jobs=(("memory", job),), device="device")
    # A streamed load prepares from the host's kernel cache, as a served load does, after the
    # measurement job has stored the executor's costs there: its bank's growth is priced on them.
    kernels = ("--kernels", "{engine_cache}/kernels") if cap else ()
    job = _measurement((*given, "--artifacts", "{artifacts}", *kernels), fresh=bool(cap))
    if kind == "dedicated" and not arguments.pressure:
        return Plan(builds=(ENGINE_BUILD, ASSESS_BUILD), jobs=(("memory", job),), device="timed",
                    acquires=Weights(arguments.model))
    measure = (("measure", Step("measure", (
        "{python}", "-c", RUNNER, "{artifacts}", ASSESS, arguments.backend, "{engine_cache}/kernels",
    ))),) if cap else ()  # the product's measurement job, for its own time (`step`)  # fmt: skip
    # The load's own footprint, not the device's: the device is held so no other load runs.
    return Plan(builds=(harness.BUILD, *((ASSESS_BUILD,) if cap else ())), jobs=(*measure, ("memory", job)),
                device="device", acquires=Weights(arguments.model),
                files=(arguments.fixture,) if arguments.residency else ())


def planned_memory(result: JobResult) -> str:
    """The device's memory the run's plan measured it as (`--memory`, which `plan` read on its
    host), from the run's own plan, fetched beside its jobs."""
    plan = json.loads((result.directory.parent / PLAN).read_text())
    (argv,) = [op["argv"] for op in plan["ops"] if op.get("job") == "memory"]
    return argv[argv.index("--memory") + 1]


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["memory"]
    output = [line for line in result.output() if not line.startswith("[stderr]")]
    written = RECOMMIT if arguments.recommit else RESULTS[planned_memory(result)]
    if result.code != 0 or not (result.directory / written).exists():
        return Answer((f"memory exited {result.code}", *result.output()[-15:]), False)
    return Answer(tuple(output), True)


INSTRUMENT = Instrument(
    name="memory",
    kind="memory",
    question="What does a load of this model hold, by the terms assessment prices?",
    observable="memory a load holds, by term",
    selectors=("backend", "host", "build", "model", "fixture", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
