"""`device`: what is this host's hardware, as the machine, the engine and the driver each state
it? Three owners, each of its own facts:

- the host record every run keeps (`host.facts`): CPU, memory, GPUs, OS;
- the engine's topology (`magnitude-assess --topology`): the devices and links the engine
  discovers, its discovery diagnostics, and the executor's costs, each with its samples and
  interval (points the host's kernel cache has no record of are timed, under the timed
  hold);
- on Vulkan, the driver's own facts and behaviour probes (`seismic/hardware/vulkan-facts`):
  identity, limits, features, memory heaps, and the FMA and specialization-constant probes.

Each job's whole output is its log; the answer shows each one's head and names the log.
"""

import argparse
import json

from .. import store
from ..engine import HOST_PACKAGE, Refused, Step
from . import Answer, Instrument, JobResult, Plan
from .binaries import ASSESS, ASSESS_BUILD

VULKAN_FACTS = "{target}/release/vulkan-facts"
SHOWN = 30


def configure(parser: argparse.ArgumentParser) -> None:
    pass


def plan(arguments, source, host) -> Plan:
    builds = [ASSESS_BUILD]
    jobs = [
        ("host", Step("host", ("{python}", "-m", "magnitude_rig.host", "facts"),
                      (("PYTHONPATH", f"{{src}}/{HOST_PACKAGE}"),))),
        ("topology", Step("topology", (ASSESS, "--device", arguments.backend, "--kernels", "{engine_cache}/kernels", "--topology"))),
    ]
    if arguments.backend == "vulkan":
        builds.append(Step("vulkan facts", ("cargo", "build", "--release", "--manifest-path",
                                            "{src}/seismic/hardware/vulkan-facts/Cargo.toml")))
        jobs.append(("vulkan", Step("vulkan", (VULKAN_FACTS,))))
    return Plan(builds=tuple(builds), jobs=tuple(jobs), device="timed")


def memory(host: str, backend: str) -> str:
    """Whose memory `host`'s `backend` device is, `unified` (the host's) or `dedicated` (its
    own), as the engine's topology in the host's latest good recorded `mrig device` run states it
    (each physical device's `memory`, `Host` or `Own`). Refused when no run of it is recorded,
    or its devices of that backend differ."""
    run, kinds = _topology(host)
    found = kinds.get(backend, set())
    if len(found) != 1:
        raise Refused(f"{host}'s topology ({run}) states {backend} devices of memory "
                      f"{sorted(found) or 'none'}, not one")
    return {"Host": "unified", "Own": "dedicated"}[next(iter(found))]


def backends(host: str) -> tuple[str, ...]:
    """The backends whose devices `host`'s latest good recorded topology states (`memory` says
    whose memory each is); Refused as `memory` is when none is recorded."""
    return tuple(sorted(_topology(host)[1]))


def _topology(host: str) -> tuple[str, dict[str, set[str]]]:
    """The run directory of `host`'s latest good recorded `mrig device` run, and the memory kinds
    its engine topology states for each backend's devices."""
    # `<time>Z-device-<hex>` (`engine.new_run_id`), latest first
    runs = sorted(store.results().glob("*Z-device-*/rig.json"), reverse=True)
    for path in runs:
        record = json.loads(path.read_text())
        if record["host"] != host or record["failed"] is not None or record["codes"]["topology"] != 0:
            continue
        topology = next((document["topology"] for line in (path.parent / "logs" / "job-topology.log").read_text().splitlines()
                         if (document := _document(line)) is not None and "topology" in document), None)
        if topology is None:
            raise Refused(f"{host}'s latest device run ({path.parent.name}) recorded no topology: "
                          f"run mrig device --host {host} again")
        # A physical device's identity is a JSON value (`{"Uuid": [...]}`): keyed by its text.
        key = lambda identity: json.dumps(identity, sort_keys=True)
        memories = {key(physical["identity"]): physical["memory"] for physical in topology["physical"]}
        kinds: dict[str, set[str]] = {}
        for executor in topology["executors"]:
            kinds.setdefault(_backend(executor["selector"]), set()).add(memories[key(executor["physical"])])
        return path.parent.name, kinds
    raise Refused(f"no topology of {host} is recorded: run mrig device --host {host} first")


def _backend(selector) -> str:
    """The backend of an executor selector as the topology serializes it (`HostCpu`, `{"Cuda": …}`)."""
    name = selector if isinstance(selector, str) else next(iter(selector))
    return name.lower().removeprefix("host")


def _document(line: str) -> dict | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


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
