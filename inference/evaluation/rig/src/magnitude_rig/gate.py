"""`mrig gate`: does this build load and step correctly on every topology?

A topology is a backend and whose memory its device computes in, `unified` (the host's) or
`dedicated` (its own, beside a host tier), as the hosts' recorded engine topologies state it
(`device.memory`): the inventory, every listed host's latest `mrig device` run, so a host joins
the gate once its topology is recorded; the CPU backend is not gated. The gate asks whether every
topology passes, so its verdict and exit status are over the recorded topologies alone: hosts
whose topology is not recorded are named on one informational line (`no topology recorded`),
each with its likely topology from its platform (`likely`), which never touches the verdict or
the exit status; recording one (`mrig device --host HOST`) is a chore of its own. A host whose
recorded devices of one gated backend disagree on whose memory they compute in has no topology to
run: it fails the gate (a line of its own), for that backend when topologies are named, else
always.

Each topology is one run of its own, all at once, each placed by the rig's placement among the
hosts of that topology (a host of another refuses the plan). Under one device hold (no thermal
gate: nothing is timed), every entry at its declared default (no tuning search) and with the
gate's model, context and weight cap (`MODEL`, `CONTEXT`, `WEIGHT_CAP`), its experts read from
storage, it runs:

- `step`: `step`'s harness `time` as served (lookahead on, no logits exported), at the fewest
  steps and samples the harness takes: it loads streamed, prefills (a cold and a warm sample) and
  decodes `STEPS` steps;
- `logits`: `kl`'s harness `logits`, streamed likewise, on the first fixture of the step's own
  rendering cut to its first `ROWS` scored rows (`trim`): one prefill, then `ROWS` decodes, each
  exporting its logits;
- `finite`: whether every logit the logits job produced is finite.

Each run's estimate is derived where it is placed: the seconds until its jobs would start there
(the queues ahead of it and its build, priced by that host's measured builds of the harness,
cold or incremental by whether the checkout it claims has built it at an ancestor:
`engine.build_seconds`), and what the latest recorded `step` run with a streamed cell on that
host took past its build and waits, with twice its streamed cell for the gate's two streamed
loads (`after_build`; for a host with none, the most of its topology's hosts).

A ref is resolved to its commit once, for every topology (`worktree` is staged by each run as it
starts, as any run's is). A run past its estimate is not a result: the gate says so as it
happens (RUNNING, past its estimate by how much) and follows it on, as `mrig wait` would, to its
end. Once every run has ended it prints a line per topology, PASS or the first error line, each
run's results named, and exits 0 only when every topology passed (`verdict`). Each topology's
run is an ordinary run: `mrig logs`, `mrig stop` and `mrig wait` take its id, and one whose
caller is gone is answered by the next command like any other.
"""

import argparse
import functools
import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import engine, selectors, sources, state, store
from .connection import HostError, connect
from .hosts import Hosts
from .instruments import Answer, JobResult, harness, kl, step
from .instruments.device import backends, memory
from .results import CHILD, OVERRUN, answer_fetched, follow, forward_stops, launch, perform

KIND = "correctness"
QUESTION = "Does this build load and step correctly on every topology?"
OBSERVABLE = "a streamed load's prefill, decode steps and logits, per topology"
MODEL = "qwen3.6-35b-a3b:gguf:q4"
CONTEXT = 4096
# The ceiling on MODEL's weight bytes: the bank grows to the most expert slots the cap holds, at
# most every expert, so a cap at or above the load's weights with every expert resident streams
# nothing and the gate would pass without exercising streaming; below the weights its bank needs
# at its minimum the load is refused. This one holds about a fifth of each layer's experts, the
# shape expert streaming is measured at, so the rest stream from storage.
WEIGHT_CAP = 6890894474
FIXTURE = "prose-repeat"
STEPS = 2  # the fewest measured steps the harness `time` takes
ROWS = 4  # the logits job's decodes
MEMORIES = ("unified", "dedicated")
UNGATED = ("cpu",)
DEFAULTS = ("--tuning-defaults",)

# The rendering's first fixture, its first scored span cut to its first ROWS rows: argv[1] the
# rendering, argv[2] the output, argv[3] ROWS. Run on the host.
TRIM = (
    "import json, sys\n"
    "with open(sys.argv[1]) as stream:\n"
    "    line = json.loads(stream.readline())\n"
    "start, end = line['scored_spans'][0]\n"
    "line['scored_spans'] = [[start, min(end, start + int(sys.argv[3]))]]\n"
    "with open(sys.argv[2], 'w') as stream:\n"
    "    stream.write(json.dumps(line) + '\\n')\n"
)
# Whether every float32 of the logits file argv[1] is finite; the first non-finite one named.
FINITE = (
    "import array, math, sys\n"
    "values = bad = 0\n"
    "first = None\n"
    "with open(sys.argv[1], 'rb') as stream:\n"
    "    for block in iter(lambda: stream.read(1 << 22), b''):\n"
    "        floats = array.array('f')\n"
    "        floats.frombytes(block)\n"
    "        for index, value in enumerate(floats):\n"
    "            if not math.isfinite(value):\n"
    "                first = values + index if first is None else first\n"
    "                bad += 1\n"
    "        values += len(floats)\n"
    "if not values:\n"
    "    sys.exit('gate: no logits were produced')\n"
    "if bad:\n"
    "    sys.exit('gate: non-finite logits: %d of %d values, the first at value %d' % (bad, values, first))\n"
    "print('gate: logits finite: %d values' % values)\n"
)


@dataclass(frozen=True, order=True)
class Topology:
    backend: str
    memory: str

    def __str__(self) -> str:
        return f"{self.backend}/{self.memory}"


def topology(text: str) -> Topology:
    backend, _, kind = text.partition("/")
    if backend not in selectors.BACKENDS or backend in UNGATED or kind not in MEMORIES:
        gated = [backend for backend in selectors.BACKENDS if backend not in UNGATED]
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a topology BACKEND/MEMORY ({'|'.join(gated)}/{'|'.join(MEMORIES)})")
    return Topology(backend, kind)


@dataclass(frozen=True)
class Inventory:
    topologies: dict[Topology, tuple[str, ...]]  # each topology's hosts, in the hosts file's order
    # Each host's gated backend whose recorded devices' memories differ: the backend, and why;
    # each a gate failure.
    disagreeing: tuple[tuple[str, str], ...]
    # The listed hosts whose topology is not recorded: each with its likely topology, or why it
    # is unknown; information, never a gate result.
    unrecorded: tuple[str, ...]


def recorded(hosts: Hosts) -> tuple[dict[Topology, tuple[str, ...]], tuple[tuple[str, str], ...],
                                    tuple[tuple[str, str], ...]]:
    """Every listed host's gated backends and their topologies, as recorded (`device.memory`):
    each topology's hosts, in the hosts file's order; each backend whose devices' memories differ,
    with why; and each host with no topology recorded, with why."""
    topologies: dict[Topology, list[str]] = {}
    unknown, disagreeing = [], []
    for alias in hosts.known():
        try:
            found = backends(alias)
        except engine.Refused as refused:
            unknown.append((alias, str(refused)))
            continue
        for backend in found:
            if backend in UNGATED:
                continue
            try:
                topologies.setdefault(Topology(backend, memory(alias, backend)), []).append(alias)
            except engine.Refused as refused:
                disagreeing.append((backend, str(refused)))
    return {key: tuple(value) for key, value in sorted(topologies.items())}, tuple(disagreeing), tuple(unknown)


def likely(system: str, machine: str, backends: tuple[str, ...]) -> tuple[str, ...]:
    """The topologies a host whose own is not recorded likely has, from its platform as `connect`
    probes it: an Apple host's GPU computes in the host's memory, an x86 host's in its own; any
    other's memory is not guessed (`?`)."""
    kind = "unified" if system == "Darwin" else "dedicated" if machine in ("x86_64", "AMD64") else "?"
    return tuple(f"{backend}/{kind}" for backend in backends if backend not in UNGATED)


def inventory(hosts: Hosts) -> Inventory:
    """The recorded topologies and the backends whose devices disagree (`recorded`), and every
    host with none recorded: with its likely topologies (`likely`) and the hosts already covering
    them, or that it could not be reached."""
    topologies, disagreeing, unknown = recorded(hosts)
    covering = {str(topology): hosts for topology, hosts in topologies.items()}

    def classify(entry):
        alias, _ = entry
        try:
            remote = connect(hosts.host(alias))
        except HostError as error:
            return f"{alias} (unreachable: {error})"
        shown = [f"{name}, covered by {', '.join(covering[name])}" if name in covering else f"{name}, no other host"
                 for name in likely(remote.system, remote.machine, remote.backends)]
        return f"{alias} (likely {'; '.join(shown) or 'no gated backend'})"

    with ThreadPoolExecutor(max_workers=8) as pool:
        classified = tuple(pool.map(classify, unknown))
    return Inventory(topologies, disagreeing, classified)


def after_build(topology: Topology, hosts: tuple[str, ...]) -> dict[str, float]:
    """The seconds a gate run of `topology` takes past its build and its waits, on each of its
    `hosts` with a recorded `step` run with a streamed cell (the latest, its `rig.json`): that
    run's phases but its waits (`queue-*`), its build (the gate's own is priced at placement,
    `engine.build_seconds`), its thermal gate and its jobs, which the gate's do not repeat; and
    twice its streamed cell (the gate's two streamed loads, each fewer steps than a cell's)."""
    found: dict[str, float] = {}
    for path in sorted(store.results().glob("*Z-step-*/rig.json"), reverse=True):  # latest first
        record = json.loads(path.read_text())
        host = record["host"]
        if host in found or host not in hosts or record.get("backend") != topology.backend:
            continue
        if record["failed"] is not None:
            continue
        phases = {phase["phase"]: phase["seconds"] for phase in record["phases"]}
        cells = [phases[f"job-{name}"] for name, code in record["codes"].items()
                 if name.startswith("streamed-") and code == 0 and f"job-{name}" in phases]
        if not cells:
            continue
        own = sum(seconds for name, seconds in phases.items()
                  if not name.startswith(("queue-", "job-", "build-")) and name != "gate")
        found[host] = own + 2 * max(cells)
    return found


def estimate(topology: Topology, hosts: tuple[str, ...]):
    """A gate run's estimate once placed (`engine.Request.estimate`): the seconds until its jobs
    start on the host placed (its queues and its build as that host measured it) and what it
    takes past them there (`after_build`), or, for a host with no record of that, the most any
    host of the topology takes. None when none has."""
    found = after_build(topology, hosts)
    if not found:
        return None
    return lambda alias, start: start + found.get(alias, max(found.values()))


def verdict(codes: list[int]) -> int:
    """The gate's exit code from its topologies' ended runs' (`results.carry_out`'s codes): 0 only
    when every one passed; else 1 when any has no answer, else 2 (one answered no). None checked
    is no answer."""
    if not codes or 1 in codes:
        return 1
    return 0 if all(code == 0 for code in codes) else 2


def headline(code: int, transcript: str, run: str) -> str:
    """A topology's line from its ended run's code and printed output: PASS, its answer's first
    line (a job's error), or its failure and the first error it shows."""
    lines = transcript.splitlines()
    if code == 0:
        return "PASS"
    if code == 2:
        at = next((index for index, line in enumerate(lines) if line.startswith("purpose: ")), None)
        if at is not None and at + 1 < len(lines):
            return lines[at + 1]
    failed = next((index for index, line in enumerate(lines) if line.startswith("FAILED in ")), None)
    if failed is not None:
        shown = lines[failed + 1].strip() if failed + 1 < len(lines) and lines[failed + 1].startswith("  ") else ""
        return lines[failed] + (f" · {shown}" if shown else "")
    errors = [line for line in lines if line.startswith("mrig: ")]
    return errors[-1] if errors else next((line for line in reversed(lines) if line.strip()), f"exited {code}")


# One topology's run ----------------------------------------------------------------------


@functools.cache
def _arguments(backend: str) -> argparse.Namespace:
    """What `step.time` and `kl.logits` read of their arguments: the step instrument's own parser
    at the gate's settings, so an option the step gains can't go missing here."""
    parser = argparse.ArgumentParser()
    selectors.add(parser, ("backend", "context", "fixture"))
    step.configure(parser)
    return parser.parse_args((
        "--backend", backend, "--context", str(CONTEXT), "--fixture", FIXTURE, "--warm", "0",
        "--steps", str(STEPS), "--prefill-samples", "1", "--attribution", "0", "--untimed",
    ))  # fmt: skip


def jobs(backend: str) -> tuple[tuple[str, engine.Step], ...]:
    arguments = _arguments(backend)
    timed = step.time(arguments, arguments.fixture.stem, *DEFAULTS, "--weight-cap", str(WEIGHT_CAP))
    return (
        ("step", engine.Step("time", timed, fresh=True)),
        ("trim", engine.Step("trim", ("{python}", "-c", TRIM, "{runs}/step/rendered.jsonl",
                                      "{run_dir}/rendered.jsonl", str(ROWS)))),
        ("logits", kl.logits(arguments, rendered="{runs}/trim/rendered.jsonl", pinning=DEFAULTS,
                             weight_cap=WEIGHT_CAP)),
        ("finite", engine.Step("finite", ("{python}", "-c", FINITE, "{runs}/logits/logits.f32"))),
    )


def _work(wanted: Topology, source: sources.Source):
    def work(host: str) -> engine.Work:
        found = memory(host, wanted.backend)
        if found != wanted.memory:
            raise engine.Refused(f"its {wanted.backend} device is {found}, not {wanted.memory}")
        return engine.Work(
            sides=(engine.Side("source", source, (harness.BUILD,)),),
            jobs=tuple(engine.Job("source", name, job) for name, job in jobs(wanted.backend)),
            device="device",
            acquires=engine.Weights(MODEL),
            files=(_arguments(wanted.backend).fixture,),
        )

    return work


def _said(result: JobResult) -> str:
    """What a job said last, a failed one its first error line: a panic's message, else the last
    line it printed (a harness and the gate's checks print their error, or their finding, last)."""
    lines = [line.removeprefix("[stderr] ") for line in result.output()]
    panic = next((index for index, line in enumerate(lines) if " panicked at " in line), None)
    if panic is not None:
        return " ".join(line.strip() for line in lines[panic:panic + 2])
    return next((line for line in reversed(lines) if line.strip()), "no output")


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    for name, result in results.items():  # in the order they ran: the first failure caused the rest
        if result.code != 0:
            return Answer((f"{name} exited {result.code}: {_said(result)}", f"  log: {result.log}"), False)
    finite = _said(results["finite"])
    return Answer((f"{MODEL} @{CONTEXT} streamed under {WEIGHT_CAP} · loaded, prefilled and decoded "
                   f"{STEPS} steps as served · {ROWS} logits rows: {finite.removeprefix('gate: ')}",), True)


def _one(arguments) -> int:
    (wanted,) = arguments.topology  # the gate starts each topology's run with it alone
    found = recorded(Hosts.load())[0].get(wanted, ())
    derived = estimate(wanted, found)
    if derived is None:
        raise SystemExit(f"gate: no recorded step run with a streamed cell on {', '.join(found) or 'any host'} "
                         f"of {wanted} to derive its estimate from")
    with sources.opened(arguments.build) as source:
        request = engine.Request(instrument="gate", purpose=arguments.purpose, estimate=derived,
                                 backend=wanted.backend, work=_work(wanted, source))
        return perform(arguments, request, lambda results: answer(arguments, results))


# The gate ----------------------------------------------------------------------------------


def _gate(arguments) -> int:
    found = inventory(Hosts.load())
    wanted = arguments.topology or list(found.topologies)
    sources.checkout()  # refused outside a git worktree, before any topology's run starts
    # A ref is resolved once, so every topology builds one commit however the ref moves.
    build = arguments.build if arguments.build == sources.WORKTREE else sources.commit(arguments.build)
    lines, codes, runs, children = [], [], [], []
    forward_stops(children)  # each as it starts
    for backend, why in found.disagreeing:
        if not arguments.topology or any(each.backend == backend for each in wanted):
            lines.append(f"{backend + '/?':<17} FAIL: {why}")
            codes.append(1)
    for each in wanted:
        hosts = found.topologies.get(each, ())
        if not hosts:
            lines.append(f"{str(each):<17} FAIL: no listed host records this topology")
            codes.append(1)
        elif not after_build(each, hosts):
            lines.append(f"{str(each):<17} FAIL: no recorded step run with a streamed cell on {', '.join(hosts)} "
                         "to derive its estimate from")
            codes.append(1)
        else:
            argv = ["gate", "--topology", str(each), "--build", build, "--purpose", arguments.purpose]
            run, results, child = launch("gate", argv)
            children.append(child)
            print(f"{str(each):<17} {run} · its estimate derived where it is placed · hosts {', '.join(hosts)}",
                  flush=True)
            runs.append((each, run, results, child))
    if found.unrecorded:
        print(unrecorded_line(found.unrecorded))
    with ThreadPoolExecutor(max_workers=max(1, len(runs))) as pool:
        ended = list(pool.map(lambda entry: followed(*entry), runs))
    for (each, run, results, _), (code, transcript) in zip(runs, ended, strict=True):
        lines.append(f"{str(each):<17} {headline(code, transcript, run)} · {results}")
        codes.append(code)
    for line in lines:
        print(line)
    print(f"{verdict_line(codes)} · purpose: {arguments.purpose}")
    return verdict(codes)


def unrecorded_line(unrecorded: tuple[str, ...]) -> str:
    """The hosts whose topology is not recorded, as information beside the gate, never in it."""
    return (f"no topology recorded (information, not part of the verdict): {', '.join(unrecorded)}; "
            "mrig device --host HOST records one")


def verdict_line(codes: list[int]) -> str:
    """The gate's verdict, over the recorded topologies' runs (`verdict`)."""
    passed = sum(code == 0 for code in codes)
    if verdict(codes) == 0:
        return f"gate: PASS · every recorded topology passed ({passed} of {len(codes)})"
    return f"gate: FAIL · {passed} of {len(codes)} recorded topologies passed"


def followed(topology: Topology, run: str, results, child) -> tuple[int, str]:
    """Run `run`'s exit code and printed output once it has ended. Past its estimate it is not a
    result: that is said as it happens, how far past, and the run followed on to its end, as
    `mrig wait` would."""
    out = io.StringIO()
    code = follow(run, results, child.pid, None, out)
    if code != OVERRUN:
        return code, out.getvalue()
    live = state.run_state(run)
    past = "its estimate" if live is None else (
        f"its {engine.duration(live['estimate'])} estimate by "
        f"{engine.duration(state.account(live, time.time())[1] - live['estimate'])}")
    print(f"{str(topology):<17} RUNNING past {past} · mrig logs {run} · mrig stop {run}", flush=True)
    out = io.StringIO()
    return follow(run, results, child.pid, math.inf, out), out.getvalue()


def run(arguments) -> int:
    if arguments.answering is not None:  # a collected topology's run: from its fetched results
        return answer_fetched(arguments, lambda results: answer(arguments, results), arguments.answering)
    # A topology's own run is this command again, in the process `results.launch` starts.
    return _one(arguments) if CHILD in os.environ else _gate(arguments)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.description = f"{QUESTION} (observes: {OBSERVABLE})"
    parser.epilog = (__doc__.split("\n\n", 1)[1]
                     .replace("(`MODEL`, `CONTEXT`, `WEIGHT_CAP`)", f"({MODEL}, {CONTEXT} tokens, weight cap {WEIGHT_CAP})")
                     .replace("`STEPS`", str(STEPS)).replace("`ROWS`", str(ROWS)))
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    selectors.add(parser, ("build", "purpose"))
    parser.add_argument("--topology", type=topology, action="append", metavar="BACKEND/MEMORY",
                        help="only this topology, e.g. cuda/dedicated (repeatable; default every recorded one)")
    # Each run's estimate is derived once placed (`estimate`), never given.
    parser.set_defaults(handler=run, estimate=None)
