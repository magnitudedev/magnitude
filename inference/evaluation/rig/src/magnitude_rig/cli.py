"""mrig: the magnitude rig. Ask the hardware a question, through the instrument that answers it.

Every run that holds a host takes --purpose (what it settles) and --estimate (how long it
should take). It prints one line when it starts (where it runs, who holds that host, whether
this checkout's build is there) and the answer when it ends; build and job output go to log
files in its results directory. Run it as one background command: its end is the notification.
It ends with the run, or at your estimate, when it says what the run has done so far and leaves
it running for you to stop, or to wait on with a new estimate and the reason it needs one
(`mrig wait`).
"""

import argparse
import json
import os
import shlex
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ab, engine, selectors, sources, state
from .connection import HostUnavailable, connect, host_status, read, setup
from .hosts import Hosts
from .instruments import KINDS, REGISTRY, Instrument
from .results import follow, perform


def _epilog() -> str:
    rows = []
    for kind, meaning in KINDS.items():
        rows.append(f"  {kind}: {meaning}")
        rows += [f"    {name:<9}{instrument.question}" for name, instrument in REGISTRY.items() if instrument.kind == kind]
    rows.append(f"  {'ab':<11}Is B better or worse than A, by any instrument that measures? "
                "(mrig ab INSTRUMENT --base A --candidate B ...)")
    tools = [
        "  inspect    Who holds and waits on each host; or run a read-only command on one (no hold)",
        "  runs       This machine's runs in flight: phase, host, time against estimate",
        "  logs RUN   Where a run's logs are, and the end of its current phase's",
        "  wait RUN   Follow a run past its estimate, with a new --estimate and its --justification",
        "  stop RUN   Stop a run (its host work is killed and its holds released)",
        "  setup HOST Prepare and verify a host or pool; runs go only to verified hosts",
    ]
    return "\n".join([
        "instruments (by the question each answers):", *rows, "",
        "host and run tools:", *tools, "",
        "A question no instrument answers: extend the instrument that owns its observable, or add",
        "one to instruments/ (mrig script runs anything meanwhile).",
    ])


def run_instrument(instrument: Instrument, arguments) -> int:
    with sources.opened(arguments.build) as source:
        return _run(instrument, arguments, source)


def _run(instrument: Instrument, arguments, source: sources.Source) -> int:
    plan = instrument.plan(arguments, source)
    side = engine.Side("source", source, plan.builds)
    jobs = tuple(engine.Job("source", name, step) for name, step in plan.jobs)
    request = engine.Request(
        instrument=instrument.name,
        purpose=arguments.purpose,
        estimate=arguments.estimate,
        backend=arguments.backend,
        sides=(side,),
        jobs=jobs,
        device=plan.device,
        host=arguments.host,
        acquires=plan.acquires,
        files=plan.files,
    )
    return perform(arguments, request, lambda results: instrument.answer(arguments, results))


# Host and run tools ----------------------------------------------------------------------


def _inspect(arguments) -> int:
    hosts = Hosts.load()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if command:
        if not arguments.host:
            raise SystemExit("inspect -- COMMAND needs --host")
        remote = connect(hosts.host(arguments.host))
        code, stdout, stderr = read(remote, remote.command(command, cwd=remote.root)[-1])
        print(stdout, end="")
        print(stderr, end="", file=sys.stderr)
        return code
    names = hosts.resolve(arguments.host) if arguments.host else tuple(hosts.host(a) for a in hosts.known())

    def one(config):
        try:
            remote = connect(config)
            return remote, host_status(remote), None
        except HostUnavailable as error:
            return None, None, f"{config.alias}: unreachable ({error})"

    with ThreadPoolExecutor(max_workers=8) as pool:
        for remote, status, error in pool.map(one, names):
            if error:
                print(error)
                continue
            ready = "ready" if remote.ready else "not set up"
            free = status["free_bytes"] / 1e9
            print(f"{remote.alias} ({'/'.join(remote.backends)}, {ready}, {free:.0f} GB free)")
            legacy = status["device_lock_held_elsewhere"]
            if legacy is not None:
                print(f"  device held outside mrig by {legacy.get('who', '?')} (run {legacy.get('run', '?')})")
            for entry in status["entries"]:
                print(
                    f"  {entry['state']:<7} {entry['kind']:<10} {entry['run']}  \"{entry.get('purpose', '')}\"  "
                    f"{engine.duration(entry['remaining'])} left of {engine.duration(entry['estimate'])}"
                    f"  ({entry['who']})"
                )
    return 0


def _runs(arguments) -> int:
    live = state.live_runs()
    if not live:
        print("no runs in flight on this machine")
    for run in live:
        elapsed = time.time() - run["started"]
        waiting = f", waiting behind {run['waiting_behind']}" if run.get("waiting_behind") else ""
        print(
            f"{run['run']}  {run['host']}  {run['phase']}{waiting}  "
            f"{engine.duration(elapsed)} of {engine.duration(run['estimate'])}  \"{run['purpose']}\""
        )
    return 0


def _results(run: str) -> Path:
    """The results directory of the run named (in full or by its end), in flight or not."""
    live = state.run_state(run)
    if live is not None:
        return Path(live["results"])
    matches = sorted(engine.RESULTS.glob(f"*{run}"))
    if not matches:
        raise SystemExit(f"no run {run} in flight or in {engine.RESULTS}")
    return matches[-1]


def _logs(arguments) -> int:
    run = state.run_state(arguments.run)
    directory = str(_results(arguments.run) / "logs")
    if run is not None:
        current = os.path.join(directory, f"{run['phase']}.log")
    else:
        logs = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
        current = os.path.join(directory, logs[-1]) if logs else None
    print(f"logs: {directory}")
    if current and os.path.exists(current):
        with open(current, errors="replace") as stream:
            tail = stream.read().splitlines()[-arguments.lines:]
        print(f"end of {os.path.basename(current)}:")
        for line in tail:
            print(f"  {line}")
    return 0


def _wait(arguments) -> int:
    results = _results(arguments.run)
    run = state.run_state(arguments.run)
    with open(results / engine.WAITS, "a") as stream:
        stream.write(json.dumps({"at": time.time(), "estimate_seconds": arguments.estimate,
                                 "justification": arguments.justification}) + "\n")
    return follow(results.name, results, run["pid"] if run else None, arguments.estimate)


def _stop(arguments) -> int:
    run = state.run_state(arguments.run)
    if run is None:
        raise SystemExit(f"no run {arguments.run} in flight on this machine")
    os.kill(run["pid"], signal.SIGTERM)
    print(f"stopping {run['run']} on {run['host']}")
    return 0


def _setup(arguments) -> int:
    hosts = Hosts.load()
    failed = 0
    for config in (c for name in arguments.names for c in hosts.resolve(name)):
        try:
            print(f"{config.alias}:\n{setup(config).rstrip()}")
        except HostUnavailable as error:
            print(f"mrig: {error}", file=sys.stderr)
            failed = 1
    return failed


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="mrig", description=__doc__, epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = top.add_subparsers(dest="action", required=True, metavar="COMMAND")
    for name, instrument in REGISTRY.items():
        command = commands.add_parser(
            name, help=instrument.question,
            description=f"{instrument.question} (observes: {instrument.observable})",
        )
        selectors.add(command, instrument.selectors)
        instrument.configure(command)
        command.set_defaults(handler=lambda arguments, i=instrument: run_instrument(i, arguments))
    ab.configure(commands.add_parser("ab", help="compare two builds by any measuring instrument"))

    command = commands.add_parser("inspect", help="host holds and queues, or a read-only command")
    command.add_argument("--host")
    command.add_argument("command", nargs=argparse.REMAINDER)
    command.set_defaults(handler=_inspect)
    commands.add_parser("runs", help="this machine's runs in flight").set_defaults(handler=_runs)
    command = commands.add_parser("logs", help="a run's logs")
    command.add_argument("run")
    command.add_argument("--lines", type=int, default=30)
    command.set_defaults(handler=_logs)
    command = commands.add_parser("wait", help="follow a run past its estimate, only when the duration is necessary")
    command.add_argument("run")
    command.add_argument("--estimate", type=selectors.seconds, required=True,
                         help="how much longer it should take, e.g. 20m")
    command.add_argument("--justification", required=True,
                         help="why the run needs this long, and why the first estimate was wrong")
    command.set_defaults(handler=_wait)
    command = commands.add_parser("stop", help="stop a run")
    command.add_argument("run")
    command.set_defaults(handler=_stop)
    command = commands.add_parser("setup", help="prepare and verify hosts")
    command.add_argument("names", nargs="+", help="hosts or pools")
    command.set_defaults(handler=_setup)
    return top


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    arguments = parser().parse_args(argv)
    arguments.argv = argv  # a run's own process is this command again (`results.perform`)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
