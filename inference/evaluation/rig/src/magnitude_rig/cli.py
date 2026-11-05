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
import contextlib
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ab, dashboard, engine, gate, selectors, sources, state, store
from .connection import HostError, HostRun, connect, host_status, read, setup
from .hosts import Hosts
from .instruments import KINDS, REGISTRY, Instrument
from .results import CHILD, answer_fetched, follow, perform, record_answer


def _epilog() -> str:
    rows = []
    for kind, meaning in KINDS.items():
        rows.append(f"  {kind}: {meaning}")
        rows += [f"    {name:<9}{instrument.question}" for name, instrument in REGISTRY.items() if instrument.kind == kind]
        if gate.KIND == kind:
            rows.append(f"    {'gate':<9}{gate.QUESTION} (every topology at once)")
    rows.append(f"  {'ab':<11}Is B better or worse than A, by any instrument that measures? "
                "(mrig ab INSTRUMENT --base A --candidate B ...)")
    tools = [
        "  inspect    Who holds and waits on each host; or run a read-only command on one (no hold)",
        "  dashboard  A live local web page of every host: holds, queue, utilization, memory, thermals",
        "  runs       This machine's runs in flight: phase, host, time against estimate",
        "  logs RUN   Where a run's logs are, and the end of its current phase's",
        "  wait RUN   Follow a run past its estimate, with a new --estimate and its --justification",
        "  stop RUN   Stop a run (its host work is killed and its holds released)",
        "  fetch      Answer the ended runs whose caller is gone (every command does this first) and fetch them whole",
        "  setup HOST Prepare and verify a host or pool; runs go only to verified hosts",
    ]
    return "\n".join([
        "instruments (by the question each answers):", *rows, "",
        "host and run tools:", *tools, "",
        "A question no instrument answers: extend the instrument that owns its observable, or add",
        "one to instruments/ (mrig script runs anything meanwhile).",
    ])


def run_instrument(instrument: Instrument, arguments) -> int:
    if arguments.answering is not None:  # a collected run: from its fetched results, no source
        return answer_fetched(arguments, lambda results: instrument.answer(arguments, results), arguments.answering)
    with sources.opened(arguments.build) as source:
        return _run(instrument, arguments, source)


def _run(instrument: Instrument, arguments, source: sources.Source) -> int:
    def work(host: str) -> engine.Work:
        plan = instrument.plan(arguments, source, host)
        return engine.Work(
            sides=(engine.Side("source", source, plan.builds),),
            jobs=tuple(engine.Job("source", name, step) for name, step in plan.jobs),
            device=plan.device,
            acquires=plan.acquires,
            files=plan.files,
        )

    request = engine.Request(
        instrument=instrument.name,
        purpose=arguments.purpose,
        estimate=arguments.estimate,
        backend=arguments.backend,
        work=work,
        host=arguments.host,
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
        except HostError as error:
            return None, None, str(error)

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
                    f"{engine.against_estimate(entry)}  ({entry['who']})"
                )
    return 0


def _runs(arguments) -> int:
    live = state.live_runs()
    if not live:
        print("no runs in flight on this machine")
    for run in live:
        queued, held = state.account(run, time.time())
        waiting = f", waiting behind {run['waiting_behind']}" if run.get("waiting_behind") else ""
        print(
            f"{run['run']}  {run['host']}  {run['phase']}{waiting}  queued {engine.duration(queued)}, "
            f"{engine.duration(held)} of {engine.duration(run['estimate'])}  \"{run['purpose']}\""
        )
    return 0


def _results(run: str) -> Path:
    """The results directory of the run named (in full or by its end), in flight or not."""
    live = state.run_state(run)
    if live is not None:
        return Path(live["results"])
    matches = sorted(store.results().glob(f"*{run}"))
    if not matches:
        raise SystemExit(f"no run {run} in flight or in {store.results()}")
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
    """Stop the run through the process following it, which delivers the stop and fetches the
    run; with none, stop its runner directly (the next command fetches it). A follower that has
    neither begun delivering its stop nor ended `STOP_SECONDS` after it is hung: it is ended,
    and its runner is stopped as a run whose caller is gone is."""
    run = state.run_state(arguments.run)
    if run is not None:
        os.kill(run["pid"], signal.SIGTERM)
        print(f"stopping {run['run']} on {run['host']}")
        if _takes_stop(run, STOP_SECONDS):
            return 0
        os.kill(run["pid"], signal.SIGKILL)
        if not _takes_stop(run, STOP_SECONDS):
            raise SystemExit(f"{run['run']}: its caller {run['pid']} did not end when killed")
        print(f"{run['run']}: its caller did not take the stop in {STOP_SECONDS:.0f} s; ended it")
    launched = next((value for value in state.Launched.gone() if value["run"].endswith(arguments.run)), None)
    if launched is None:
        raise SystemExit(f"no run {arguments.run} in flight on this machine")
    remote = connect(Hosts.load().host(launched["host"]))
    stopped = HostRun(remote, launched["run"], launched["agent"]).stop()
    print(f"stopping {launched['run']} on {launched['host']}: {stopped}")
    return 0


# How long a run's caller has to begin delivering its stop (`engine._stop_when`).
STOP_SECONDS = 30.0


def _takes_stop(run: dict, seconds: float) -> bool:
    """Whether run `run`'s caller, within `seconds`, has ended or begun delivering its stop."""
    deadline = time.monotonic() + seconds
    while True:
        current = state.run_state(run["run"])
        if current is None or current["pid"] != run["pid"] or "stopping" in current:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def _collect(out, *, rest: bool) -> int:
    """Answer the ended runs whose caller is gone; how many. A run is answered by its own
    command again, answering from its fetched results alone (`results.answer_fetched`: no source
    is opened), or by its record where this checkout cannot read the command. With `rest`, fetch
    the rest of every answered run here; without, in a `mrig fetch` of its own, which this
    command does not wait for."""
    def answer(launched: dict) -> None:
        printed = io.StringIO()
        try:
            with contextlib.redirect_stderr(io.StringIO()) as refused:
                arguments = parser().parse_args(launched["command"])
        except SystemExit:
            # Launched records are this machine's, any checkout's: one whose command this
            # checkout's mrig cannot read is answered by its record, never left owed forever.
            record = json.loads((Path(launched["results"]) / "rig.json").read_text())
            printed.write(f"mrig: this checkout's mrig cannot read {launched['run']}'s command "
                          f"({refused.getvalue().strip().splitlines()[-1]}); its record: failed "
                          f"{record['failed']}, job codes {record['codes']}; results in {launched['results']}\n")
            code = 1
        else:
            arguments.argv, arguments.answering = launched["command"], launched["run"]
            with contextlib.redirect_stdout(printed):
                code = arguments.handler(arguments)
        record_answer(Path(launched["results"]), printed.getvalue(), code)
        print(f"mrig: {launched['run']} ended without its caller; fetched, and its answer:", file=out)
        print(printed.getvalue().rstrip(), file=out, flush=True)

    answered, unfetched = engine.collect(Hosts.load(), answer, rest=rest)
    if unfetched:
        _fetch_detached()
    return len(answered)


def _fetch_detached() -> None:
    """`mrig fetch` in a session of its own, its output in the rig's `fetch.log`."""
    state.HOME.mkdir(parents=True, exist_ok=True)
    with open(state.HOME / "fetch.log", "a") as log:
        subprocess.Popen([sys.executable, "-m", "magnitude_rig.cli", "fetch"], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def _fetch(arguments) -> int:
    if not _collect(sys.stdout, rest=True):
        print("nothing to answer")
    return 0


def _setup(arguments) -> int:
    hosts = Hosts.load()
    failed = 0
    for config in (c for name in arguments.names for c in hosts.resolve(name)):
        try:
            print(f"{config.alias}:\n{setup(config).rstrip()}")
        except HostError as error:
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
    gate.configure(commands.add_parser("gate", help=gate.QUESTION))

    command = commands.add_parser("inspect", help="host holds and queues, or a read-only command")
    command.add_argument("--host")
    command.add_argument("command", nargs=argparse.REMAINDER)
    command.set_defaults(handler=_inspect)
    dashboard.configure(commands.add_parser("dashboard", help="live web page of hosts and their queues"))
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
    commands.add_parser("fetch", help="answer the ended runs whose caller is gone and fetch them whole").set_defaults(handler=_fetch)
    command = commands.add_parser("setup", help="prepare and verify hosts")
    command.add_argument("names", nargs="+", help="hosts or pools")
    command.set_defaults(handler=_setup)
    return top


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    arguments = parser().parse_args(argv)
    arguments.argv = argv  # a run's own process is this command again (`results.perform`)
    arguments.answering = None  # set only to answer a collected run (`_collect`)
    if CHILD not in os.environ and arguments.handler is not _fetch:
        _collect(sys.stderr, rest=False)  # runs whose caller is gone, now ended
    try:
        return arguments.handler(arguments)
    except sources.SourceUnavailable as error:  # outside a git worktree, or a ref it cannot build
        raise SystemExit(f"mrig: {error}") from None


if __name__ == "__main__":
    sys.exit(main())
