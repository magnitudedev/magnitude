"""magnitude-remote: run an evaluation job on a host (or locally), or inspect host locks.

    magnitude-remote run --tool harness.time [--host POOL] [--model qwen3.5-4b:gguf:q4] \\
        [--build "cargo build --release -p magnitude-harness"] [--no-gate] -- \\
        {target}/release/magnitude-harness time --artifacts {artifacts} --output {run_dir} \\
        --build-identity {build_file}
    magnitude-remote probe [--host ALIAS] [--seconds 30] [--gate]
    magnitude-remote status [ALIAS|POOL ...]
    magnitude-remote setup ALIAS|POOL     # once per host (and after instance storage resets)
    magnitude-remote build               # {"commit", "dirty_digest"} of the working tree
    magnitude-remote hosts
"""

import argparse
import json
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

from .build import identify
from .connection import HostUnavailable, connect, holder_status, setup
from .flow import Job, StepFailed, execute
from .host.gate import GateSettings
from .hosts import Hosts


def _gate_arguments(parser: argparse.ArgumentParser) -> None:
    for f in fields(GateSettings):
        parser.add_argument(
            f"--gate-{f.name.replace('_', '-')}", dest=f"gate_{f.name}", type=float, default=f.default
        )


def _gate(arguments, enabled: bool) -> GateSettings | None:
    if not enabled:
        return None
    return GateSettings(**{f.name: getattr(arguments, f"gate_{f.name}") for f in fields(GateSettings)})


def _finish(job: Job, host: str | None) -> int:
    try:
        outcome = execute(job, host)
    except (StepFailed, HostUnavailable) as error:
        print(f"magnitude-remote: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": outcome.run_id, "host": outcome.host,
                      "results": str(outcome.results), "code": outcome.code}))
    return outcome.code


def _run(arguments) -> int:
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        raise SystemExit("run needs a job command after --")
    job = Job(
        tool=arguments.tool,
        command=tuple(command),
        build=tuple(tuple(shlex.split(b)) for b in arguments.build),
        model=arguments.model,
        gate=_gate(arguments, not arguments.no_gate),
        max_hold_seconds=arguments.max_hold,
    )
    return _finish(job, arguments.host)


def _probe(arguments) -> int:
    sleep = f"import time; time.sleep({arguments.seconds})"
    job = Job(tool="probe", command=("{python}", "-c", sleep), gate=_gate(arguments, arguments.gate))
    return _finish(replace(job, max_hold_seconds=arguments.seconds + 3600), arguments.host)


def _status(arguments) -> int:
    hosts = Hosts.load()
    configs = [c for name in (arguments.names or hosts.known()) for c in hosts.resolve(name)]

    def one(config):
        try:
            remote = connect(config)
            return {"host": config.alias, "system": f"{remote.system} {remote.machine}", **holder_status(remote)}
        except HostUnavailable as error:
            return {"host": config.alias, "error": str(error)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(one, configs):
            print(json.dumps(row))
    return 0


def _setup(arguments) -> int:
    hosts = Hosts.load()
    failed = 0
    for config in (c for name in arguments.names for c in hosts.resolve(name)):
        try:
            print(f"{config.alias}:\n{setup(config).rstrip()}")
        except HostUnavailable as error:
            print(f"magnitude-remote: {error}", file=sys.stderr)
            failed = 1
    return failed


def _build(arguments) -> int:
    print(json.dumps(identify().to_json()))
    return 0


def _hosts(arguments) -> int:
    hosts = Hosts.load()
    print(json.dumps({"defaults": {"root": hosts.defaults.root, "models": hosts.defaults.models},
                      "pools": hosts.pools,
                      "hosts": {a: {"root": h.root, "models": h.models} for a, h in hosts.hosts.items()}},
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="magnitude-remote", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)

    run = commands.add_parser("run", help="run a job under a host's lock, or locally")
    run.add_argument("--tool", required=True, help="result tool name, e.g. harness.time")
    run.add_argument("--host", help="ssh alias or pool; omitted runs locally")
    run.add_argument("--model", help="catalog reference <model>:<variant> to acquire")
    run.add_argument("--build", action="append", default=[], help="build command (repeatable)")
    run.add_argument("--no-gate", action="store_true", help="skip the thermal gate")
    run.add_argument("--max-hold", type=float, default=Job.max_hold_seconds, help="seconds")
    _gate_arguments(run)
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=_run)

    probe = commands.add_parser("probe", help="record host facts and a thermal trace")
    probe.add_argument("--host")
    probe.add_argument("--seconds", type=float, default=30.0)
    probe.add_argument("--gate", action="store_true", help="pass the thermal gate first")
    _gate_arguments(probe)
    probe.set_defaults(handler=_probe)

    status = commands.add_parser("status", help="show lock holders")
    status.add_argument("names", nargs="*", help="hosts or pools (default: every listed host)")
    status.set_defaults(handler=_status)

    setup_hosts = commands.add_parser(
        "setup", help="one-time: create cache root and model store (sudo where needed)"
    )
    setup_hosts.add_argument("names", nargs="+", help="hosts or pools")
    setup_hosts.set_defaults(handler=_setup)

    commands.add_parser("build", help="print the working tree's build identity").set_defaults(
        handler=_build
    )
    commands.add_parser("hosts", help="show the hosts file").set_defaults(handler=_hosts)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
