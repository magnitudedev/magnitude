"""Host-side commands, run by the rig on the evaluated host:

    python3 -m magnitude_rig.host facts [--alias A]
    python3 -m magnitude_rig.host probe
    python3 -m magnitude_rig.host watch --root ROOT [--alias A] [--interval S]
    python3 -m magnitude_rig.host gate --log PATH [settings]
    python3 -m magnitude_rig.host record --run-dir DIR [--alias A] -- ARGV...
    python3 -m magnitude_rig.host fresh --artifacts FILE --output FILE --clones DIR -- ARGV...
    python3 -m magnitude_rig.host compiler --root ROOT
    python3 -m magnitude_rig.host provenance --output FILE --source DIR... --target DIR [--tests FILE] -- TEXT...
    python3 -m magnitude_rig.host header-artifacts --export DIR --model REF --output FILE
    python3 -m magnitude_rig.host tests --list FILE -- ARGS...

`record` writes `host.json`, records `thermal.jsonl` around the job, and completes the
job's `result.json` (when it wrote one) with the host facts and the thermal trace path.
`watch` streams JSON Lines until stdin closes: the host facts once (`{"facts": ...}`), then
every interval the agent's status of the rig at ROOT with a probe reading and its group
aggregates (`{"at", "status", "reading", "groups_c"}`).
`compiler` installs the recording rustc wrapper and prints its path; `provenance` writes the
build provenance of the cargo-built executables the job's texts name and the test executables
the build produced (`--tests`, which the suite's jobs run) and exits 1 unless each is verified,
and withdraws every other executable of the target it cannot verify (see `provenance.py`).
`fresh` runs a job that reads its model from storage (see `fresh.py`). `tests` runs a cargo suite's test executables from a build output instead of
cargo: each one FILE lists (`host/agent.py`), with ARGS, from its package's directory with
`CARGO_MANIFEST_DIR` naming it, as cargo runs them; it exits with the first failure's code, 0
when every one passed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

from . import agent, facts, fresh, gate, provenance, trace
from .probes import open_probe

GATE_TIMEOUT_EXIT = 3


def _compiler(arguments) -> int:
    print(provenance.install(arguments.root))
    return 0


def _provenance(arguments) -> int:
    tests = json.loads(Path(arguments.tests).read_text()) if arguments.tests is not None else []
    record = provenance.verify(arguments.texts, arguments.source, arguments.target, tests)
    trace.atomic(Path(arguments.output), record)
    for executable in record["executables"]:
        for reason in executable["reasons"]:
            sys.stderr.write(f"{executable['path']}: {executable['status']}: {reason}\n")
    summary = ", ".join(f"{e['path']} {e['status']}" for e in record["executables"])
    print(f"{record['status']}: {summary}" if summary else record["status"])
    for executable in record["withdrawn"]:
        print(f"withdrew {executable['path']} ({executable['status']}: no job runs it until a build "
              f"of this source produces it)")
    return 1 if record["status"] == "refused" else 0


def _tests(arguments) -> int:
    failed = 0
    for test in json.loads(Path(arguments.list).read_text()):
        sys.stdout.write("     Running %s\n" % test["executable"])
        sys.stdout.flush()
        code = subprocess.run([test["executable"], *arguments.argv], cwd=test["directory"],
                              env=dict(os.environ, CARGO_MANIFEST_DIR=test["directory"])).returncode
        # A test binary a signal ended exits as a shell reports it, 128 + the signal: an exit
        # status never negative, and never 0.
        failed = failed or (128 - code if code < 0 else code)
    return failed


def _header_artifacts(arguments) -> int:
    """The acquisition object of a catalog model's headers in a header export
    (`catalog-tool headers`): its target's and drafter's header files, which open as the
    files they were read from, with zero weights."""
    export = Path(arguments.export)
    index = json.loads((export / "index.json").read_text())

    def component(role):
        for entry in index["components"]:
            if entry["model"] == arguments.model and entry["role"] == role:
                return str(export / entry["repository"] / entry["revision"] / entry["path"])
        return None

    target = component("target")
    if target is None:
        sys.stderr.write("the header export has no target %s\n" % arguments.model)
        return 1
    draft = component("draft")
    artifacts = {
        "model": arguments.model,
        "target": target,
        "target_sha256": "headers-only",
        "drafter": {"path": draft} if draft else None,
        "projector": None,
    }
    trace.atomic(Path(arguments.output), artifacts)
    return 0


def _facts(arguments) -> int:
    print(json.dumps(facts.capture(arguments.alias).to_json(), indent=2))
    return 0


def _probe(arguments) -> int:
    probe = open_probe()
    try:
        time.sleep(1.0)
        reading, at, started, ended = trace.timed_read(probe, None)
        line = trace.sample_line(reading, at, 0.0, "probe", ended - started)
        line["source"] = probe.source
        print(json.dumps(line, indent=2))
    finally:
        probe.close()
    return 0


def _watch(arguments) -> int:
    closed = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), closed.set()), daemon=True).start()
    print(json.dumps({"facts": facts.capture(arguments.alias).to_json()}), flush=True)
    probe, unavailable = None, None
    try:
        probe = open_probe()
    except Exception as error:
        unavailable = f"{type(error).__name__}: {error}"
    try:
        while not closed.wait(arguments.interval):
            reading, at, _, _ = trace.timed_read(probe, unavailable)
            sample = {
                "at": at,
                "status": agent.report(arguments.root, []),
                "reading": reading.to_json(),
                "groups_c": trace.group_aggregates(reading),
            }
            print(json.dumps(sample), flush=True)
    finally:
        if probe is not None:
            probe.close()
    return 0


def _gate(arguments) -> int:
    settings = gate.GateSettings(
        **{f.name: getattr(arguments, f.name) for f in fields(gate.GateSettings)}
    )
    log = Path(arguments.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    # The machine's CPU signals are the run's own only where its CPUs all are.
    cpu_own = facts.cpus_allowed() >= os.cpu_count()
    probe = open_probe()
    try:
        verdict = gate.wait(probe, settings, log, cpu_own)
    except gate.GateTimeout as timeout:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(timeout),
                    "verdict": asdict(timeout.verdict),
                    "latest": timeout.latest.to_json(),
                }
            )
        )
        return GATE_TIMEOUT_EXIT
    finally:
        probe.close()
    print(json.dumps(asdict(verdict)))
    return 0


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


# How often a recorded job looks whether its runner still runs.
PARENT_POLL_SECONDS = 1.0


def _record(arguments) -> int:
    """Run the job recorded with the host's facts and thermal trace. The job lives no longer
    than the runner that started this: once this process's parent is gone (a runner killed
    outright, which ends nothing of its steps), the job is ended and this exits."""
    parent = os.getppid()
    run_dir = Path(arguments.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    host = facts.capture(arguments.alias).to_json()
    trace.atomic(run_dir / "host.json", host)
    signal.signal(signal.SIGTERM, _terminate)
    with trace.ThermalRecorder(run_dir, interval_seconds=arguments.interval):
        child = subprocess.Popen(arguments.argv)
        try:
            while True:
                try:
                    code = child.wait(timeout=PARENT_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if os.getppid() != parent or parent == 1:  # adopted, at once or since
                        raise SystemExit("record: its runner is gone; its job is ended") from None
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    result = run_dir / "result.json"
    if result.exists():
        record = json.loads(result.read_text())
        record["host"] = host
        record["traces"] = {**record.get("traces", {}), "thermal": trace.TRACE}
        trace.atomic(result, record)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="magnitude_rig.host")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("facts")
    command.add_argument("--alias")
    command.set_defaults(handler=_facts)
    commands.add_parser("probe").set_defaults(handler=_probe)
    command = commands.add_parser("watch")
    command.add_argument("--root", required=True)
    command.add_argument("--alias")
    command.add_argument("--interval", type=float, default=2.0)
    command.set_defaults(handler=_watch)
    command = commands.add_parser("gate")
    command.add_argument("--log", required=True)
    for field in fields(gate.GateSettings):
        command.add_argument(f"--{field.name.replace('_', '-')}", type=float, default=field.default)
    command.set_defaults(handler=_gate)
    command = commands.add_parser("record")
    command.add_argument("--run-dir", required=True)
    command.add_argument("--alias")
    command.add_argument("--interval", type=float, default=1.0)
    command.add_argument("argv", nargs=argparse.REMAINDER)
    command.set_defaults(handler=_record)
    command = commands.add_parser("fresh")
    command.add_argument("--artifacts", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--clones", required=True, type=Path)
    command.add_argument("argv", nargs=argparse.REMAINDER)
    command.set_defaults(handler=lambda arguments: fresh.run(
        arguments.artifacts, arguments.output, arguments.clones, arguments.argv))
    command = commands.add_parser("header-artifacts")
    command.add_argument("--export", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_header_artifacts)
    command = commands.add_parser("tests")
    command.add_argument("--list", required=True)
    command.add_argument("argv", nargs=argparse.REMAINDER)
    command.set_defaults(handler=_tests)
    command = commands.add_parser("compiler")
    command.add_argument("--root", required=True)
    command.set_defaults(handler=_compiler)
    command = commands.add_parser("provenance")
    command.add_argument("--output", required=True)
    command.add_argument("--source", action="append", default=[])
    command.add_argument("--target", required=True)
    command.add_argument("--tests", help="the test executables the build produced, as JSON (`host/agent.py`)")
    command.add_argument("texts", nargs=argparse.REMAINDER)
    command.set_defaults(handler=_provenance)
    arguments = parser.parse_args(argv)
    if arguments.command in ("record", "fresh", "tests") and arguments.argv[:1] == ["--"]:
        arguments.argv = arguments.argv[1:]
    if arguments.command in ("record", "fresh"):
        if not arguments.argv:
            parser.error(f"{arguments.command} needs a job command after --")
    if arguments.command == "provenance" and arguments.texts[:1] == ["--"]:
        arguments.texts = arguments.texts[1:]
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
