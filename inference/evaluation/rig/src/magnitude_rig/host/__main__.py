"""Host-side commands, run by the rig on the evaluated host:

    python3 -m magnitude_rig.host facts [--alias A]
    python3 -m magnitude_rig.host probe
    python3 -m magnitude_rig.host gate --log PATH [settings]
    python3 -m magnitude_rig.host record --run-dir DIR [--alias A] -- ARGV...
    python3 -m magnitude_rig.host compiler --root ROOT
    python3 -m magnitude_rig.host provenance --output FILE --source DIR... -- TEXT...
    python3 -m magnitude_rig.host size --checkout DIR
    python3 -m magnitude_rig.host header-artifacts --export DIR --model REF --output FILE

`record` writes `host.json`, records `thermal.jsonl` around the job, and completes the
job's `result.json` (when it wrote one) with the host facts and the thermal trace path.
`compiler` installs the recording rustc wrapper and prints its path; `provenance` writes the
build provenance of the cargo-built executables the job's texts name and exits 1 unless
each is verified (see `provenance.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from . import facts, gate, provenance, trace
from .probes import open_probe

GATE_TIMEOUT_EXIT = 3


def _compiler(arguments) -> int:
    print(provenance.install(arguments.root))
    return 0


def _provenance(arguments) -> int:
    record = provenance.verify(arguments.texts, arguments.source)
    trace.atomic(Path(arguments.output), record)
    for executable in record["executables"]:
        for reason in executable["reasons"]:
            sys.stderr.write(f"{executable['path']}: {executable['status']}: {reason}\n")
    summary = ", ".join(f"{e['path']} {e['status']}" for e in record["executables"])
    print(f"{record['status']}: {summary}" if summary else record["status"])
    return 1 if record["status"] == "refused" else 0


def _size(arguments) -> int:
    """Record the checkout's size (`<checkout>/size`): eviction keeps the largest free."""
    total = 0
    for directory, _, files in os.walk(arguments.checkout):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    path = Path(arguments.checkout) / "size"
    staged = path.with_suffix(f".{os.getpid()}.tmp")  # runs of one checkout may record at once
    staged.write_text(str(total))
    staged.replace(path)
    print(total)
    return 0


def _header_artifacts(arguments) -> int:
    """The acquisition object of a catalog model's headers in a header export
    (`catalog-tool headers`): its target's and drafter's header files, which open as the
    files they were read from, with zero weights."""
    export = Path(arguments.export)
    index = json.loads((export / "index.json").read_text())
    model, variant = arguments.model.split(":", 1)

    def component(role):
        for entry in index["components"]:
            if entry["model"] == model and entry["role"] == role:
                return str(export / entry["repository"] / entry["revision"] / entry["path"])
        return None

    target = component({"target": {"variant": variant}})
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


def _gate(arguments) -> int:
    settings = gate.GateSettings(
        **{f.name: getattr(arguments, f.name) for f in fields(gate.GateSettings)}
    )
    log = Path(arguments.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    probe = open_probe()
    try:
        verdict = gate.wait(probe, settings, log)
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


def _record(arguments) -> int:
    run_dir = Path(arguments.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    host = facts.capture(arguments.alias).to_json()
    trace.atomic(run_dir / "host.json", host)
    signal.signal(signal.SIGTERM, _terminate)
    with trace.ThermalRecorder(run_dir, interval_seconds=arguments.interval):
        child = subprocess.Popen(arguments.argv)
        try:
            code = child.wait()
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
    command = commands.add_parser("header-artifacts")
    command.add_argument("--export", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_header_artifacts)
    command = commands.add_parser("size")
    command.add_argument("--checkout", required=True)
    command.set_defaults(handler=_size)
    command = commands.add_parser("compiler")
    command.add_argument("--root", required=True)
    command.set_defaults(handler=_compiler)
    command = commands.add_parser("provenance")
    command.add_argument("--output", required=True)
    command.add_argument("--source", action="append", default=[])
    command.add_argument("texts", nargs=argparse.REMAINDER)
    command.set_defaults(handler=_provenance)
    arguments = parser.parse_args(argv)
    if arguments.command == "record":
        if arguments.argv[:1] == ["--"]:
            arguments.argv = arguments.argv[1:]
        if not arguments.argv:
            parser.error("record needs a job command after --")
    if arguments.command == "provenance" and arguments.texts[:1] == ["--"]:
        arguments.texts = arguments.texts[1:]
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
