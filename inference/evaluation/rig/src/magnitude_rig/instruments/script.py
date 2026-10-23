"""`script`: anything no instrument answers. Runs a local file on a host of the backend, with
the checkout's warm build, the caches, a run directory and a record. A question a script
answers a second time becomes an instrument, or a case or option of one.

The file runs by its `#!` line (`sh` without one), from the checkout's source root, with
`MRIG_SRC`, `MRIG_TARGET`, `MRIG_VENV`, `MRIG_RUN` (its run directory, fetched back),
`MRIG_MODELS`, `MRIG_ROOT` and, with `--model`, `MRIG_ARTIFACTS` (the acquired model's paths).
"""

import argparse
import base64
import shlex
from pathlib import Path

from .. import selectors
from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Plan

RUNNER = (
    "import base64, os, stat, sys; d = os.environ['MRIG_RUN']; os.makedirs(d, exist_ok=True); "
    "p = os.path.join(d, 'script'); open(p, 'wb').write(base64.b64decode(sys.argv[1])); "
    "os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR); "
    "first = open(p, 'rb').readline(); "
    "os.execv(p, [p]) if first.startswith(b'#!') else os.execvp('sh', ['sh', p])"
)
ENV = (
    ("MRIG_SRC", "{src}"), ("MRIG_TARGET", "{target}"), ("MRIG_VENV", "{venv}"),
    ("MRIG_RUN", "{run_dir}"), ("MRIG_MODELS", "{models}"), ("MRIG_ROOT", "{root}"),
    ("MRIG_ARTIFACTS", "{artifacts}"),
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", type=Path, help="the script to run on the host")
    parser.add_argument("--hold", choices=("device", "timed", "build"), default="device",
                        help="what the script needs from the host (default device)")
    parser.add_argument("--model", type=selectors.model, help="catalog model to acquire first")
    parser.add_argument("--prepare", action="append", default=[], type=shlex.split,
                        help="a build command to run first under the build hold (repeatable), "
                             "e.g. 'cargo build --release -p magnitude-harness'")


def plan(arguments, source) -> Plan:
    if not arguments.file.is_file():
        raise SystemExit(f"script: {arguments.file} is not a file")
    content = base64.b64encode(arguments.file.read_bytes()).decode()
    builds = tuple(Step("prepare", tuple(argv)) for argv in arguments.prepare)
    job = Step("script", ("{python}", "-c", RUNNER, content), ENV)
    device = None if arguments.hold == "build" else arguments.hold
    acquires = None if arguments.model is None else Weights(arguments.model)
    return Plan(builds=builds, jobs=(("script", job),), device=device, acquires=acquires)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["script"]
    lines = result.output()
    shown = lines[-40:]
    head = f"script {arguments.file.name} exited {result.code}" + (
        f" (last {len(shown)} of {len(lines)} lines)" if len(lines) > len(shown) else ""
    )
    return Answer((head, *shown), result.code == 0)


INSTRUMENT = Instrument(
    name="script",
    kind="anything",
    question="Anything no instrument answers yet (then make it one if it recurs)",
    observable="a script's own output",
    selectors=("backend", "host", "build", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
