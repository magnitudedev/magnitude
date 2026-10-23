"""`check`: does it compile on this backend's host, including backend-only code?"""

import argparse

from ..engine import Step
from . import Answer, Instrument, JobResult, Plan


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("cargo", nargs=argparse.REMAINDER,
                        help="cargo check arguments after --, default --workspace --all-targets")


def _arguments(arguments) -> list[str]:
    given = arguments.cargo[1:] if arguments.cargo[:1] == ["--"] else arguments.cargo
    return given or ["--workspace", "--all-targets"]


def plan(arguments, source) -> Plan:
    argv = ("cargo", "check", "--message-format=short", *_arguments(arguments))
    return Plan(builds=(Step("check", argv),), jobs=(), device=None)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    return Answer((f"compiles: cargo check {' '.join(_arguments(arguments))} ({arguments.backend})",), True)


INSTRUMENT = Instrument(
    name="check",
    kind="correctness",
    question="Does it compile, including code only this backend's host builds?",
    observable="compile result",
    selectors=("backend", "host", "build", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
