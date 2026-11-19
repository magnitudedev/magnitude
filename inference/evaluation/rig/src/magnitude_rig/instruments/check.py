"""`check`: does it compile and pass clippy on this backend's host, including backend-only code?

It runs `cargo clippy`, the same compile as `cargo check` with the lints the workspace
denies (`[workspace.lints]` in inference/Cargo.toml), which the release also runs."""

import argparse

from ..engine import Step
from . import Answer, Instrument, JobResult, Plan


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("cargo", nargs=argparse.REMAINDER,
                        help="cargo clippy arguments after --, default --workspace --all-targets")


def _arguments(arguments) -> list[str]:
    given = arguments.cargo[1:] if arguments.cargo[:1] == ["--"] else arguments.cargo
    return given or ["--workspace", "--all-targets"]


def plan(arguments, source, host) -> Plan:
    argv = ("cargo", "clippy", "--message-format=short", *_arguments(arguments))
    return Plan(builds=(Step("check", argv),), jobs=(), device=None)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    return Answer((f"compiles and passes clippy: cargo clippy {' '.join(_arguments(arguments))} ({arguments.backend})",), True)


INSTRUMENT = Instrument(
    name="check",
    kind="correctness",
    question="Does it compile and pass clippy, including code only this backend's host builds?",
    observable="compile and clippy result",
    selectors=("backend", "host", "build", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
