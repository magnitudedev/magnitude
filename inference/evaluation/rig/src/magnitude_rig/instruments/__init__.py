"""The registry: every instrument, the one thing it observes and the question it answers.

An instrument is defined by its observable; what it is applied to (a backend, a model, an
entry, a shape, a context, a fixture, a build) is a selector. A question whose observable an
instrument already owns becomes a selector, an option or a case of that instrument, never a
second instrument. Every instrument is a thin front end over the shared cores: the engine
(placement, holds, build, records), the selectors, and the measurement code it runs on the
host. `mrig --help` is generated from `REGISTRY`.

An instrument plans one source's work (its build steps and its job) on a candidate host and
answers from the job's results. `ab` runs any instrument that reports measures over two builds, alternating.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..engine import NOTE, Headers, Step, Weights
from ..sources import Source

# The packages of the evaluation workspace (`evaluation/pyproject.toml`'s members).
WORKSPACE = ("magnitude-fixtures", "magnitude-references", "magnitude-bench", "magnitude-rig")


def environment(name: str, *selection: str) -> Step:
    """The build step that syncs the side's environment (`{venv}`, in its build output) from the
    evaluation workspace's lock, `selection` choosing what (`--package P`, `--all-packages`).
    Its packages are installed, not editable, so the environment reads nothing of the checkout
    once the side releases it, and every workspace package is reinstalled, built from the tree
    being built: uv keys a local package's built wheel by its project file alone, so it would
    otherwise install the wheel some other tree built."""
    reinstall = tuple(argument for package in WORKSPACE for argument in ("--reinstall-package", package))
    return Step(name, ("uv", "sync", "--frozen", "--no-editable", *reinstall, "--project", "{src}/evaluation",
                       *selection))


@dataclass(frozen=True)
class Measure:
    """One measured quantity, with what the instrument could resolve at its budget."""

    value: float
    unit: str
    resolution: float | None  # half-width in `unit`; None when the value is exact
    lower_is_better: bool


@dataclass(frozen=True)
class Answer:
    lines: tuple[str, ...]
    ok: bool
    measures: dict[str, Measure] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """One source's work: build steps under the build hold, then its jobs, in order."""

    builds: tuple[Step, ...]
    jobs: tuple[tuple[str, Step], ...]  # (run directory name, step)
    device: str | None  # the jobs' hold: "device", "timed", or None for host work
    acquires: Weights | Headers | None = None  # what the jobs read of the catalog
    files: tuple[Path, ...] = ()  # local files the job reads, as `{inputs}/<name>`


@dataclass(frozen=True)
class JobResult:
    """What a job left: its fetched run directory, its output log and its exit code."""

    directory: Path
    log: Path
    code: int

    def output(self) -> list[str]:
        """The job's own output lines (the rig's notes in the same log left out)."""
        lines = self.log.read_text(errors="replace").splitlines() if self.log.exists() else []
        return [line for line in lines if not line.startswith(NOTE)]


def lacking(record: dict, *paths: str) -> list[str]:
    """The dotted `paths` (`detail.window.wall_ms_per_step`) a harness record lacks. Each side
    runs its own harness, versioned with its engine (`sources.py`), so one build's record may
    lack a field another's has: an answer requires the fields that are its measurement and
    names the ones a build lacks, and renders an optional section only when it lacks none of
    that section's fields, saying so otherwise."""
    missing = []
    for path in paths:
        value = record
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                missing.append(path)
                break
            value = value[key]
    return missing


def unrecorded(name: str, missing: list[str]) -> str:
    return f"  {name} not recorded by this build's harness (no {', '.join(missing)})"


@dataclass(frozen=True)
class Pairing:
    """How an instrument compares two builds itself, when alternating its own jobs is not the
    comparison (`kl` measures the candidate on the base's tokens and judges the pair)."""

    # (arguments, base side label, candidate side label) -> (side, run name, step) in order
    jobs: Callable[[argparse.Namespace, str, str], tuple[tuple[str, str, Step], ...]]
    answer: Callable[[argparse.Namespace, dict[str, "JobResult"]], Answer]


KINDS = {
    "correctness": "does it build, and is it correct",
    "fidelity": "does the engine compute the model as released",
    "speed": "how fast: one kernel entry, one model step, serving",
    "memory": "what a load holds of the device",
    "decision": "what the product plans and predicts",
    "hardware": "what the device is",
    "anything": "a question no instrument answers yet",
}


@dataclass(frozen=True)
class Instrument:
    name: str
    kind: str  # one of KINDS: the kind of question it answers, for `mrig --help`
    question: str
    observable: str
    selectors: tuple[str, ...]  # shared selectors it takes (`selectors.SELECTORS`)
    configure: Callable[[argparse.ArgumentParser], None]
    # (arguments, source, host alias) -> its work on that host, planned once the host is a
    # candidate; raises `engine.Refused` for a host whose fact it reads is not recorded
    plan: Callable[[argparse.Namespace, Source, str], Plan]
    # from each job's result by name; an instrument with no jobs gets its build's, as "build"
    answer: Callable[[argparse.Namespace, dict[str, "JobResult"]], Answer]
    comparable: bool = False  # reports measures `ab` can compare across alternating rounds
    # Under `ab`, one side's jobs in one round, when they differ from its plan's: (arguments,
    # side, round) -> (name, step). `step` pins every round to the base's first tuning.
    alternate: Callable[[argparse.Namespace, str, int], tuple[tuple[str, Step], ...]] | None = None
    pairing: Pairing | None = None  # compares two builds its own way under `ab`


def _registry() -> dict[str, Instrument]:
    from . import assess, bench, check, device, draft, kernel, kl, memory, script, served, step, test

    found = (check, test, kl, draft, kernel, step, bench, served, memory, assess, device, script)
    return {module.INSTRUMENT.name: module.INSTRUMENT for module in found}


REGISTRY = _registry()
