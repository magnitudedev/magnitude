"""The harness runs of a comparison, in execution order.

Speed runs come first, right after the thermal gate: `rounds` rounds, each running every
family on both builds, base first on even rounds and candidate first on odd ones, so drift
such as heat affects both sides alike. Quality runs follow, per family and context:
- the base's `logits`, which renders the fixtures (the base rendering is the cell's input);
- the candidate's `render` at the same context, to detect a rendering change;
- the candidate's `logits` on the base's exact tokens (`--rendered`), so its quality is
  measurable on identical input even when it renders differently.

The first engine run of a side tunes afresh; how later runs tune is the plan's tuning mode.
"""

import os
from dataclasses import dataclass
from typing import Literal

from .plan import SIDES, Plan, Side

PINS = "tuning-pins.json"
RENDERED = "rendered.jsonl"
TOOLS = "tools"  # the build identity of the measurement tools' tree


@dataclass(frozen=True)
class HostPaths:
    """Where a comparison's files are on the executing host."""

    inputs: str
    run_dir: str
    artifacts: str
    root: str
    targets: dict[str, str]  # side -> cargo target directory of its harness build
    executable_suffix: str = ".exe" if os.name == "nt" else ""

    def harness(self, side: Side) -> str:
        return f"{self.targets[side]}/release/magnitude-harness{self.executable_suffix}"

    def kernels(self, side: Side) -> str:
        """The side's engine cache (formed kernels and tuning results), kept across comparisons."""
        return f"{self.root}/compare/kernels/{side}"

    def identity(self, name: str) -> str:
        """A build identity file: a side's, or the tools' (`TOOLS`)."""
        return f"{self.inputs}/build-{name}.json"

    def fixtures(self, family: str) -> str:
        return f"{self.inputs}/{family}.jsonl"

    @property
    def references(self) -> str:
        return f"{self.root}/references"

    def run(self, name: str) -> str:
        return f"{self.run_dir}/{name}"


@dataclass(frozen=True)
class EngineRun:
    name: str  # run directory, relative to the comparison's
    kind: Literal["time", "logits", "render"]
    side: Side
    family: str
    context: int | None  # logits and render only
    round: int | None  # time only
    argv: tuple[str, ...]

    @property
    def tunes(self) -> bool:
        """Whether the run loads the engine (and so records tuning)."""
        return self.kind != "render"


@dataclass(frozen=True)
class QualityCell:
    family: str
    context: int
    runs: dict[str, EngineRun]  # side -> its logits run
    render: EngineRun  # the candidate's own rendering

    @property
    def name(self) -> str:
        return f"quality/{self.family}-{self.context}"


@dataclass(frozen=True)
class Schedule:
    speed: tuple[EngineRun, ...]
    quality: tuple[QualityCell, ...]


def _order(round_index: int) -> tuple[Side, ...]:
    return SIDES if round_index % 2 == 0 else SIDES[::-1]


class _Tuner:
    """The tuning arguments of each engine run, in execution order."""

    def __init__(self, plan: Plan, paths: HostPaths):
        self.plan = plan
        self.pins = paths.run(PINS)
        self.tuned: set[str] = set()

    def arguments(self, side: Side) -> list[str]:
        budget = ["--tuning-budget", f"{self.plan.tuning.seconds:g}"]
        first = side not in self.tuned
        self.tuned.add(side)
        if self.plan.tuning.mode == "fresh":
            return [*budget, "--fresh-tuning"] if first else budget
        if side == "base" and first:
            return [*budget, "--fresh-tuning", "--tuning-record", self.pins]
        if "base" not in self.tuned:
            raise ValueError("replay-base needs the base's run before the candidate's")
        return [*budget, "--tuning-replay", self.pins]


def schedule(plan: Plan, paths: HostPaths) -> Schedule:
    tuner = _Tuner(plan, paths)

    def inputs(side: Side, family: str, name: str) -> list[str]:
        return [
            "--artifacts", paths.artifacts,
            "--fixtures", paths.fixtures(family),
            "--output", paths.run(name),
            "--build-identity", paths.identity(side),
        ]  # fmt: skip

    def engine(side: Side, family: str, name: str) -> list[str]:
        return [*inputs(side, family, name), "--cache-dir", paths.kernels(side)]

    speed = []
    if plan.speed:
        for index in range(plan.rounds):
            for family in plan.families:
                for side in _order(index):
                    name = f"speed/r{index}-{family}-{side}"
                    argv = [
                        paths.harness(side), "time", *engine(side, family, name),
                        "--contexts", ",".join(map(str, plan.contexts)),
                        "--concurrency", ",".join(map(str, plan.concurrency)),
                        *tuner.arguments(side),
                    ]  # fmt: skip
                    speed.append(EngineRun(name, "time", side, family, None, index, tuple(argv)))
    quality = []
    if plan.quality:
        for family in plan.families:
            for context in plan.contexts:
                cell = f"quality/{family}-{context}"
                base = f"{cell}/base"
                base_argv = [
                    paths.harness("base"), "logits", *engine("base", family, base),
                    "--context", str(context), *tuner.arguments("base"),
                ]  # fmt: skip
                render = f"{cell}/candidate-render"
                render_argv = [
                    paths.harness("candidate"), "render", *inputs("candidate", family, render),
                    "--context", str(context),
                ]  # fmt: skip
                candidate = f"{cell}/candidate"
                candidate_argv = [
                    paths.harness("candidate"), "logits", *engine("candidate", family, candidate),
                    "--rendered", f"{paths.run(base)}/{RENDERED}",
                    *tuner.arguments("candidate"),
                ]  # fmt: skip
                runs = {
                    "base": EngineRun(
                        base, "logits", "base", family, context, None, tuple(base_argv)
                    ),
                    "candidate": EngineRun(
                        candidate,
                        "logits",
                        "candidate",
                        family,
                        context,
                        None,
                        tuple(candidate_argv),
                    ),
                }
                rendering = EngineRun(
                    render, "render", "candidate", family, context, None, tuple(render_argv)
                )
                quality.append(QualityCell(family, context, runs, rendering))
    return Schedule(tuple(speed), tuple(quality))
