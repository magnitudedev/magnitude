"""The controlling side of a comparison: inputs, builds and one remote job per model.

The job runs under the host's lock (or locally): it mirrors the working tree and each ref
side's tree, builds both harnesses and the Python environment, acquires the model, passes the
thermal gate and then runs `magnitude-compare host`, which measures and judges everything on
the host. Fixture sets are never built here: the prebuilt sets (`magnitude-fixtures build`)
travel with the plan, recorded by digest, so their renderings match the hosts' prebuilt
references.
"""

import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from magnitude_references.digest import sha256_file
from magnitude_remote.flow import Job, Outcome, Tree, execute
from magnitude_remote.host.gate import GateSettings
from magnitude_remote.hosts import Hosts

from . import builds, report
from .plan import SIDES, Plan
from .runner import REPORT, RESULT
from .steps import TOOLS, HostPaths, schedule

INPUTS = "compare-inputs"
PROJECT = "{src}/evaluation"
UV_ENVIRONMENT = (("UV_PROJECT_ENVIRONMENT", "{root}/venv"),)


@dataclass(frozen=True)
class Prepared:
    plan: Plan
    trees: tuple[Tree, ...]


def target(plan: Plan, side: str) -> str:
    """The side's cargo target directory on the host (a placeholder string)."""
    return (
        "{target}" if plan.builds[side].worktree else f"{{root}}/targets/{builds.tree_name(side)}"
    )


def host_paths(plan: Plan) -> HostPaths:
    """The host paths as the job's placeholders, filled in on the host."""
    return HostPaths(
        inputs=f"{{tree:{INPUTS}}}",
        run_dir="{run_dir}",
        artifacts="{artifacts}",
        root="{root}",
        targets={side: target(plan, side) for side in SIDES},
    )


def build_commands(plan: Plan) -> tuple[tuple[str, ...], ...]:
    features = ("--features", "pinned-tuning") if plan.tuning.mode == "replay-base" else ()
    commands: list[tuple[str, ...]] = [
        ("uv", "sync", "--frozen", "--project", PROJECT, "--package", "magnitude-compare"),
    ]
    for side in SIDES:
        manifest = (
            ()
            if plan.builds[side].worktree
            else ("--manifest-path", f"{{tree:{builds.tree_name(side)}}}/Cargo.toml")
        )
        commands.append(
            (
                "cargo",
                "build",
                "--release",
                *manifest,
                "-p",
                "magnitude-harness",
                *features,
                "--target-dir",
                target(plan, side),
            )
        )
    return tuple(commands)


def job(plan: Plan, model: str, trees: tuple[Tree, ...], gate: GateSettings | None) -> Job:
    paths = host_paths(plan)
    command = (
        "uv",
        "run",
        "--frozen",
        "--project",
        PROJECT,
        "--package",
        "magnitude-compare",
        "magnitude-compare",
        "host",
        "--inputs",
        paths.inputs,
        "--run-dir",
        paths.run_dir,
        "--artifacts",
        paths.artifacts,
        "--root",
        paths.root,
        "--base-target",
        paths.targets["base"],
        "--candidate-target",
        paths.targets["candidate"],
    )
    return Job(
        tool="compare",
        command=command,
        build=build_commands(plan),
        model=model,
        gate=gate,
        trees=trees,
        env=UV_ENVIRONMENT,
    )


class FixturesUnavailable(Exception):
    """A requested fixture set is not in the fixture directory."""


def fixture_sets(directory: Path, families: tuple[str, ...]) -> dict[str, str]:
    """Each family's prebuilt set (`<directory>/<family>.jsonl`) and its digest."""
    missing = [family for family in families if not (directory / f"{family}.jsonl").is_file()]
    if missing:
        raise FixturesUnavailable(
            f"no fixture sets for {', '.join(missing)} in {directory}; build them with "
            "`magnitude-fixtures build <family> --output <dir>/<family>.jsonl`"
        )
    return {family: sha256_file(directory / f"{family}.jsonl") for family in families}


def prepare(plan: Plan, root: Path) -> Prepared:
    """Write the inputs (plan, build identities, the plan's fixture sets) and materialize the
    ref sides."""
    trees = [
        builds.materialize(side, plan.builds[side], root)
        for side in SIDES
        if not plan.builds[side].worktree
    ]
    inputs = root / "trees" / INPUTS
    if inputs.exists():
        shutil.rmtree(inputs)
    inputs.mkdir(parents=True)
    plan.write(inputs / "plan.json")
    identities = {
        **{side: plan.builds[side].identity for side in SIDES},
        TOOLS: plan.tools,
    }
    for name, identity in identities.items():
        (inputs / f"build-{name}.json").write_text(json.dumps(identity.to_json()) + "\n")
    for family, digest in plan.fixture_sets.items():
        shipped = inputs / f"{family}.jsonl"
        shutil.copyfile(Path(plan.fixtures) / f"{family}.jsonl", shipped)
        if sha256_file(shipped) != digest:
            raise FixturesUnavailable(f"the {family} fixture set changed after it was planned")
    paths = tuple(sorted(path.name for path in inputs.iterdir()))
    return Prepared(plan, (*trees, Tree(INPUTS, inputs, paths)))


def describe(
    plan: Plan, models: tuple[str, ...], host: str | None, gate: GateSettings | None
) -> str:
    """The planned jobs and each job's host schedule, without running anything."""
    trees = tuple(
        Tree(
            builds.tree_name(side), Path(f"<git archive {plan.builds[side].commit}:inference>"), ()
        )
        for side in SIDES
        if not plan.builds[side].worktree
    ) + (Tree(INPUTS, Path("<plan, build identities, fixture sets>"), ()),)
    lines = [
        f"plan: {json.dumps(plan.to_json(), indent=2)}",
        "",
    ]
    for model in models:
        planned = job(plan, model, trees, gate)
        lines += [
            f"job: tool {planned.tool}, model {planned.model}, on {host or 'this machine'}",
            f"  trees: {', '.join(f'{tree.name} <- {tree.local}' for tree in planned.trees)}",
            f"  env: {' '.join(f'{name}={value}' for name, value in planned.env)}",
            *(f"  build: {shlex.join(command)}" for command in planned.build),
            f"  acquire: {model} into {{models}}",
            f"  gate: {'off' if gate is None else gate}",
            f"  job: {shlex.join(planned.command)}",
            "  host schedule:",
        ]
        scheduled = schedule(plan, host_paths(plan))
        for run in scheduled.speed:
            lines.append(f"    {run.name}: {shlex.join(run.argv)}")
        for cell in scheduled.quality:
            for run in (cell.runs["base"], cell.render, cell.runs["candidate"]):
                lines.append(f"    {run.name}: {shlex.join(run.argv)}")
            lines.append(
                f"    {cell.name}: compare renderings (a finding); reference from the base "
                "rendering (cached under {root}/references); KL base/candidate vs reference "
                "and candidate vs base"
            )
        lines.append("")
    return "\n".join(lines)


def finish(outcome: Outcome) -> str | None:
    """Re-render the fetched report (the record now holds the host facts); None without a record."""
    result = outcome.results / RESULT
    if not result.is_file():
        return None
    text = report.render(json.loads(result.read_text()))
    (outcome.results / REPORT).write_text(text)
    return text


def run(
    prepared: Prepared, model: str, host: str | None, gate: GateSettings | None, hosts: Hosts
) -> Outcome:
    return execute(job(prepared.plan, model, prepared.trees, gate), host, hosts=hosts)
