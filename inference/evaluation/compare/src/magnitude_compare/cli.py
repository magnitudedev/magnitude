"""magnitude-compare: did a change make the engine better or worse?

    magnitude-compare run --base REF [--candidate worktree|REF] --model REF[,REF] \\
        --family tools,prose-repeat [--fixtures DIR] --contexts 4096,16384 [--host ALIAS|POOL] \\
        [--tuning-seconds 60] [--tuning fresh|replay-base] [--rounds 3] \\
        [--quality|--no-quality] [--speed|--no-speed] [--dry-run]
    magnitude-compare report RUN_DIR

`run` measures the base and candidate builds on one host (or locally) under its lock: quality
as KL against a cached reference and between the builds, speed as alternating rounds of the
harness `time`. It writes one result record (tool `compare`) and `report.md` per model and
exits 0 when nothing regressed, 2 on a regression and 1 on any failure.

Fixtures are the prebuilt sets `<family>.jsonl` in `--fixtures` (by default the shared
`evaluation/results/fixture-sets`, which the hosts' prebuilt references are keyed by);
compare never builds fixture sets itself.

`host` is the job the remote flow runs on the host; it is not meant to be run by hand.
"""

import argparse
import json
import sys
from pathlib import Path

from magnitude_fixtures import FAMILIES
from magnitude_remote.connection import HostUnavailable
from magnitude_remote.flow import RESULTS, StepFailed
from magnitude_remote.host.gate import GateSettings
from magnitude_remote.hosts import Hosts

from . import controller, report
from .builds import BuildUnavailable, check_ref, identify, identify_worktree
from .plan import WORKTREE, Plan, Tuning
from .runner import EXIT, RESULT, Runner
from .steps import HostPaths

FIXTURE_SETS = RESULTS / "fixture-sets"


def _list(value: str) -> tuple[str, ...]:
    items = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    if not all(items):
        raise argparse.ArgumentTypeError(f"empty item in {value!r}")
    return items


def _families(value: str) -> tuple[str, ...]:
    families = _list(value)
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown families {unknown}; choose from {', '.join(FAMILIES)}"
        )
    return families


def _counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in _list(value))
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("values must be positive")
    return counts


def _run(arguments, argv: list[str]) -> int:
    if arguments.base == arguments.candidate:
        raise SystemExit("the base and the candidate are the same source")
    if not (arguments.quality or arguments.speed):
        raise SystemExit("nothing to compare: both --no-quality and --no-speed")
    try:
        sides = {"base": identify(arguments.base), "candidate": identify(arguments.candidate)}
    except BuildUnavailable as error:
        print(f"magnitude-compare: {error}", file=sys.stderr)
        return EXIT["failed"]
    for build in sides.values():
        try:
            if not build.worktree:
                check_ref(build)
        except BuildUnavailable as error:
            # A dry run still shows what would run.
            print(f"magnitude-compare: {error}", file=sys.stderr)
            if not arguments.dry_run:
                return EXIT["failed"]
    try:
        fixture_sets = controller.fixture_sets(arguments.fixtures, arguments.family)
    except controller.FixturesUnavailable as error:
        print(f"magnitude-compare: {error}", file=sys.stderr)
        return EXIT["failed"]
    candidate = sides["candidate"]
    plan = Plan(
        builds=sides,
        tools=candidate.identity if candidate.worktree else identify_worktree(),
        fixtures=str(arguments.fixtures.resolve()),
        fixture_sets=fixture_sets,
        contexts=arguments.contexts,
        concurrency=arguments.concurrency,
        rounds=arguments.rounds,
        tuning=Tuning(arguments.tuning_seconds, arguments.tuning),
        quality=arguments.quality,
        speed=arguments.speed,
        quality_bar=arguments.quality_bar,
        speed_bar=arguments.speed_bar,
        command=("magnitude-compare", *argv),
    )
    gate = None if arguments.no_gate else GateSettings()
    if arguments.dry_run:
        print(controller.describe(plan, arguments.model, arguments.host, gate))
        return EXIT["pass"]
    hosts = Hosts.load()
    prepared = controller.prepare(plan, Path(hosts.defaults.root).expanduser())
    code = EXIT["pass"]
    for model in arguments.model:
        try:
            outcome = controller.run(prepared, model, arguments.host, gate, hosts)
        except (StepFailed, HostUnavailable, BuildUnavailable) as error:
            print(f"magnitude-compare: {model}: {error}", file=sys.stderr)
            code = EXIT["failed"]
            continue
        text = controller.finish(outcome)
        if text is None:
            print(
                f"magnitude-compare: {model}: the host job exited {outcome.code} without a result "
                f"({outcome.results})",
                file=sys.stderr,
            )
            code = EXIT["failed"]
            continue
        print(text)
        print(f"magnitude-compare: {model}: {outcome.results}", file=sys.stderr)
        code = _worse(code, outcome.code)
    return code


def _worse(code: int, other: int) -> int:
    """The more severe exit code: a failure (any code the host runner does not define, such as a
    killed job) over a regression over a pass."""
    severity = (EXIT["pass"], EXIT["regressed"], EXIT["failed"])
    return max(code, other if other in severity else EXIT["failed"], key=severity.index)


def _host(arguments) -> int:
    plan = Plan.read(Path(arguments.inputs) / "plan.json")
    paths = HostPaths(
        inputs=arguments.inputs,
        run_dir=arguments.run_dir,
        artifacts=arguments.artifacts,
        root=arguments.root,
        targets={"base": arguments.base_target, "candidate": arguments.candidate_target},
    )
    return Runner(plan, paths).run()


def _report(arguments) -> int:
    print(report.render(json.loads((Path(arguments.run_dir) / RESULT).read_text())))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="magnitude-compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="compare a base and a candidate build")
    run.add_argument("--base", required=True, help="git ref (e.g. HEAD, origin/main) or worktree")
    run.add_argument("--candidate", default=WORKTREE, help="git ref or worktree (default)")
    run.add_argument(
        "--model",
        type=_list,
        required=True,
        help="catalog references <model>:<variant>, comma-separated",
    )
    run.add_argument(
        "--family", type=_families, required=True, help=f"fixture families: {', '.join(FAMILIES)}"
    )
    run.add_argument(
        "--contexts",
        type=_counts,
        default=(4096, 16384),
        help="context lengths (default 4096,16384)",
    )
    run.add_argument(
        "--concurrency", type=_counts, default=(1, 4), help="speed concurrency (default 1,4)"
    )
    run.add_argument(
        "--fixtures",
        type=Path,
        default=FIXTURE_SETS,
        help="directory of prebuilt fixture sets <family>.jsonl (magnitude-fixtures build); "
        f"default {FIXTURE_SETS}",
    )
    run.add_argument("--host", help="ssh alias or pool; omitted runs locally")
    run.add_argument(
        "--tuning-seconds", type=float, default=60.0, help="tuning budget per side (default 60)"
    )
    run.add_argument(
        "--tuning",
        choices=("fresh", "replay-base"),
        default="fresh",
        help="fresh: each side tunes afresh with the same budget; replay-base: the candidate replays "
        "the base's tuned configurations, comparing kernel code alone (builds with pinned-tuning)",
    )
    run.add_argument("--rounds", type=int, default=3, help="alternating speed rounds (default 3)")
    run.add_argument("--quality", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--speed", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--quality-bar",
        type=float,
        default=0.02,
        help="relative change a quality metric must exceed (default 0.02)",
    )
    run.add_argument(
        "--speed-bar",
        type=float,
        default=0.0,
        help="relative change a speed cell must exceed beyond its noise (default 0)",
    )
    run.add_argument("--no-gate", action="store_true", help="skip the thermal gate")
    run.add_argument(
        "--dry-run", action="store_true", help="print the planned jobs and host schedule only"
    )
    run.set_defaults(handler=lambda arguments: _run(arguments, argv))

    host = commands.add_parser("host", help="(the remote job) measure and judge on this host")
    for name in ("inputs", "run-dir", "artifacts", "root", "base-target", "candidate-target"):
        host.add_argument(f"--{name}", required=True)
    host.set_defaults(handler=_host)

    rendered = commands.add_parser(
        "report", help="print the report of a comparison's run directory"
    )
    rendered.add_argument("run_dir")
    rendered.set_defaults(handler=_report)

    arguments = parser.parse_args(argv)
    if arguments.command == "run" and arguments.rounds < 1:
        parser.error("--rounds must be at least 1")
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
