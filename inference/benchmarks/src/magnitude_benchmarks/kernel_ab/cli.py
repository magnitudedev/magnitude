"""kernel-ab: compare two engine builds' kernel speed and output on this host.

    kernel-ab run --baseline origin/main --candidate working-tree [--models a,b] [--rounds 3]
        [--own-tuning]
    kernel-ab report RUN_DIRECTORY [--bar 0.03]
    kernel-ab build --baseline origin/main --candidate working-tree
    kernel-ab models
    kernel-ab fetch [--models a,b]

Builds, models and runs live under --root (default ~/magnitude-bench). Builds are named by commit
and, for the working tree, a digest of its uncommitted inference changes, so an unchanged side is
not rebuilt. The exit status is 0 only when every cell is within the bar, every logits comparison
passes and every invocation succeeded.
"""

import argparse
import json
import sys
from pathlib import Path

from . import builds, compare, runner
from . import models as model_set


def workspace() -> Path:
    """The `inference/` directory of this checkout."""
    return Path(__file__).resolve().parents[4]


def log(message: str) -> None:
    print(f"kernel-ab: {message}", file=sys.stderr, flush=True)


def report(directory: Path, bar: float) -> int:
    analysis = compare.analyze(directory, bar)
    text = compare.markdown(analysis)
    (directory / "report.md").write_text(text)
    (directory / "report.json").write_text(
        json.dumps(
            {key: value for key, value in analysis.items() if key != "run"}, indent=2, default=str
        )
    )
    print(text)
    return 0 if analysis["pass"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kernel-ab", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path.home() / "magnitude-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--baseline", default="origin/main")
    run.add_argument("--candidate", default=builds.WORKING_TREE)
    run.add_argument("--models", help="comma-separated names from `kernel-ab models`; default all")
    run.add_argument("--rounds", type=int, default=3, help="measured rounds per build and model")
    run.add_argument(
        "--own-tuning",
        action="store_true",
        help="also tune the candidate and measure each build on its own choice",
    )
    run.add_argument("--device", default="auto")
    run.add_argument("--kv-codec", default="affine-k8v4", choices=("affine-k8v4", "dense"))
    run.add_argument("--histories", default=",".join(map(str, runner.HISTORIES)))
    run.add_argument("--bar", type=float, default=compare.DEFAULT_BAR)
    show = commands.add_parser("report")
    show.add_argument("directory", type=Path)
    show.add_argument("--bar", type=float, default=compare.DEFAULT_BAR)
    prepare = commands.add_parser("build", help="build both sides and print their directories")
    prepare.add_argument("--baseline", default="origin/main")
    prepare.add_argument("--candidate", default=builds.WORKING_TREE)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--models")
    commands.add_parser("models")
    args = parser.parse_args(argv)
    root = args.root.expanduser()
    inference = workspace()
    if args.command == "models":
        for model in model_set.MODELS:
            print(f"{model.name:24} {model.covers}")
        return 0
    if args.command == "fetch":
        for model in model_set.select(args.models):
            print(model_set.fetch(inference, root / "models", model, log))
        return 0
    if args.command == "report":
        return report(args.directory, args.bar)
    repository = inference.parent
    if args.command == "build":
        for side, spec in (("baseline", args.baseline), ("candidate", args.candidate)):
            print(side, builds.build(repository, spec, root, log)[1])
        return 0
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    built = {
        side: builds.build(repository, spec, root, log)
        for side, spec in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    histories = tuple(int(value) for value in args.histories.split(",") if value)
    directory = runner.run(
        inference,
        root,
        built,
        model_set.select(args.models),
        args.own_tuning,
        args.rounds,
        args.device,
        args.kv_codec,
        histories,
        log,
    )
    log(f"results in {directory}")
    return report(directory, args.bar)


if __name__ == "__main__":
    raise SystemExit(main())
